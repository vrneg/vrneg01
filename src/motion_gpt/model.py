"""Encoder-decoder transformer over the unified motion/text vocabulary.

MotionGPT uses a T5-style sequence-to-sequence model rather than T2M-GPT's decoder-only
stack: a bidirectional encoder reads the instruction and motion, and an autoregressive
decoder writes the answer.  That shape is what lets one model serve span denoising, motion
prediction, in-betweening, and motion-to-text at once.

Relative position buckets are replaced by learned absolute positions.  With windows of
roughly ten tokens the bucketing has nothing to generalize over, and the learned table is
smaller and simpler.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .config import MotionLanguageConfig
from .tasks import LABEL_IGNORE_INDEX
from .vocabulary import MotionLanguageVocabulary


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention over explicit boolean masks."""

    def __init__(self, config: MotionLanguageConfig) -> None:
        super().__init__()
        self.num_heads = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.dropout_probability = config.dropout
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = tensor.shape
        return tensor.view(batch_size, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``attention_mask`` is ``[batch, 1, queries, keys]`` and True means visible."""

        batch_size, query_length, d_model = queries.shape
        attended = functional.scaled_dot_product_attention(
            self._split(self.query(queries)),
            self._split(self.key(keys)),
            self._split(self.value(keys)),
            attn_mask=attention_mask,
            dropout_p=self.dropout_probability if self.training else 0.0,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, query_length, d_model)
        )
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Sequential):
    def __init__(self, config: MotionLanguageConfig) -> None:
        super().__init__(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )


class EncoderBlock(nn.Module):
    """Pre-norm block with bidirectional self-attention."""

    def __init__(self, config: MotionLanguageConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = FeedForward(config)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        hidden = hidden + self.attention(normalized, normalized, attention_mask)
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class DecoderBlock(nn.Module):
    """Pre-norm block with causal self-attention and cross-attention to the encoder."""

    def __init__(self, config: MotionLanguageConfig) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention_norm = nn.LayerNorm(config.d_model)
        self.cross_attention = MultiHeadAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = FeedForward(config)

    def forward(
        self,
        hidden: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_attention_norm(hidden)
        hidden = hidden + self.self_attention(normalized, normalized, self_attention_mask)
        hidden = hidden + self.cross_attention(
            self.cross_attention_norm(hidden), memory, cross_attention_mask
        )
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class MotionLanguageModel(nn.Module):
    """Sequence-to-sequence model over motion codes and the small text vocabulary."""

    def __init__(
        self,
        vocabulary: MotionLanguageVocabulary,
        config: MotionLanguageConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.vocabulary = vocabulary
        self.config = config

        self.embedding = nn.Embedding(vocabulary.size, config.d_model)
        self.encoder_positions = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.d_model)
        )
        self.decoder_positions = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.d_model)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder_blocks = nn.ModuleList(
            EncoderBlock(config) for _ in range(config.num_encoder_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(config) for _ in range(config.num_decoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(config.d_model, vocabulary.size, bias=False)
        self.classifier_head = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.language_model_head.weight = self.embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Every parameter shared by pretraining and the supervised task."""

        head_ids = {id(parameter) for parameter in self.classifier_head.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in head_ids
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier_head.parameters())

    def _embed(
        self,
        token_ids: torch.Tensor,
        positions: nn.Parameter,
    ) -> torch.Tensor:
        length = token_ids.shape[1]
        if length > positions.shape[1]:
            raise ValueError(
                f"sequence length {length} exceeds max_sequence_length "
                f"{positions.shape[1]}"
            )
        if int(token_ids.max()) >= self.vocabulary.size or int(token_ids.min()) < 0:
            raise ValueError("token ids fall outside the vocabulary")
        return self.embedding_dropout(self.embedding(token_ids) + positions[:, :length])

    def encode(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return bidirectional encoder states for the instruction and motion."""

        hidden = self._embed(encoder_input, self.encoder_positions)
        attention_mask = encoder_mask[:, None, None, :]
        for block in self.encoder_blocks:
            hidden = block(hidden, attention_mask)
        return self.encoder_norm(hidden)

    def decode(
        self,
        decoder_input: torch.Tensor,
        decoder_mask: torch.Tensor,
        memory: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return decoder states under causal self-attention and encoder cross-attention."""

        length = decoder_input.shape[1]
        causal = torch.ones(
            (length, length), dtype=torch.bool, device=decoder_input.device
        ).tril()
        # Padded key positions are hidden; the causal diagonal keeps every query able to
        # attend to itself, so no row can be fully masked.
        self_attention_mask = causal[None, None, :, :] & decoder_mask[:, None, None, :]
        cross_attention_mask = encoder_mask[:, None, None, :]

        hidden = self._embed(decoder_input, self.decoder_positions)
        for block in self.decoder_blocks:
            hidden = block(hidden, memory, self_attention_mask, cross_attention_mask)
        return self.decoder_norm(hidden)

    def forward_seq2seq(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[batch, decoder_length, vocabulary]`` logits."""

        memory = self.encode(encoder_input, encoder_mask)
        hidden = self.decode(decoder_input, decoder_mask, memory, encoder_mask)
        return self.language_model_head(hidden)

    def seq2seq_loss(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward_seq2seq(
            encoder_input, encoder_mask, decoder_input, decoder_mask
        )
        return functional.cross_entropy(
            logits.reshape(-1, self.vocabulary.size),
            labels.reshape(-1),
            ignore_index=LABEL_IGNORE_INDEX,
        )

    def answer_logit(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the log-odds between the two answer words at the first decoder step.

        Because both answers are scored at the same position, the difference of their raw
        logits already equals the difference of their log-probabilities.
        """

        batch_size = encoder_input.shape[0]
        start = torch.full(
            (batch_size, 1),
            self.vocabulary.bos_id,
            dtype=torch.long,
            device=encoder_input.device,
        )
        decoder_mask = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=encoder_input.device
        )
        logits = self.forward_seq2seq(
            encoder_input, encoder_mask, start, decoder_mask
        )[:, 0]
        negative_id, positive_id = self.vocabulary.answer_ids[:2]
        return logits[:, positive_id] - logits[:, negative_id]

    def forward_classifier(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one binary logit from the mean-pooled encoder state."""

        memory = self.encode(encoder_input, encoder_mask)
        weights = encoder_mask.unsqueeze(-1).to(memory.dtype)
        pooled = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier_head(pooled).squeeze(-1)

    def forward(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the decision score for the configured head."""

        if self.config.head == "discriminative":
            return self.forward_classifier(encoder_input, encoder_mask)
        return self.answer_logit(encoder_input, encoder_mask)

    @torch.no_grad()
    def generate(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        max_new_tokens: int = 16,
    ) -> torch.Tensor:
        """Greedily decode an answer, for inspecting what the model writes."""

        was_training = self.training
        self.eval()
        try:
            batch_size = encoder_input.shape[0]
            memory = self.encode(encoder_input, encoder_mask)
            sequence = torch.full(
                (batch_size, 1),
                self.vocabulary.bos_id,
                dtype=torch.long,
                device=encoder_input.device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=encoder_input.device)
            for _ in range(max_new_tokens):
                decoder_mask = torch.ones_like(sequence, dtype=torch.bool)
                hidden = self.decode(sequence, decoder_mask, memory, encoder_mask)
                next_token = self.language_model_head(hidden[:, -1]).argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.vocabulary.pad_id),
                    next_token,
                )
                sequence = torch.cat((sequence, next_token.unsqueeze(1)), dim=1)
                finished = finished | (next_token == self.vocabulary.eos_id)
                if bool(finished.all()):
                    break
            return sequence[:, 1:]
        finally:
            self.train(was_training)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


__all__ = [
    "DecoderBlock",
    "EncoderBlock",
    "MotionLanguageModel",
    "MultiHeadAttention",
    "count_parameters",
]
