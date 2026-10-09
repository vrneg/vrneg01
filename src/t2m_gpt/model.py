"""Stage-two transformer over motion tokens.

T2M-GPT conditions an autoregressive transformer on a text embedding placed at the first
sequence position.  Here the anchor label takes that position instead, which gives two
usable formulations of the same architecture:

* the discriminative head reads the sequence behind a ``[BOS]`` prefix and pools one
  binary logit, and
* the generative head learns ``p(tokens | class)`` and classifies by the log-likelihood
  ratio between the two class prefixes.

Both share one backbone, so unconditional next-token pretraining initializes either.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .config import GPTConfig


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional causal mask."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.causal = config.causal
        self.dropout_probability = config.dropout
        self.query_key_value = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = hidden.shape
        projected = self.query_key_value(hidden)
        queries, keys, values = projected.split(d_model, dim=2)
        shape = (batch_size, sequence_length, self.num_heads, self.head_dim)
        queries = queries.view(shape).transpose(1, 2)
        keys = keys.view(shape).transpose(1, 2)
        values = values.view(shape).transpose(1, 2)

        attended = functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=self.causal,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, d_model)
        )
        return self.residual_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = SelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class MotionTokenGPT(nn.Module):
    """Transformer over discrete motion tokens with classification and LM heads."""

    def __init__(
        self,
        num_codes: int,
        num_tokens: int,
        config: GPTConfig,
    ) -> None:
        super().__init__()
        config.validate()
        if num_codes < 2:
            raise ValueError("num_codes must be at least 2")
        if num_tokens < 1:
            raise ValueError("num_tokens must be at least 1")

        self.config = config
        self.num_codes = num_codes
        self.num_tokens = num_tokens
        # The generative head needs one prefix per class; every other mode conditions on
        # a single [BOS] prefix.
        self.num_prefix_tokens = 2 if config.head == "generative" else 1

        self.token_embedding = nn.Embedding(num_codes, config.d_model)
        self.prefix_embedding = nn.Embedding(self.num_prefix_tokens, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_tokens + 1, config.d_model)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(config.d_model, num_codes, bias=False)
        self.classifier_head = nn.Sequential(
            nn.Linear(config.d_model, config.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )
        self.apply(self._initialize_weights)

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
        """Parameters shared by pretraining and fine-tuning."""

        head_parameters = {id(parameter) for parameter in self.classifier_head.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in head_parameters
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier_head.parameters())

    def synchronize_prefix_embeddings(self) -> None:
        """Copy the pretrained prefix row into every class prefix.

        Unconditional pretraining only ever sees prefix 0.  Broadcasting it gives the
        generative head two identical, already-trained starting points instead of one
        trained and one random.
        """

        with torch.no_grad():
            self.prefix_embedding.weight.copy_(
                self.prefix_embedding.weight[0].unsqueeze(0).expand_as(
                    self.prefix_embedding.weight
                )
            )

    def corrupt(self, indices: torch.Tensor, corruption_rate: float) -> torch.Tensor:
        """Replace a fraction of tokens with uniformly random codes.

        This is T2M-GPT's corrupted-sequence strategy: teacher forcing on clean codes
        leaves the model unprepared for the imperfect prefixes it meets at inference.
        """

        if corruption_rate <= 0.0:
            return indices
        replace = torch.rand(indices.shape, device=indices.device) < corruption_rate
        random_indices = torch.randint(
            self.num_codes, indices.shape, device=indices.device
        )
        return torch.where(replace, random_indices, indices)

    def _hidden_states(
        self,
        indices: torch.Tensor,
        prefix_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [batch, tokens]")
        if indices.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens per window, got {indices.shape[1]}"
            )
        if indices.numel() and (
            int(indices.max()) >= self.num_codes or int(indices.min()) < 0
        ):
            raise ValueError("token indices fall outside the codebook")

        batch_size = indices.shape[0]
        if prefix_ids is None:
            prefix_ids = indices.new_zeros(batch_size)
        prefix = self.prefix_embedding(prefix_ids).unsqueeze(1)
        tokens = self.token_embedding(indices)
        hidden = torch.cat((prefix, tokens), dim=1) + self.position_embedding
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward_language_model(
        self,
        indices: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        """Return ``[batch, tokens, num_codes]`` next-token logits.

        Position ``i`` of the returned tensor predicts ``indices[:, i]`` from the prefix
        and the tokens before it, which requires the causal mask.
        """

        if not self.config.causal:
            raise RuntimeError("next-token prediction requires causal attention")
        model_input = self.corrupt(indices, corruption_rate) if self.training else indices
        hidden = self._hidden_states(model_input, prefix_ids)
        return self.language_model_head(hidden[:, :-1])

    def language_model_loss(
        self,
        indices: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        logits = self.forward_language_model(
            indices, prefix_ids=prefix_ids, corruption_rate=corruption_rate
        )
        return functional.cross_entropy(
            logits.reshape(-1, self.num_codes), indices.reshape(-1)
        )

    def conditional_log_likelihood(
        self,
        indices: torch.Tensor,
        prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``log p(tokens | prefix)`` summed over the window."""

        logits = self.forward_language_model(indices, prefix_ids=prefix_ids)
        log_probabilities = functional.log_softmax(logits, dim=-1)
        gathered = log_probabilities.gather(2, indices.unsqueeze(2)).squeeze(2)
        return gathered.sum(dim=1)

    def log_likelihood_ratio(self, indices: torch.Tensor) -> torch.Tensor:
        """Score windows by ``log p(tokens | positive) - log p(tokens | negative)``."""

        if self.num_prefix_tokens < 2:
            raise RuntimeError("the log-likelihood ratio requires class prefixes")
        positive = self.conditional_log_likelihood(
            indices, indices.new_ones(indices.shape[0])
        )
        negative = self.conditional_log_likelihood(
            indices, indices.new_zeros(indices.shape[0])
        )
        return positive - negative

    def forward_classifier(
        self,
        indices: torch.Tensor,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        """Return one binary logit per window."""

        model_input = self.corrupt(indices, corruption_rate) if self.training else indices
        hidden = self._hidden_states(model_input, prefix_ids=None)
        if self.config.pooling == "last":
            pooled = hidden[:, -1]
        else:
            pooled = hidden.mean(dim=1)
        return self.classifier_head(pooled).squeeze(-1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the decision score for the configured head."""

        if self.config.head == "generative":
            return self.log_likelihood_ratio(indices)
        return self.forward_classifier(indices)


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def uniform_token_log_likelihood(num_tokens: int, num_codes: int) -> float:
    """Log-likelihood a uniform code distribution assigns to any window.

    Useful as the reference point when reading generative-head scores.
    """

    return -float(num_tokens) * math.log(float(num_codes))
