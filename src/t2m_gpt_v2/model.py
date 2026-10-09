"""Stage-two transformer over continuous motion latents.

A direct continuous-input counterpart of ``t2m_gpt.model.MotionTokenGPT``: the causal
transformer backbone (``TransformerBlock``/``SelfAttention``) is reused unchanged from
``t2m_gpt.model`` -- both operate purely on ``[batch, seq, d_model]`` hidden states and
have no dependency on how those hidden states were produced. The only architectural
difference is the input layer: a linear projection of each continuous latent vector
instead of a discrete embedding-table lookup, since there is no vocabulary to index into.

Only the discriminative head is supported (see ``t2m_gpt_v2.config.ExperimentConfig``'s
validation for why), so this class has one class prefix, not two, and no log-likelihood-
ratio machinery. Autoregressive pretraining has a direct continuous analogue: instead of
cross-entropy over a codebook, the model predicts the next latent vector by regression
(smooth-L1), which plays the same role of giving the classifier an initialization that
already models motion-latent structure.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

try:
    from t2m_gpt.config import GPTConfig
    from t2m_gpt.model import TransformerBlock, count_parameters
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "t2m_gpt":
        raise
    from ..t2m_gpt.config import GPTConfig
    from ..t2m_gpt.model import TransformerBlock, count_parameters


class MotionLatentGPT(nn.Module):
    """Transformer over continuous motion latents with a classification head."""

    def __init__(
        self,
        latent_dim: int,
        num_tokens: int,
        config: GPTConfig,
    ) -> None:
        super().__init__()
        config.validate()
        if config.head != "discriminative":
            raise ValueError(
                "MotionLatentGPT only supports gpt.head='discriminative'"
            )
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if num_tokens < 1:
            raise ValueError("num_tokens must be at least 1")

        self.config = config
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens

        self.latent_projection = nn.Linear(latent_dim, config.d_model)
        self.prefix_embedding = nn.Embedding(1, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_tokens + 1, config.d_model)
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.latent_prediction_head = nn.Linear(config.d_model, latent_dim)
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

    def corrupt(self, latents: torch.Tensor, corruption_rate: float) -> torch.Tensor:
        """Replace a fraction of latent positions with scaled Gaussian noise.

        Continuous analogue of ``MotionTokenGPT.corrupt``'s random-code substitution:
        teacher forcing on clean latents leaves the model unprepared for the imperfect
        prefixes it would meet if used generatively, so a fraction of positions are
        replaced with noise scaled to the batch's own latent magnitude.
        """

        if corruption_rate <= 0.0:
            return latents
        replace_mask = (
            torch.rand(latents.shape[:2], device=latents.device) < corruption_rate
        )
        noise_scale = latents.detach().std().clamp_min(1e-6)
        noise = torch.randn_like(latents) * noise_scale
        return torch.where(replace_mask.unsqueeze(-1), noise, latents)

    def _hidden_states(
        self,
        latents: torch.Tensor,
        prefix_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, tokens, latent_dim]")
        if latents.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens per window, got {latents.shape[1]}"
            )
        if latents.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim {self.latent_dim}, got {latents.shape[2]}"
            )

        batch_size = latents.shape[0]
        if prefix_ids is None:
            prefix_ids = latents.new_zeros(batch_size, dtype=torch.long)
        prefix = self.prefix_embedding(prefix_ids).unsqueeze(1)
        projected = self.latent_projection(latents)
        hidden = torch.cat((prefix, projected), dim=1) + self.position_embedding
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def forward_next_latent(
        self,
        latents: torch.Tensor,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        """Return ``[batch, tokens, latent_dim]`` next-latent predictions.

        Position ``i`` predicts ``latents[:, i]`` from the prefix and the latents
        before it, which requires the causal mask -- the continuous analogue of
        ``MotionTokenGPT.forward_language_model``.
        """

        if not self.config.causal:
            raise RuntimeError("next-latent prediction requires causal attention")
        model_input = self.corrupt(latents, corruption_rate) if self.training else latents
        hidden = self._hidden_states(model_input, prefix_ids=None)
        return self.latent_prediction_head(hidden[:, :-1])

    def next_latent_loss(
        self,
        latents: torch.Tensor,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        predicted = self.forward_next_latent(latents, corruption_rate=corruption_rate)
        return functional.smooth_l1_loss(predicted, latents)

    def forward_classifier(
        self,
        latents: torch.Tensor,
        corruption_rate: float = 0.0,
    ) -> torch.Tensor:
        """Return one binary logit per window."""

        model_input = self.corrupt(latents, corruption_rate) if self.training else latents
        hidden = self._hidden_states(model_input, prefix_ids=None)
        if self.config.pooling == "last":
            pooled = hidden[:, -1]
        else:
            pooled = hidden.mean(dim=1)
        return self.classifier_head(pooled).squeeze(-1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.forward_classifier(latents)


__all__ = ["MotionLatentGPT", "count_parameters"]
