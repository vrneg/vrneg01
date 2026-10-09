"""Linear-chain CRF over BIO tags.

A per-frame softmax treats each frame independently, which lets it emit sequences BIO
forbids (``O`` followed by ``I-CUE``) and, more importantly, ignores the strong temporal
smoothness of the target: negation spans are contiguous runs of several frames, never
isolated frames. A CRF models the transitions explicitly, so the decoded output is a set
of well-formed spans rather than a per-frame vote that needs cleaning up afterwards.

The implementation is the standard forward algorithm for the partition function plus
Viterbi for decoding, with an option to hard-constrain transitions that BIO disallows.
"""

from __future__ import annotations

import torch
from torch import nn

from .labels import TAG_NAMES

IMPOSSIBLE = -1.0e4


def bio_transition_mask(tag_names: tuple[str, ...] = TAG_NAMES) -> torch.Tensor:
    """``[num_tags, num_tags]`` boolean mask of transitions BIO permits.

    ``I-X`` may only follow ``B-X`` or ``I-X``. Everything else is free.
    """

    size = len(tag_names)
    allowed = torch.ones(size, size, dtype=torch.bool)
    for target_index, target in enumerate(tag_names):
        target_prefix, _, target_type = target.partition("-")
        if target_prefix != "I":
            continue
        for source_index, source in enumerate(tag_names):
            source_prefix, _, source_type = source.partition("-")
            if not (source_prefix in {"B", "I"} and source_type == target_type):
                allowed[source_index, target_index] = False
    return allowed


def bio_start_mask(tag_names: tuple[str, ...] = TAG_NAMES) -> torch.Tensor:
    """Boolean mask of tags a sequence may begin with (no ``I-X`` opening a sequence)."""

    return torch.tensor(
        [not name.startswith("I-") for name in tag_names], dtype=torch.bool
    )


class LinearChainCRF(nn.Module):
    """Conditional random field over a fixed tag set.

    ``constrain_transitions`` freezes BIO-illegal transitions at a large negative
    constant, so they are neither learned nor decoded. The constant is finite rather than
    ``-inf`` to keep the forward algorithm's log-sum-exp free of ``nan``.
    """

    def __init__(
        self,
        num_tags: int,
        constrain_transitions: bool = True,
        tag_names: tuple[str, ...] = TAG_NAMES,
    ) -> None:
        super().__init__()
        if num_tags != len(tag_names):
            raise ValueError(
                f"num_tags={num_tags} does not match {len(tag_names)} tag names"
            )
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        if constrain_transitions:
            self.register_buffer("transition_mask", bio_transition_mask(tag_names))
            self.register_buffer("start_mask", bio_start_mask(tag_names))
        else:
            self.register_buffer(
                "transition_mask", torch.ones(num_tags, num_tags, dtype=torch.bool)
            )
            self.register_buffer("start_mask", torch.ones(num_tags, dtype=torch.bool))

    def _constrained_transitions(self) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = self.transitions.masked_fill(~self.transition_mask, IMPOSSIBLE)
        start = self.start_transitions.masked_fill(~self.start_mask, IMPOSSIBLE)
        return transitions, start

    def _validate(self, emissions: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if emissions.dim() != 3:
            raise ValueError("emissions must have shape [batch, frames, tags]")
        if emissions.shape[2] != self.num_tags:
            raise ValueError(
                f"emissions has {emissions.shape[2]} tags; expected {self.num_tags}"
            )
        if mask is None:
            return emissions.new_ones(emissions.shape[:2], dtype=torch.bool)
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask must have shape [batch, frames]")
        return mask.bool()

    def _sequence_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        transitions, start = self._constrained_transitions()
        batch, frames, _ = emissions.shape
        float_mask = mask.to(emissions.dtype)

        score = start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for step in range(1, frames):
            step_score = (
                transitions[tags[:, step - 1], tags[:, step]]
                + emissions[:, step].gather(1, tags[:, step : step + 1]).squeeze(1)
            )
            score = score + step_score * float_mask[:, step]

        lengths = mask.sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0)
        last_tags = tags.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _log_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        transitions, start = self._constrained_transitions()
        frames = emissions.shape[1]

        alpha = start.unsqueeze(0) + emissions[:, 0]
        for step in range(1, frames):
            candidate = (
                alpha.unsqueeze(2) + transitions.unsqueeze(0) + emissions[:, step].unsqueeze(1)
            )
            updated = torch.logsumexp(candidate, dim=1)
            keep = mask[:, step].unsqueeze(1)
            alpha = torch.where(keep, updated, alpha)
        return torch.logsumexp(alpha + self.end_transitions.unsqueeze(0), dim=1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Negative log-likelihood of ``tags`` under the emissions."""

        mask = self._validate(emissions, mask)
        if tags.shape != emissions.shape[:2]:
            raise ValueError("tags must have shape [batch, frames]")
        negative_log_likelihood = self._log_partition(emissions, mask) - self._sequence_score(
            emissions, tags, mask
        )
        if reduction == "none":
            return negative_log_likelihood
        if reduction == "sum":
            return negative_log_likelihood.sum()
        if reduction == "mean":
            return negative_log_likelihood.mean()
        if reduction == "token_mean":
            return negative_log_likelihood.sum() / mask.to(emissions.dtype).sum()
        raise ValueError(f"Unknown reduction: {reduction!r}")

    def decode(
        self, emissions: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Viterbi decode, returning ``[batch, frames]`` tag ids."""

        mask = self._validate(emissions, mask)
        transitions, start = self._constrained_transitions()
        batch, frames, num_tags = emissions.shape

        score = start.unsqueeze(0) + emissions[:, 0]
        history: list[torch.Tensor] = []
        for step in range(1, frames):
            candidate = score.unsqueeze(2) + transitions.unsqueeze(0)
            best_score, best_source = candidate.max(dim=1)
            updated = best_score + emissions[:, step]
            keep = mask[:, step].unsqueeze(1)
            score = torch.where(keep, updated, score)
            history.append(best_source)

        score = score + self.end_transitions.unsqueeze(0)
        best_last = score.argmax(dim=1)

        lengths = mask.sum(dim=1)
        paths = torch.zeros(batch, frames, dtype=torch.long, device=emissions.device)
        for index in range(batch):
            length = int(lengths[index].item())
            if length == 0:
                continue
            tag = int(best_last[index].item())
            path = [tag]
            for step in range(length - 1, 0, -1):
                tag = int(history[step - 1][index, tag].item())
                path.append(tag)
            path.reverse()
            paths[index, :length] = torch.tensor(
                path, dtype=torch.long, device=emissions.device
            )
        return paths

    def marginal_positive_probability(
        self,
        emissions: torch.Tensor,
        positive_tags: tuple[int, ...],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-frame probability that a frame carries one of ``positive_tags``.

        Computed with forward-backward rather than from the Viterbi path, because a
        threshold-free score is what an AUROC comparison against the window-level models
        needs.
        """

        mask = self._validate(emissions, mask)
        transitions, start = self._constrained_transitions()
        batch, frames, num_tags = emissions.shape

        alpha = torch.empty(batch, frames, num_tags, device=emissions.device, dtype=emissions.dtype)
        alpha[:, 0] = start.unsqueeze(0) + emissions[:, 0]
        for step in range(1, frames):
            candidate = (
                alpha[:, step - 1].unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, step].unsqueeze(1)
            )
            alpha[:, step] = torch.logsumexp(candidate, dim=1)

        beta = torch.empty_like(alpha)
        beta[:, frames - 1] = self.end_transitions.unsqueeze(0)
        for step in range(frames - 2, -1, -1):
            candidate = (
                transitions.unsqueeze(0)
                + emissions[:, step + 1].unsqueeze(1)
                + beta[:, step + 1].unsqueeze(1)
            )
            beta[:, step] = torch.logsumexp(candidate, dim=2)

        log_marginal = alpha + beta
        log_marginal = log_marginal - torch.logsumexp(log_marginal, dim=2, keepdim=True)
        marginal = log_marginal.exp()
        selected = marginal[:, :, list(positive_tags)].sum(dim=2)
        return selected * mask.to(selected.dtype)


__all__ = [
    "IMPOSSIBLE",
    "LinearChainCRF",
    "bio_start_mask",
    "bio_transition_mask",
]
