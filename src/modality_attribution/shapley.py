"""Reproducible sampled Shapley values over modality coalitions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


CoalitionPredictor = Callable[[tuple[str, ...]], np.ndarray]


@dataclass(frozen=True, slots=True)
class ShapleyResult:
    modality_names: tuple[str, ...]
    full_logits: np.ndarray
    empty_logits: np.ndarray
    values: np.ndarray
    standard_errors: np.ndarray
    num_orderings: int
    num_coalitions_evaluated: int

    @property
    def reconstruction_error(self) -> np.ndarray:
        return self.values.sum(axis=1) - (self.full_logits - self.empty_logits)


def _orderings(
    num_modalities: int, num_orderings: int, seed: int
) -> list[np.ndarray]:
    if num_orderings < 2 or num_orderings % 2:
        raise ValueError("num_orderings must be a positive even number of at least 2")
    generator = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    for _ in range(num_orderings // 2):
        ordering = generator.permutation(num_modalities)
        result.extend((ordering, ordering[::-1]))
    return result


def sampled_modality_shapley(
    modality_names: Sequence[str],
    predict: CoalitionPredictor,
    *,
    num_orderings: int = 32,
    seed: int = 42,
) -> ShapleyResult:
    """Allocate the full-minus-empty logit using antithetic random orderings."""

    names = tuple(modality_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("modality_names must be non-empty and unique")
    orderings = _orderings(len(names), num_orderings, seed)
    cache: dict[tuple[str, ...], np.ndarray] = {}

    def evaluate(indices: frozenset[int]) -> np.ndarray:
        coalition = tuple(name for index, name in enumerate(names) if index in indices)
        if coalition not in cache:
            scores = np.asarray(predict(coalition), dtype=np.float64)
            if scores.ndim != 1:
                raise ValueError("Coalition predictor must return one logit per sample")
            if cache and scores.shape != next(iter(cache.values())).shape:
                raise ValueError("Coalition predictions returned inconsistent sample counts")
            cache[coalition] = scores
        return cache[coalition]

    empty = evaluate(frozenset())
    contributions = np.empty(
        (num_orderings, empty.shape[0], len(names)), dtype=np.float64
    )
    for ordering_index, ordering in enumerate(orderings):
        coalition: frozenset[int] = frozenset()
        previous = empty
        for modality_index in ordering:
            expanded = coalition | {int(modality_index)}
            current = evaluate(expanded)
            contributions[ordering_index, :, modality_index] = current - previous
            coalition = expanded
            previous = current

    values = contributions.mean(axis=0)
    # Antithetic forward/reverse paths form one sampling unit.  Estimate Monte
    # Carlo error from pair means rather than treating negatively correlated paths
    # as independent observations.
    paired = contributions.reshape(
        num_orderings // 2, 2, empty.shape[0], len(names)
    ).mean(axis=1)
    standard_errors = (
        paired.std(axis=0, ddof=1) / np.sqrt(paired.shape[0])
        if paired.shape[0] > 1
        else np.zeros_like(values)
    )
    full = evaluate(frozenset(range(len(names))))
    return ShapleyResult(
        modality_names=names,
        full_logits=full,
        empty_logits=empty,
        values=values,
        standard_errors=standard_errors,
        num_orderings=num_orderings,
        num_coalitions_evaluated=len(cache),
    )
