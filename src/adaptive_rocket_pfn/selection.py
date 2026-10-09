"""Nested stability selection, structured budgets, and redundancy suppression."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import AdaptiveRocketPFNConfig
from .features import EXPERT_NAMES


@dataclass(frozen=True, slots=True)
class ExpertSelection:
    """Selected local column indices and diagnostics for one semantic expert."""

    expert: str
    family_order: tuple[str, ...]
    family_indices: dict[str, np.ndarray]
    family_budgets: dict[str, int]
    family_utilities: dict[str, float]

    @property
    def num_features(self) -> int:
        return int(sum(indices.size for indices in self.family_indices.values()))


def _safe_f_scores(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        scores, _ = f_classif(values, labels)
    finite_scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=0.0,
    )
    # ANOVA F is nonnegative analytically, but nearly constant float features can
    # produce tiny negative values through cancellation in sklearn's implementation.
    return np.maximum(finite_scores, 0.0)


def _allocate_family_budgets(
    utilities: dict[str, float],
    capacities: dict[str, int],
    config: AdaptiveRocketPFNConfig,
) -> dict[str, int]:
    ranked = sorted(utilities, key=lambda name: (-utilities[name], name))
    maximum_active = min(config.active_families_per_expert, len(ranked))
    affordable_active = max(
        1,
        config.selected_features_per_expert
        // config.minimum_features_per_active_family,
    )
    active = ranked[: min(maximum_active, affordable_active)]
    budgets = {
        name: min(capacities[name], config.minimum_features_per_active_family)
        for name in active
    }
    remaining = config.selected_features_per_expert - sum(budgets.values())
    if remaining <= 0:
        return budgets

    utility_values = np.asarray([utilities[name] for name in active], dtype=np.float64)
    utility_values = utility_values - np.max(utility_values)
    weights = np.exp(utility_values / config.allocation_temperature)
    weights = weights / max(float(weights.sum()), np.finfo(np.float64).eps)

    # Weighted fair allocation is deterministic, respects capacity, and avoids
    # handing the entire residual budget to a single family after integer rounding.
    extra = np.zeros(len(active), dtype=np.int64)
    for _ in range(remaining):
        available = np.asarray(
            [budgets[name] + extra[index] < capacities[name] for index, name in enumerate(active)]
        )
        if not np.any(available):
            break
        priorities = np.where(available, weights / (extra + 1), -np.inf)
        winner = int(np.argmax(priorities))
        extra[winner] += 1
    for index, name in enumerate(active):
        budgets[name] += int(extra[index])
    return budgets


def _redundancy_suppressed_indices(
    values: np.ndarray,
    ranked_indices: np.ndarray,
    budget: int,
    subset_indices: np.ndarray,
    config: AdaptiveRocketPFNConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if budget < 1 or ranked_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    pool_size = min(
        ranked_indices.size,
        max(budget, budget * config.selection_pool_multiplier),
    )
    pool = np.asarray(ranked_indices[:pool_size], dtype=np.int64)
    rows = np.asarray(subset_indices, dtype=np.int64)
    if rows.size > config.redundancy_sample_count:
        rows = np.sort(
            rng.choice(rows, size=config.redundancy_sample_count, replace=False)
        )
    matrix = np.asarray(values[np.ix_(rows, pool)], dtype=np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(matrix, axis=0)
    nonconstant = norms > np.finfo(np.float64).eps
    matrix[:, nonconstant] /= norms[nonconstant]

    selected_positions: list[int] = []
    for position in range(pool.size):
        if not nonconstant[position]:
            continue
        if selected_positions:
            correlations = np.abs(matrix[:, selected_positions].T @ matrix[:, position])
            if np.max(correlations) >= config.redundancy_correlation_threshold:
                continue
        selected_positions.append(position)
        if len(selected_positions) == budget:
            break
    return pool[np.asarray(selected_positions, dtype=np.int64)]


def select_expert_features(
    candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    subset_indices: np.ndarray,
    expert: str,
    family_names: tuple[str, ...],
    config: AdaptiveRocketPFNConfig,
    seed: int,
) -> ExpertSelection:
    """Select features using repeated folds contained within ``subset_indices``."""

    if expert not in EXPERT_NAMES:
        raise ValueError(f"Unknown semantic expert: {expert!r}")
    subset = np.asarray(subset_indices, dtype=np.int64)
    subset_labels = np.asarray(labels)[subset]
    classes, counts = np.unique(subset_labels, return_counts=True)
    if classes.size != 2:
        raise ValueError("Adaptive selection requires both classes")
    n_splits = min(config.selection_inner_folds, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough examples per class for stable feature selection")

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=config.selection_repeats,
        random_state=seed,
    )
    split_train_indices = [
        subset[local_train]
        for local_train, _ in splitter.split(np.zeros(subset.size), subset_labels)
    ]
    num_votes = len(split_train_indices)
    ranking_scores: dict[str, np.ndarray] = {}
    utilities: dict[str, float] = {}
    capacities: dict[str, int] = {}

    for family_name in family_names:
        values = np.asarray(candidates[family_name], dtype=np.float32)
        capacities[family_name] = int(values.shape[1])
        frequency = np.zeros(values.shape[1], dtype=np.float64)
        score_sum = np.zeros(values.shape[1], dtype=np.float64)
        vote_utilities: list[float] = []
        screen_count = min(
            values.shape[1],
            max(
                32,
                config.selected_features_per_expert
                * config.selection_pool_multiplier
                // max(1, config.active_families_per_expert),
            ),
        )
        for train_indices in split_train_indices:
            scores = _safe_f_scores(values[train_indices], labels[train_indices])
            ranked = np.argsort(-scores, kind="stable")
            frequency[ranked[:screen_count]] += 1.0
            score_sum += np.log1p(scores)
            top_count = min(16, scores.size)
            vote_utilities.append(
                float(np.mean(np.partition(np.log1p(scores), -top_count)[-top_count:]))
            )
        mean_scores = score_sum / num_votes
        scale = max(float(np.max(mean_scores)), np.finfo(np.float64).eps)
        ranking_scores[family_name] = frequency / num_votes + 0.05 * mean_scores / scale
        utilities[family_name] = float(np.median(vote_utilities))

    family_budgets = _allocate_family_budgets(utilities, capacities, config)
    active_order = tuple(
        sorted(family_budgets, key=lambda name: (-utilities[name], name))
    )
    rng = np.random.default_rng(seed)
    selected: dict[str, np.ndarray] = {}
    for family_name in active_order:
        ranked = np.argsort(-ranking_scores[family_name], kind="stable")
        selected[family_name] = _redundancy_suppressed_indices(
            candidates[family_name],
            ranked,
            family_budgets[family_name],
            subset,
            config,
            rng,
        )
    return ExpertSelection(
        expert=expert,
        family_order=active_order,
        family_indices=selected,
        family_budgets=family_budgets,
        family_utilities=utilities,
    )


def select_all_experts(
    candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    subset_indices: np.ndarray,
    expert_families: dict[str, tuple[str, ...]],
    config: AdaptiveRocketPFNConfig,
    seed: int,
) -> dict[str, ExpertSelection]:
    return {
        expert: select_expert_features(
            candidates,
            labels,
            subset_indices,
            expert,
            expert_families[expert],
            config,
            seed + index,
        )
        for index, expert in enumerate(EXPERT_NAMES)
    }


def assemble_expert_matrix(
    candidates: dict[str, np.ndarray],
    selection: ExpertSelection,
    rows: np.ndarray | slice | None = None,
) -> np.ndarray:
    row_selector = slice(None) if rows is None else rows
    blocks = [
        np.asarray(candidates[name][row_selector][:, selection.family_indices[name]])
        for name in selection.family_order
        if selection.family_indices[name].size
    ]
    if not blocks:
        raise RuntimeError(f"No features survived for expert {selection.expert}")
    return np.ascontiguousarray(np.concatenate(blocks, axis=1), dtype=np.float32)


__all__ = [
    "ExpertSelection",
    "assemble_expert_matrix",
    "select_all_experts",
    "select_expert_features",
]
