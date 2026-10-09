"""Small, dependency-free helpers for cross-validation orchestration."""

from __future__ import annotations


def resolve_fold_worker_count(
    pending_fold_count: int,
    parallel_folds: bool,
    max_parallel_folds: int | None,
) -> int:
    """Return the number of outer-fold processes to launch concurrently.

    ``max_parallel_folds=None`` deliberately means no outer-process cap. The
    number of workers is still bounded by the number of pending folds.
    """

    if isinstance(pending_fold_count, bool) or not isinstance(pending_fold_count, int):
        raise TypeError("pending_fold_count must be an integer")
    if pending_fold_count < 0:
        raise ValueError("pending_fold_count must be non-negative")
    if not isinstance(parallel_folds, bool):
        raise TypeError("parallel_folds must be a bool")
    if max_parallel_folds is not None:
        if isinstance(max_parallel_folds, bool) or not isinstance(
            max_parallel_folds, int
        ):
            raise TypeError("max_parallel_folds must be an integer or None")
        if max_parallel_folds < 1:
            raise ValueError("max_parallel_folds must be at least 1 or None")

    if pending_fold_count == 0:
        return 0
    if not parallel_folds:
        return 1
    if max_parallel_folds is None:
        return pending_fold_count
    return min(max_parallel_folds, pending_fold_count)
