"""Reusable sweep controller: any model x any (target, window) dataset already exported
to the Hub, without duplicating each model's tuned hyperparameters.

Every ``train_main_<model>.py`` script in this directory is an "edit this file" entry
point (this repo's convention: plain dataclass configs, no CLI) whose dataset selection
-- ``DATASET_PREFIX`` and, for the Hub-backed models, the window/grid constants -- are
plain module-level globals read at call time by that script's own
``build_experiment_config``/``train_cross_validation``. This controller imports those
modules and rebinds those globals for the duration of one run, then restores them --
the same monkeypatch-and-restore pattern ``src/minirocket_representation.py`` already
uses to swap in representation-aware data loading. Every other tuned hyperparameter
(ridge grid, VQVAE/transformer architecture, training schedule, ...) stays exactly what
that script's author set; this file only ever changes which dataset it points at, and
optionally the mirror-augmentation toggle and (for the two motion-token models) a
capacity step.

Usage::

    from run_dataset_matrix import DatasetSpec, run_cross_validation

    run_cross_validation("minirocket", DatasetSpec("cue", 1000))
    run_cross_validation("motion_gpt", DatasetSpec("cue", 1000), mirror_augment_train=True)
    run_cross_validation("t2m_gpt", DatasetSpec("cueandscope", 500), capacity="large")

Or from the command line::

    python src/run_dataset_matrix.py minirocket cue 1000
    python src/run_dataset_matrix.py motion_gpt cue 1000 --mirror
    python src/run_dataset_matrix.py t2m_gpt cueandscope 500 --capacity large
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "data/trainsets"
DATASET_NAMESPACE = "VR-Faces-Neg"

# Each train_main_*.py is written to be run as ``python src/train_main_x.py``, so its
# own imports (``from minirocket import ...``) assume src/ is on sys.path. Guarantee
# that here, so importing this controller as ``src.run_dataset_matrix`` (from a test, or
# from another module) resolves those entry points the same way running them directly
# would.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

Target = Literal["cue", "scope", "cueandscope"]
ModelName = Literal["minirocket", "t2m_gpt", "motion_gpt", "frame_tagger"]

#: Models that read data/trainsets/<prefix>_fold-<n> directly (src/minirocket/data.py's
#: load_fold is a plain load_from_disk with no Hub fallback) and so need
#: ensure_local_dataset before training. The other three load a Hub repository id
#: straight through the datasets library's own loader/cache -- no local copy needed.
LOCAL_PATH_MODELS: frozenset[str] = frozenset({"minirocket"})

#: Models whose DataConfig is ``t2m_gpt.config.DataConfig``, the only one carrying
#: ``mirror_augment_train``. MiniRocket builds ``minirocket.config.DataConfig`` and loads
#: through its own untouched pipeline, so mirroring is not reachable from this controller
#: for it (``src/minirocket_representation.py`` is the bridge that would do it).
MIRROR_CAPABLE_MODELS: frozenset[str] = frozenset({"t2m_gpt", "motion_gpt", "frame_tagger"})

#: ``train_main_frame_tagger.train_cross_validation`` takes only ``folds`` -- it has no
#: resume logic of its own -- while the other three also accept
#: ``resume_completed_folds``. Passing an unsupported keyword would be a TypeError.
RESUME_CAPABLE_MODELS: frozenset[str] = frozenset({"minirocket", "t2m_gpt", "motion_gpt"})

#: One frame every ~31ms (32 points over the +-0.5s window-500 grid), preserved when
#: scaling to other window sizes so token/frame rates stay comparable across datasets
#: (see the "keep the token rate constant" note in train_main_t2m_gpt.py).
_POINTS_PER_SECOND = 32.0


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """One (target, window) dataset exported to the Hub (VR-Faces-Neg org, 2026-08-10).

    Repository ids are ``target-{target}_window-{window_ms}_splits-{splits}[_fold-N]``.
    ``window_ms`` is the half-window used on both sides of the anchor word, matching
    FINDINGS.md's "window-500" = +-0.5s convention.
    """

    target: Target
    window_ms: int
    splits: int = 10

    @property
    def prefix(self) -> str:
        return f"target-{self.target}_window-{self.window_ms}_splits-{self.splits}"

    def hub_repo(self, fold: int) -> str:
        return f"{DATASET_NAMESPACE}/{self.prefix}_fold-{fold}"

    def local_path(self, fold: int) -> Path:
        return DATASET_ROOT / f"{self.prefix}_fold-{fold}"

    @property
    def window_start_seconds(self) -> float:
        return -self.window_ms / 1000.0

    @property
    def window_end_seconds(self) -> float:
        return self.window_ms / 1000.0

    @property
    def num_time_points(self) -> int:
        span_seconds = 2.0 * self.window_ms / 1000.0
        return round(span_seconds * _POINTS_PER_SECOND)


AVAILABLE_DATASETS: tuple[DatasetSpec, ...] = tuple(
    DatasetSpec(target=target, window_ms=window_ms)
    for target in ("cue", "scope", "cueandscope")
    for window_ms in (500, 1000)
)


def ensure_local_fold(spec: DatasetSpec, fold: int) -> Path:
    """Materialize one fold as a local ``save_to_disk`` directory, if not cached yet."""

    path = spec.local_path(fold)
    if (path / "dataset_dict.json").exists():
        return path
    dataset = load_dataset(spec.hub_repo(fold))
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    return path


def ensure_local_dataset(spec: DatasetSpec) -> None:
    for fold in range(spec.splits):
        ensure_local_fold(spec, fold)


@dataclass(frozen=True, slots=True)
class GPTCapacity:
    """Sizing for the two motion-token models' VQ-VAE and stage-two transformer.

    ``"small"`` is exactly what train_main_t2m_gpt.py / train_main_motion_gpt.py already
    use (the size that was found to generalize on window-500's ~740 training windows);
    passing it is a byte-for-byte no-op. ``"large"`` is a deliberately modest step up to
    *screen* now that mirror augmentation and/or a bigger dataset (cueandscope, more
    folds' worth of pretraining data) give the model more to learn from -- not a
    replacement for validating it the way this project validates everything else: run a
    quick fold or two, and only trust it at full 10-fold with paired significance testing
    against the "small" baseline.
    """

    vqvae_num_codes: int
    vqvae_code_dim: int
    vqvae_width: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int


GPT_CAPACITY_VARIANTS: dict[str, GPTCapacity] = {
    "small": GPTCapacity(
        vqvae_num_codes=128, vqvae_code_dim=64, vqvae_width=128,
        d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
    ),
    "large": GPTCapacity(
        vqvae_num_codes=256, vqvae_code_dim=96, vqvae_width=192,
        d_model=96, nhead=8, num_layers=3, dim_feedforward=384,
    ),
}


@contextmanager
def _rebind(module: Any, **overrides: Any):
    """Set module-level globals for the duration of the block, then restore them."""

    missing = [name for name in overrides if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{module.__name__} has no attribute(s): {missing}")
    original = {name: getattr(module, name) for name in overrides}
    for name, value in overrides.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


@contextmanager
def _mirror_augmented(module: Any, enabled: bool):
    """Wrap ``module.build_experiment_config`` to set ``data.mirror_augment_train``.

    There is no existing module-level global for this (mirror augmentation postdates
    every train_main_*.py script), so this patches the function itself instead of a
    constant -- same monkeypatch-and-restore shape, one level deeper.
    """

    if not enabled:
        yield
        return

    original_build = module.build_experiment_config

    def patched_build(fold: int, **kwargs: Any):
        config = original_build(fold, **kwargs)
        return dataclasses.replace(
            config, data=dataclasses.replace(config.data, mirror_augment_train=True)
        )

    module.build_experiment_config = patched_build
    try:
        yield
    finally:
        module.build_experiment_config = original_build


@contextmanager
def _gpt_capacity(module: Any, capacity: str, *, is_motion_gpt: bool):
    if capacity not in GPT_CAPACITY_VARIANTS:
        raise ValueError(
            f"Unknown capacity {capacity!r}; expected one of {sorted(GPT_CAPACITY_VARIANTS)}"
        )
    sizing = GPT_CAPACITY_VARIANTS[capacity]
    vqvae = dataclasses.replace(
        module.VQVAE_CONFIG,
        num_codes=sizing.vqvae_num_codes,
        code_dim=sizing.vqvae_code_dim,
        width=sizing.vqvae_width,
    )
    if is_motion_gpt:
        stage_two = dataclasses.replace(
            module.MODEL_CONFIG,
            d_model=sizing.d_model,
            nhead=sizing.nhead,
            num_encoder_layers=sizing.num_layers,
            num_decoder_layers=sizing.num_layers,
            dim_feedforward=sizing.dim_feedforward,
        )
        with _rebind(module, VQVAE_CONFIG=vqvae, MODEL_CONFIG=stage_two):
            yield
    else:
        stage_two = dataclasses.replace(
            module.GPT_CONFIG,
            d_model=sizing.d_model,
            nhead=sizing.nhead,
            num_layers=sizing.num_layers,
            dim_feedforward=sizing.dim_feedforward,
        )
        with _rebind(module, VQVAE_CONFIG=vqvae, GPT_CONFIG=stage_two):
            yield


def run_cross_validation(
    model: ModelName,
    spec: DatasetSpec,
    *,
    folds: Sequence[int] | None = None,
    mirror_augment_train: bool = False,
    capacity: str = "small",
    resume_completed_folds: bool = True,
) -> Any:
    """Run one model's full cross-validation against one dataset spec.

    Reuses that model's own ``train_main_<model>.py`` orchestration (resume checking,
    aggregation, printing) unchanged; only the dataset/window/grid globals (and,
    optionally, mirror augmentation or GPT capacity) are rebound for this call.
    """

    if model not in ("minirocket", "t2m_gpt", "motion_gpt", "frame_tagger"):
        raise ValueError(f"Unknown model {model!r}")
    if capacity != "small" and model not in ("t2m_gpt", "motion_gpt"):
        raise ValueError(f"capacity is only meaningful for t2m_gpt/motion_gpt, not {model!r}")
    if mirror_augment_train and model not in MIRROR_CAPABLE_MODELS:
        raise ValueError(
            f"mirror_augment_train is not reachable for {model!r}: it builds "
            "minirocket.config.DataConfig and loads through src/minirocket/data.py, "
            "which has no mirror toggle. Use src/minirocket_representation.py instead."
        )

    if model in LOCAL_PATH_MODELS:
        ensure_local_dataset(spec)

    module = importlib.import_module(f"train_main_{model}")
    resolved_folds = tuple(range(spec.splits)) if folds is None else tuple(folds)

    # Every train_main_*.py derives its output directory as
    # OUTPUT_ROOT / DATASET_PREFIX / EXPERIMENT_VARIANT at call time, so the dataset is
    # already namespaced -- but mirror/capacity are not, and would silently overwrite the
    # baseline's artifacts (and defeat its resume check) under the same run_name.
    variant = module.EXPERIMENT_VARIANT
    if mirror_augment_train:
        variant = f"{variant}_mirror"
    if capacity != "small":
        variant = f"{variant}_cap-{capacity}"

    grid_overrides = dict(
        DATASET_PREFIX=spec.prefix,
        EXPERIMENT_VARIANT=variant,
        WINDOW_START_SECONDS=spec.window_start_seconds,
        WINDOW_END_SECONDS=spec.window_end_seconds,
        NUM_TIME_POINTS=spec.num_time_points,
    )
    run_kwargs: dict[str, Any] = {"folds": resolved_folds}
    if model in RESUME_CAPABLE_MODELS:
        run_kwargs["resume_completed_folds"] = resume_completed_folds

    with _rebind(module, **grid_overrides):
        with _mirror_augmented(module, mirror_augment_train):
            if model in ("t2m_gpt", "motion_gpt"):
                with _gpt_capacity(module, capacity, is_motion_gpt=(model == "motion_gpt")):
                    return module.train_cross_validation(**run_kwargs)
            return module.train_cross_validation(**run_kwargs)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("minirocket", "t2m_gpt", "motion_gpt", "frame_tagger"))
    parser.add_argument("target", choices=("cue", "scope", "cueandscope"))
    parser.add_argument("window_ms", type=int)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--mirror", action="store_true", help="Mirror-augment the training split")
    parser.add_argument("--capacity", choices=tuple(GPT_CAPACITY_VARIANTS), default="small")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Any:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    spec = DatasetSpec(target=args.target, window_ms=args.window_ms, splits=args.splits)
    return run_cross_validation(
        args.model,
        spec,
        folds=args.folds,
        mirror_augment_train=args.mirror,
        capacity=args.capacity,
        resume_completed_folds=not args.no_resume,
    )


if __name__ == "__main__":
    main()
