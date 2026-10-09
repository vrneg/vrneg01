"""Bridges MiniRocket's transform+ridge pipeline to the representation layer.

MiniRocket's own data pipeline (`minirocket.data.prepare_data`) has no
`RepresentationConfig` field, and `src/minirocket/` is not touched here. Instead this
module builds the fixed grid through `t2m_gpt.data.prepare_data`, which already
supports `RepresentationConfig`, and substitutes it for MiniRocket's own data-loading
step for the duration of one training call.

That substitution is safe because `t2m_gpt.data.FixedGridSplit`/`DataBundle` are
structurally identical to `minirocket.data.TimeSeriesSplit`/`DataBundle` -- same field
names (`values`, `labels`, `sample_ids`, `group_ids`; `train`, `validation`, `test`,
`normalizer`, `time_grid`, `channel_names`), same shapes. `minirocket.training` never
checks the concrete type of what it receives, so passing the `t2m_gpt` bundle through
duck typing needs no conversion. With the identity `RepresentationConfig`, the two
pipelines build the array from the same underlying `EventSequenceEncoder` and the same
`minirocket.data.window_to_array`/`channel_names` functions (`t2m_gpt.data` imports
them directly), so results are expected to match MiniRocket's native path exactly --
useful as a correctness check before trusting any non-identity variant.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

try:
    import minirocket.training as minirocket_training
    from minirocket.config import ExperimentConfig as MiniRocketExperimentConfig
    from minirocket.data import _collapse_near_constant_traces
    from minirocket.training import TrainingResult, load_training_result
    from t2m_gpt.config import DataConfig as RepresentationDataConfig
    from t2m_gpt.data import prepare_data as prepare_representation_data
    from representation import RepresentationConfig
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"minirocket", "t2m_gpt", "representation"}:
        raise
    from .minirocket import training as minirocket_training
    from .minirocket.config import ExperimentConfig as MiniRocketExperimentConfig
    from .minirocket.data import _collapse_near_constant_traces
    from .minirocket.training import TrainingResult, load_training_result
    from .t2m_gpt.config import DataConfig as RepresentationDataConfig
    from .t2m_gpt.data import prepare_data as prepare_representation_data
    from .representation import RepresentationConfig

REPRESENTATION_SIDECAR = "representation.json"


def _representation_data_config(
    minirocket_config: MiniRocketExperimentConfig,
    dataset: str,
    representation: RepresentationConfig,
) -> RepresentationDataConfig:
    """Mirror MiniRocket's grid settings into a representation-aware ``DataConfig``."""

    data = minirocket_config.data
    return RepresentationDataConfig(
        dataset=dataset,
        num_time_points=data.num_time_points,
        window_start_seconds=data.window_start_seconds,
        window_end_seconds=data.window_end_seconds,
        max_events_per_modality=data.max_events_per_modality,
        normalize_features=data.normalize_features,
        cache_in_memory=data.cache_in_memory,
        actor_scope=data.actor_scope,
        include_presence_channels=data.include_presence_channels,
        positive_label=data.positive_label,
        negative_label=data.negative_label,
        representation=representation,
    )


def train_minirocket_with_representation(
    minirocket_config: MiniRocketExperimentConfig,
    dataset: str,
    representation: RepresentationConfig,
) -> TrainingResult:
    """Train and evaluate one MiniRocket fold on a non-default channel representation.

    A ``representation.json`` sidecar recording ``representation`` is written next to
    the usual MiniRocket artifacts. MiniRocket's own saved ``experiment_config`` has no
    field for it, so without the sidecar a resumed run could not tell which
    representation actually produced a cached checkpoint; :func:`load_matching_result`
    checks it before trusting a cache hit.
    """

    representation.validate()
    data_config = _representation_data_config(minirocket_config, dataset, representation)
    bundle = prepare_representation_data(data_config)
    # t2m_gpt.data.prepare_data skips this cleanup since PyTorch does not care about
    # near-zero-variance channels; aeon's MiniRocket transform rejects them outright
    # (interpolation and float32 rounding leave a few traces with a tiny nonzero
    # standard deviation). minirocket.data.prepare_data applies it for exactly this
    # reason -- reused unchanged here rather than duplicated.
    for split in (bundle.train, bundle.validation, bundle.test):
        split.values = _collapse_near_constant_traces(split.values)

    original_prepare_data = minirocket_training.prepare_data
    minirocket_training.prepare_data = lambda _config: bundle
    try:
        result = minirocket_training.train_minirocket(minirocket_config)
    finally:
        minirocket_training.prepare_data = original_prepare_data

    sidecar_path = result.checkpoint_path.parent / REPRESENTATION_SIDECAR
    sidecar_path.write_text(
        json.dumps(asdict(representation), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def load_matching_result(
    minirocket_config: MiniRocketExperimentConfig, representation: RepresentationConfig
) -> TrainingResult:
    """Load a cached result only if it was produced with this exact representation.

    Raises ``FileNotFoundError`` otherwise (including when the sidecar is missing, which
    means the run predates this module or was written by a different mechanism), which
    the caller should treat identically to "no cached run" and retrain.
    """

    run_directory = Path(minirocket_config.output_dir) / minirocket_config.run_name
    sidecar_path = run_directory / REPRESENTATION_SIDECAR
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"No {REPRESENTATION_SIDECAR} in {run_directory}")
    saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(asdict(representation), default=str))
    if saved != expected:
        raise FileNotFoundError(
            f"{run_directory} was produced with a different representation; not reusing"
        )
    return load_training_result(minirocket_config)


__all__ = [
    "REPRESENTATION_SIDECAR",
    "load_matching_result",
    "train_minirocket_with_representation",
]
