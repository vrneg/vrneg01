"""Grouped/per-modality motion tokenization: an alternative to one joint codebook.

Elsewhere in this package, all eight modalities share one codebook: the encoder reads
all 698 channels at once and every motion token is a joint code over all of them. This
module instead gives each channel group its own encoder, codebook, and decoder, trained
independently on its own reconstruction objective.

Codes from different groups are concatenated into one longer token sequence with
disjoint id ranges -- group *g*'s code *c* becomes the global id ``offset[g] + c`` -- so
the existing :class:`~t2m_gpt.model.MotionTokenGPT`, pretraining, and instruction-tuning
code runs completely unchanged. It only ever sees integers in ``[0, total_codes)`` and
has no notion of which group produced them.

This trades a larger token budget (``num_groups * tokens_per_group`` instead of one
``tokens_per_group``) and a larger vocabulary (the sum of each group's codebook) for
groups that no longer compete for codebook capacity. In the joint codebook, the 452
finger channels and 130 facial channels can dominate what the codebook represents at the
expense of head and body motion; grouping removes that competition, at the cost of
training one tokenizer per group instead of one tokenizer overall.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from .config import MODALITY_CHOICES, ExperimentConfig
from .data import DataBundle, MotionTokenDataset, make_loader, prepare_data
from .model import MotionTokenGPT, count_parameters
from .training import (
    CHECKPOINT_VERSION,
    EvaluationOutput,
    ProgressCallback,
    TrainingResult,
    _config_signature,
    _json_ready,
    _prediction_rows,
    _write_lines,
    binary_classification_metrics,
    evaluate_with_predictions,
    optimize_binary_threshold,
    pretrain_motion_gpt,
    resolve_device,
    set_reproducible_seed,
    token_statistics,
    tokenize_split,
    train_motion_gpt,
    train_motion_vqvae,
)
from .vqvae import MotionVQVAE

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(slots=True)
class GroupTokenizer:
    """One trained per-group tokenizer and where its codes sit in the global id space."""

    modalities: tuple[str, ...]
    model: MotionVQVAE
    num_codes: int
    offset: int
    history: list[dict[str, Any]]
    metrics: dict[str, float]


def validate_grouping(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Check that ``groups`` is a non-overlapping partition of every modality."""

    if not groups:
        raise ValueError("At least one modality group is required")
    normalized = tuple(tuple(group) for group in groups)
    seen: set[str] = set()
    for group in normalized:
        if not group:
            raise ValueError("Modality groups must be non-empty")
        overlap = seen & set(group)
        if overlap:
            raise ValueError(f"Modality groups overlap: {sorted(overlap)}")
        seen.update(group)
    if seen != set(MODALITY_CHOICES):
        missing = set(MODALITY_CHOICES) - seen
        raise ValueError(
            f"Modality groups must partition all modalities; missing {sorted(missing)}"
        )
    return normalized


def train_group_tokenizers(
    config: ExperimentConfig,
    groups: Sequence[Sequence[str]],
    device: torch.device,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GroupTokenizer], list[DataBundle]]:
    """Train one independent :class:`MotionVQVAE` per channel group.

    Every group reuses ``config.vqvae``/``config.vqvae_training`` unchanged, so the only
    variable between groups -- and between this and the single-codebook baseline -- is
    which channels each tokenizer sees.
    """

    tokenizers: list[GroupTokenizer] = []
    group_bundles: list[DataBundle] = []
    offset = 0
    for group in groups:
        data = prepare_data(replace(config.data, modalities=tuple(group)))
        model, history, metrics = train_motion_vqvae(
            data, config, device, progress_callback
        )
        tokenizers.append(
            GroupTokenizer(
                modalities=tuple(group),
                model=model,
                num_codes=config.vqvae.num_codes,
                offset=offset,
                history=history,
                metrics=metrics,
            )
        )
        group_bundles.append(data)
        offset += config.vqvae.num_codes
    return tokenizers, group_bundles


def _concatenated_tokens(
    tokenizers: Sequence[GroupTokenizer],
    group_bundles: Sequence[DataBundle],
    split: str,
    batch_size: int,
    device: torch.device,
) -> MotionTokenDataset:
    """Tokenize one split per group and concatenate into one global-id sequence."""

    per_group = [
        tokenize_split(tokenizer.model, getattr(bundle, split), batch_size, device)
        for tokenizer, bundle in zip(tokenizers, group_bundles, strict=True)
    ]
    reference = per_group[0]
    for dataset in per_group[1:]:
        if dataset.sample_ids != reference.sample_ids:
            raise RuntimeError(
                "Per-group tokenization produced misaligned window order; every group "
                "must tokenize the same windows in the same order"
            )
        if not torch.equal(dataset.labels, reference.labels):
            raise RuntimeError("Per-group tokenization produced mismatched labels")

    offset_indices = [
        dataset.indices + tokenizer.offset
        for tokenizer, dataset in zip(tokenizers, per_group, strict=True)
    ]
    return MotionTokenDataset(
        indices=torch.cat(offset_indices, dim=1),
        labels=reference.labels,
        sample_ids=reference.sample_ids,
    )


def _grouped_checkpoint(
    model_state: dict[str, torch.Tensor],
    config: ExperimentConfig,
    groups: tuple[tuple[str, ...], ...],
    total_num_codes: int,
    num_tokens: int,
    best_epoch: int,
    decision_threshold: float,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    model_parameter_count: int,
    history: list[dict[str, Any]],
    pretraining_history: list[dict[str, Any]],
    group_tokenizer_metrics: list[dict[str, float]],
    token_statistics_by_split: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_type": f"t2m_gpt_grouped_{config.gpt.head}_classifier",
        "model_state_dict": model_state,
        "gpt_config": _json_ready(asdict(config.gpt)),
        "vqvae_config": _json_ready(asdict(config.vqvae)),
        "experiment_config": _config_signature(config),
        "groups": [list(group) for group in groups],
        "num_codes": total_num_codes,
        "num_tokens": num_tokens,
        "label_convention": {
            config.data.negative_label: 0,
            config.data.positive_label: 1,
        },
        "best_epoch": best_epoch,
        "decision_threshold": decision_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_parameter_count": model_parameter_count,
        "history": history,
        "pretraining_history": pretraining_history,
        "group_tokenizer_metrics": group_tokenizer_metrics,
        "token_statistics": token_statistics_by_split,
    }


def train_t2m_gpt_grouped(
    config: ExperimentConfig,
    groups: Sequence[Sequence[str]],
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Run the grouped-tokenizer variant for one fold.

    Mirrors :func:`t2m_gpt.training.train_t2m_gpt`'s checkpoint/prediction outputs so
    both are read by the same aggregation and analysis code; only phase one -- how
    motion becomes tokens -- differs.
    """

    config.validate()
    normalized_groups = validate_grouping(groups)
    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    device = resolve_device(config.device)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = run_dir / "motion_vqvae_groups.pt"
    pretrained_path = run_dir / "pretrained_backbone.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    tokenizers, group_bundles = train_group_tokenizers(
        config, normalized_groups, device, progress_callback
    )
    total_num_codes = sum(tokenizer.num_codes for tokenizer in tokenizers)

    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_type": "t2m_gpt_grouped_motion_vqvae",
            "groups": [list(tokenizer.modalities) for tokenizer in tokenizers],
            "offsets": [tokenizer.offset for tokenizer in tokenizers],
            "num_codes_per_group": [tokenizer.num_codes for tokenizer in tokenizers],
            "model_state_dicts": [
                tokenizer.model.state_dict() for tokenizer in tokenizers
            ],
            "vqvae_config": _json_ready(asdict(config.vqvae)),
            "history_per_group": [tokenizer.history for tokenizer in tokenizers],
            "metrics_per_group": [tokenizer.metrics for tokenizer in tokenizers],
        },
        tokenizer_path,
    )

    token_datasets = {
        split: _concatenated_tokens(
            tokenizers,
            group_bundles,
            split,
            config.vqvae_training.evaluation_batch_size,
            device,
        )
        for split in SPLIT_NAMES
    }
    token_statistics_by_split = {
        split: token_statistics(dataset, total_num_codes)
        for split, dataset in token_datasets.items()
    }
    num_tokens = token_datasets["train"].num_tokens

    set_reproducible_seed(config.seed, config.training.deterministic_algorithms)
    model = MotionTokenGPT(total_num_codes, num_tokens, config.gpt).to(device)

    pretraining_history: list[dict[str, Any]] = []
    if config.use_pretraining:
        pretraining_history = pretrain_motion_gpt(
            model,
            token_datasets["train"],
            token_datasets["validation"],
            config,
            device,
            progress_callback,
        )
        model.synchronize_prefix_embeddings()
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "t2m_gpt_grouped_pretrained_backbone",
                "model_state_dict": model.state_dict(),
                "gpt_config": _json_ready(asdict(config.gpt)),
                "num_codes": total_num_codes,
                "num_tokens": num_tokens,
                "pretraining_history": pretraining_history,
            },
            pretrained_path,
        )

    history, best_state, best_epoch = train_motion_gpt(
        model, token_datasets, config, device, progress_callback
    )
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        token_datasets["validation"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        token_datasets["test"],
        batch_size=config.training.evaluation_batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )

    validation_output = evaluate_with_predictions(
        model, validation_loader, config, device
    )
    decision_threshold = config.training.threshold
    if config.training.calibrate_threshold_on_validation:
        decision_threshold = optimize_binary_threshold(
            validation_output.logits,
            validation_output.labels,
            config.training.threshold_metric,
        )
        validation_output = EvaluationOutput(
            metrics=binary_classification_metrics(
                validation_output.logits,
                validation_output.labels,
                validation_output.metrics["loss"],
                decision_threshold,
            ),
            sample_ids=validation_output.sample_ids,
            labels=validation_output.labels,
            logits=validation_output.logits,
        )
    test_output = evaluate_with_predictions(
        model, test_loader, config, device, threshold=decision_threshold
    )

    _write_lines(
        validation_predictions_path,
        _prediction_rows(validation_output, decision_threshold),
    )
    _write_lines(test_predictions_path, _prediction_rows(test_output, decision_threshold))

    model_parameter_count = count_parameters(model, trainable_only=False)
    group_tokenizer_metrics = [tokenizer.metrics for tokenizer in tokenizers]
    torch.save(
        _grouped_checkpoint(
            model_state=best_state,
            config=config,
            groups=normalized_groups,
            total_num_codes=total_num_codes,
            num_tokens=num_tokens,
            best_epoch=best_epoch,
            decision_threshold=decision_threshold,
            validation_metrics=validation_output.metrics,
            test_metrics=test_output.metrics,
            model_parameter_count=model_parameter_count,
            history=history,
            pretraining_history=pretraining_history,
            group_tokenizer_metrics=group_tokenizer_metrics,
            token_statistics_by_split=token_statistics_by_split,
        ),
        checkpoint_path,
    )

    history_path.write_text(
        json.dumps(
            _json_ready(
                {
                    "run_name": config.run_name,
                    "experiment_config": _config_signature(config),
                    "groups": [list(group) for group in normalized_groups],
                    "num_codes": total_num_codes,
                    "num_tokens": num_tokens,
                    "split_sizes": {
                        split: len(token_datasets[split]) for split in SPLIT_NAMES
                    },
                    "best_epoch": best_epoch,
                    "decision_threshold": decision_threshold,
                    "model_parameter_count": model_parameter_count,
                    "group_tokenizer_metrics": group_tokenizer_metrics,
                    "token_statistics": token_statistics_by_split,
                    "validation_metrics": validation_output.metrics,
                    "test_metrics": test_output.metrics,
                    "pretraining_history": pretraining_history,
                    "history": history,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=test_predictions_path,
        run_name=config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_output.metrics,
        test_metrics=test_output.metrics,
        model_parameter_count=model_parameter_count,
        history=history,
        tokenizer_checkpoint_path=tokenizer_path,
        tokenizer_history=[tokenizer.history for tokenizer in tokenizers],
        tokenizer_metrics={
            f"group_{index}_{'-'.join(tokenizer.modalities)}": value
            for index, tokenizer in enumerate(tokenizers)
            for value in [tokenizer.metrics]
        },
        pretrained_checkpoint_path=pretrained_path if config.use_pretraining else None,
        pretraining_history=pretraining_history,
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version in {path}: "
            f"{checkpoint.get('checkpoint_version')!r}"
        )
    return checkpoint


def load_grouped_training_result(
    config: ExperimentConfig,
    groups: Sequence[Sequence[str]],
) -> TrainingResult:
    """Reload a finished grouped fold, refusing a mismatched configuration or grouping."""

    config.validate()
    normalized_groups = validate_grouping(groups)
    run_dir = Path(config.output_dir) / config.run_name
    checkpoint = _load_checkpoint(run_dir / "best_model.pt")
    if checkpoint.get("experiment_config") != _config_signature(config):
        raise ValueError(
            f"Checkpoint in {run_dir} was produced by a different configuration"
        )
    if [tuple(group) for group in checkpoint.get("groups", [])] != list(normalized_groups):
        raise ValueError(f"Checkpoint in {run_dir} used a different modality grouping")
    for required in (
        "validation_predictions.jsonl",
        "test_predictions.jsonl",
        "metrics.json",
    ):
        if not (run_dir / required).exists():
            raise FileNotFoundError(f"Missing {required} in {run_dir}")

    pretrained_path = run_dir / "pretrained_backbone.pt"
    return TrainingResult(
        checkpoint_path=run_dir / "best_model.pt",
        history_path=run_dir / "metrics.json",
        validation_predictions_path=run_dir / "validation_predictions.jsonl",
        predictions_path=run_dir / "test_predictions.jsonl",
        run_name=config.run_name,
        best_epoch=int(checkpoint["best_epoch"]),
        decision_threshold=float(checkpoint["decision_threshold"]),
        validation_metrics=dict(checkpoint["validation_metrics"]),
        test_metrics=dict(checkpoint["test_metrics"]),
        model_parameter_count=int(checkpoint["model_parameter_count"]),
        history=list(checkpoint["history"]),
        tokenizer_checkpoint_path=run_dir / "motion_vqvae_groups.pt",
        tokenizer_history=[],
        tokenizer_metrics={},
        pretrained_checkpoint_path=pretrained_path if pretrained_path.exists() else None,
        pretraining_history=list(checkpoint.get("pretraining_history", [])),
    )


__all__ = [
    "GroupTokenizer",
    "load_grouped_training_result",
    "train_group_tokenizers",
    "train_t2m_gpt_grouped",
    "validate_grouping",
]
