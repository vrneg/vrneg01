"""Pretrain the T2M-GPT motion tokenizer on the expanded, unlabeled allWords corpus.

`target-allWords_source-speaker_windowL-500_windowR-500` (57,971 windows across 34
experiments, uploaded 2026-08-14) is ~76x the ~750 training windows any single
`target-cue_window-500` fold has, and every row is unlabeled (`label == "none"` for all
of it -- it was built by `main_utils.dataset_creation.create_dataset_all_words`, not
`create_dataset`). This is option #1 from the earlier "what could help au_neg_drop"
discussion: give the VQ-VAE tokenizer a much larger, task-agnostic motion corpus to
learn its codebook from, independent of how much labeled data exists for classification.

The tokenizer's architecture (`MotionVQVAE`) depends only on `input_dim` and
`VQVAEConfig`, not on which data trained it -- see `t2m_gpt.training._tokenizer_checkpoint`,
which stores no reference to a specific fold. That is what makes swapping it in for a
fold's own from-scratch tokenizer possible without touching `t2m_gpt/training.py`: this
module trains one on the expanded corpus and calls the same `tokenize_split`,
`pretrain_motion_gpt`, `train_motion_gpt`, `evaluate_with_predictions` primitives
`train_t2m_gpt` itself calls, in the same order, just skipping its internal
`train_motion_vqvae` step.

Why one tokenizer per fold, not one shared tokenizer
-----------------------------------------------------
The allWords corpus and `target-cue_window-500`'s ten folds are drawn from the same
~34 experiments. Training one tokenizer on the full corpus and reusing it across every
fold would let each fold's held-out validation and test experiments' motion leak into
tokenizer pretraining -- unsupervised, but still leakage: the tokenizer would have seen
those exact participants' movement statistics before being asked to encode them at
evaluation time. `leak_safe_corpus_experiment_ids` excludes each fold's validation and
test experiment ids from the corpus before training that fold's tokenizer, so no held-out
experiment is ever touched during pretraining. Measured retention is ~77-85% of the
corpus per fold (~45,000-47,000 windows), still far larger than any fold's own split.

Normalization: a tokenizer must see fine-tuning windows standardized on the *same* scale
it was trained on -- `tokenize_split` feeds `split.values` straight into the encoder with
no re-normalization step. The corpus and a fine-tuning fold fit their own
`FeatureNormalizer` independently, so `_renormalize` un-normalizes a fold's grid with its
own fitted (mean, std) and re-normalizes with the tokenizer corpus's, using the same
per-channel inversion `representation.channel_normalization_arrays` already provides for
representation and mirror math.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Json, Value

try:
    from event_transformer.metrics import binary_classification_metrics, optimize_binary_threshold
    from representation import channel_normalization_arrays
    from t2m_gpt.config import DataConfig, ExperimentConfig, VQVAEConfig, VQVAETrainingConfig
    from t2m_gpt.data import DataBundle, make_loader, prepare_data
    from t2m_gpt.model import MotionTokenGPT, count_parameters
    from t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        TrainingResult,
        _checkpoint,
        _json_ready,
        _prediction_rows,
        _tokenizer_checkpoint,
        _write_lines,
        evaluate_with_predictions,
        load_training_result,
        pretrain_motion_gpt,
        resolve_device,
        set_reproducible_seed,
        token_statistics,
        tokenize_split,
        train_motion_gpt,
        train_motion_vqvae,
    )
    from t2m_gpt.vqvae import MotionVQVAE
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name not in {"representation", "t2m_gpt", "event_transformer"}:
        raise
    from .event_transformer.metrics import binary_classification_metrics, optimize_binary_threshold
    from .representation import channel_normalization_arrays
    from .t2m_gpt.config import DataConfig, ExperimentConfig, VQVAEConfig, VQVAETrainingConfig
    from .t2m_gpt.data import DataBundle, make_loader, prepare_data
    from .t2m_gpt.model import MotionTokenGPT, count_parameters
    from .t2m_gpt.training import (
        CHECKPOINT_VERSION,
        SPLIT_NAMES,
        EvaluationOutput,
        TrainingResult,
        _checkpoint,
        _json_ready,
        _prediction_rows,
        _tokenizer_checkpoint,
        _write_lines,
        evaluate_with_predictions,
        load_training_result,
        pretrain_motion_gpt,
        resolve_device,
        set_reproducible_seed,
        token_statistics,
        tokenize_split,
        train_motion_gpt,
        train_motion_vqvae,
    )
    from .t2m_gpt.vqvae import MotionVQVAE


TOKENIZER_SIDECAR = "expanded_tokenizer.json"
VALIDATION_FRACTION = 0.1
CORPUS_SPLIT_SEED = 42

CORPUS_FEATURES = Features({"context": Json(), "word": Json(), "label": Value("string")})


def _experiment_id(word: Any) -> int:
    record = word if isinstance(word, dict) else json.loads(word)
    return int(record["experiment"]["id"])


def leak_safe_corpus_experiment_ids(
    corpus_experiment_ids: set[int], fold_dataset: DatasetDict
) -> set[int]:
    """Corpus experiment ids with the fold's validation and test experiments removed."""

    held_out: set[int] = set()
    for split_name in ("validation", "test"):
        held_out.update(_experiment_id(word) for word in fold_dataset[split_name]["word"])
    return corpus_experiment_ids - held_out


def build_leak_safe_corpus_fold(
    corpus: Dataset,
    held_out_experiment_ids: set[int],
    destination: Path,
    seed: int = CORPUS_SPLIT_SEED,
) -> Path:
    """Materialize a train/validation split of the corpus, excluding held-out experiments.

    Written as a save_to_disk DatasetDict so it can be loaded through the exact same
    `t2m_gpt.data.prepare_data` path every other dataset in this repository uses -- no
    separate loading code for the tokenizer corpus. ``test`` is a copy of ``validation``:
    nothing in this module reads it, but `prepare_data` requires all three splits.
    """

    if (destination / "dataset_dict.json").exists():
        return destination

    keep_indices = [
        index
        for index, word in enumerate(corpus["word"])
        if _experiment_id(word) not in held_out_experiment_ids
    ]
    filtered = corpus.select(keep_indices)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(filtered))
    num_validation = max(1, int(len(permutation) * VALIDATION_FRACTION))
    validation_indices = permutation[:num_validation].tolist()
    train_indices = permutation[num_validation:].tolist()

    train_split = filtered.select(train_indices)
    validation_split = filtered.select(validation_indices)
    fold = DatasetDict(
        {"train": train_split, "validation": validation_split, "test": validation_split}
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fold.save_to_disk(str(destination))
    return destination


def _renormalize(
    values: np.ndarray,
    source_channel_names: tuple[str, ...],
    source_normalizer: object | None,
    target_channel_names: tuple[str, ...],
    target_normalizer: object | None,
) -> np.ndarray:
    """Convert a grid standardized by one FeatureNormalizer into another's scale."""

    if source_channel_names != target_channel_names:
        raise ValueError(
            "the tokenizer corpus and the fine-tuning fold produced different channel "
            "layouts; they must share the same modalities, representation, and "
            "presence-channel settings"
        )
    source_means, source_scales = channel_normalization_arrays(
        source_channel_names, source_normalizer
    )
    target_means, target_scales = channel_normalization_arrays(
        target_channel_names, target_normalizer
    )
    raw = values.astype(np.float64) * source_scales[None, :, None] + source_means[None, :, None]
    renormalized = (raw - target_means[None, :, None]) / target_scales[None, :, None]
    return renormalized.astype(values.dtype, copy=False)


def train_expanded_tokenizer(
    corpus_data_config: DataConfig,
    vqvae_config: VQVAEConfig,
    vqvae_training_config: VQVAETrainingConfig,
    *,
    seed: int,
    device: str = "auto",
) -> tuple[MotionVQVAE, DataBundle, list[dict[str, Any]], dict[str, float]]:
    """Fit a VQ-VAE tokenizer on the (already leak-filtered) expanded corpus."""

    corpus_data_config.validate()
    set_reproducible_seed(seed)
    resolved_device = resolve_device(device)
    corpus = prepare_data(corpus_data_config)

    probe_config = ExperimentConfig(
        data=corpus_data_config,
        output_dir="unused",
        run_name="unused",
        vqvae=vqvae_config,
        vqvae_training=vqvae_training_config,
        seed=seed,
        device=device,
    )
    tokenizer, history, metrics = train_motion_vqvae(corpus, probe_config, resolved_device)
    return tokenizer, corpus, history, metrics


def train_fold_with_expanded_tokenizer(
    fold_config: ExperimentConfig,
    tokenizer: MotionVQVAE,
    tokenizer_data: DataBundle,
    tokenizer_history: list[dict[str, Any]],
    tokenizer_metrics: dict[str, float],
    tokenizer_corpus_signature: dict[str, Any],
) -> TrainingResult:
    """Run pretraining + classification for one fold against a frozen, externally
    trained tokenizer -- mirrors `t2m_gpt.training.train_t2m_gpt` but skips its internal
    `train_motion_vqvae` call and re-normalizes the fold's data into the tokenizer's
    normalization space first.
    """

    fold_config.validate()
    set_reproducible_seed(fold_config.seed, fold_config.training.deterministic_algorithms)
    device = resolve_device(fold_config.device)

    run_dir = Path(fold_config.output_dir) / fold_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = run_dir / "motion_vqvae.pt"
    pretrained_path = run_dir / "pretrained_backbone.pt"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "metrics.json"
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    test_predictions_path = run_dir / "test_predictions.jsonl"

    data = prepare_data(fold_config.data)
    for split_name in SPLIT_NAMES:
        split = getattr(data, split_name)
        split.values = _renormalize(
            split.values,
            data.channel_names,
            data.normalizer,
            tokenizer_data.channel_names,
            tokenizer_data.normalizer,
        )
    data.normalizer = tokenizer_data.normalizer

    torch.save(
        _tokenizer_checkpoint(tokenizer, fold_config, data, tokenizer_history, tokenizer_metrics),
        tokenizer_path,
    )
    (run_dir / TOKENIZER_SIDECAR).write_text(
        json.dumps(tokenizer_corpus_signature, indent=2, default=str) + "\n"
    )

    token_datasets = {
        split_name: tokenize_split(
            tokenizer,
            getattr(data, split_name),
            fold_config.vqvae_training.evaluation_batch_size,
            device,
        )
        for split_name in SPLIT_NAMES
    }
    token_statistics_by_split = {
        split_name: token_statistics(dataset, fold_config.vqvae.num_codes)
        for split_name, dataset in token_datasets.items()
    }
    num_tokens = token_datasets["train"].num_tokens

    set_reproducible_seed(fold_config.seed, fold_config.training.deterministic_algorithms)
    model = MotionTokenGPT(fold_config.vqvae.num_codes, num_tokens, fold_config.gpt).to(device)

    pretraining_history: list[dict[str, Any]] = []
    if fold_config.use_pretraining:
        pretraining_history = pretrain_motion_gpt(
            model,
            token_datasets["train"],
            token_datasets["validation"],
            fold_config,
            device,
            None,
        )
        model.synchronize_prefix_embeddings()
        torch.save(
            {
                "checkpoint_version": CHECKPOINT_VERSION,
                "checkpoint_type": "t2m_gpt_pretrained_backbone",
                "model_state_dict": model.state_dict(),
                "gpt_config": _json_ready(asdict(fold_config.gpt)),
                "num_codes": fold_config.vqvae.num_codes,
                "num_tokens": num_tokens,
                "pretraining_history": pretraining_history,
            },
            pretrained_path,
        )

    history, best_state, best_epoch = train_motion_gpt(model, token_datasets, fold_config, device, None)
    model.load_state_dict(best_state)

    pin_memory = device.type == "cuda"
    validation_loader = make_loader(
        token_datasets["validation"],
        batch_size=fold_config.training.evaluation_batch_size,
        shuffle=False,
        seed=fold_config.seed,
        num_workers=fold_config.training.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        token_datasets["test"],
        batch_size=fold_config.training.evaluation_batch_size,
        shuffle=False,
        seed=fold_config.seed,
        num_workers=fold_config.training.num_workers,
        pin_memory=pin_memory,
    )

    validation_output = evaluate_with_predictions(model, validation_loader, fold_config, device)
    decision_threshold = fold_config.training.threshold
    if fold_config.training.calibrate_threshold_on_validation:
        decision_threshold = optimize_binary_threshold(
            validation_output.logits, validation_output.labels, fold_config.training.threshold_metric
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
        model, test_loader, fold_config, device, threshold=decision_threshold
    )

    _write_lines(
        validation_predictions_path, _prediction_rows(validation_output, decision_threshold)
    )
    _write_lines(test_predictions_path, _prediction_rows(test_output, decision_threshold))

    model_parameter_count = count_parameters(model, trainable_only=False)
    torch.save(
        _checkpoint(
            model_state=best_state,
            config=fold_config,
            data=data,
            num_tokens=num_tokens,
            best_epoch=best_epoch,
            decision_threshold=decision_threshold,
            validation_metrics=validation_output.metrics,
            test_metrics=test_output.metrics,
            model_parameter_count=model_parameter_count,
            history=history,
            pretraining_history=pretraining_history,
            tokenizer_history=tokenizer_history,
            tokenizer_metrics=tokenizer_metrics,
            token_statistics_by_split=token_statistics_by_split,
        ),
        checkpoint_path,
    )
    history_path.write_text(json.dumps(_json_ready(history), indent=2))

    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validation_predictions_path=validation_predictions_path,
        predictions_path=test_predictions_path,
        run_name=fold_config.run_name,
        best_epoch=best_epoch,
        decision_threshold=decision_threshold,
        validation_metrics=validation_output.metrics,
        test_metrics=test_output.metrics,
        model_parameter_count=model_parameter_count,
        history=history,
        tokenizer_checkpoint_path=tokenizer_path,
        tokenizer_history=tokenizer_history,
        tokenizer_metrics=tokenizer_metrics,
        pretrained_checkpoint_path=pretrained_path if fold_config.use_pretraining else None,
        pretraining_history=pretraining_history,
    )


def load_matching_fold_result(
    fold_config: ExperimentConfig, tokenizer_corpus_signature: dict[str, Any]
) -> TrainingResult:
    """Reuse a completed fold run only if it used this exact tokenizer corpus."""

    run_dir = Path(fold_config.output_dir) / fold_config.run_name
    sidecar_path = run_dir / TOKENIZER_SIDECAR
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"No {TOKENIZER_SIDECAR} in {run_dir}")
    saved = json.loads(sidecar_path.read_text())
    expected = json.loads(json.dumps(tokenizer_corpus_signature, default=str))
    if saved != expected:
        raise FileNotFoundError(f"{run_dir} was trained against a different tokenizer corpus")
    return load_training_result(fold_config)


__all__ = [
    "CORPUS_FEATURES",
    "TOKENIZER_SIDECAR",
    "build_leak_safe_corpus_fold",
    "leak_safe_corpus_experiment_ids",
    "load_matching_fold_result",
    "train_expanded_tokenizer",
    "train_fold_with_expanded_tokenizer",
]
