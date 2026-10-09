"""Editable entry point: T2M-GPT with a tokenizer pretrained on the expanded allWords corpus.

Runs the exact same VQ-VAE architecture, stage-two transformer, and training schedule as
`train_main_t2m_gpt.py` (imported from it directly, so the two configurations cannot
drift apart), but replaces each fold's from-scratch tokenizer -- fit on that fold's ~750
labeled training windows -- with one fit on `target-allWords_source-speaker_windowL-500_windowR-500`
(57,971 unlabeled windows across 34 experiments), with that fold's own validation and
test experiments excluded from the corpus first. See `src/expanded_tokenizer.py` for why
a per-fold tokenizer is necessary rather than one shared across all ten folds.

Edit ``CROSS_VALIDATION_FOLDS`` or the tokenizer's own training schedule below, then run::

    python src/train_main_t2m_gpt_expanded_tokenizer.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from datasets import load_dataset

import train_main_t2m_gpt as baseline
from expanded_tokenizer import (
    build_leak_safe_corpus_fold,
    leak_safe_corpus_experiment_ids,
    load_matching_fold_result,
    train_expanded_tokenizer,
    train_fold_with_expanded_tokenizer,
)
from t2m_gpt import (
    CrossValidationArtifacts,
    DataConfig,
    TrainingResult,
    VQVAETrainingConfig,
    aggregate_cross_validation_results,
)
from t2m_gpt.data import load_fold_dataset

PROJECT_ROOT = baseline.PROJECT_ROOT
CORPUS_HUB_REPO = "VR-Faces-Neg/target-allWords_source-speaker_windowL-500_windowR-500"

# The raw corpus is loaded straight off the Hub every time -- `datasets` already caches
# it under the standard HF cache (~/.cache/huggingface, wherever that resolves to), so
# there is no reason to keep our own second full copy. Only the per-fold *filtered*
# subset is genuinely new, derived data with no Hub home of its own; it is written to a
# scratch directory (preferring a well-provisioned shared volume if one is mounted, since
# it can be tens of GB) and deleted immediately after that fold's tokenizer is trained --
# see train_fold() below. A prior version of this script kept every fold's filtered
# corpus around under the repo's own data/ directory indefinitely, which multiplied the
# 15-20GB corpus by up to 10 folds and filled a shared cluster home directory to 100%.
_SHARED_SCRATCH = Path("/mnt/conda/staff/pschrott_negation_scratch")
SCRATCH_ROOT = _SHARED_SCRATCH if _SHARED_SCRATCH.is_dir() else Path(tempfile.gettempdir())

EXPERIMENT_VARIANT = "discriminative_expanded_tokenizer"
OUTPUT_ROOT = baseline.OUTPUT_ROOT
CROSS_VALIDATION_FOLDS: tuple[int, ...] = tuple(range(10))
RESUME_COMPLETED_FOLDS = True

# The tokenizer's own training schedule on the (much larger) corpus. Kept separate from
# baseline.VQVAE_TRAINING_CONFIG (which is tuned for ~750 windows per fold) since more
# data can support more epochs before the early-stopping patience triggers; the
# architecture (VQVAEConfig) is shared with the baseline so codebook capacity is not a
# second, uncontrolled variable.
CORPUS_VQVAE_TRAINING_CONFIG = VQVAETrainingConfig(
    max_epochs=100,
    batch_size=64,
    evaluation_batch_size=128,
    learning_rate=2e-4,
    reconstruction_loss="smooth_l1",
    velocity_loss_weight=0.5,
    commitment_loss_weight=0.02,
    early_stopping_patience=10,
    lr_scheduler_patience=5,
)


def experiment_output_dir() -> Path:
    return OUTPUT_ROOT / baseline.DATASET_PREFIX / EXPERIMENT_VARIANT


def corpus_data_config(fold_directory: Path) -> DataConfig:
    return DataConfig(
        dataset=str(fold_directory),
        num_time_points=baseline.NUM_TIME_POINTS,
        window_start_seconds=baseline.WINDOW_START_SECONDS,
        window_end_seconds=baseline.WINDOW_END_SECONDS,
        max_events_per_modality=baseline.MAX_EVENTS_PER_MODALITY,
        normalize_features=baseline.NORMALIZE_FEATURES,
        cache_in_memory=baseline.CACHE_DATASET_IN_MEMORY,
        # This auxiliary corpus is always source-speaker, independently of the
        # labeled fold dataset selected by a sweep launcher.
        actor_scope="anchor",
        include_presence_channels=baseline.INCLUDE_PRESENCE_CHANNELS,
        modalities=baseline.MODALITIES,
        # The corpus has no negation labels; every row is "none". positive_label is
        # unreachable but must still differ from negative_label to pass validation.
        positive_label="neg",
        negative_label="none",
    )


def tokenizer_corpus_signature(fold: int, held_out_experiment_ids: set[int]) -> dict:
    return {
        "corpus": CORPUS_HUB_REPO,
        "fold": fold,
        "held_out_experiment_ids": sorted(held_out_experiment_ids),
        "vqvae_config": asdict(baseline.VQVAE_CONFIG),
        "corpus_vqvae_training_config": asdict(CORPUS_VQVAE_TRAINING_CONFIG),
    }


def train_fold(fold: int) -> TrainingResult:
    # ExperimentConfig is frozen; only output_dir needs to change from the baseline's
    # own (dataset_prefix, "discriminative") directory to this variant's.
    fold_config = replace(
        baseline.build_experiment_config(fold, show_progress=False),
        output_dir=str(experiment_output_dir()),
    )

    fold_dataset = load_fold_dataset(baseline.dataset_source(fold))
    # Uses the standard HF cache -- no local copy of our own.
    corpus = load_dataset(CORPUS_HUB_REPO)["train"]
    all_experiment_ids = {
        int((w if isinstance(w, dict) else json.loads(w))["experiment"]["id"])
        for w in corpus["word"]
    }
    safe_experiment_ids = leak_safe_corpus_experiment_ids(all_experiment_ids, fold_dataset)
    held_out = all_experiment_ids - safe_experiment_ids
    signature = tokenizer_corpus_signature(fold, held_out)

    try:
        result = load_matching_fold_result(fold_config, signature)
    except (FileNotFoundError, ValueError):
        pass
    else:
        if RESUME_COMPLETED_FOLDS:
            print(
                f"[expanded-tokenizer] reusing completed fold {fold}: "
                f"test_auc={result.test_metrics['roc_auc']:.4f}",
                flush=True,
            )
            return result

    # Only materialize the (tens-of-GB) filtered corpus once we know this fold actually
    # needs (re)training, and always delete it afterwards -- it is fully consumed by
    # train_expanded_tokenizer() below and nothing later in this function reads it again.
    corpus_fold_directory = Path(
        tempfile.mkdtemp(prefix=f"leak_safe_fold{fold}_", dir=SCRATCH_ROOT)
    )
    try:
        build_leak_safe_corpus_fold(corpus, held_out, corpus_fold_directory)
        print(f"[expanded-tokenizer] fold {fold}: training tokenizer on {len(safe_experiment_ids)} "
              f"experiments ({held_out} held out) ...", flush=True)
        tokenizer, tokenizer_data, tokenizer_history, tokenizer_metrics = train_expanded_tokenizer(
            corpus_data_config(corpus_fold_directory),
            baseline.VQVAE_CONFIG,
            CORPUS_VQVAE_TRAINING_CONFIG,
            seed=baseline.SEED,
            device=baseline.DEVICE,
        )
    finally:
        shutil.rmtree(corpus_fold_directory, ignore_errors=True)
    print(
        f"[expanded-tokenizer] fold {fold}: tokenizer trained, "
        f"val_loss={tokenizer_metrics.get('loss', float('nan')):.4f}, "
        f"val_frame_loss={tokenizer_metrics.get('frame_loss', float('nan')):.4f}, "
        f"codebook_usage={tokenizer_metrics.get('codebook_usage_fraction', float('nan')):.2f}",
        flush=True,
    )

    result = train_fold_with_expanded_tokenizer(
        fold_config, tokenizer, tokenizer_data, tokenizer_history, tokenizer_metrics, signature
    )
    print(
        f"[expanded-tokenizer] fold {fold}: test_auc={result.test_metrics['roc_auc']:.4f} "
        f"test_macro_f1={result.test_metrics['macro_f1']:.4f}",
        flush=True,
    )
    return result


def train_cross_validation(
    folds: Sequence[int] = CROSS_VALIDATION_FOLDS,
) -> tuple[dict[int, TrainingResult], CrossValidationArtifacts]:
    fold_results = {fold: train_fold(fold) for fold in folds}
    summary_dir = experiment_output_dir() / f"cross_validation_seed-{baseline.SEED}"
    artifacts = aggregate_cross_validation_results(
        fold_results=fold_results,
        output_dir=summary_dir,
        threshold=baseline.TRAINING_CONFIG.threshold,
        artifact_prefix=f"{baseline.DATASET_PREFIX}_{EXPERIMENT_VARIANT}",
    )
    pooled = artifacts.summary["pooled_out_of_fold"]["metrics"]
    stats = artifacts.summary["fold_statistics"]["test"]["roc_auc"]
    print(
        f"=== expanded_tokenizer: pooled_auc={pooled['roc_auc']:.4f} "
        f"macro_f1={pooled['macro_f1']:.4f} "
        f"fold_mean={stats['mean']:.4f}+-{stats['standard_deviation']:.4f} ===",
        flush=True,
    )
    return fold_results, artifacts


def main() -> tuple[dict[int, TrainingResult], CrossValidationArtifacts]:
    return train_cross_validation()


if __name__ == "__main__":
    main()
