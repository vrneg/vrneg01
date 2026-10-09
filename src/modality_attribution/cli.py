"""Command-line entry point for cross-validated modality attribution."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .analysis import (
    AnalysisConfig,
    analyze_cross_validation,
    attach_retraining_summary,
    load_analysis_result,
)
from .retraining import retrain_cross_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank input modalities for one classifier using held-out ablation, "
            "sampled Shapley attribution, and optional subset retraining."
        )
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        required=True,
        help="Two or more fold-matched model.joblib/best_model.pt checkpoints.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-shapley",
        action="store_true",
        help="Run held-out modality ablation without sampled Shapley analysis.",
    )
    parser.add_argument("--shapley-orderings", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console progress verbosity (default: INFO).",
    )
    parser.add_argument(
        "--allow-duplicate-samples",
        action="store_true",
        help=(
            "Preserve duplicate held-out rows and disambiguate them by fold/index. "
            "By default duplicates fail validation."
        ),
    )
    parser.add_argument(
        "--post-hoc-only",
        action="store_true",
        help=(
            "Do not train subset models. This is now the default unless "
            "a retraining flag is provided."
        ),
    )
    parser.add_argument(
        "--leave-one-out",
        action="store_true",
        help=(
            "Train whole-modality omissions plus the lower-face facial omission "
            "for each fold."
        ),
    )
    parser.add_argument(
        "--train-only-one-modality",
        "--modality-only",
        dest="train_only_one_modality",
        action="store_true",
        help="Train one model per fold and modality using only that modality.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Maximum number of subset retraining jobs to run in parallel "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help=(
            "Maximum inner CPU threads per retraining worker. By default the "
            "Slurm/OS CPU affinity count is divided across --n-jobs."
        ),
    )
    parser.add_argument(
        "--retraining-only",
        action="store_true",
        help=(
            "Reuse completed post-hoc outputs in --output-dir and run/resume only "
            "the requested subset retraining."
        ),
    )
    parser.add_argument(
        "--tabpfn-batch-groups",
        action="store_true",
        default=None,
        help=(
            "Opt into TabPFN's dataset-batch API across ROCKET feature groups. "
            "This can be faster on large GPUs but raises peak GPU memory and can "
            "slightly change probabilities."
        ),
    )
    parser.add_argument(
        "--no-tabpfn-progress",
        action="store_true",
        help="Disable each TabPFN estimator progress bar during retraining.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing retraining manifest and rerun subset models.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    retraining_variants = []
    if args.train_only_one_modality:
        retraining_variants.append("modality_only")
    if args.leave_one_out:
        retraining_variants.append("leave_one_out")
    if args.post_hoc_only and retraining_variants:
        parser.error("--post-hoc-only cannot be combined with retraining flags")
    if args.retraining_only and not retraining_variants:
        parser.error("--retraining-only requires a retraining flag")
    if args.retraining_only and args.post_hoc_only:
        parser.error("--retraining-only cannot be combined with --post-hoc-only")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least 1")
    if args.threads_per_job is not None and args.threads_per_job < 1:
        parser.error("--threads-per-job must be at least 1")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting modality attribution for %d fold checkpoints; output=%s",
        len(args.checkpoints),
        args.output_dir,
    )
    try:
        analysis = (
            load_analysis_result(args.output_dir, args.checkpoints)
            if args.retraining_only
            else analyze_cross_validation(
                args.checkpoints,
                AnalysisConfig(
                    output_dir=args.output_dir,
                    device=args.device,
                    batch_size=args.batch_size,
                    run_shapley=not args.skip_shapley,
                    shapley_orderings=args.shapley_orderings,
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                    allow_duplicate_samples=args.allow_duplicate_samples,
                ),
            )
        )
        if retraining_variants:
            if args.tabpfn_batch_groups and args.n_jobs > 1:
                logger.warning(
                    "Combining --tabpfn-batch-groups with %d GPU workers raises "
                    "peak device memory; benchmark a smaller --n-jobs first.",
                    args.n_jobs,
                )
            retraining = retrain_cross_validation(
                args.checkpoints,
                args.output_dir / "retraining",
                device=args.device,
                resume=not args.no_resume,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                variant_types=retraining_variants,
                n_jobs=args.n_jobs,
                threads_per_job=args.threads_per_job,
                tabpfn_batch_groups=args.tabpfn_batch_groups,
                tabpfn_show_progress_bar=(
                    False if args.no_tabpfn_progress else None
                ),
            )
            analysis = attach_retraining_summary(analysis, retraining.summary)
    except (FileNotFoundError, ValueError) as error:
        logger.error("Analysis stopped: %s", error)
        duplicate_audit = args.output_dir / "duplicate_samples.csv"
        if duplicate_audit.is_file() and duplicate_audit.stat().st_size:
            logger.error(
                "Duplicate audit: %s. After reviewing it, rerun with "
                "--allow-duplicate-samples to retain those rows.",
                duplicate_audit,
            )
        return 2
    print(f"Wrote attribution summary: {analysis.summary_path}")
    print(f"Wrote attribution report: {analysis.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
