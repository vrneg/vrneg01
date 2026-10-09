"""CLI for cross-validated DTW separability and checkpoint timing robustness."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .analysis import run_analysis
from .config import DTWAnalysisConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze modality-wise temporal separability with Aeon DTW and/or "
            "stress fitted checkpoints with controlled timing perturbations."
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
    parser.add_argument(
        "--mode", choices=("dtw", "robustness", "both"), default="both"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--fold-jobs",
        type=int,
        default=1,
        help=(
            "Robustness fold checkpoints to evaluate concurrently in spawned "
            "processes (default: 1; DTW folds remain sequential)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console progress verbosity (default: INFO).",
    )
    parser.add_argument(
        "--dtw-windows",
        nargs="+",
        type=float,
        default=(0.0, 0.1, 0.2),
        help="Sakoe-Chiba window proportions selected on each validation split.",
    )
    parser.add_argument(
        "--neighbors",
        nargs="+",
        type=int,
        default=(1, 3, 5),
        help="Odd inverse-distance k-NN counts selected on validation.",
    )
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--pca-max-components", type=int, default=16)
    parser.add_argument(
        "--shift-steps", nargs="+", type=int, default=(-4, -2, -1, 1, 2, 4)
    )
    parser.add_argument(
        "--time-scales", nargs="+", type=float, default=(0.8, 1.2)
    )
    parser.add_argument("--event-num-time-points", type=int, default=32)
    parser.add_argument("--event-window-start-seconds", type=float, default=-0.5)
    parser.add_argument("--event-window-end-seconds", type=float, default=0.5)
    parser.add_argument(
        "--attribution-summary",
        type=Path,
        help="Optional modality-attribution summary.json for rank comparison.",
    )
    parser.add_argument(
        "--allow-duplicate-samples",
        action="store_true",
        help="Retain audited duplicate split rows and add unique sample keys.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger(__name__).info(
        "Starting %s analysis for %d fold checkpoints; output=%s",
        args.mode,
        len(args.checkpoints),
        args.output_dir,
    )
    try:
        result = run_analysis(
            args.checkpoints,
            DTWAnalysisConfig(
                output_dir=args.output_dir,
                mode=args.mode,
                device=args.device,
                batch_size=args.batch_size,
                n_jobs=args.n_jobs,
                fold_jobs=args.fold_jobs,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                pca_variance=args.pca_variance,
                pca_max_components=args.pca_max_components,
                dtw_windows=tuple(args.dtw_windows),
                neighbor_counts=tuple(args.neighbors),
                shift_steps=tuple(args.shift_steps),
                time_scales=tuple(args.time_scales),
                event_num_time_points=args.event_num_time_points,
                event_window_start_seconds=args.event_window_start_seconds,
                event_window_end_seconds=args.event_window_end_seconds,
                allow_duplicate_samples=args.allow_duplicate_samples,
                attribution_summary_path=args.attribution_summary,
            ),
        )
    except ValueError as error:
        logging.getLogger(__name__).error("Analysis stopped: %s", error)
        duplicate_audit = args.output_dir / "duplicate_samples.csv"
        if duplicate_audit.is_file() and duplicate_audit.stat().st_size:
            logging.getLogger(__name__).error(
                "Duplicate audit: %s. After reviewing it, rerun with "
                "--allow-duplicate-samples to retain those rows.",
                duplicate_audit,
            )
        return 2
    print(f"Wrote DTW analysis summary: {result.summary_path}")
    print(f"Wrote DTW analysis report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
