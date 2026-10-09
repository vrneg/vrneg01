"""Significance-test the mirror-augmentation and window/capacity variants.

Which comparisons are legitimate here is not uniform, and the distinction matters:

* **Same dataset, different training recipe** (mirror on/off, small/large capacity) evaluate on
  the *identical* held-out windows, so these are paired: paired bootstrap on the shared sample
  ids plus McNemar on the thresholded decisions.
* **Different dataset** (window-500 vs window-1000, cue vs cueandscope) do *not* share their
  windows -- the non-negation words are an independent ``rng.sample`` draw per export, so
  window-500 and window-1000 share all 463 positives but only 4 of ~460 negatives. Pairing
  those is invalid; ``comparison_stats._align_by_sample_id`` refuses it by design. They are
  reported as independent bootstrap CIs and compared by whether those CIs overlap.

Run::

    python src/compare_mirror_and_window.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from comparison_stats import (
        PooledPredictions,
        bootstrap_metric_ci,
        load_pooled_predictions_from_output_dir,
        mcnemar_test,
        paired_bootstrap_comparison,
    )
except ModuleNotFoundError as error:  # pragma: no cover - import-path fallback
    if error.name != "comparison_stats":
        raise
    from .comparison_stats import (
        PooledPredictions,
        bootstrap_metric_ci,
        load_pooled_predictions_from_output_dir,
        mcnemar_test,
        paired_bootstrap_comparison,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
NUM_RESAMPLES = 4000
SEED = 42
METRIC = "roc_auc"
EXPECTED_FOLDS = 10

#: Every contrast examined, in one place, because the *count* is part of interpreting them.
#: The window-1000 2x2 decomposition below tests five contrasts among four cells; at
#: alpha = 0.05 that expects roughly one spurious "significant" result, and exactly one
#: contrast (both-vs-neither, the largest of the five) clears zero while neither main
#: effect does. A Bonferroni-corrected alpha = 0.05/5 = 0.01 does not sustain it. Keeping
#: the list explicit is what makes that correction possible to state honestly.
COMPARISONS: tuple[tuple[str, str], ...] = (
    # Paired: same dataset, different training recipe.
    ("w500_small_mirror", "w500_small_plain"),
    # The window-1000 2x2: main effects, the combination, and the two conditionals.
    ("w1000_large_plain", "w1000_small_plain"),
    ("w1000_small_mirror", "w1000_small_plain"),
    ("w1000_large_mirror", "w1000_small_plain"),
    ("w1000_large_mirror", "w1000_large_plain"),
    ("w1000_large_mirror", "w1000_small_mirror"),
    # Unpaired: different dataset exports (see module docstring).
    ("w1000_small_plain", "w500_small_plain"),
    ("minirocket_cue_w1000", "minirocket_cue_w500"),
    ("minirocket_cueandscope_w500", "minirocket_cue_w500"),
)


@dataclass(frozen=True, slots=True)
class RunRef:
    """One completed cross-validation run, identified by its output directory."""

    name: str
    directory: Path
    dataset: str

    def load(self) -> PooledPredictions:
        return load_pooled_predictions_from_output_dir(self.name, self.directory)

    def is_complete(self) -> bool:
        """Whether all ``EXPECTED_FOLDS`` folds hold pooled test predictions.

        Two distinct traps this guards against. A fold directory appears as soon as
        training starts, so its existence is not evidence of a finished fold. And a
        single-fold verification run leaves behind a directory that pools perfectly
        happily into a confident-looking CI over one fold's ~85 windows -- which is the
        exact "2-fold screening result" mistake recorded in FINDINGS.md section 2, in a
        new disguise. Only a full 10-fold run is comparable.
        """

        folds = list(self.directory.glob("fold-*_seed-*"))
        if len(folds) != EXPECTED_FOLDS:
            return False
        return all((fold / "test_predictions.jsonl").is_file() for fold in folds)


def t2m_gpt_runs() -> list[RunRef]:
    """The factorial cells, skipping any that have not been run yet."""

    root = OUTPUT_ROOT / "t2m_gpt"
    candidates = [
        RunRef("w500_small_plain", root / "target-cue_window-500_splits-10/discriminative", "cue_w500"),
        RunRef("w500_small_mirror", root / "target-cue_window-500_splits-10/discriminative_mirror", "cue_w500"),
        RunRef("w1000_small_plain", root / "target-cue_window-1000_splits-10/discriminative", "cue_w1000"),
        RunRef(
            "w1000_small_mirror",
            root / "target-cue_window-1000_splits-10/discriminative_mirror",
            "cue_w1000",
        ),
        RunRef(
            "w1000_large_plain",
            root / "target-cue_window-1000_splits-10/discriminative_cap-large",
            "cue_w1000",
        ),
        RunRef(
            "w1000_large_mirror",
            root / "target-cue_window-1000_splits-10/discriminative_mirror_cap-large",
            "cue_w1000",
        ),
    ]
    return [run for run in candidates if run.is_complete()]


def minirocket_runs() -> list[RunRef]:
    root = OUTPUT_ROOT / "minirocket"
    variant = "ridge-alpha-1e-3-to-1e6"
    candidates = [
        RunRef("minirocket_cue_w500", root / f"target-cue_window-500_splits-10/{variant}", "cue_w500"),
        RunRef("minirocket_cue_w1000", root / f"target-cue_window-1000_splits-10/{variant}", "cue_w1000"),
        RunRef(
            "minirocket_cueandscope_w500",
            root / f"target-cueandscope_window-500_splits-10/{variant}",
            "cueandscope_w500",
        ),
    ]
    return [run for run in candidates if run.is_complete()]


def _interval(pooled: PooledPredictions) -> dict[str, Any]:
    result = bootstrap_metric_ci(pooled, METRIC, num_resamples=NUM_RESAMPLES, seed=SEED)
    return {
        "num_windows": int(len(pooled.labels)),
        "positive_rate": float(pooled.labels.mean()),
        "point_estimate": result.point_estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
    }


def _paired(first: PooledPredictions, second: PooledPredictions) -> dict[str, Any]:
    comparison = paired_bootstrap_comparison(
        first, second, METRIC, num_resamples=NUM_RESAMPLES, seed=SEED
    )
    mcnemar = mcnemar_test(first, second)
    return {
        "comparison": f"{first.name} - {second.name}",
        "paired": True,
        "difference": comparison.difference_point_estimate,
        "ci_lower": comparison.difference_lower,
        "ci_upper": comparison.difference_upper,
        "probability_first_better": comparison.probability_first_better,
        "num_windows": comparison.num_windows,
        "mcnemar_p_value": mcnemar.p_value,
    }


def _unpaired(first: dict[str, Any], second: dict[str, Any], label: str) -> dict[str, Any]:
    overlap = not (
        first["ci_lower"] > second["ci_upper"] or second["ci_lower"] > first["ci_upper"]
    )
    return {
        "comparison": label,
        "paired": False,
        "reason": (
            "different dataset exports do not share their negative windows "
            "(independent rng.sample draw per export), so a paired test is invalid"
        ),
        "difference": first["point_estimate"] - second["point_estimate"],
        "confidence_intervals_overlap": overlap,
    }


def main() -> dict[str, Any]:
    report: dict[str, Any] = {"metric": METRIC, "num_resamples": NUM_RESAMPLES, "per_run": {}}

    runs = t2m_gpt_runs() + minirocket_runs()
    loaded = {run.name: run.load() for run in runs}
    datasets = {run.name: run.dataset for run in runs}
    for name, pooled in loaded.items():
        report["per_run"][name] = {**_interval(pooled), "dataset": datasets[name]}
        row = report["per_run"][name]
        print(
            f"{name:30s} n={row['num_windows']:5d} AUROC={row['point_estimate']:.4f} "
            f"[{row['ci_lower']:.4f},{row['ci_upper']:.4f}]"
        )

    comparisons: list[dict[str, Any]] = []
    print()
    for first_name, second_name in COMPARISONS:
        if first_name not in loaded or second_name not in loaded:
            print(f"skipping {first_name} vs {second_name}: not run yet")
            continue
        same_dataset = datasets[first_name] == datasets[second_name]
        if same_dataset:
            entry = _paired(loaded[first_name], loaded[second_name])
            excludes_zero = entry["ci_lower"] > 0.0 or entry["ci_upper"] < 0.0
            entry["ci_excludes_zero"] = excludes_zero
            print(
                f"{entry['comparison']:45s} PAIRED   delta={entry['difference']:+.4f} "
                f"[{entry['ci_lower']:+.4f},{entry['ci_upper']:+.4f}]"
                f"{'*' if excludes_zero else ' '} "
                f"McNemar p={entry['mcnemar_p_value']:.4f}"
            )
        else:
            entry = _unpaired(
                report["per_run"][first_name],
                report["per_run"][second_name],
                f"{first_name} - {second_name}",
            )
            print(
                f"{entry['comparison']:45s} UNPAIRED delta={entry['difference']:+.4f} "
                f"CIs overlap={entry['confidence_intervals_overlap']}"
            )
        comparisons.append(entry)

    paired = [entry for entry in comparisons if entry["paired"]]
    flagged = [entry for entry in paired if entry.get("ci_excludes_zero")]
    report["multiple_comparisons"] = {
        "num_paired_contrasts": len(paired),
        "num_ci_excludes_zero": len(flagged),
        "expected_false_positives_at_alpha_0.05": round(0.05 * len(paired), 2),
        "bonferroni_alpha": round(0.05 / max(len(paired), 1), 4),
        "note": (
            "The 95% CIs above are uncorrected. Divide alpha by the number of paired "
            "contrasts before calling any single one significant; a contrast that only "
            "just excludes zero will not survive it."
        ),
    }
    print(
        f"\n* = uncorrected 95% CI excludes zero "
        f"({len(flagged)} of {len(paired)} paired contrasts; "
        f"~{0.05 * len(paired):.1f} expected by chance at alpha=0.05, "
        f"Bonferroni alpha={0.05 / max(len(paired), 1):.4f})"
    )

    report["comparisons"] = comparisons
    destination = OUTPUT_ROOT / "comparison_mirror_and_window.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {destination}")
    return report


if __name__ == "__main__":
    main()
