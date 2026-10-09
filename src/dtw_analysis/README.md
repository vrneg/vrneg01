# DTW temporal analysis

The implementation uses
[`aeon.distances.dtw_alignment_path`](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.distances.dtw_alignment_path.html)
from the repository's pinned Aeon dependency. It does not add `dtw-python` as a
dependency.

This standalone package answers two separate questions:

- **DTW separability:** does an observed modality trajectory discriminate `neg` from
  `none`, allowing modest timing misalignment?
- **Timing robustness:** how much does a fitted classifier's prediction change when
  selected modality timing is shifted or compressed/expanded?

The DTW experiment does not run or retrain the saved classifiers. Checkpoints provide
the fold-specific data configuration. Within every outer fold, scaling/PCA is fitted
on training trajectories, DTW window and neighbor count are selected on validation,
and preprocessing plus references are rebuilt from train+validation before untouched
test evaluation. The robustness experiment does run checkpoint inference, but never
updates model weights.

Run both analyses:

```bash
PYTHONPATH=src python -m dtw_analysis.cli \
  --checkpoints outputs/minirocket/target-cue_window-500_splits-10/fold-*/model.joblib \
  --output-dir outputs/dtw_analysis/minirocket \
  --mode both \
  --n-jobs 4 \
  --fold-jobs 1
```

Add `--attribution-summary path/to/modality_attribution/summary.json` to produce
rank correlations and a concordance/discrepancy table. Scores are never combined.

The primary DTW ranking is pooled conditional macro F1. “Conditional” is essential:
an absent stream is not converted into a zero trajectory. Coverage and a separate
presence-only baseline accompany every modality. Confidence intervals resample the
recording/session groups used to create the folds. Duplicate sample IDs within any
fold split or across outer-test rows fail by default; expected training-set overlap
between different CV folds is not flagged. Prefer repairing duplicates upstream;
use `--allow-duplicate-samples` only after auditing the source rows.

When that override is used, `duplicate_samples.csv` records every retained occurrence,
its fold, session, label, and unique analysis key.

Per fold and modality, observed feature rows are standardized and projected
with PCA (95% variance, at most 16 components by default). The validation split
selects the Sakoe-Chiba window from 0%, 10%, and 20% and inverse-distance k from
1, 3, and 5. Aeon's accumulated squared DTW cost is reported as root-mean-square
cost per alignment-path step, which makes unequal observed lengths more comparable.
All these defaults are exposed by `python -m dtw_analysis.cli --help`.
The CLI logs fold loading, every modality/window computation, every timing condition,
cache hits, aggregation, and elapsed times at `INFO` level. Use `--log-level WARNING`
for quieter batch runs or `--log-level DEBUG` when diagnosing execution.

Timing robustness evaluates shifts of ±1, ±2, and ±4 grid steps plus scales 0.8 and
1.2 around the target-word anchor for every modality and for all modalities together.
Continuous fixed-grid channels use linear interpolation, presence and finger-status
channels use nearest-neighbor interpolation, and values moved outside the observation
window are zeroed. Raw event-transformer timestamps are shifted/scaled before event
encoding. Predictions are cached below the output directory, so interrupted runs can
resume—particularly important for HIVE-COTE and TabPFN checkpoints.

ROCKET-PFN robustness uses a repeated-inference path: within each fold, the unchanged
training ROCKET representations are computed once, perturbation query rows are scored
together, and each TabPFN feature-group context is fitted once. `--fold-jobs N` adds a
second level of parallelism by evaluating up to `N` fold checkpoints in CUDA-safe
spawned processes inside the same Slurm job. Each worker loads its own model and uses
the configured device, so high values can exhaust GPU memory or oversubscribe a single
GPU; reduce `--fold-jobs` if that occurs. Fold caches are disjoint and remain resumable.
Caches written before the repeated-inference implementation are treated as stale to
avoid mixing inference schemes in one result summary.

Main artifacts are `summary.json` and `report.md`, plus pooled and per-fold CSV
tables, row-level JSONL predictions, representative PCA-space alignment paths, and
fingerprinted robustness caches. Run one complete CV repetition at a time; checkpoints
from repeated seeds share outer test rows and should be reported as separate repeated
CV analyses rather than pooled as independent samples.
