# HIVE-COTE 2.0 upper-bound experiment

This package wraps aeon 1.x's `HIVECOTEV2`, an implementation maintained by the
algorithm's research group. It combines the four HIVE-COTE 2.0 components using the
paper's fourth-power CAWPE weighting:

1. Shapelet Transform Classifier (STC);
2. Diverse Representation Canonical Interval Forest (DrCIF);
3. Arsenal, an ensemble of ROCKET classifiers;
4. Temporal Dictionary Ensemble (TDE).

The model uses the same normalized 698-channel, 32-step arrays and the same saved
outer folds as the other experiments. The DrCIF component retains the project's
interval-level numerical safeguard; its statistical features and forest behavior
are otherwise unchanged. HC2 converts the prepared arrays to contiguous `float64`
once because aeon 1.5's STC Numba kernel combines them with double-precision
normalized shapelets; fold 0's training tensor occupies about 129 MiB at this dtype.

## Compute policy

The aeon HC2 components are NumPy, Numba, and scikit-learn implementations. They do
not expose a CUDA backend, so this entry point deliberately uses CPU parallelism.
Moving the input through PyTorch to a GPU would not accelerate the underlying
algorithms.

The train-main default is `contract-6h-paper-caps`:

- an approximate six-hour total contract per fold;
- four internal CPU workers;
- paper-default component sizes as hard caps;
- one fold at a time by default to avoid multiplying a very large memory footprint;
- component start/finish and elapsed-time messages;
- a `status.json` file that records loading, training, prediction, saving, failure,
  or completion;
- atomic model checkpoint replacement and completed-fold resumption.

Aeon's time contract is approximate. It allocates one sixth of the requested total
to each component and a component must finish its current work unit, so short
contracts can overrun. Set `time_limit_in_minutes=0` to request all paper-default
component counts without a time limit. This may take many hours or days per fold on
the full multivariate data.

Component probabilities are saved alongside ensemble predictions, making it
possible to inspect the four members or build later error-diversity analyses without
refitting HC2.

### Running all outer folds concurrently

Outer-fold parallelism is separate from aeon's internal component parallelism. In
`src/train_main_hivecote_v2.py`, set:

```python
PARALLEL_FOLDS = True
MAX_PARALLEL_FOLDS = None
```

`None` means all selected folds that are not already complete may run at once. For
ten folds with the default `n_jobs=4`, this can use roughly 40 CPU workers at peak,
plus native-library threads, and up to ten times the memory of one fold. To limit
only the outer concurrency, set (for example) `MAX_PARALLEL_FOLDS = 4`. To target a
roughly one-hour contract for every independently running fold, also set
`HIVECOTE_CONFIG.time_limit_in_minutes=60.0`.

Every process writes to its own fold directory and `status.json`; completed-fold
resumption and final out-of-fold aggregation work in both modes. Console messages
from concurrent components can be interleaved.

Edit `src/train_main_hivecote_v2.py`, then run from the repository root:

```bash
venv/bin/python src/train_main_hivecote_v2.py
```

Artifacts are written under
`outputs/hivecote_v2/<DATASET_PREFIX>/contract-6h-paper-caps/`.

Paper: <https://arxiv.org/abs/2104.07551>
