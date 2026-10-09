# Multivariate CASTOR

This module applies [CASTOR](https://arxiv.org/abs/2403.13176), the competing
dilated shapelet transform, to the same multivariate fixed-grid arrays and saved
group-separated folds used by MiniRocket and MultiRocket. CASTOR samples actual
training subsequences rather than random convolution kernels, groups the shapelets
so they compete at every time position, and passes three features per shapelet and
dilation to a ridge classifier.

The default configuration follows the paper rather than Wildboar's older package
defaults:

- 128 groups with 16 shapelets per group and shapelet length 9
- Euclidean distance, 50% normalized groups, and occurrence bounds 0.01 to 0.2
- soft minimum, hard maximum, and independent occurrence features
- 64 groups on the raw series and 64 groups on its first difference
- CASTOR's square-root sparse feature scaling
- leave-one-out `RidgeClassifierCV` over alpha values 0.01, 1, and 10

At 32 time points this yields 12,288 features: each raw/difference branch has two
exponential dilation levels, and each shapelet contributes three features. Settings
are editable in `src/train_main_castor.py`.

## Installation

CASTOR uses the official compiled `wildboar.transform.CastorTransform`. Wildboar
1.2.1 declares scikit-learn `<1.6`, while this project needs aeon 1.5 with
scikit-learn `>=1.6`. Install Wildboar without letting pip replace the project's
working scikit-learn version:

```bash
venv/bin/pip install --no-deps -r requirements-castor.txt
```

The module contains a narrow compatibility bridge for the single renamed validation
argument used by Wildboar 1.2.1. It uses the project's Ridge implementation instead
of Wildboar's older classifier wrapper.

## Running

From the repository root:

```bash
venv/bin/python src/train_main_castor.py
```

The default consumes the existing datasets
`target-cue_window-500_splits-10_fold-0` through `..._fold-9`; it does not generate
new folds. Runs are sequential and resumable by default because each fold fits a
large transform over all 698 channels. Per-fold models, metrics, validation/test
predictions, and the aggregated out-of-fold summary are stored below
`outputs/castor/`.

For a quick check, set `RUN_MODE = "single"`. For a smaller exploratory transform,
reduce `n_groups` and `n_shapelets`, but keep `n_groups` even while first differences
are enabled.
