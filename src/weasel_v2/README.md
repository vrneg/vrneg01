# Multivariate WEASEL 2.0

This module implements WEASEL 2.0's randomized dilated SFA dictionaries on the same
fixed-grid representation and ten saved folds as the ROCKET experiments. Each
configuration randomly chooses window size, dilation, word length, binning, and
raw/first-difference input. The resulting symbolic word counts are classified with
the paper's `RidgeClassifierCV` alpha grid.

The published algorithm and aeon implementation are univariate. Applying one full
WEASEL ensemble to each of this dataset's 698 channels would destroy the controlled
memory property. This module therefore ranks channels using class-trajectory
separation on the training split only, retains 32 by default, and distributes the
single fixed WEASEL ensemble across them. Validation and test labels never influence
channel selection. Selected indices, names, scores, and configuration counts are
stored in every fold artifact.

The remaining defaults follow WEASEL 2.0:

- minimum window 4, word lengths 7 or 8, binary alphabet, no bigrams
- equi-depth/equi-width binning and variance-based Fourier coefficient selection
- raw plus first-difference dictionaries
- automatic ensemble size 50, 100, or 150; these folds use 100
- chi-squared top-k feature control and Ridge alphas from `10^-1` through `10^5`

Run from the repository root:

```bash
venv/bin/python src/train_main_weasel_v2.py
```

The entry point consumes the existing
`target-cue_window-500_splits-10_fold-0` through `..._fold-9`, runs sequentially,
resumes configuration-matched folds, and writes results under `outputs/weasel_v2/`.
Set `RUN_MODE = "single"` for a one-fold run.
