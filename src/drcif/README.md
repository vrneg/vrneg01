# Diverse Representation Canonical Interval Forest

This package evaluates aeon 1.x's multivariate `DrCIFClassifier` on exactly the same
normalized 698-channel, 32-step fixed-grid arrays and experiment-separated folds as
the ROCKET and TCN experiments.

For each tree, DrCIF samples intervals from three representations:

1. the normalized raw series;
2. first-order differences;
3. a periodogram representation.

It extracts a random subset of catch22 and summary features from those intervals and
fits a decision tree. Probabilities are averaged over the forest. The standard run
uses 200 trees, the aeon `(4, "sqrt-div")` interval rule, ten sampled attributes per
interval, and four internal workers.

aeon accepts exactly constant traces but rejects a nonconstant trace whose standard
deviation is at or below `1e-7`. Differences and periodograms can create that kind of
numerical residue, and a short random slice can expose it even when the full
representation passes validation. The default safeguard collapses only those
effectively constant traces to their mean after representation construction and
again immediately before Catch22 extraction. It is applied identically during
fitting and prediction and is stored as part of the experiment configuration.

Edit `src/train_main_drcif.py`, then run from the repository root:

```bash
venv/bin/python src/train_main_drcif.py
```

The entry point defaults to sequential ten-fold cross-validation. Completed folds
resume only when their full saved configuration matches. Artifacts are written under
`outputs/drcif/<DATASET_PREFIX>/standard-200-trees/`.

DrCIF is substantially slower than ROCKET on this high-dimensional input. To perform
a time-contracted exploratory run, set `time_limit_in_minutes` and optionally lower
`contract_max_n_estimators` in the train-main script; use a separate experiment
variant name so contracted and standard results cannot be confused.
