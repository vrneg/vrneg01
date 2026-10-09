# MASHT

This module implements [MASHT: In-Context Time Series Classification with Random
Convolutional Features](https://arxiv.org/abs/2607.19234v1), using the authors'
[reference implementation](https://github.com/joschac/masht) to resolve the paper's
exact feature-budget and TabPFN settings.

```text
                     +-> HYDRA ------> sparse scaling --+
multivariate series -+                                  +-> concatenate -> TabPFN-3
                     +-> MultiRocket -> no scaling -----+
```

This is one combined tabular representation and one TabPFN prediction, not the
feature-group probability ensemble used by RocketPFN. There is no supervised feature
selection. The random transforms are fitted only on the outer training split, and
validation/test labels are never passed to either transform or TabPFN.

## Paper-aligned defaults

- adaptive nominal feature budget: 10,000 below 1,000 samples, 2,000 below 100,000,
  and 200 otherwise;
- budget split equally between MultiRocket and HYDRA;
- MultiRocket budget split equally across the raw series and first differences,
  with four pooling features per kernel;
- HYDRA uses eight competing kernels per group and its sparse square-root scaler;
- one explicitly selected TabPFN-3 classifier with 8 estimators, automatic estimator
  scaling, `fit_mode="low_memory"`, automatic precision, 8 preprocessing jobs, no
  tuning, hidden progress bars, and ignored pretraining limits;
- CUDA in the editable entry point, matching the benchmark hardware configuration.

The paper selected the adaptive budget from train plus test count. Because this
project has separate validation and test splits, `feature_budget_scope="all_splits"`
uses `train + validation + test` sizes. This reveals no labels or feature values. For
the current 758/90/78 folds, the nominal budget is 10,000. With 32 time points,
integer allocation in the reference method yields 4,992 HYDRA and 4,704 MultiRocket
features, or 9,696 effective features.

The implementation reuses aeon 1.5's multivariate MultiRocket and experimental
multivariate HYDRA transforms instead of copying the authors' vendored GPL source.
The paper used aeon 1.4. The reference method's 200-feature large-dataset budget is
below aeon's minimum MultiRocket allocation; this module raises a clear error for
that case rather than failing inside compiled transform code.

## Run

```bash
venv/bin/python src/train_main_masht.py
```

The entry point uses the project-local checkpoint:

```text
data/tabpfn/tabpfn-v3-classifier-v3_default.ckpt
```

That directory is git-ignored because TabPFN weights have their own license. Set
`TABPFN_DEVICE="auto"` in the entry point for CPU fallback; TabPFN-3 with roughly
10,000 features is strongly recommended on a CUDA GPU. Folds are sequential by
default so multiple TabPFN instances do not contend for GPU memory.

Each fold artifact stores the fitted random transforms and training context, but not
another copy of the 203 MiB pretrained TabPFN checkpoint. Reloaded inference creates
TabPFN-3 lazily from the explicit local path.
