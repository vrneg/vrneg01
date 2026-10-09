# Sparse MultiRocket-HYDRA

This package extends the existing MultiRocket+HYDRA benchmark for the small-sample,
high-dimensional setting. Each outer fold follows this pipeline:

```text
time series -> MultiRocket + HYDRA -> branch standardization -> top-k F selection
            -> inner-CV linear-model comparison -> prediction
```

The expensive convolutional transforms are fitted once using only the outer fold's
training split. Branch standardization, supervised `SelectKBest(f_classif)`, feature
count, classifier, and regularization are fitted or selected inside stratified inner
cross-validation. Neither the outer validation split nor the test split participates
in feature or model selection. The validation split is used only when optional
decision-threshold calibration is enabled.

The default search compares:

- Ridge classification;
- L2 logistic regression;
- elastic-net logistic regression, optimized with stochastic gradient descent;
- linear SVM.

Top-k candidates are 5,000, 10,000, and 15,000 features. If the transform produces
fewer features, candidates are clipped and deduplicated automatically. The default
inner objective is balanced accuracy. Per-fold artifacts record the winning model,
regularization, selected MultiRocket/HYDRA feature counts, exact non-zero coefficient
counts, and the best inner-CV result for every classifier family. A temporary joblib
cache reuses standardized matrices and selected subsets across regularization values;
it is deleted after each outer fold finishes.

The repository entry point also reuses each completed baseline fold's fitted
MultiRocket and HYDRA random transforms. The selected features are still recomputed
inside every inner training split. Set `REUSE_FITTED_MULTIROCKET_HYDRA=False` only
when changing transform or data settings; fitting fresh transforms can be much slower.
Inner search uses three workers by default and prints every completed fit.

Edit the constants in `src/train_main_sparse_multirocket_hydra.py`, then run from the
repository root:

```bash
venv/bin/python src/train_main_sparse_multirocket_hydra.py
```

Ten outer folds run sequentially by default. The inner search uses three workers,
bounded by `SEARCH_PRE_DISPATCH=3`, which is appropriate for the measured fold size
on this workstation. Reduce both values together if running under tighter memory
limits; parallel inner fits can duplicate the roughly 50,000-feature training matrix.
