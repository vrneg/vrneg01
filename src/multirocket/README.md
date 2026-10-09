# Multivariate MultiRocket with optional Hydra

This package applies aeon's MultiRocket transform to the same fixed-grid arrays and
saved group-separated folds as the MiniRocket experiment. It reuses the MiniRocket
data representation so model comparisons do not introduce preprocessing differences.

MultiRocket generates convolutional features from both the raw series and its first
difference, using four pooling features per kernel. The default 6,250 kernels
produce approximately 49,728 features before ridge classification.

Set `USE_HYDRA` in `src/train_main_multirocket.py`:

- `False` runs plain MultiRocket.
- `True` concatenates a separately scaled Hydra feature branch before ridge fitting.

Both variants use the extended ridge grid from `1e-3` through `1e6`. Plain and Hydra
runs are stored in separate output directories and completed configuration-matched
folds resume automatically.

From the repository root, run:

```bash
venv/bin/python src/train_main_multirocket.py
```

The default is sequential ten-fold cross-validation because transformed feature
matrices can consume substantially more memory than MiniRocket. Increase fold-level
parallelism only after observing peak RAM use for one fold.
