# Multivariate MrSQM

This module uses the official `mrsqm.MrSQMTransformer` to create multiple symbolic
representations, mine subsequences, apply the default RS random-then-chi-squared
feature selection, and train a simple logistic-regression classifier. It uses the
same fixed-grid arrays and saved ten-fold split as the other models.

MrSQM is natively multivariate, but it repeats every symbolic representation and its
feature budget for every channel. With 698 channels that produces an impractically
large dense feature matrix. The module therefore performs leakage-safe supervised
channel screening on the training split and retains eight channels by default. The
selected channel names and scores are saved with each model.

Defaults are:

- RS selection with 2,000 random candidates and 500 chi-squared-selected features
  per representation
- five SFA representation families, SFA normalization, and randomized first
  differences
- balanced L2 logistic regression with `C=1`

The project package is named `mrsqm_model` so it does not shadow the third-party
Python package named `mrsqm`. Version 0.0.7 is listed in `requirements.txt`; on
Python 3.12 pip builds its Cython/C++ extension from source. The dependency is
GPL-3.0, while this repository only wraps its public transformer interface.

Run from the repository root:

```bash
venv/bin/python src/train_main_mrsqm.py
```

The default consumes folds 0 through 9, runs sequentially and resumably, and writes
outputs under `outputs/mrsqm/`. Set `RUN_MODE = "single"` for a one-fold run.
