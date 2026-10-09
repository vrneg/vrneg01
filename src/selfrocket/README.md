# Multivariate SelF-Rocket

This package adapts [SelF-Rocket](https://arxiv.org/abs/2409.01115) to the
project's fixed-grid multivariate event representation. It uses MiniRocket's 84
fixed length-nine kernels through aeon's multivariate MultiRocket machinery, then
constructs the feature candidates described by the paper:

- input representations: `BASE`, first difference (`DIFF`), and their concatenation
  (`MIX`);
- pooling operators: PPV, zero crossings (ZC), MPV, MIPV, and LSPV;
- 15 representation/pooling candidates in the full configuration.

The wrapper-selection stage evaluates random feature subsets with repeated
stratified ridge classifiers, selects the highest median validation accuracy, and
applies the paper's vote-validation fallback to `PPV_MIX` or `ZC_MIX`. The selected
complete feature set is used by the final ridge classifier. Selection diagnostics
are stored in both `model.joblib` and `metrics.json`.

Edit `src/train_main_selfrocket.py`, then run from the repository root:

```bash
venv/bin/python src/train_main_selfrocket.py
```

The defaults match the paper's v5 experimental settings: 10,000 nominal kernels,
two selection folds, ten repeated runs, 2,500 random features per mini-classifier,
a 500-sample selection cap, and top-5 vote validation at a 0.9 threshold. Set
`only_mix=True` for the faster five-candidate MIX ablation.

SelF-Rocket temporarily materializes ten full pooling/representation blocks. Keep
fold-level process parallelism disabled until peak memory use for one fold has been
measured.
