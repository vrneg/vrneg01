# RocketPFN

This experimental package combines random convolutional time-series features with
in-context classification by a pretrained TabPFN model. Its default mode follows
[RocketPFN (O'Rourke, Trisovic, and Bertsimas, 2026)](https://arxiv.org/abs/2606.21786v1):

```text
time series -> 10 independent ROCKET groups (1,000 kernels each)
            -> 10 matrices with 2,000 features each
            -> TabPFN v2.5 per group
            -> mean predicted probability
```

Each ROCKET kernel produces max and PPV features. The transforms use per-instance
z-normalization through aeon's `Rocket(normalise=True)`. TabPFN performs its own
tabular preprocessing, so this package does not add a feature scaler. Validation and
test inputs are predicted together in one batch per group because separate TabPFN
calls would recompute the training context.

The ten feature groups run sequentially by default. This preserves the original
independent TabPFN prediction path and keeps peak memory low. Setting
`TABPFN_BATCH_GROUPS=True` in `src/train_main_rocket_pfn.py` enables the experimental
batched-dataset path, but TabPFN 8.1 handles internal constant columns differently in
multi-dataset batches. Consequently, batched probabilities are not guaranteed to
equal sequential ones and peak memory is substantially higher. Use
`src/benchmark_rocket_pfn_group_batching.py` to compare both paths explicitly.

The default explicitly selects TabPFN v2.5 and eight internal estimators, matching the
paper rather than silently following the latest TabPFN default. Set
`TABPFN_VERSION="3"` in the entry point for the paper's v3 ablation.

## MultiRocket+HYDRA extension

`FEATURE_REPRESENTATION="multirocket_hydra"` enables a project-specific extension:

```text
time series -> MultiRocket + HYDRA raw features
            -> training-only ANOVA top-k selection
            -> rank-balanced groups of at most 2,000 features
            -> TabPFN per group -> mean probability
```

Feature ranking is fitted using outer-training labels only. Validation and test labels
never participate. A compatible fold-matched baseline artifact can supply the already
fitted random transforms; its ridge classifier and scalers are ignored.

## Installation and execution

Install `requirements.txt`, then run:

```bash
venv/bin/python src/train_main_rocket_pfn.py
```

The project entry point explicitly uses the version-matched checkpoint under
`data/tabpfn/`, rather than the user-level TabPFN cache. The default v2.5 checkpoint
is `data/tabpfn/tabpfn-v2.5-classifier-v2.5_default.ckpt`. This directory is ignored
by Git because the weights have their own non-commercial license. GPU execution is
strongly recommended; `TABPFN_DEVICE="auto"` selects it when available. Folds run
sequentially by default because each TabPFN inference can occupy most of a GPU.

TabPFN's local weights require one-time Prior Labs license acceptance. Visit
<https://ux.priorlabs.ai>, accept the license, and copy the API key from the account
page. For PyCharm, either add `TABPFN_TOKEN` to the run configuration's environment
variables or create the following project-root `.env` file:

```dotenv
TABPFN_TOKEN=<your-api-key>
```

The repository already ignores `.env`. Never commit or share the token. RocketPFN
checks access before loading a fold, so a missing or rejected credential fails quickly.

The saved fold artifact contains the fitted random transforms and training context,
but not another copy of the large pretrained TabPFN network. Calling the reloaded
model's `predict_proba` recreates TabPFN from the explicitly configured model version.
