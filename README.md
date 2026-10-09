# VR_NEG_FACES

VR_NEG_FACES is an experimental framework for detecting verbal negation from
multimodal VR behavior around an anchor word. It compares random-convolution,
shapelet, dictionary, interval-forest, TabPFN, temporal-convolution, event-attention,
and motion-token classifiers under the same group-separated cross-validation setup.

The event streams are `Eye`, `Facial`, `Head`, `Body`, `LeftHand`, `RightHand`,
`LeftFinger`, and `RightFinger`; the binary target is `neg` versus `none`. Most models
consume a shared fixed-grid representation, while the Event Transformer operates
directly on irregular event times. Every model has an editable
`src/train_main_<suffix>.py` controller.

## Contents

- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Running models](#running-models)
- [Model catalog](#model-catalog)
- [Outputs and result collection](#outputs-and-result-collection)
- [Modality attribution](#modality-attribution)
- [DTW separability and timing robustness](#dtw-separability-and-timing-robustness)
- [Repository layout](#repository-layout)

## Quick start

With Python 3.12, from the repository root:

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

CASTOR needs one additional compiled dependency whose published requirements conflict
with the project's scikit-learn version. Install it without dependency resolution
only when using CASTOR:

```bash
venv/bin/pip install --no-deps -r requirements-castor.txt
```

Place locally saved Hugging Face `DatasetDict` folds under `data/trainsets/`, then
start a configured controller or several at once:

```bash
venv/bin/python src/train_main_minirocket.py

venv/bin/python src/train_main_all.py \
  --dataset-family tgt-cue_bal-05_src-speaker_wL-1000_wR-1000_spl-10 \
  --max-workers 2 \
  minirocket multirocket inception_tcn
```

Run the test suite with:

```bash
venv/bin/python -m unittest discover -s tests
```

## Datasets

### Families, folds, and windows

A dataset family is the shared part of the fold names, without `_fold-N`:

```text
data/trainsets/<DATASET_PREFIX>_fold-<N>
tgt-<target>_bal-<balance>_src-<event source>_wL-<left>_wR-<right>_spl-<n folds>[_ctrl-...]
```

- `<target>`: `cue`, `scope`, or `cueandscope`.
- `<event source>`: `speaker` (events of the anchor's speaker), `listener` (events of
  the dialogue partner), or `both`.
- `<left>`, `<right>`: signed context distances in milliseconds. The left value is
  subtracted from the anchor onset and the right value is added; negative values are
  written with an `m` prefix. `wL-1000_wR-m500` selects $[-1000,-500]$ ms and
  `wL-m500_wR-1000` selects $[500,1000]$ ms. The start must be earlier than the end.
- `_ctrl-...`: control and anchor selection, see below.

Folds use stratified group splitting with the recording session as the group. For
outer fold *k*, partition *k* is the test set, partition *k*+1 the validation set,
and the remaining partitions the training set, so no recording crosses a split
boundary. Normalization, channel and feature selection, random transforms, and
learned representations are fitted on training data according to each model's
documented protocol.

T2M-GPT and MotionGPT also accept parquet directories or Hub dataset IDs through their
data configuration; `DATASET_NAMESPACE` in their train-main files sets the Hub
namespace.

### Creating datasets

[`src/main_utils/dataset_creation.py`](src/main_utils/dataset_creation.py) reads the
source events from SurrealDB, selects anchor and control words, builds the context
windows, and writes group-separated folds to `data/trainsets/`. Database and optional
Hugging Face credentials live in the ignored `data/surreal/` and `data/hf/`
directories. Call `create_dataset(...)` with the target, event source, window, number
of splits, control and anchor modes, and upload policy (`upload=True` additionally
pushes every fold to the configured Hugging Face organization; `upload=False` keeps
them local). The training entry point's dataset family must match the generated
prefix.

`_dataset_creation_main()` and `_dataset_creation_main_sliding_window()` generate the
configured window grids (one-sided cumulative windows and ten contiguous 500 ms
windows from -2500 to +2500 ms, for `speaker`, `listener`, and `both`). Executed as a
script, the module first inventories the configured Hugging Face organization and
skips configurations whose fold repositories already exist there.

### Control and anchor selection

The control mode decides which non-cue words become negative examples. It is set per
call with `control_mode=...` or for the whole module with `CONTROL_SELECTION_MODE`:

| Mode | Controls |
|---|---|
| `coarse` | Random words outside any negation cue or scope that lie at least `min_distance_ms` from every negation cue; fixed seed; no speaker matching. |
| `strict` | One control per cue, sampled without replacement from the same experiment and speaker and at least 1 s from the cue onset. The search starts in the cue's audio chunk and then widens around the cue (±10 s, ±20 s, …); cues without any eligible control are dropped. |
| `very_strict` | One control per cue from the same experiment and speaker with the same coarse part of speech (spaCy Universal POS) and the same early/middle/late position within the speaker's turn; any audio chunk, no minimum distance. |
| `neg` | One word from the cue's own negation scope, at least 1.1 s from the cue onset (cue target only); cues without an eligible scope word are skipped. |

`anchor_mode=...` (`all`, `coarse`, `strict`, `very_strict`, or `neg`) fixes the set
of positive anchors independently, namely the cues that are retainable under that
mode. This compares control modes on identical anchors; a control mode that cannot
match every fixed anchor raises an error instead of shrinking the set. If only one of
the two modes is given, it is used for both. `strict`, `very_strict`, and `neg`
require a balance of `None` or `0.5`. Selection is seeded, so the same anchors and
controls are used across windows and event sources.

The mode appears in the dataset name as follows:

| Modes | Name suffix |
|---|---|
| different anchor and control modes | `_ctrl-<control>_anchors-<anchor>` |
| `very_strict` | `_ctrl-very-strict` |
| `neg` | `_ctrl-neg` |
| `strict` via `anchor_mode="strict"` | `_ctrl-strict` |
| `coarse`, or `strict` via `control_mode`/`CONTROL_SELECTION_MODE` only | none |

Unsuffixed names therefore do not identify their control mode. Keep datasets and
outputs of different modes in separate destinations (or use the suffixed spelling),
because a new dataset with the same name replaces the existing folders.

## Running models

Each `train_main_<suffix>.py` is an editable experiment controller rather than a
command-line interface. Review its configuration block before a long run:

| Setting | Purpose |
|---|---|
| `RUN_MODE` | One fold, full cross-validation, or a model-specific mode such as repeated-seed CV. |
| `DATASET_PREFIX` | Dataset family. |
| `SINGLE_FOLD`, `CROSS_VALIDATION_FOLDS` | Outer folds. |
| `SEED` | Model and split-dependent randomness. |
| `OUTPUT_ROOT`, `EXPERIMENT_VARIANT` | Separate artifact directories per variant. |
| `DEVICE`, `GPU_DEVICES`, `TABPFN_DEVICE` | CPU/CUDA execution where supported. |
| `USE_PROCESS_POOL`, `MAX_PARALLEL_FOLDS` | Fold-level concurrency. |
| `RESUME_COMPLETED_FOLDS` | Reuse configuration-matched completed folds where supported. |
| Model/training config objects | Representation size, estimator parameters, optimization, early stopping, evaluation. |

TabPFN-based models additionally need the version-matched checkpoint under
`data/tabpfn/` and one-time acceptance of the Prior Labs model license. `TABPFN_TOKEN`
is read from the environment or `.env`; never commit tokens or checkpoints. See the
[RocketPFN guide](src/rocket_pfn/README.md).

### Several models or several datasets in parallel

[`src/train_main_all.py`](src/train_main_all.py) runs several controllers for one
dataset family; [`src/train_main_all_one_model.py`](src/train_main_all_one_model.py)
runs one controller across several families:

```bash
venv/bin/python src/train_main_all.py \
  -d tgt-cue_bal-05_src-speaker_wL-1000_wR-1000_spl-10 -j 2 \
  minirocket castor weasel_v2

venv/bin/python src/train_main_all_one_model.py --model minirocket -j 2 \
  tgt-cue_bal-05_src-speaker_wL-100_wR-100_spl-10 \
  tgt-cue_bal-05_src-speaker_wL-1000_wR-1000_spl-10
```

Each job runs in a fresh interpreter: the runner imports the controller, overrides
`DATASET_PREFIX`, sets `WINDOW_START_SECONDS`/`WINDOW_END_SECONDS` from the family
name, and calls `main()`. Families for fixed-grid models must contain
`wL-LEFT_wR-RIGHT` (or the older `window-MS` / `windowL-LEFT_windowR-RIGHT`
spellings); `wL-2500_wR-500` becomes $[-2.5,+0.5]$ s and `wL-1000_wR-m500`
becomes $[-1.0,-0.5]$ s. Everything else (run mode, folds, seeds, devices, internal
concurrency, hyperparameters, resume behavior) stays in the individual controller.

Invalid or duplicate suffixes and families are rejected before training starts.
`-j/--max-workers` bounds the number of concurrent jobs; without it, every job gets
its own worker. Controllers may additionally use fold-level processes, estimator
threads, or GPU workers (notably the Event Transformer and HIVE-COTE), so measure
RAM, CPU, and GPU usage before running many jobs at once. Console output can
interleave; the runner summarizes each exit status and exits nonzero if any job
fails.

## Model catalog

The suffix is the value accepted by `train_main_all.py` and by `--model` of
`train_main_all_one_model.py`.

| Suffix | Model and role | Compute profile | Guide |
|---|---|---|---|
| `minirocket` | Multivariate MiniRocket features with ridge classification. | CPU; moderate RAM. | [MiniRocket](src/minirocket/README.md) |
| `multirocket` | MultiRocket over raw and differenced series, optionally concatenated with HYDRA features. | CPU; high feature-matrix RAM. | [MultiRocket + HYDRA](src/multirocket/README.md) |
| `sparse_multirocket_hydra` | MultiRocket+HYDRA with leakage-safe top-k feature selection and inner-CV linear-model comparison. | CPU; expensive inner search, high RAM. | [Sparse MultiRocket-HYDRA](src/sparse_multirocket_hydra/README.md) |
| `selfrocket` | SelF-Rocket representation/pooling candidate selection with ridge classification. | CPU; high temporary feature RAM. | [SelF-Rocket](src/selfrocket/README.md) |
| `castor` | Competing dilated shapelets (Wildboar CASTOR) with ridge classification. | CPU; compiled optional dependency. | [CASTOR](src/castor/README.md) |
| `weasel_v2` | Randomized dilated SFA dictionaries with training-only channel screening and ridge classification. | CPU; sequential folds recommended. | [WEASEL 2.0](src/weasel_v2/README.md) |
| `mrsqm` | Multivariate symbolic subsequence mining with channel screening and logistic regression. | CPU; compiled MrSQM extension. | [MrSQM](src/mrsqm_model/README.md) |
| `drcif` | Diverse Representation Canonical Interval Forest over raw, differenced, and periodogram views. | CPU; slower than ROCKET. | [DrCIF](src/drcif/README.md) |
| `hivecote_v2` | HIVE-COTE 2.0 ensemble of STC, DrCIF, Arsenal, and TDE under contracts. | CPU; multi-hour contracts, large RAM. | [HIVE-COTE 2.0](src/hivecote_v2/README.md) |
| `rocket_pfn` | ROCKET feature groups classified by TabPFN, with a MultiRocket+HYDRA extension. | CUDA strongly recommended; sequential folds. | [RocketPFN](src/rocket_pfn/README.md) |
| `masht` | One combined MultiRocket+HYDRA representation classified by TabPFN-3. | CUDA strongly recommended. | [MASHT](src/masht/README.md) |
| `adaptive_rocket_pfn` | Multi-view candidate banks, stability selection, semantic TabPFN-3 experts, and OOF ensemble weighting. | CPU features plus CUDA inference; very expensive. | [Adaptive RocketPFN](src/adaptive_rocket_pfn/README.md) |
| `inception_tcn` | Eight modality-specific Inception/TCN branches with presence-aware pooling. | CUDA recommended; compact. | [Inception-TCN](src/inception_tcn/README.md) |
| `compact_fusion_tcn` | Cue-aware shared fusion-TCN with modality dropout, pre/post-cue pooling, and repeated-seed CV. | CUDA recommended; compact. | [Compact fusion-TCN](src/compact_fusion_tcn/README.md) |
| `transformer` | Event Transformer over irregular modality observations, actor relations, and relative timing, with optional masked-modality pretraining. | CUDA recommended; cost grows with event count. | [Event Transformer](src/event_transformer/README.md) |
| `t2m_gpt` | VQ-VAE motion tokenizer followed by a causal token Transformer for discriminative or generative classification. | CUDA recommended; two-stage training. | [T2M-GPT](src/t2m_gpt/README.md) |
| `motion_gpt` | Shared motion tokenizer plus a T5-style encoder-decoder trained on motion-language pretext tasks and classification. | CUDA recommended; three-stage training. | [MotionGPT](src/motion_gpt/README.md) |

## Outputs and result collection

Artifacts are written under `outputs/<model>/<DATASET_PREFIX>/<variant>/`. A fold
directory contains a `model.joblib` or `best_model.pt` checkpoint, `metrics.json`
with configuration and diagnostics, held-out validation and test predictions, and
model-specific artifacts. After cross-validation, the aggregators write fold metrics,
metric summaries, and a JSON summary with metrics recomputed from pooled out-of-fold
predictions; exact filenames are listed in each model guide. `data/`, `outputs/`,
virtual environments, `.env`, and TabPFN weights are ignored by Git.

Three collectors gather fold-average test and validation AUROC and macro-F1 for the
cue target. Each run in their JSON output keeps a sorted `fold_results` list with the
complete per-fold metrics, run name, and decision threshold for interval estimation;
experiment variants and seeds remain separate runs. `--outputs-dir` and `--output`
set the input and destination paths.

| Collector | Compares | Options |
|---|---|---|
| `collect_model_results_symmetric_model_comparison.py [speaker\|listener\|both]` | All models over symmetric windows; ranks models across windows and windows across models (variants and seeds averaged per model/window first). | `--symmetric-windows 100 500 1000 2500` selects the window sizes in ms. |
| `collect_model_results_asymmetric_source_temp_comparison.py [model]` | One model over all asymmetric windows and the three event sources; ranks sources and windows. | Positional model output-directory name; `--event-sources`. |
| `collect_model_results_selected_windows.py <model> --windows L,R ...` | One model over an ordered list of windows, written as `LEFT,RIGHT` with `mN` for negative values (`2500,m2000` is $[-2.5,-2.0]$ s). Missing windows are listed in `missing_windows`. | `--event-sources`, `--strict`. |

Dataset families without a `src-...` or `source-...` field are treated as speaker
datasets. Compare models only when they share family, fold definition, actor scope,
label convention, time grid/event policy, and held-out sample set, and prefer pooled
out-of-fold metrics over conclusions drawn from a single small test fold.

## Modality attribution

[`src/modality_attribution`](src/modality_attribution/README.md) keeps three questions
separate: held-out full-model ablation (reliance on each modality), sampled Shapley
values (per-sample logit effects), and modality-only / leave-one-out retraining
(sufficiency and unique value). Run it over at least two fold-matched checkpoints:

```bash
PYTHONPATH=src venv/bin/python -m modality_attribution.cli \
  --checkpoints "outputs/minirocket/<DATASET_PREFIX>/<variant>/fold-*_seed-<seed>/model.joblib" \
  --output-dir outputs/modality_attribution/minirocket \
  --post-hoc-only
```

`--post-hoc-only` runs ablation and Shapley analysis without fitting new models;
without it, 16 additional variants per fold are trained (one modality-only and one
leave-one-out model per modality), which is resumable but can be prohibitive for
HIVE-COTE, TabPFN, and deep models. `--device cuda` runs supported CUDA checkpoints.
Shapley values are on the logit scale (positive supports `neg`) and describe
predictive associations, not causal effects. Outputs include `summary.json`,
`report.md`, pooled tables, and `sample_attributions.jsonl`. Adapters exist for
`model.joblib` estimators, the Event Transformer, Inception-TCN, and compact
fusion-TCN; T2M-GPT and MotionGPT checkpoints are not registered.

## DTW separability and timing robustness

[`src/dtw_analysis`](src/dtw_analysis/README.md) offers three modes: `dtw` tests
whether modality trajectories separate `neg` from `none` under modest temporal
misalignment, `robustness` shifts or rescales modality timing and measures the change
in a fitted classifier's predictions, and `both` runs the two analyses without
combining their scores.

```bash
PYTHONPATH=src venv/bin/python -m dtw_analysis.cli \
  --checkpoints "outputs/minirocket/<DATASET_PREFIX>/<variant>/fold-*_seed-<seed>/model.joblib" \
  --output-dir outputs/dtw_analysis/minirocket \
  --mode both --n-jobs 4 \
  --shift-steps -2 2 --time-scales 0.8 1.2
```

DTW preprocessing and PCA are fitted inside each outer fold, and the validation split
selects the Sakoe-Chiba window and neighbor count before test evaluation. Missing
streams are not converted into zero trajectories; coverage and a presence-only
baseline are reported separately. Timing robustness only runs inference and caches its
predictions for resumption. Adding
`--attribution-summary outputs/modality_attribution/minirocket/summary.json` reports
rank correlations and a concordance table against modality attribution. Use one
complete cross-validation repetition at a time; repeated seeds share test rows and
must not be pooled as independent observations.

Both interpretation workflows reject duplicate held-out sample IDs and write an audit
when duplicates occur; `--allow-duplicate-samples` permits them after review.

## Repository layout

```text
VR_NEG_FACES/
├── README.md
├── requirements.txt
├── requirements-castor.txt
├── src/
│   ├── train_main_all.py
│   ├── train_main_all_one_model.py
│   ├── train_main_<suffix>.py
│   ├── <model_package>/
│   ├── modality_attribution/
│   ├── dtw_analysis/
│   └── main_utils/
├── tests/
├── data/       # local folds, credentials, and checkpoints; ignored
└── outputs/    # training and analysis artifacts; ignored
```

The source code is licensed under the [MIT License](LICENSE). External datasets,
compiled dependencies, pretrained checkpoints, and third-party model weights retain
their own licenses and access terms.
