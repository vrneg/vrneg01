# T2M-GPT

A two-stage T2M-GPT pipeline for the VR negation windows: a discrete motion tokenizer
(VQ-VAE) trained on the sensor streams, then a transformer over its codes that decides
whether the anchor word was a negation cue. There is no command-line parser; build the
configuration dataclasses and call `train_t2m_gpt` directly.

The reference method is Zhang et al., *T2M-GPT: Generating Human Motion from Textual
Descriptions with Discrete Representations* (CVPR 2023). Stage one is unchanged in
substance. Stage two replaces its CLIP text condition with the class label, because the
task here is classification rather than text-to-motion generation.

## Module layout

- `config.py`: data, VQ-VAE, tokenizer-training, GPT, pretraining, and optimization
  dataclasses.
- `data.py`: fold loading, the irregular-to-fixed-grid conversion, and torch adapters.
- `vqvae.py`: convolutional encoder/decoder and the EMA + reset codebook.
- `model.py`: causal transformer with autoregressive and classification heads.
- `training.py`: both stages, early stopping, checkpoints, and metric histories.
- `grouped.py`: an alternative tokenizer with one codebook per channel group instead of
  one joint codebook; see "Grouped/per-modality tokenization" below.
- `metrics` and `aggregation.py`: reused from `event_transformer`, so folds aggregate
  exactly like the other models in this repository.
- `../comparison_stats.py`: paired bootstrap and McNemar significance testing over any
  two models' pooled out-of-fold predictions, used for every comparison in this README.
- `run_experiment.py`: an editable direct-call script.
- `../train_main_t2m_gpt.py`: the complete single-fold/cross-validation controller.

## Loading folds

`DataConfig.dataset` is resolved in this order: a local `save_to_disk` directory, a local
directory of parquet files, then a Hugging Face Hub repository id. The Hub path uses the
ordinary datasets cache, so nothing is copied into the repository:

```python
DataConfig(dataset="VR-Faces-Neg/target-cue_window-500_splits-10_fold-1")
```

## Representation

Irregular sensor events become a dense `[channels, frames]` array with the same per-event
feature schema, actor scope, presence channels, and linear interpolation used by the
ROCKET and TCN baselines, so the tokenizer sees the representation those models saw. With
all eight modalities that is **698 channels**: continuous measurements plus motion
derivatives, the 13 finger status/pinch flags, and one presence channel per modality.

`modalities` restricts the tokenizer to a subset of streams for ablations, for example
`("Head", "Facial", "LeftHand", "RightHand")`. The channel order stays modality-major, so
`modality_channel_slices` gives contiguous bounds per stream.

Interpolation only fills the interval that a modality actually observed; frames outside it
stay at zero and the presence channel marks them. Because normalization is fitted on
events before interpolation, those zeros sit at the per-feature mean rather than at a
neutral value. This matches the baselines and is why the presence channels matter.

`num_time_points` frames become `num_time_points / 2 ** num_downsample_layers` motion
tokens. The default 32 frames over a ±0.5 s window yield **8 tokens**, a token rate close
to the original paper's (it downsamples 20 fps motion by 4). Use 64 frames for the
window-1000 datasets to hold the token rate constant. Treat both as hyperparameters and
keep them identical across compared systems.

## Stage one: motion tokenizer

The encoder is a strided 1D convolutional stack with dilated residual blocks; the decoder
mirrors it with nearest-neighbour upsampling. The loss is the paper's:

```
smooth_l1(reconstruction, target)
  + velocity_loss_weight * smooth_l1(diff(reconstruction), diff(target))
  + commitment_loss_weight * ||encoder_output - sg[code]||^2
```

The velocity term is not decoration: without it the decoder can score well by emitting
average poses and discarding the movement that carries the communicative signal.

The codebook is updated by exponential moving averages and is never touched by the
optimizer — gradients reach the encoder through the straight-through estimator. Codes that
attract fewer than `code_reset_threshold` vectors in a batch are re-seeded from current
encoder outputs. On a dataset this small that reset is what keeps the codebook from
collapsing onto a handful of entries.

Only the current fold's training split is used, and the epoch is selected on validation
reconstruction loss, so no test window influences the tokenizer.

**Watch the codebook.** `metrics.json` records `num_used_codes`,
`codebook_usage_fraction`, `token_entropy_nats`, and `mean_unique_codes_per_window` per
split. A usage fraction near zero, or a mean-unique count near 1, means the tokenizer
collapsed and stage two is reading noise. Lower `num_codes` before anything else.

## Stage two: transformer over motion tokens

Both heads share one backbone, so unconditional next-token pretraining initializes either.

`head="discriminative"` (default)
: A `[BOS]` prefix is prepended, the causal transformer reads the sequence, and the final
  position is pooled into one binary logit trained with `BCEWithLogitsLoss`.

`head="generative"`
: The class label takes the position T2M-GPT gives the text embedding. The model learns
  `p(tokens | class)` by cross-entropy on the true class prefix and classifies by the
  log-likelihood ratio `log p(tokens | neg) - log p(tokens | none)`. Because the objective
  is generative rather than discriminative, the reported `loss` is token cross-entropy,
  and the score is an unnormalized ratio — set
  `calibrate_threshold_on_validation=True` when using it. With balanced classes a
  threshold of 0.5 corresponds to the natural ratio-of-one decision.

Both apply the paper's corrupted-sequence strategy during training: a
`token_corruption_rate` fraction of input tokens is replaced by uniformly random codes,
which narrows the gap between teacher forcing and inference on imperfect prefixes.
Corruption is disabled in evaluation.

`use_pretraining=True` runs unconditional next-token prediction on the training split
first. Afterwards the pretrained prefix row is broadcast into every class prefix, so the
generative head starts from two trained conditions instead of one trained and one random.

## Running one fold

From the repository root:

```python
from src.t2m_gpt.run_experiment import run_training

result = run_training(
    dataset_prefix="target-cue_window-500_splits-10",
    fold=1,
    seed=42,
    head="discriminative",
)
print(result.test_metrics)
```

All folds, with aggregation:

```python
from src.t2m_gpt.run_experiment import run_cross_validation

results, artifacts = run_cross_validation(
    dataset_prefix="target-cue_window-500_splits-10",
    fold_numbers=range(10),
)
```

For the complete controller, edit the configuration block in
`src/train_main_t2m_gpt.py`, set `RUN_MODE` to `"single"` or `"cross_validation"`, then:

```bash
python src/train_main_t2m_gpt.py
```

Cross-validation uses spawn-based process workers when `USE_PROCESS_POOL` is enabled.
`MAX_PARALLEL_FOLDS` controls concurrency; processes are unrestricted and may exhaust
VRAM, so one job per GPU is the safe starting point.

## Outputs

Results are written below `outputs/t2m_gpt/<DATASET_PREFIX>/<VARIANT>/<RUN_NAME>/`:
`motion_vqvae.pt`, `pretrained_backbone.pt` (when enabled), `best_model.pt`,
`metrics.json`, `validation_predictions.jsonl`, and `test_predictions.jsonl`. Checkpoints
carry the configuration signature, normalization statistics, channel names, time grid,
label convention, decision threshold, and both stages' histories. `load_training_result`
refuses a checkpoint produced by a different configuration, which is what makes
`RESUME_COMPLETED_FOLDS` safe.

After cross-validation the shared aggregator writes `<PREFIX>_fold_metrics.csv`,
`<PREFIX>_metric_summary.csv`, and `<PREFIX>_cross_validation_summary.json`, including
metrics recomputed from the pooled out-of-fold predictions.

## Scale of the data, and what it means here

One fold of `target-cue_window-500_splits-10` holds **741 training, 95 validation, and 90
test windows** across 26/4/3 sessions, with roughly 18 observations per modality per
window. The paper's setting is a different regime entirely: HumanML3D has tens of
thousands of sequences and uses a 512x512 codebook.

A single-fold run with the paper-shaped defaults (256 codes, `d_model=128`, 3 layers,
670K parameters) overfit within one epoch (test AUROC 0.52, chance). Shrinking to the
current defaults (128 codes, `d_model=64`, 2 layers, ~120K parameters; see
`train_main_t2m_gpt.py`) fixed the immediate overfitting — best epoch moved from 1 to 2-11
across folds — but the pooled 10-fold cross-validation result below shows that this alone
does not make motion tokens competitive with a shallow feature-engineering baseline on
this amount of data.

### Full comparison, `target-cue_window-500_splits-10`, pooled out-of-fold AUROC (10 folds, 919 windows)

| Model | AUROC | 95% bootstrap CI |
|---|---:|---|
| **MiniRocket** (10,000 random-convolution kernels + ridge) | **0.686** | [0.652, 0.722] |
| Inception-TCN (modality-aware, supervised only) | 0.590 | [0.554, 0.627] |
| T2M-GPT discriminative (this package) | 0.572 | [0.536, 0.609] |
| MotionGPT discriminative | 0.545 | [0.508, 0.582] |
| MotionGPT motion-to-text | 0.542 | [0.506, 0.581] |

MiniRocket beats every other model here, including the two deep supervised baselines, at
McNemar p ≤ 0.0004 in every pairwise comparison. This is a known pattern for ROCKET-family
methods against learned architectures below a few thousand training instances (Middlehurst
et al., *Bake off redux*, 2023): a bank of thousands of fixed random kernels needs only a
linear readout, while every architecture here has to learn a representation from scratch
on 741 examples. None of the three token/attention-based models differ from each other at
conventional significance — a single fold is not enough to rank them, and the full pooled
comparison reverses what fold 1 alone suggested (MotionGPT initially looked ahead of
T2M-GPT there; pooled across all ten folds the order flips, though not significantly).

**Codebook variants did not close the gap.** Two follow-ups, both evaluated against the
identical fold subset of the baseline for a valid paired test:

| Variant | Folds | AUROC | vs. matched baseline | McNemar p |
|---|---:|---:|---|---:|
| Grouped codebook, 4 semantic groups (see `grouped.py`) | 5 | 0.570 | diff -0.010, CI [-0.069, +0.049] | 0.33 |
| Grouped codebook, 8 per-modality groups | 3 | 0.561 | diff +0.008, CI [-0.056, +0.082] | 0.40 |

Neither grouping scheme moved AUROC outside noise. `target-cueandscope_window-500` (≈3.7x
more windows via a broader label criterion) and `target-cue_window-1000` with
`num_time_points=64` (twice the tokens per window) were also run; **the two window sizes
turned out to share only 17% of their sample ids** — evidently different word selections
despite the deterministic `seed=42` used elsewhere in the pipeline, not the identical
label sets this package initially assumed — so the two are not validly paired-comparable,
and neither result (cueandscope 0.551 [0.525, 0.582]; window-1000 0.587 [0.537, 0.638],
each on 5 folds) is distinguishable from the baseline's own confidence interval either way.

Read this as a genuine negative result for this amount of data, not as a flaw in the
tokenizer: `metrics.json`'s codebook usage stayed healthy throughout (55-85% of codes in
active use, no collapse), so the bottleneck is what a learned sequence model can extract
from 741 examples, not a broken discretization. Before spending more compute on
architecture variants, the highest-value next step is a cheap, non-deep diagnostic: a
cross-correlation or DTW scan between individual kinematic channels and the negation label
to check which channels (if any) carry a detectable signal at what lag, and compare that
against what MiniRocket's fitted kernels are actually keying on. If MiniRocket's advantage
traces back to a handful of localized shapelet-like features, that argues for feature
engineering over more sequence-model capacity; if it does not, that argues for more data
before any architecture comparison here is conclusive.

## Grouped/per-modality tokenization

`grouped.py` gives each channel group its own encoder, codebook, and decoder instead of
one joint codebook over all 698 channels. Group *g*'s code *c* becomes the global token id
`offset[g] + c`, so the concatenated per-group token sequences feed the existing
`MotionTokenGPT`, pretraining, and instruction-tuning code completely unchanged — it never
sees anything but integers in `[0, total_codes)`.

```python
from src.t2m_gpt.grouped import train_t2m_gpt_grouped

SEMANTIC_GROUPS = (
    ("Eye", "Facial"),
    ("Head", "Body"),
    ("LeftHand", "RightHand"),
    ("LeftFinger", "RightFinger"),
)
result = train_t2m_gpt_grouped(config, SEMANTIC_GROUPS)
```

`groups` must partition all eight modality names with no overlap (`validate_grouping`
checks this). Every group reuses `config.vqvae`/`config.vqvae_training` unchanged, so
grouping scheme is the only variable between groups and between this and the
single-codebook baseline. `load_grouped_training_result` mirrors `load_training_result`
but also refuses a checkpoint trained with a different grouping.

This is markedly more expensive than the joint tokenizer: training *N* independent
tokenizers costs roughly *N* times the tokenizer phase alone, which measured about 3.5x
wall time for 4 groups on this dataset. The measured result (above) found no significant
difference from the joint codebook at either 4 or 8 groups, so the added cost is not
currently justified by an accuracy gain on this dataset size — but the mechanism is
useful independent of that: it is also how you would give the 452 finger channels their
own capacity rather than competing with the other modalities for a shared 128-code budget
in a larger dataset.

## Deviations from the paper, and why

- **Class condition instead of text.** The task is binary classification of an anchor
  word, so there is no caption to encode. The generative head keeps the conditional
  formulation; the discriminative head drops it for a pooled logit.
- **One joint codebook over all channels.** The paper tokenizes a single 263-dimensional
  pose vector. Here 698 heterogeneous channels share one codebook, which lets
  high-dimensional finger and face streams dominate the reconstruction. Per-modality
  codebooks are the natural next step and the reason `modality_channel_slices` exists;
  until then, `modalities` allows the same comparison by restriction.
- **Smaller capacity defaults** than the published 512x512 codebook, for the reasons
  above.
- **No generation-quality metrics.** FID and diversity against a motion feature extractor
  have no counterpart here; codebook usage and reconstruction loss serve as the stage-one
  diagnostics instead.
