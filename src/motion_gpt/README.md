# MotionGPT

MotionGPT treats motion as a language: the VQ-VAE turns each window into discrete tokens,
those tokens share one vocabulary with text, and an encoder-decoder transformer is trained
on a mixture of motion-language tasks before being instruction-tuned on the target task.
There is no command-line parser; build the configuration dataclasses and call
`train_motion_gpt` directly.

The reference method is Jiang et al., *MotionGPT: Human Motion as a Foreign Language*
(NeurIPS 2023). Compared with [`../t2m_gpt/`](../t2m_gpt/README.md) the differences that
matter here are the **bidirectional encoder-decoder** in place of a decoder-only stack, the
**unified motion/text vocabulary**, and **multi-task self-supervised pretraining** instead
of plain next-token prediction.

## Relationship to the T2M-GPT package

Phase one is deliberately shared. `DataConfig`, `VQVAEConfig`, `VQVAETrainingConfig`,
`TrainingConfig`, the fixed-grid conversion, and the tokenizer itself are imported from
`t2m_gpt`, exactly as `minirocket` imports from `event_transformer` elsewhere in this
repository. Both models therefore consume the **same channels, the same grid, the same
folds, and the same codes**, so a comparison between them isolates the stage-two
architecture rather than the preprocessing. `src/train_main_motion_gpt.py` keeps the
tokenizer block byte-identical to `src/train_main_t2m_gpt.py` for that reason.

## Module layout

- `config.py`: architecture, pretraining-mixture, and experiment dataclasses; re-exports
  the shared data/tokenizer/optimization configs.
- `vocabulary.py`: the unified identifier layout for motion codes, specials, task tokens,
  and answer words.
- `tasks.py`: the four task formats, span sampling, datasets, and the padding collator.
- `model.py`: T5-style encoder-decoder with autoregressive and classification heads.
- `data.py`: re-exports the shared fold pipeline and adds the instruction-sequence loader.
- `training.py`: all three phases, early stopping, checkpoints, and metric histories.
- `aggregation.py`: the shared cross-validation aggregator.
- `run_experiment.py` and `../train_main_motion_gpt.py`: entry points.

## The unified vocabulary

Motion codes keep the identifiers the VQ-VAE assigns; everything else is appended after
them:

| Range | Contents |
|---|---|
| `[0, num_codes)` | motion codes |
| next 5 | `<pad>` `<eos>` `<bos>` `<som>` `<eom>` |
| next `num_sentinels` | `<extra_id_i>` span-corruption sentinels |
| next 4 | `<task_denoise>` `<task_predict>` `<task_inbetween>` `<task_classify>` |
| last 2 | `none` `negation` |

The text side is intentionally tiny. The only natural language this task needs is a task
name and a two-word answer, so a pretrained subword tokenizer would add parameters without
adding information. `MotionLanguageVocabulary.describe` renders any sequence for
inspection, which is the fastest way to check a task format:

```
<task_denoise> <som> <motion_3> <motion_7> <motion_1> <extra_id_0> <eom>
  -> <extra_id_0> <motion_5> <motion_4> <eos>
```

## Tasks

Every task is one sequence-to-sequence problem, which is what lets a single model absorb
all of them.

| Task | Encoder sees | Decoder writes |
|---|---|---|
| `denoise` | motion with spans replaced by sentinels | the hidden spans, each behind its sentinel |
| `predict` | the head of the window | the tail |
| `inbetween` | both ends, middle replaced by a sentinel | the removed middle |
| `classify` | the whole window | `negation` or `none` |

The first three are self-supervised and form the pretraining mixture; one is drawn per
window per epoch. Corruption is resampled every epoch through
`PretrainingTaskDataset.set_epoch`, seeded from the run seed, so each window is corrupted
differently across epochs while the run stays reproducible.

On a few hundred windows these auxiliary tasks matter more than they do at the paper's
scale: they multiply the supervision extracted from each window without needing a label,
which is the most promising lever this dataset offers. Span sampling always leaves at least
one visible token and always produces at least one target, so no batch degenerates.

## Heads

`head="motion_to_text"` (default)
: MotionGPT's motion-to-text task with a two-word caption. The decoder is trained by
  cross-entropy to write the label word, and the score is the log-odds between the two
  answer tokens at the first decoder step. Because both answers are read at the same
  position, the difference of their raw logits already equals the difference of their
  log-probabilities. That score is an unnormalized ratio, so
  `calibrate_threshold_on_validation=True` is the default in the controller.

`head="discriminative"`
: A mean-pooled encoder state feeds a binary classifier trained with
  `BCEWithLogitsLoss`. The encoder is bidirectional, which suits classification better
  than T2M-GPT's causal pooling, at the cost of leaving the generative formulation behind.
  This is the more likely performer on small data; the motion-to-text head is the
  faithful one.

## Running one fold

```python
from src.motion_gpt.run_experiment import run_training

result = run_training(
    dataset_prefix="target-cue_window-500_splits-10",
    fold=1,
    head="motion_to_text",
)
print(result.test_metrics)
```

All folds, with aggregation:

```python
from src.motion_gpt.run_experiment import run_cross_validation

results, artifacts = run_cross_validation(fold_numbers=range(10))
```

Or edit the configuration block in `src/train_main_motion_gpt.py` and run:

```bash
python src/train_main_motion_gpt.py
```

## Outputs

Written below `outputs/motion_gpt/<DATASET_PREFIX>/<VARIANT>/<RUN_NAME>/`:
`motion_vqvae.pt`, `pretrained_backbone.pt`, `best_model.pt`, `metrics.json`,
`validation_predictions.jsonl`, `test_predictions.jsonl`. Histories are recorded per phase
under the stage names `tokenizer`, `motion_language_pretraining`, and
`instruction_tuning`. Checkpoints carry the configuration signature, normalization
statistics, channel names, time grid, vocabulary layout, and label convention;
`load_training_result` refuses a checkpoint from a different configuration, which is what
makes `RESUME_COMPLETED_FOLDS` safe.

Cross-validation writes `<PREFIX>_fold_metrics.csv`, `<PREFIX>_metric_summary.csv`, and
`<PREFIX>_cross_validation_summary.json`, including metrics recomputed from the pooled
out-of-fold predictions.

## Capacity, and the measured 10-fold result

An initial paper-shaped T2M-GPT stage two (~670K parameters) reached its best epoch on the
**first** pass over this dataset's 741 training windows, then overfit monotonically to
chance test AUROC — see [`../t2m_gpt/README.md`](../t2m_gpt/README.md). The dataclass
defaults here keep a paper-shaped model, but `src/train_main_motion_gpt.py` deliberately
runs a smaller one (`d_model=64`, 2+2 layers, `dim_feedforward=256`, `dropout=0.3`, ~120K
parameters, same right-sized tokenizer as T2M-GPT for comparability).

That single-fold comparison initially suggested MotionGPT clearly outperformed T2M-GPT
(test AUROC 0.62-0.63 vs 0.52 on fold 1). **The full 10-fold pooled result reverses this**:

| Model | Pooled OOF AUROC (10 folds, 919 windows) | 95% bootstrap CI |
|---|---:|---|
| T2M-GPT discriminative | 0.572 | [0.536, 0.609] |
| MotionGPT discriminative | 0.545 | [0.508, 0.582] |
| MotionGPT motion-to-text | 0.542 | [0.506, 0.581] |

None of these three differ from each other at conventional significance (paired
bootstrap/McNemar over shared windows), and all three sit below a MiniRocket baseline on
the identical grid (pooled AUROC 0.686 [0.652, 0.722], McNemar p ≤ 0.0004 against every
deep model here) — see the full comparison table and discussion in
[`../t2m_gpt/README.md`](../t2m_gpt/README.md). Fold 1 alone was simply favorable to
MotionGPT; a 90-window single-fold test split has a standard error near 0.06 on AUROC, well
within the swing observed here. Only pooled out-of-fold numbers across many folds are
worth interpreting, and even those currently favor a shallow feature-engineering baseline
over either sequence-model architecture on this amount of data.

## Deviations from the paper, and why

- **No pretrained language model.** MotionGPT initializes from T5 and inherits real
  language competence. There is no caption here to exploit it on, so the text side is a
  handful of learned tokens trained from scratch. This is the largest deviation and it
  removes the transfer that motivates the original architecture.
- **Learned absolute positions instead of relative position buckets.** With roughly ten
  tokens per sequence the bucketing has nothing to generalize over.
- **Instruction tuning on one task.** The paper tunes across many motion-language tasks
  and reports on all of them; the supervised phase here has a single task, so the
  multi-task benefit only enters through pretraining.
- **One joint codebook over all 698 channels**, inherited from the shared tokenizer. See
  the T2M-GPT README for why per-modality codebooks are the natural next step.
- **No generation metrics.** FID, diversity, and R-precision have no counterpart in a
  binary classification setting.
