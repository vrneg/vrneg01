# Frame tagger

Frame-level BIO sequence labelling over the VR sensor grid: instead of "does this window
contain a negation cue", label every frame `O` / `B-CUE` / `I-CUE` / `B-SCOPE` / `I-SCOPE`.
This is the NER-shaped formulation of the same annotation, and it is the direct approach to
"locate a temporal span in a sensor stream" — no motion tokenizer, no codebook, no
pretraining, and no temporal downsampling to blur a boundary.

A bidirectional encoder (BiLSTM or Transformer) reads the continuous fixed-grid channels and
a linear-chain CRF turns per-frame scores into well-formed spans. Bidirectionality is
deliberate: a negation gesture can begin before the word it belongs to, so a frame's label
depends on frames after it too.

## The labels come from data already in the folds

Window-level datasets carry one binary label, but the annotation behind them is finer. Each
row's `word["neg"]` holds the negation record with `cue_tokens` and `scope_tokens`, and a
SurrealDB `Word` id is the composite `[timeMs, Player, index]` — **the onset timestamp is
literally its first element**. Frame targets are therefore derived from the existing fold
datasets with no database access.

| Quantity | Status |
|---|---|
| Every token's **onset** | exact, read from its record id |
| The **anchor word's offset** | exact (`endTime - timeMs == duration * 1000`, verified) |
| **Non-anchor token offsets** | **assumed** — the token references carry no duration |

Non-anchor tokens get the median anchor-word duration of the training split (200ms on fold 0).
`FrameLabelStats` reports how many spans used the assumption so it is visible in every result:
on fold 0's training split, 397 of 1560 spans have a measured duration and 1163 are assumed.

Sanity check that the time alignment is right: an anchor cue span opens on the first frame at
a non-negative offset — frame 16 of 32 on the ±0.5s grid — and scope follows it ~180ms later,
matching the annotation.

## Why overlap-based span F1, not just exact match

Exact span matching is the standard NER criterion and it is reported. But it is too harsh for
*these* labels: three quarters of spans have an assumed offset, and the 32-frame grid quantizes
time to ~32ms, so a perfect prediction can still miss the reference end frame. Overlap matching
(intersection-over-union ≥ `minimum_span_overlap`, default 0.5) measures what the model can
actually be held to. Both are reported so the gap is visible rather than hidden by the choice.

Frame accuracy is reported but should be ignored as a headline: about 80% of frames are `O`, so
predicting `O` everywhere scores 0.8 while finding nothing. `TaggerMetrics.primary` is overlap
micro F1, and that is what model selection uses.

## Comparability with the window models

Every run also writes a window-level score — each window's maximum per-frame positive
marginal — into `test_predictions.jsonl` using the **same schema** as every other model here
(`sample_id`, `label`, `logit`, `probability`, `prediction`). `src/comparison_stats.py` reads it
directly, so a tagger can be bootstrap- and McNemar-compared against MiniRocket, T2M-GPT, and
MotionGPT on the same folds instead of being an isolated experiment.

The per-frame score comes from CRF forward–backward marginals, not from the Viterbi path,
because a threshold-free score is what an AUROC comparison needs.

## Running

```python
from src.frame_tagger import train_frame_tagger
from src.train_main_frame_tagger import build_experiment_config

result = train_frame_tagger(build_experiment_config(fold=0))
print(result.test_metrics["span_overlap"]["micro"])
```

Or edit the configuration block in `src/train_main_frame_tagger.py` and run:

```bash
python src/train_main_frame_tagger.py
```

The data half is a `t2m_gpt.DataConfig`, so `representation=RepresentationConfig(...)` switches
the channel representation exactly as it does for the window models — see
[`../representation/README.md`](../representation/README.md).

## Outputs

Written below `outputs/frame_tagger/<DATASET_PREFIX>/<VARIANT>/<RUN_NAME>/`: `best_model.pt`,
`metrics.json`, `validation_predictions.jsonl`, `test_predictions.jsonl`. Prediction rows carry
the decoded `tags` list alongside the window-level columns. `metrics.json` records the label
statistics and the assumed token duration used, so a run can be interpreted without re-deriving
its targets.

## Module layout

- `labels.py`: recovers cue/scope spans from token record ids and renders BIO frame targets.
- `crf.py`: linear-chain CRF — forward algorithm, Viterbi, forward–backward marginals, and the
  BIO transition constraints (`I-X` may only follow `B-X`/`I-X`).
- `model.py`: `FrameEncoder` (BiLSTM or Transformer) and `FrameTagger` (encoder + CRF).
- `metrics.py`: exact and overlap span scoring, frame metrics, `evaluate_tagging`.
- `config.py`, `training.py`, `../train_main_frame_tagger.py`: configuration, training loop, entry point.

## Known limitations

- **`none` windows are all-`O` by construction.** They were selected to be ≥20s from any
  negation, so they contain no spans. The tagger therefore never sees a hard negative where a
  negation is nearby but outside the window.
- **Assumed offsets** dominate scope spans; a scope span built from three assumed-duration
  tokens is a coarser target than a cue span anchored on a measured word.
- **Class imbalance is real but mild**: ~20% of frames are in-span, because windows are 1s and
  the window-level dataset is roughly balanced. `TaggerMetrics.primary` and the CRF's
  sequence-level loss handle this without needing frame reweighting.
