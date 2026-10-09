# Event Transformer

This package trains a binary classifier directly on the irregular VR sensor events in
the saved Hugging Face `DatasetDict` folds. It contains no command-line parser; use the
Python configuration objects and call `train_event_transformer` directly.

## Module layout

- `config.py`: data, model, pretraining, optimization, and experiment dataclasses.
- `features.py`: fixed feature schemas for the eight stored event types.
- `data.py`: chronological encoding, training-only normalization, anchor insertion,
  per-modality downsampling, and padded batching.
- `model.py`: modality encoders, time encoder, Transformer, and classifier.
- `pretraining.py`: masked-modality reconstruction using the training split only.
- `metrics.py`: binary, macro/per-class, ranking, and threshold-calibration metrics.
- `training.py`: early stopping, evaluation, checkpoints, and metric histories.
- `run_experiment.py`: an editable direct-call training script.
- `../train_main.py`: complete single-fold/cross-validation controller with optional
  process-pool execution.
- `aggregation.py`: fold statistics and pooled out-of-fold evaluation.

## Representation

Events with the same sensor timestamp and actor are encoded separately by modality and
then fused into one temporal Transformer token. In the current data this reduces the median
sequence from roughly 360 modality observations to 44 temporal tokens. The continuous
feature sizes include raw measurements plus derived motion:

| Modality | Raw features | Motion features | Total |
|---|---|---|---:|
| Eye | two gaze confidences, validity flags, poses | linear/angular eye velocity | 30 |
| Facial | 63 expression weights, confidences, validity | blendshape velocity | 130 |
| Head, Body, LeftHand, RightHand | position and quaternion | linear/angular velocity | 13 each |
| LeftFinger, RightFinger | tracked poses, bones, confidences, strengths | pose/bone/pinch velocity | 226 each |

Finger `status` and `pinches` are bitmasks rather than ordinal measurements. Their
8 and 5 binary flags are sent through a separate bias-free flag encoder before temporal
fusion. They are therefore not counted as continuous input dimensions.

Finger records without `boneRotations` are not tokens. In the inspected saved data,
every such record also has `status=0`, zero confidence/scale, and zero pointer/root
poses: it reports unavailable tracking rather than an observed hand event. Filtering
these records avoids filling a large feature vector with synthetic zeros, prevents
normalization from turning missing zeros into artificial signals, and shortens event
sequences. Normalization for finger measurements is consequently fitted only on valid
tracked-hand records.

Database identifiers, absolute timestamps, counters, and raw participant IDs are not
features. Each modality observation receives a learned relation embedding indicating whether
it belongs to the anchor-word speaker, the other participant, or an unknown actor.

The time encoder receives three values in seconds:

1. time relative to the anchor word;
2. time since the previous distinct sensor timestamp;
3. time until the next distinct sensor timestamp.

An `[ANCHOR]` token is inserted chronologically at `t=0`, and a `[CLS]` token supplies
the pooled window representation.

## Running one fold

From the repository root:

```python
from pathlib import Path

from src.event_transformer.run_experiment import run_training

result = run_training(
    dataset_path=Path(
        "data/trainsets/target-cue_window-1000_splits-10_fold-0"
    ),
    run_name="cue-window1000-fold0-seed42",
    seed=42,
    use_pretraining=True,
)
print(result.test_metrics)
```

To train the same model across all predefined folds:

```python
from src.event_transformer.run_experiment import run_cross_validation

fold_results = run_cross_validation(
    dataset_prefix="target-cue_window-1000_splits-10",
    fold_numbers=range(10),
    seed=42,
    use_pretraining=True,
)
```

`use_pretraining` is a keyword flag and defaults to `True` in the single-fold,
cross-validation, and lower-level training functions. Pass `use_pretraining=False` for
a run initialized from random weights. Pretraining masks complete modality/time slots
and reconstructs their normalized continuous features from the remaining modalities,
timing, actor relation, and temporal context. Finger status/pinch flags are hidden when
their finger observation is masked. Only the current fold's training split is used, so
validation and test examples do not leak into initialization.

Alternatively, edit `run_experiment.py` and run:

```bash
python -m src.event_transformer.run_experiment
```

For the complete training controller, edit the configuration block in
`src/train_main.py`.
Set `RUN_MODE` to `"single"` or `"cross_validation"`, then run:

```bash
python src/train_main.py
```

CUDA (`cuda:0`) is the default. Cross-validation uses spawn-based process workers,
which are safe for CUDA. `MAX_PARALLEL_FOLDS` controls concurrency and `GPU_DEVICES`
assigns jobs round-robin. Ten jobs can target one GPU, but all processes are unrestricted
and may use the available VRAM; if their combined allocations do not fit, PyTorch raises
a CUDA out-of-memory error. One job per GPU is the safe starting point.

When `show_progress=True`, the main process renders one persistent progress line per
fold. Worker processes send epoch results back through a spawn-safe queue, so concurrent
training logs do not overwrite one another. Each line first reports masked-modality
reconstruction loss, then training loss, validation loss, validation AUROC, validation
macro F1, the current best epoch, and final test macro F1.

All results are stored below `outputs/event_transformer/<DATASET_PREFIX>/`. Each fold
writes `pretrained_backbone.pt` (when enabled), `best_model.pt`, `metrics.json`,
`validation_predictions.jsonl`, and `test_predictions.jsonl`. The pretraining history
is included in both checkpoints and in `metrics.json`. After
cross-validation, the pipeline writes files whose names also begin with `DATASET_PREFIX`:

- `<DATASET_PREFIX>_fold_metrics.csv`: validation/test metrics for every fold;
- `<DATASET_PREFIX>_metric_summary.csv`: mean, sample standard deviation, standard
  error, median, min, and max for every score metric;
- `<DATASET_PREFIX>_cross_validation_summary.json`: all fold statistics and metrics
  recomputed from the pooled out-of-fold predictions, including AUROC and average
  precision.

Validation loss controls fine-tuning early stopping and `ReduceLROnPlateau` in the
provided experiment scripts. It is less variable than per-epoch AUROC on validation
folds containing only a few sessions. `TrainingConfig` also supports a short
`freeze_backbone_epochs` classifier-only warm-up and a separate
`classifier_learning_rate`; after warm-up, the complete model is fine-tuned with the
backbone learning rate. The freeze is skipped for random-initialization runs. Training
histories record both rates and the frozen state.

After restoring the selected checkpoint, validation and test metrics use the fixed
decision threshold of `0.5` by default. This avoids high-variance threshold fitting on
a small validation fold. The checkpoint contains this threshold,
model/data/pretraining/training configuration, label convention, feature dimensions,
normalization statistics, and pretraining history.

## Sequence length

Attention is quadratic in the number of temporal tokens. The default first retains at
most 128 uniformly spaced observations from each modality and then fuses observations
that share a timestamp. Set `max_events_per_modality=None` in `DataConfig` to retain
every observation. Treat this value as a model hyperparameter and keep it identical
across compared systems.
