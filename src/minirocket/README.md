# Multivariate MiniRocket

This package provides a CPU-based MiniRocket baseline over the same saved
`DatasetDict` folds used by the event Transformer. It uses aeon's faithful multivariate
MiniRocket transform, training-only normalization, `StandardScaler(with_mean=False)`,
and `RidgeClassifierCV`.

The ridge search uses half-decade steps from `1e-3` through `1e6`. The expanded
upper range is intentional: all folds of the original experiment selected the old
maximum of `1e3`.

## Representation

Each irregular window is converted to an equal-length array with shape
`[channels, time]`. Continuous event features and derived motion features reuse the
event-transformer feature extractor. Finger status/pinch flags are retained as binary
channels. Each modality optionally receives a presence channel.

Continuous streams are linearly interpolated only between their first and last observed
timestamps. Values outside that range remain zero after training-only normalization.
Traces containing only sub-`1e-7` floating-point residue are collapsed to a constant;
this avoids aeon's low-variation guard without rescaling numerical noise.
The default 500 ms experiment uses 32 points on the fixed interval `[-0.5, 0.5]`
seconds. MiniRocket requires at least nine time points.

`actor_scope` controls which observations are used:

- `"anchor"`: the participant speaking the anchor word;
- `"other"`: the interlocutor;
- `"all"`: all actors, averaging collisions in the same modality/time channel.

## Running

Edit `src/train_main_minirocket.py`, then run from the repository root:

```bash
python src/train_main_minirocket.py
```

The controller supports a single fold and complete cross-validation. Sequential folds
are the safe default because dense fixed-grid and MiniRocket feature arrays can use
substantial RAM. Output is stored under
`outputs/minirocket/<DATASET_PREFIX>/` and includes a joblib model artifact, metrics,
validation/test predictions, and the same cross-validation CSV/JSON summaries as the
Transformer pipeline.

The default entry point writes the wider-alpha run below the
`ridge-alpha-1e-3-to-1e6/` variant directory, preserving the original baseline.

Cross-validation resumes completed, configuration-matched folds by default. Set
`RESUME_COMPLETED_FOLDS = False` in the entry point to deliberately retrain them.

Decision values from the ridge classifier are stored as logits. Their sigmoid is
reported as a monotonic probability-like score; it is not a calibrated probability.
Consequently, AUROC and average precision are the primary comparison metrics.
