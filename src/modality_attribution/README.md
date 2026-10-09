# Modality attribution

This package compares the complete sensor groups `Eye`, `Facial`, `Head`, `Body`,
`LeftHand`, `RightHand`, `LeftFinger`, and `RightFinger` for any classifier trained
by this repository. It operates across held-out cross-validation folds and keeps
three questions separate:

- full-model ablation measures what a fitted classifier relies on;
- sampled Shapley values explain individual logits while sharing interactions;
- modality-only and leave-one-out retraining measure sufficiency and unique value.

Run post-hoc ablation and Shapley analysis with at least two fold checkpoints:

```bash
PYTHONPATH=src python -m modality_attribution.cli \
  --checkpoints outputs/minirocket/target-cue_window-500_splits-10/fold-*/model.joblib \
  --output-dir outputs/modality_attribution/minirocket
```

Use `--skip-shapley` to run only the fitted-model ablation pass. Retraining is
opt-in: `--leave-one-out` trains eight whole-modality omissions plus one
`lower_face` facial-part omission per fold, and `--train-only-one-modality` trains
one model per modality using only that modality. The `lower_face` omission masks
raw and velocity channels for facial blendshapes whose names start with `Jaw_`,
`Lip_`, `Lips_`, `Lower_Lip_`, `Upper_Lip_`, or `Mouth_`, together with
`Cheek_Puff_L/R`, `Cheek_Raiser_L/R`, `Cheek_Suck_L/R`, and `Chin_Raiser_B/T`.
The corresponding 39 raw blendshape and 39 velocity channels are masked; nose,
brow, lid, gaze, confidence, validity, and presence channels are kept. Enabling
both families creates 17 additional models per fold; enabling only leave-one-out
creates 9 additional models per fold; enabling only modality-only creates 8
additional models per fold. Use `--n-jobs N` to train up to `N` subset models in
parallel.
Parallel workers use the multiprocessing `spawn` context so CUDA is not inherited
from the post-hoc process. Inner estimator and transform threads are automatically
capped to the CPUs in the Slurm/OS affinity mask divided by the active worker count;
`--threads-per-job N` overrides that budget. The manifest is updated after every
completed run, so the command is safe to resume. Use `--device cuda` for
PyTorch/TabPFN workflows.

ROCKET+TabPFN workers are fold-affine: one worker handles the pending variants for a
fold and prepares its full fixed-grid train/validation/test representation once.
Each variant copies that prepared representation and applies only its channel mask,
eliminating the repeated raw-event decoding visible in older parallel runs.

If post-hoc analysis already completed, avoid rerunning its full-model ablations when
resuming subset training:

```bash
PYTHONPATH=src python -m modality_attribution.cli \
  --checkpoints outputs/rocket_pfn/experiment/fold-*/model.joblib \
  --output-dir data/modality_attribution/rocket_pfn_loo \
  --skip-shapley --leave-one-out --retraining-only \
  --n-jobs 2 --device cuda --no-tabpfn-progress
```

One CUDA worker per physical GPU is the safe starting point. Several workers aimed
at the same `cuda` device replicate TabPFN weights and inference state, so increase
`--n-jobs` gradually while watching GPU memory. Slurm `gres/shard` allocations make
a GPU shareable but do not give each process isolated device memory.

ROCKET+TabPFN can additionally opt into `--tabpfn-batch-groups`, which sends
same-shaped ROCKET feature groups through TabPFN's dataset-batch API. This is often
faster on large GPUs, but it raises peak GPU memory and TabPFN 8.1 can return slightly
different probabilities because constant columns are handled differently in a
multi-dataset batch. Benchmark it before a final attribution run; do not combine it
immediately with a large same-GPU `--n-jobs` value.
Duplicate held-out sample IDs fail validation by default. If duplicates in a legacy
dataset have been inspected and are intentional, `--allow-duplicate-samples` keeps
them and adds a unique fold/row `sample_key` to every explanation.

The primary global ranking is the decrease in pooled held-out macro F1 after a
modality is removed. When Shapley is enabled, `sample_attributions.jsonl` contains
signed logit-scale Shapley values: positive values support the `neg` class and
negative values oppose it. These are predictive associations, not causal effects.
