# Representation

Three switchable feature groups layered on the shared fixed grid. Default is the identity:
a `RepresentationConfig()` with nothing enabled produces the exact channel array and channel
names the repository already used, so every recorded ROCKET, TCN, T2M-GPT, and MotionGPT
result stays reproducible while alternatives are compared against it.

`DataConfig.representation` carries the configuration, and `t2m_gpt.prepare_data` applies it,
so **T2M-GPT, MotionGPT, and the frame tagger all pick it up from one field**.

## The three options

| Option | What it adds | Why |
|---|---|---|
| `root_relative_positions` | `<Modality>.relpos_<body>_{x,y,z}` per non-reference position triple | Absolute positions let a model key on *where in the room* someone stood. That is a recording confound, not a communicative signal. |
| `append_acceleration` | `<Modality>.accel_<i>`, one per existing first-derivative channel | The pipeline already computes velocity; second derivatives are the next term, and movement onsets are often what carries a gesture. |
| `append_action_units` | `Facial.AU<n>_<name>` (+ `_velocity`) | Raw blendshapes are 63 uninterpreted floats. FACS units are a small, linguistically grounded space that connects to the literature on negation and disagreement. |

What was **already** true before this package and needs no option: rotations are normalized,
hemisphere-fixed **quaternions** everywhere (never Euler, so no discontinuity problem), and
per-channel **velocity** is already half of every modality's channels
(`event_transformer.features._rate`).

## Correctness: the three steps, in order

1. **Invert the fitted normalizer.** The grid arrives standardized per channel, so
   differencing two positions or averaging several blendshapes would otherwise mix
   incompatible units. Everything is derived in raw sensor units.
2. **Respect presence.** Interpolation fills unobserved frames with zeros, which
   un-normalize to the *training mean* rather than to "missing". Derived channels are
   computed only where source frames are real and left at zero elsewhere. This is why
   `include_presence_channels=True` is enforced, and why `masked_time_derivative` uses
   one-sided differences at the edge of a present run instead of differencing straight
   through a gap and manufacturing a spike.
3. **Standardize on training statistics only**, accumulated over present training frames,
   then applied unchanged to validation and test.

`RepresentationConfig` is part of the experiment config signature, so `load_training_result`
refuses a checkpoint trained under a different representation — resuming stays safe.

## Blendshape ordering is verified, not assumed

`facial_units.py` names the 63 `OVRPlugin.FaceExpression` weights. The ordering is pinned by
this repository's own `neo/replay/export_replay.py`: its `FACE_MAP` maps indices `0..49`
straight through, expands index `50` into four SDK slots (identifying `Lips_Toward`), and its
tail lands `Upper_Lid_Raiser_L/R` on SDK slots `68/69` — which `neo/README.md` independently
names. Two independent anchors agree, and `_validate_tables()` asserts them at import time.

Pooling is a mean over member blendshapes. It is deliberately lossy (left/right detail for
interpretability) and is **not** a calibrated FACS coder: read `AU4` as "the brow-lowering
shape pair", not as a certified AU4 intensity.

## Measured channel counts (fold 0, 32 frames)

| Configuration | Channels |
|---|---:|
| identity | 698 |
| `root_relative_positions` | 725 |
| `root_relative_positions` + `drop_absolute_positions` | 696 |
| `append_action_units` (all 27, + velocity) | 752 |
| `append_action_units` (negation subset) + `drop_raw_blendshapes` | 594 |
| `append_acceleration` (Head, LeftHand, RightHand) | 716 |
| all three combined | 831 |

## Usage

```python
from dataclasses import replace
from representation import RepresentationConfig
from t2m_gpt import DataConfig
from t2m_gpt.data import prepare_data

config = DataConfig(
    dataset="VR-Faces-Neg/target-cue_window-500_splits-10_fold-0",
    representation=RepresentationConfig(
        root_relative_positions=True,
        drop_absolute_positions=True,
        append_action_units=True,
        action_unit_subset="negation",
    ),
)
bundle = prepare_data(config)
```

Or apply it directly to arrays:

```python
from representation import fit_representation, grid_delta_seconds

transform = fit_representation(
    bundle.train.values, bundle.channel_names, bundle.normalizer,
    config.representation, grid_delta_seconds(bundle.time_grid),
)
test_values = transform.apply(bundle.test.values)
```

## Module layout

- `config.py`: `RepresentationConfig`, its validation, and its signature string.
- `facial_units.py`: the 63 blendshape names, the Action Unit tables, and their import-time checks.
- `layout.py`: recovers per-modality slices and position offsets from the grid's channel names.
- `transforms.py`: `masked_time_derivative`, the three derived blocks, and the fitted `RepresentationTransform`.
