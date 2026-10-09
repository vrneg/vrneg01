# Compact cue-aware fusion-TCN

This follow-up addresses the failure modes observed in the modality-aware
Inception-TCN while preserving the exact ROCKET fixed-grid data representation.

The model:

1. assigns more projection capacity to Facial and Finger streams than pose streams;
2. fuses all projected modalities before a shared temporal encoder;
3. uses a roughly 50k-parameter depthwise-separable TCN instead of eight encoders;
4. pools mean, maximum, and standard deviation separately over the complete,
   pre-cue, and post-cue regions;
5. applies modality dropout, input noise, cue jitter, dropout, and stronger AdamW
   weight decay during training.

The independent entry point defaults to three complete cross-validation repetitions:

```bash
venv/bin/python src/train_main_compact_fusion_tcn.py
```

Each seed receives its own fold-level and pooled out-of-fold summary. A final
multi-seed summary treats complete CV repetitions—not overlapping folds—as the
uncertainty unit. Configuration-matched folds resume automatically. Artifacts are
written beneath
`outputs/compact_fusion_tcn/<DATASET_PREFIX>/cue-aware-shared-fusion/`.
