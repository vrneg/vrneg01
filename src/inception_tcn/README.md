# Modality-aware Inception-TCN

This package implements a compact learned temporal model over the same normalized
fixed-grid arrays and experiment-separated folds as MiniRocket and MultiRocket.

The model has one branch for each of the eight sensor modalities. Every branch uses:

1. a small `1x1` feature projection;
2. parallel temporal convolutions with kernel sizes 3, 5, and 9;
3. residual depthwise-separable TCN blocks at dilations 1, 2, and 4;
4. presence-mask-aware mean and maximum pooling.

A normalized relative-time channel retains cue alignment through global pooling.
The modality embeddings are concatenated only in the final small classification head.
The default model intentionally stays compact for 926 examples and 33 independent
experiment groups.

Training uses AdamW, validation-loss checkpoint selection, learning-rate reduction,
gradient clipping, early stopping, and optional CUDA mixed precision. The test split
is evaluated only after the best validation checkpoint has been selected.

Edit `src/train_main_inception_tcn.py`, then run from the repository root:

```bash
venv/bin/python src/train_main_inception_tcn.py
```

The entry point supports a single fold or sequential ten-fold cross-validation.
Configuration-matched completed folds resume automatically. Artifacts are written to
`outputs/inception_tcn/<DATASET_PREFIX>/modality-aware/`.
