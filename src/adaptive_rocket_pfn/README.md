# Adaptive Multi-Representation RocketPFN

This experimental module implements the architecture proposed for this project:

```text
raw / first diff / second diff / smooth / high-pass / local-normalized
       × short-scale and full-dilation MultiRocket banks
       × base/difference branches × PPV/LSPV/MPV/MIPV
                    + HYDRA + prototype distances
                                  |
                    structured candidate families
                                  |
          nested repeated stability selection + redundancy pruning
                                  |
                 500 features per semantic expert
                                  |
             morphology / dynamics / prototype TabPFN-3
                                  |
                 OOF-learned nonnegative weights
                                  |
                              prediction
```

The implementation is informed by [SelF-Rocket](https://arxiv.org/abs/2409.01115),
[Detach-ROCKET](https://arxiv.org/abs/2309.14518),
[RocketPFN](https://arxiv.org/abs/2606.21786), and
[MASHT](https://arxiv.org/abs/2607.19234), but this combined architecture is an
unpublished project experiment rather than a reproduction of a published estimator.

## What is adaptive

Each MultiRocket bank is split into explicit metadata families:

```text
(external view, dilation regime, internal base/difference, pooling operator)
```

Repeated inner folds rank features and estimate a stable utility for every family.
The final fixed expert budget is allocated to the most useful families through a
temperature-controlled distribution. Highly correlated candidates are suppressed.
This learns a compact empirical prior over representation, scale, and pooling while
leaving the random convolution weights untouched.

The three experts are:

- `morphology`: raw, smoothed, locally normalized, and HYDRA competition banks;
- `dynamics`: first/second derivatives and high-frequency residuals;
- `prototypes`: random dilated shapelet distances, occurrences, and locations.

Every OOF ensemble prediction refits both the candidate transforms and feature
selection inside that OOF training partition. Final selection uses the complete
outer-training split. Outer validation and test rows never enter feature generation,
selection, TabPFN fitting, or ensemble-weight learning.

## Deliberate scope boundaries

This version does not learn individual convolution weights and does not perform
supervised MiniRocket bias optimization. Bias learning needs an additional nested
cross-fitting layer and should be evaluated as a separate ablation.

The prototype expert is not labelled SPROCKET. The public SPROCKET repository is
alpha research code with no declared software license or installable package. Copying
it would be inappropriate. This module instead uses aeon's maintained unsupervised
`RandomDilatedShapeletTransform`, which supplies a reproducible distance/prototype
branch. It can be replaced by an officially licensed SPROCKET transformer later
without changing the selector or semantic ensemble interfaces.

## Default computational profile

With 32 time points, six views, two dilation regimes, and 625 requested MultiRocket
kernels per bank, the initial bank is expected to contain roughly 50,000–70,000
features. Exact width depends on aeon's integer allocation and shapelet similarity
filtering. Only up to 500 selected features reach each TabPFN expert.

The OOF stage defaults to three folds and two-estimator TabPFN experts. Final experts
use eight estimators. This means one outer fold fits four candidate banks (three OOF
banks and one final bank), nine small OOF TabPFN experts, and three final experts.
Outer folds therefore run sequentially on CUDA by default.

## Run

```bash
venv/bin/python src/train_main_adaptive_rocket_pfn.py
```

The entry point uses the existing project-local TabPFN-3 checkpoint:

```text
data/tabpfn/tabpfn-v3-classifier-v3_default.ckpt
```

Set `TABPFN_DEVICE = "auto"` in the entry point for automatic CPU fallback. The
candidate bank is CPU-generated; TabPFN experts run on the configured CUDA device.
