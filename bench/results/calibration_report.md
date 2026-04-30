# Calibration report

## Headline numbers
- **acc(originals_only)= 0.9833**  (n=60, HONEST — original captures only)
- acc(full_aug_test) = 0.9769  (n=780, BIASED — augmented variants of test set)
- Δacc = acc(full_aug_test) - acc(originals_only) = -0.0064 (-0.64 pp)

## Per-class F1

| class | full_aug_test F1 | originals_only F1 |
|---|---:|---:|
| Amine | 0.9685 | 0.9744 |
| Rifki | 0.9756 | 0.9756 |
| Jakub | 0.9865 | 1.0000 |
| **macro** | **0.9769** | **0.9833** |

## Confusion matrices

### Full augmented test (biased)

```text
actual\pred  Amine  Rifki  Jakub
Amine    246     11      3
Rifki      0    260      0
Jakub      2      2    256
```

### Originals-only test (honest)

```text
actual\pred  Amine  Rifki  Jakub
Amine     19      1      0
Rifki      0     20      0
Jakub      0      0     20
```

## MAC-count summary

- Total compute MACs (Conv2D + DepthwiseConv2D + FullyConnected): 10,695,184
- CONV_2D: count=35, est_macs=9,074,304
- DEPTHWISE_CONV_2D: count=17, est_macs=1,579,824
- FULLY_CONNECTED: count=2, est_macs=41,056
- Source: `bench/results/mac_count.csv` generated from `python/gen/model.tflite`.

## Stats

Phase 3 statistical artifacts are in [`stats_summary.md`](stats_summary.md): Wilson accuracy CI, capture-cluster bootstrap macro-F1 CI, exact-McNemar helper status, and the rejection-threshold sweep. Machine-readable outputs: [`bootstrap_ci.json`](bootstrap_ci.json), [`rejection_sweep.csv`](rejection_sweep.csv), and [`rejection_sweep.png`](rejection_sweep.png).

## Interpretation

The honest originals-only evaluation is the headline result: 0.9833 accuracy / 0.9833 macro-F1 on 60 independent captures. The full augmented test reports 0.9769, but its 780 rows are augmented variants of the same 60 captures and should not be treated as independent evidence. Here Δacc = full_aug - originals = -0.0064 (-0.64 pp), which is below the 2 pp concern threshold and below the 5 pp mandatory-retraining threshold. Therefore, the existing QAT model is credible for Phase 3 calibration; Phase 2 retraining remains useful insurance against earlier validation leakage, not a Δacc-driven requirement.
