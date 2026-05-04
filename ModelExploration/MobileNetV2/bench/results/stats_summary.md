# Phase 3 statistical summary

Input artifacts: `jakubs_qat_originals_test.npz` for headline statistics and `jakubs_qat_full_aug_test.npz` only for the biased augmentation-robustness panel.

## Headline: originals-only capture test

| metric | value |
|---|---:|
| independent captures | 60 |
| correct captures | 59 |
| accuracy | 98.33% |
| Wilson 95% CI | [91.14%, 99.71%] |
| macro-F1 | 0.9833 |
| cluster-bootstrap macro-F1 95% CI | [0.9433, 1.0000] |

The headline accuracy is 59/60 = 98.33%, matching `calibration_report.md` (98.33%). The macro-F1 is 0.9833, also matching the existing headline after rounding.

## Rejection threshold sweep

Thresholds were swept from 0.00 to 0.99 in steps of 0.01. The chosen operating point maximizes accuracy on accepted samples subject to `accept_rate >= 0.70`; ties retain the most captures, then use the lower threshold.

Recommended threshold: **q = 0.77**. It accepts 58/60 captures (96.67%), rejects 2, and gives 100.00% accuracy on accepted captures (ECE=0.0364).

Firmware mapping:
- Current firmware compares dequantized floats, so use `best_conf >= 0.77f`.
- If comparing raw softmax int8 output, use `floor(q / OUTPUT_SCALE) + OUTPUT_ZERO_POINT = 69` with OUTPUT_SCALE=0.00390625, OUTPUT_ZERO_POINT=-128.
- The requested input-scale formula gives `floor(q / INPUT_SCALE) + INPUT_ZP = 98` with INPUT_SCALE=0.00784314, INPUT_ZP=0.

| threshold | n accepted | accept rate | accuracy on accepted | ECE on accepted |
|---:|---:|---:|---:|---:|
| 0.00 | 60 | 100.00% | 98.33% | 0.0487 |
| 0.77 | 58 | 96.67% | 100.00% | 0.0364 |
| 0.80 | 57 | 95.00% | 100.00% | 0.0334 |
| 0.85 | 57 | 95.00% | 100.00% | 0.0334 |
| 0.90 | 53 | 88.33% | 100.00% | 0.0266 |
| 0.95 | 46 | 76.67% | 100.00% | 0.0191 |

Tradeoff: q=0.77 removes the single known Amine→Rifki error while retaining 96.67% of captures. The previous q=0.90 also reaches 100% accepted accuracy but rejects 7/60 captures, so q=0.77 is the less aggressive operating point.

## Augmentation-robustness panel (biased; informational only)

The full augmented test has n=780 files but only derives from the same 60 original captures, so it is not independent evidence. Its file-level accuracy is 97.69% and macro-F1 is 0.9769.

| threshold | n accepted | accept rate | accuracy on accepted | ECE on accepted |
|---:|---:|---:|---:|---:|
| 0.00 | 780 | 100.00% | 97.69% | 0.0357 |
| 0.80 | 726 | 93.08% | 98.76% | 0.0247 |
| 0.85 | 704 | 90.26% | 99.29% | 0.0256 |
| 0.90 | 658 | 84.36% | 99.54% | 0.0219 |
| 0.93 | 618 | 79.23% | 100.00% | 0.0228 |
| 0.95 | 560 | 71.79% | 100.00% | 0.0192 |

Under the same accept-rate constraint, the biased file-level sweep first reaches 100% accepted accuracy at q=0.93 with 79.23% acceptance. This is reported only as an augmentation-robustness diagnostic, not as headline statistical evidence.

## Caveats

- n=60 captures; CIs are wide. Wilson score 95% CI on 59/60 = [91.14%, 99.71%].
- Cluster bootstrap is by original capture, n_boot=10000, seed=42.
- Rejection threshold q must be mapped with the tensor scale used at the firmware comparison point; softmax output int8 is the relevant raw-output mapping, while the input-scale value above is included for project convention traceability.
- Single-model evaluation. Head-to-head McNemar requires a second candidate; deferred. `exact_mcnemar(b, c)` is implemented in `python/bench/stats.py` for future paired tests.
- Paired permutation testing is also deferred until a second candidate exists; the capture-level method is noted in `python/bench/stats.py`.
