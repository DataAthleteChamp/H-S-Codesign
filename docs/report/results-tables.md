# Results cheat-sheet — every reportable number in one place

This document **only aggregates** numbers already produced by the
benchmark scripts under `bench/results/` and the figures under
`docs/figures/`. It exists so the report writer doesn't have to hunt
across eight files. Re-running `make eval compare stats figures
footprint bench-firmware-check` regenerates every source artefact.

## 1 · Headline accuracy (deployed `python/gen/model.tflite`)

| metric | value | 95 % CI | source |
|---|---:|---|---|
| accuracy on n=60 originals | 98.33 % (59/60) | Wilson [91.14 %, 99.71 %] | `bench/results/stats_summary.md` |
| macro-F1 on n=60 originals | 0.9833 | cluster bootstrap [0.9433, 1.0000] | `bench/results/stats_summary.md` |
| accuracy on n=780 augmented | 97.69 % | – | `bench/results/calibration_report.md` |
| Δacc (augmented − originals) | −0.64 pp | – | `bench/results/calibration_report.md` |

## 2 · Per-class F1 (originals-only, honest)

| class | F1 | source |
|---|---:|---|
| Amine | 0.9744 | `bench/results/calibration_report.md` |
| Rifki | 0.9756 | `bench/results/calibration_report.md` |
| Jakub | 1.0000 | `bench/results/calibration_report.md` |
| **macro** | **0.9833** | `bench/results/calibration_report.md` |

## 3 · Confusion matrix (originals-only, n=60)

```text
actual\pred  Amine  Rifki  Jakub
Amine          19      1      0
Rifki           0     20      0
Jakub           0      0     20
```
Single error: one Amine capture predicted Rifki. Source: `bench/results/calibration_report.md`.

## 4 · Head-to-head: challenger vs F2-clean baseline

| model | acc | Wilson 95 % CI | macro-F1 | bootstrap 95 % CI |
|---|---:|---|---:|---|
| baseline (F2-clean) | 93.33 % | [84.07 %, 97.38 %] | 0.9329 | [0.8602, 0.9842] |
| challenger (deployed QAT) | 98.33 % | [91.14 %, 99.71 %] | 0.9833 | [0.9433, 1.0000] |

| Δaccuracy | Δmacro-F1 | discordant b / c | exact McNemar p |
|---:|---:|---:|---:|
| +5.00 pp | +0.0504 | 0 / 3 | **0.2500** |

Lexicographic decision rule (Δ macro-F1 ≥ 0.02 **and** McNemar p < 0.05) is **not** met because p = 0.25 > 0.05. Tie-break is therefore by TFLite size → MAC count → maturity, signed off by the team in `docs/decision.md`.

Source: `bench/results/mcnemar_comparison.md`.

## 5 · Calibration & rejection threshold

| metric | value | source |
|---|---:|---|
| ECE @ q = 0.00 (no rejection) | 0.0487 | `bench/results/stats_summary.md` |
| **chosen operating point q = 0.77** | **100.00 % accuracy on accepted** | `bench/results/stats_summary.md` |
| accept rate at q = 0.77 | 96.67 % (58/60) | `bench/results/stats_summary.md` |
| ECE @ q = 0.77 | 0.0364 | `bench/results/stats_summary.md` |
| firmware float threshold | `best_conf >= 0.77f` | `bench/results/stats_summary.md` |
| firmware INT8-output threshold | `out_q >= 69` (OUTPUT_SCALE=0.00390625, OUTPUT_ZP=−128) | `bench/results/stats_summary.md` |

## 6 · Embedded footprint

| metric | value | source |
|---|---:|---|
| TFLite file size | 662,056 B (646.54 KiB) | `bench/results/footprint.md` |
| total parameters | 430,184 | `bench/results/footprint.md` |
| INT8 weights / INT32 biases | 423,107 / 7,077 | `bench/results/footprint.md` |
| input | (1, 96, 96, 3) INT8, scale 0.007843, zp 0 | `bench/results/footprint.md` |
| output | (1, 3) INT8, scale 0.00390625, zp −128 | `bench/results/footprint.md` |
| firmware `TENSOR_ARENA_SIZE` | 1,048,576 B (1024 KiB), in PSRAM | `esp32/main/inference.cpp` |
| flash use vs 8 MiB ceiling | 7.89 % | `bench/results/footprint.md` |
| PSRAM arena vs 8 MiB ceiling | 12.50 % (declared) | `bench/results/footprint.md` |
| total compute MACs (Conv + DWConv + FC) | 10,695,184 | `bench/results/calibration_report.md`, `bench/results/mac_count.csv` |

## 7 · Hyperparameter sweep (Keras-Tuner, 30 completed trials)

| metric | value | source |
|---|---:|---|
| best val-accuracy | **96.79 %** (trial 0025) | `bench/results/tuner_summary.md` |
| best config | α=0.5, dense=32, dropout=0.2/0.2, lr=5e-4, smooth=0.10, 20 ep | `bench/results/tuner_summary.md` |
| smallest within 1 pp of best | α=0.35 (trial 0029, 96.41 %) | `bench/results/tuner_summary.md` |
| α tradeoff | α=0.35 dominates 0.5 and 0.75 within 1 pp band | `bench/results/tuner_summary.md` |

Full data: `bench/results/tuner_all.csv`, `bench/results/tuner_top10.csv`, `bench/results/tuner_pareto.csv`.

## 8 · Training history (deployed model, seed 42)

| phase | epochs | final loss | final acc | best val loss | best val acc |
|---|---:|---:|---:|---:|---:|
| float feature-extraction | 20 | 0.2085 | 0.9966 | 0.2440 | 0.9808 |
| QAT | 10 | 0.2119 | 0.9955 | 0.2440 | 0.9786 |

Source: `bench/results/baseline_retrain_report.md`, raw curves in `bench/results/baseline_training_history_seed42.json`, plotted as `docs/figures/f07_training_curves.png`.

## 9 · F3 firmware-preprocess regression check

| assertion | result | value |
|---|:---:|---:|
| fixed mean ≈ 0 (\|μ\|<5) | **PASS** | −1.07 |
| fixed ≥ 40 % negatives | **PASS** | 50.3 % |
| fixed ≥ 40 % positives | **PASS** | 49.2 % |
| fixed min ≤ −50 | **PASS** | −127 |
| fixed max ≥ +50 | **PASS** | 127 |
| buggy min ≥ 0 (proves bug) | **PASS** | 0 |
| buggy < 5 % negatives (proves bug) | **PASS** | 0.0 % |

Source: `bench/results/firmware_preprocess_check.md`.

## 10 · Figures index (`docs/figures/`)

| figure | content | report section |
|---|---|---|
| `f01_dataspace.png` | capture counts × class × split | dataset |
| `f02_calibration.png` | reliability diagram (ECE) | verification |
| `f03_confusion.png` | confusion matrices (challenger and baseline) | verification |
| `f04_rejection.png` | accept-rate vs accuracy threshold sweep | design space |
| `f05_tuner_sweep.png` | tuner val-acc vs α with 1-pp Pareto | design space |
| `f06_compare_macroF1.png` | challenger vs baseline macro-F1 with bootstrap CI | verification |
| `f07_training_curves.png` | float head + QAT loss/accuracy per epoch | design / training |

## 11 · Pending real-world (still to capture)

| metric | source planned | status |
|---|---|---|
| desktop webcam top-1 accuracy | `python/realworld_webcam_test.py` → `bench/results/realworld_webcam.md` | scaffold ready, run before report |
| on-device latency p50/p95 | `python/tools/serial_latency_logger.py` → `bench/results/onboard_latency.{csv,png}` | scaffold ready, run on hardware |
| live demo recording | `docs/report/live-demo-checklist.md` | checklist ready, capture on demo day |

## 12 · Reproduction one-liner

```bash
make originals && make eval && make compare && make stats && \
make figures && make footprint && make bench-firmware-check
```

Each target writes its source-of-truth file inside `bench/results/`; this cheat-sheet links to those files rather than restating numbers, so re-running the pipeline keeps everything in sync.
