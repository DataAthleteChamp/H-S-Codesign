# Design space and trade-offs

This page documents the design-space we explored for the XIAO ESP32-S3 Sense
face-recognition project, the search method we used to evaluate it, and the
configuration that ships in the deployed model. It is the analytical
companion to `docs/decision.md` (which records the team's pick) and
`bench/results/tuner_summary.md` (the raw tuner output).

## 1. Knobs we explored

| Knob | Search range | Effect |
|---|---|---|
| `alpha` (MobileNetV2 depth multiplier) | 0.35 / 0.5 / 0.75 | Width of every conv stage. Drives parameter count, MACs and TFLite size roughly quadratically. |
| `dense_units` (head width) | 32 / 64 / 128 | Capacity of the top dense layer; bigger heads risk overfitting our 60×13 train captures. |
| `dropout_1` (after GAP) | 0.2 / 0.3 / 0.4 | Regularises the GAP -> dense path. |
| `dropout_2` (after dense) | 0.1 / 0.2 | Regularises the dense -> softmax path. |
| `learning_rate` | 1e-4 / 5e-4 / 1e-3 | Adam learning rate during the head warm-up + fine-tune phases. |
| `label_smoothing` | 0.0 / 0.05 / 0.10 / 0.15 | Smooths softmax targets; shifts confidence calibration which matters for the rejection threshold. |
| `epochs` | 3 / 7 / 20 (Hyperband) | Hyperband budget; large brackets only run on promising configs. |
| Quantization | PTQ vs QAT | INT8 either way; QAT recovers ~1.5 pp at the same flash size. |
| `IMG_SIZE` | 96 / 160 | Locked to 96 by the PSRAM tensor-arena budget once the camera double-buffer and USB streaming buffer are accounted for. |
| Rejection threshold `q` | softmax in [0.0, 0.99] sweep | Trades acceptance rate against accuracy on accepted captures. |

`IMG_SIZE` is *not* searched by the tuner. The 160×160 alternative comes from
`origin/main`'s `MobileNetV2_3ClassKeras_Crop.ipynb`; we keep it as a
documented design-space reference but do not retrain it because it does not
fit our memory budget on the device.

## 2. Search method

* **Keras-Tuner Hyperband** over the 6 numeric knobs above, 30 trials, single
  random seed (= 42). Output: `python/gen/tuner/face_recognition/`.
* Tuner used the *biased* validation set (= the contaminated test set under
  finding F2). Absolute `val_accuracy` numbers are therefore over-stated by
  ~3–5 pp on the cleaned originals; **rankings remain informative**, which is
  why the tuner still drives design-space evidence rather than the final
  performance claim.
* Final performance numbers are reported on the cleaned originals-only test
  set (n = 60 captures, 20 per class) — see `bench/results/stats_summary.md`.

## 3. Top-10 trials

From `bench/results/tuner_summary.md`:

| Rank | Trial | val_acc | alpha | dense | drop1 | drop2 | lr | smooth | epochs |
|------|-------|---------|-------|-------|-------|-------|--------|--------|--------|
| 1 | 0025 | 96.79 % | 0.5  | 32  | 0.2 | 0.2 | 5e-4 | 0.10 | 20 |
| 2 | 0029 | 96.41 % | 0.35 | 32  | 0.4 | 0.1 | 5e-4 | 0.05 | 20 |
| 3 | 0017 | 96.15 % | 0.5  | 64  | 0.2 | 0.1 | 1e-3 | 0.00 | 20 |
| 4 | 0024 | 96.15 % | 0.5  | 32  | 0.2 | 0.2 | 5e-4 | 0.00 | 20 |
| 5 | 0016 | 96.03 % | 0.35 | 128 | 0.3 | 0.2 | 1e-3 | 0.15 | 20 |
| 6 | 0026 | 96.03 % | 0.35 | 64  | 0.4 | 0.2 | 5e-4 | 0.05 | 20 |
| 7 | 0028 | 96.03 % | 0.75 | 32  | 0.4 | 0.1 | 1e-3 | 0.10 | 20 |
| 8 | 0023 | 95.64 % | 0.5  | 32  | 0.2 | 0.2 | 5e-4 | 0.00 | 7  |
| 9 | 0015 | 95.51 % | 0.35 | 128 | 0.3 | 0.2 | 1e-3 | 0.15 | 7  |
| 10 | 0009 | 95.51 % | 0.5  | 64  | 0.2 | 0.1 | 1e-3 | 0.00 | 3  |

## 4. Pareto frontier (1 pp band)

Trial `0029` (`alpha=0.35`) is the smallest configuration that lies within
1 pp of the best observed score (trial `0025` at 96.79 %). For a flash- and
arena-constrained target like the ESP32-S3, the 1 pp Pareto frontier is
therefore degenerate and consists of a single point:

| Trial | val_acc | alpha | params | TFLite size (INT8) |
|-------|---------|-------|--------|--------------------|
| 0029  | 96.41 % | 0.35  | ~410 K | **662 KB** |

The next-smaller `alpha` we considered (0.25) is documented in the literature
to drop accuracy by another ~3–4 pp on similarly-sized datasets and was not
included in the tuner search. The next-larger `alpha` (0.5) increases TFLite
size by ~30 % for ≤ 0.4 pp gain.

Visual versions of this frontier are in `docs/figures/f05_tuner_sweep.png`
(score vs. trial index, coloured by `alpha`) and `docs/figures/f01_dataspace.png`
(per-class capture budget which constrains the reachable model size).

## 5. Hyperparameter trends from the tuner

Hyperband mixes brackets and epoch budgets so the per-knob means below are
trend indicators, not controlled ablations:

| Knob | Best single trial | Best mean | Course-theory note |
|---|---|---|---|
| `alpha` | 0.5 | 0.5 | Smaller alpha shrinks every conv proportionally; we trade a modest accuracy drop for sub-1 MB tensor arena. |
| `dense_units` | 32 | 64 | A 32-unit head is the smallest that still hits the 1 pp band; bigger heads add parameters without improving the originals-only error. |
| `learning_rate` | 5e-4 | 5e-4 | Slow LR is preferred during fine-tuning; 1e-4 underfits, 1e-3 occasionally diverges. |
| `label_smoothing` | 0.10 (best single) / 0 (best mean) | mixed | Non-zero smoothing tightens the softmax distribution, which matters for the rejection threshold; we choose 0.05 as a calibration trade-off. |
| Quantization | QAT | — | QAT recovers ~1.5 pp over PTQ at identical flash size; on n=60 honest captures, both fall within Wilson 95 % CIs of each other. We keep QAT for principled INT8 simulation during training. |

## 6. Final design point (deployed)

| Setting | Value | Source |
|---|---|---|
| Backbone | MobileNetV2, alpha = 0.35 | trial 0029 |
| Head | GAP -> Dropout(0.4) -> Dense(32) -> Dropout(0.1) -> Softmax(3) | trial 0029 |
| Input | 96 × 96 × 3, MobileNetV2 [-1, 1] | preprocess.py |
| Optimizer | Adam, lr = 5e-4, label_smoothing = 0.05 | trial 0029 |
| Quantization | INT8 QAT (full-integer) | main.py |
| TFLite size | **662 KB** | bench/results/mac_count.csv |
| MACs | ~10.7 M | bench/results/mac_count.csv |
| Rejection threshold | softmax q ≥ 0.77 | bench/results/stats_summary.md |
| Honest accuracy (n=60) | **98.33 %**, F1 = 0.9833 | bench/results/mcnemar_comparison.md |

## 7. What we did *not* explore (and why)

* `IMG_SIZE = 128`: the tuner search was budget-limited; one input size at a
  time keeps the comparison apples-to-apples. 128 sits between 96 and 160 with
  no obvious memory benefit over 96 once tensor-arena overhead is added.
* `alpha = 0.25`: dropped from the search after a pilot showed > 4 pp
  degradation; the smallest reasonable backbone for this task is 0.35.
* On-device latency sweeps: we have no XIAO ESP32-S3 hardware available, so
  on-device timing is reported as an analytical estimate from MAC count and
  the ESP32-S3 datasheet, not a measurement.

## 8. References

* `bench/results/tuner_summary.md` — full distillation, including alpha
  trade-offs and 1 pp Pareto.
* `bench/results/tuner_all.csv`, `bench/results/tuner_top10.csv`,
  `bench/results/tuner_pareto.csv` — raw data.
* `python/bench/distill_tuner.py` — the script that produced the summary.
* `docs/decision.md` — team-signed lexicographic decision rationale.
