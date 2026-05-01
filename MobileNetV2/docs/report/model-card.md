# Model Card — H-S-Codesign three-class face classifier

This card follows the [Mitchell et al. (2019) "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993) skeleton, scoped to the open-source artefacts in this repo. Every quoted number traces back to a file under `bench/results/` so the card stays in sync with the reproduction pipeline.

## Model details

| field | value |
|---|---|
| name | `H-S-Codesign / model.tflite` |
| architecture | MobileNetV2 (α=0.35) feature extractor + GlobalAveragePooling + Dense(32, ReLU) + Dropout(0.4) + Dropout(0.1) + Dense(3, softmax) |
| input | 96 × 96 × 3 RGB image, MobileNetV2 normalisation `(x / 127.5) − 1` (INT8 scale 0.007843, zero-point 0) |
| output | 3-way softmax over `{Amine, Rifki, Jakub}` (INT8 scale 0.00390625, zero-point −128) |
| training pipeline | float head fit (20 ep, lr 5e-4, label-smoothing 0.05) → QAT fine-tune (10 ep) → INT8 representative-dataset TFLite export |
| size on flash | 662 056 B (646.54 KiB) |
| parameters | 430 184 (423 107 INT8 weights + 7 077 INT32 biases) |
| target hardware | XIAO ESP32-S3 Sense (8 MiB PSRAM, 8 MiB flash) running TensorFlow Lite Micro |
| date | exported 2026-04-09; baseline retrain 2026-04-30 (seed 42) |
| license | MIT (see `LICENSE`) |
| citation | see `CITATION.cff` |

## Intended use

- **Primary:** demonstrate an end-to-end embedded-ML co-design pipeline (data collection → augmentation → transfer learning → INT8 QAT → on-device inference) for the DTU 02214 Hardware/Software Co-design course.
- **Secondary:** serve as a reproducible reference open-source implementation for face-class verification on a battery-powered ESP32-S3 device.

## Out-of-scope use

- General face-recognition. The model only knows three identities and is not a face-identification system.
- Identification of a person not in `{Amine, Rifki, Jakub}` — the model will emit one of the three classes regardless. The deployed firmware mitigates this with a confidence threshold (q = 0.77 ⇒ reject low-confidence predictions); see `bench/results/stats_summary.md`.
- Liveness, anti-spoofing, age, gender, emotion, or any other inference beyond the three identities.
- Authentication or access-control decisions: a 60-sample test cannot bound the false-accept rate tightly enough for a security context.
- Any deployment that captures, transmits, or persists images of bystanders without consent.

## Factors

| factor | coverage in dataset |
|---|---|
| identities | 3 (Amine, Rifki, Jakub) |
| capture device | XIAO ESP32-S3 Sense onboard OV2640 camera, 320 × 240 RGB JPEG frames |
| illumination | indoor, mixed natural + artificial; no explicit night / low-light split |
| pose | mostly frontal with limited yaw / pitch variation |
| occlusion | augmentation introduces synthetic random box occlusions |
| capture range | desk-distance (≈ 30–80 cm); not designed for far-field |
| augmentation | Albumentations 2.0: hflip, rotation, shift-scale, brightness, blur, JPEG compression, occlusion, grayscale, four `combo*` chains (12 variants per original) |

## Training data

- **Source:** captures collected by the team using the target hardware, plus shared captures from the course's optional SharePoint pool (used responsibly per course rules).
- **Per-class splits (seed 42, F2-clean):** 80 train captures + 20 test captures per class; 12 × 80 = 960 augmented variants per class for training.
- **Validation:** 12 captures per class held out from `data/<class>/train/` only — 15 % of the train set, stratified by class and grouped by capture prefix to prevent within-capture leakage. Manifest: `python/gen/val_split_seed42.json`. The test set is **never** used for early stopping or model selection (this is the F2 fix).
- **Datasheet:** `docs/report/dataset-datasheet.md` documents collection process, environments, motivations, and consent context per Gebru et al. (2018).

## Evaluation data

- **n=60 originals-only test set** (20 per class), built by `python/bench/build_originals_test.py` and persisted to `bench/results/x_test_originals_96_pm1.npy` + `y_test_originals.npy`.
- This is the **honest** test set: pure originals, no augmented variants, no intersection with train/val.
- A second n=780 set with augmented variants is reported only as a **biased** comparison panel; we do not use it for the headline number.

## Quantitative analyses

### Headline (n=60 originals)

| metric | value | 95 % CI |
|---|---:|---|
| accuracy | 98.33 % (59/60) | Wilson [91.14 %, 99.71 %] |
| macro-F1 | 0.9833 | bootstrap [0.9433, 1.0000] |
| ECE @ no-reject | 0.0487 | – |
| ECE @ q = 0.77 | 0.0364 | – |

### Per-class F1

Amine 0.9744 · Rifki 0.9756 · Jakub 1.0000 — see `bench/results/calibration_report.md`.

### Versus F2-clean baseline

ΔF1 = +0.0504, McNemar p = 0.25 (not significant at α = 0.05). The team made the deployment decision on the lexicographic tie-breaker chain documented in `docs/decision.md`.

### Compute & footprint

- 10 695 184 MACs (Conv + DWConv + FC).
- TFLite size 662 KiB. Declared `TENSOR_ARENA_SIZE` 1 024 KiB in PSRAM. Live `arena_used_bytes()` is logged by firmware and tightens this bound at runtime.

## Calibration & rejection

The deployed firmware compares the dequantised top-1 probability against `q = 0.77f`. At this threshold the accept-rate is 96.67 % and accuracy on accepted captures is 100 % on the n=60 test. See the threshold sweep in `bench/results/stats_summary.md` and figure `docs/figures/f04_rejection.png`.

## Ethical considerations

- The model was trained on faces of the three team members with their explicit consent for use in this academic project. The repository contains no images, only code and metrics.
- The class set is closed and small; the model is **not** suitable as a face-recognition or surveillance system.
- The shared SharePoint dataset agreement requires that data is used for training only and not redistributed. Anything checked into this repository is code/metric only.
- The model has not been audited for demographic bias because the three-class design does not admit a meaningful subgroup analysis.

## Caveats and recommendations

- The 95 % CI on n=60 captures is wide ([91.14 %, 99.71 %]). Numbers should not be over-stated.
- The McNemar p-value (0.25) means we cannot statistically separate the deployed model from the F2-clean baseline on this test set; the deploy decision is qualitative (smaller maturity-tested pipeline, identical TFLite size).
- Real-world tests (live capture session and on-device latency p50/p95) are tracked in `docs/report/live-demo-checklist.md` and `docs/report/results-tables.md` § 11; once captured, update this card.
- F3 firmware-preprocess regression is guarded by `python/bench/firmware_preprocess_check.py` (CI-enforced).

## How to reproduce

```bash
make originals      # rebuild honest n=60 test arrays
make eval           # evaluate model.tflite
make compare        # paired McNemar vs baseline_model.tflite
make stats          # Wilson CI, bootstrap, threshold sweep
make figures        # all PNGs into docs/figures/
make footprint      # flash/params/arena summary
make bench-firmware-check   # F3 regression assertion
```
