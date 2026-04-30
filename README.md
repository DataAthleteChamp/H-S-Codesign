# Face Recognition for the XIAO ESP32-S3 Sense

> DTU 02214 embedded-ML project for the XIAO ESP32-S3 Sense.
> Recognises Amine, Rifki, and Jakub; rejects unknown faces with a confidence threshold.

[![CI](https://img.shields.io/github/actions/workflow/status/DataAthleteChamp/H-S-Codesign/ci.yml?branch=jakubs-solution&label=ci)](https://github.com/DataAthleteChamp/H-S-Codesign/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](python/requirements.txt)
[![ESP-IDF 5.x](https://img.shields.io/badge/ESP--IDF-5.x-red.svg)](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)

## What this is

This repository is a DTU 02214 Machine Learning for Embedded Systems / Hardware-Software Codesign course project targeting the Seeed Studio XIAO ESP32-S3 Sense: it trains and deploys a compact face-recognition model for three team members (`Amine`, `Rifki`, and `Jakub`) and uses a softmax confidence threshold to reject unknown faces rather than forcing every frame into one of the three known classes.

## Quick stats

| Item | Current value | Source / note |
| --- | ---: | --- |
| Final model | MobileNetV2, `alpha=0.35`, `IMG_SIZE=96`, dense head `32` | `python/main.py`, `python/qat_export.py` |
| Quantization | Full INT8, QAT + representative dataset | `python/qat_export.py` |
| TFLite size | **662,056 bytes** / **646.5 KiB** / **662 KB** | `python/gen/model.tflite` |
| MACs | **10,695,184** total compute MACs | `bench/results/mac_count.csv` |
| Honest test accuracy | **98.33%** (`59/60`) | `bench/results/calibration_report.md` |
| Honest macro-F1 | **0.9833** | `bench/results/calibration_report.md` |
| Default rejection threshold | **0.90** firmware default | `REJECTION_THRESHOLD` in Python export |
| ESP32-S3 latency | **TBD** | Real-hardware latency has not yet been measured[^latency] |

[^latency]: The repository contains analytical and desktop-proxy evidence, but no final measured latency on the XIAO ESP32-S3 Sense hardware yet. Treat latency as a required follow-up measurement before making real-time claims.

## Repo layout

```text
.
├── README.md                         # public project overview and reproduction guide
├── Makefile                          # common developer commands
├── LICENSE                           # MIT license
├── CITATION.cff                      # citation metadata
├── CONTRIBUTING.md                   # contribution notes for the course repo
├── python/                           # training, preprocessing, QAT export, and evaluation helpers
│   ├── augment.py                    # Albumentations image augmentation
│   ├── preprocess.py                 # MediaPipe face crop, resize, and [-1,1] normalization
│   ├── main.py                       # MobileNetV2 transfer learning and export path
│   ├── qat_export.py                 # legacy-Keras QAT export path
│   ├── bench/                        # reproducible evaluation harnesses
│   ├── utils/                        # TFLite export and train/validation helpers
│   └── gen/                          # generated models and caches; mostly gitignored
├── esp32/                            # ESP-IDF firmware for XIAO ESP32-S3 Sense
│   ├── sdkconfig.defaults            # checked-in ESP32-S3 defaults
│   └── main/                         # camera, inference, app entry, and model C array
├── bench/                            # published benchmark artefacts
│   └── results/                      # markdown/csv/json/png/npz evidence used by the report
├── docs/                             # architecture notes and report support material
│   ├── architecture.md               # Mermaid pipeline diagrams
│   └── report/                       # report outline, datasheet, AI note, bug notes
└── data/                             # private face images; gitignored and not redistributed
```

## Hardware setup

The target board is the [Seeed Studio XIAO ESP32-S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/), which combines an ESP32-S3 module with a small camera expansion board.

| Part | Role |
| --- | --- |
| ESP32-S3 MCU | Dual-core Xtensa LX7 target for TFLite Micro inference. |
| PSRAM | Required for the camera frame buffer and TFLite Micro tensor arena. |
| OV camera module | Captures RGB565 frames through the XIAO Sense camera connector. |
| USB-C | Flashing, serial monitor, and `ESP_LOGI` output. |
| BOOT / RESET buttons | Used for flashing and recovery, depending on host tooling. |

Pin routing for the camera is board-specific and is summarized in the Seeed wiki and mirrored by the firmware camera configuration under `esp32/main/`. Use the Sense camera module supplied for the board; a plain XIAO ESP32-S3 without the Sense expansion does not provide the same camera path.

## Reproduce: training

The private face dataset is intentionally not committed. Place images under `data/` using the exact class names expected by `python/preprocess.py`.

```text
data/
├── Amine/
│   ├── train/
│   └── test/
├── Rifki/
│   ├── train/
│   └── test/
└── Jakub/
    ├── train/
    └── test/
```

Recommended local environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

Generate image-space augmentations. The script is idempotent and creates 8 single-transform variants plus 4 combined variants per original image.

```bash
python python/augment.py
```

Build the cached NumPy arrays. `python/preprocess.py` detects faces with MediaPipe, applies a padded crop, resizes to `96x96` RGB, and normalizes pixels to MobileNetV2 `[-1, 1]`.

```bash
python python/preprocess.py
```

Train the MobileNetV2 transfer-learning model and export artefacts. The selected configuration is `alpha=0.35`, `IMG_SIZE=96`, `dense_units=32`, label smoothing `0.05`, and QAT enabled.

```bash
python python/main.py
```

If the modern Keras path cannot run QAT because of TensorFlow Model Optimization compatibility, use the legacy-Keras QAT exporter. This is the path that writes the final INT8 `.tflite` and C-array headers.

```bash
TF_USE_LEGACY_KERAS=1 python python/qat_export.py
```

Run the honest originals-only evaluation harness. The checked-in benchmark artefacts already contain the cleaned `n=60` arrays, but the command below documents the intended invocation.

```bash
python -m python.bench.eval_branches \
  --model python/gen/model.tflite \
  --x bench/results/x_test_originals_96_pm1.npy \
  --y bench/results/y_test_originals.npy \
  --capture-ids bench/results/capture_ids_originals.npy \
  --norm pm1 \
  --out bench/results/jakubs_qat_originals_test.npz
```

Check the firmware-preprocessing regression test after changing either training preprocessing or firmware quantization.

```bash
python -m python.bench.firmware_preprocess_check \
  --tflite python/gen/model.tflite
```

The same commands are wrapped by the top-level Makefile:

```bash
make venv
make augment
make train
make qat
make eval
make bench-firmware-check
```

## Reproduce: firmware build

Install ESP-IDF v5.x using Espressif's official guide, then export the environment in every terminal session that builds firmware.

```bash
. "$IDF_PATH/export.sh"
```

Configure and build for the ESP32-S3 target:

```bash
cd esp32
idf.py set-target esp32s3
idf.py menuconfig    # optional: inspect camera, PSRAM, partition, and log settings
idf.py build
```

Flash and monitor the board:

```bash
idf.py flash monitor
```

If your serial port is not auto-detected, pass it explicitly:

```bash
idf.py -p /dev/cu.usbmodemXXXX flash monitor
```

The firmware logs predictions through `ESP_LOGI`. The final path is camera capture, RGB565 conversion, MobileNetV2 `[-1,1]` preprocessing, INT8 quantization, TFLite Micro inference, confidence-threshold rejection, and serial output.

## Evaluation methodology

The headline result is the honest originals-only test, not the augmented file-level test. See [`bench/results/calibration_report.md`](bench/results/calibration_report.md) for the summary and confusion matrices.

The key correction was finding **F1**: the on-disk test split contained augmented variants of the same 60 original captures. Those 780 files are useful as an augmentation-robustness panel, but they are not 780 independent test examples. The reported headline therefore uses only the de-contaminated originals-only set: 20 original captures per class, `n=60` total.

On that honest set the final QAT MobileNetV2 model reaches **98.33% accuracy** and **0.9833 macro-F1**. The only observed error is one `Amine` capture predicted as `Rifki`; `Rifki` and `Jakub` are perfect in the originals-only confusion matrix.

The calibration report also compares the biased full augmented test (`97.69%`, `n=780`) against the honest test (`98.33%`, `n=60`). The difference is small, but the interpretation is very different: the augmented set checks robustness to transformations, while the originals-only set supports statistical claims.

Additional result files include:

- [`bench/results/stats_summary.md`](bench/results/stats_summary.md) for Wilson confidence intervals, cluster bootstrap, and rejection sweep.
- [`bench/results/rejection_sweep.csv`](bench/results/rejection_sweep.csv) for threshold tradeoffs.
- [`bench/results/mac_count.csv`](bench/results/mac_count.csv) for compute estimates.
- [`bench/results/firmware_preprocess_check.md`](bench/results/firmware_preprocess_check.md) for the F3 regression check.

## Design space and tradeoffs

The design space is summarized in [`bench/results/tuner_summary.md`](bench/results/tuner_summary.md) and connected to the written report outline in [`docs/report/outline.md`](docs/report/outline.md).

Explored axes included:

| Axis | Values considered | Tradeoff |
| --- | --- | --- |
| MobileNetV2 width `alpha` | `{0.35, 0.5, 0.75}` | Accuracy margin vs flash, RAM pressure, and MACs. |
| Input size `IMG_SIZE` | `{96, 160}` | Face detail vs tensor arena, preprocessing cost, and compute. |
| Dense head width | `{32, 64, 128}` | Classifier capacity vs parameters and overfitting risk. |
| Quantization | PTQ and QAT | Conversion simplicity vs INT8 accuracy. |
| Rejection threshold | `0.80`, `0.85`, `0.90`, `0.95` plus finer sweep | False accept vs false reject. |

The best Keras-Tuner validation trial was `alpha=0.5`, but the smallest model within one percentage point was `alpha=0.35`, dense `32`. The final release picks **MobileNetV2 `alpha=0.35`, `IMG_SIZE=96`, dense `32`** because it is the tightest embedded fit while preserving the honest `98.33%` result and keeping the TFLite artefact to about **662 KB** with about **10.7M MACs**.

## Known issues / fixed bugs

| Finding | Description | Commit | Status |
| --- | --- | --- | --- |
| F3 — firmware preprocess mismatch | Firmware normalized RGB bytes to `[0,1]`, while training used MobileNetV2 `[-1,1]`; this shifted INT8 input into only the non-negative half of the expected range. | `ff18dcd` | **Fixed**; regression check in `python/bench/firmware_preprocess_check.py`. |
| F1 — test-set contamination | The test directory contained augmented variants of the same originals, so file-level `n=780` overstated independence. | `c3174dd` | **Fixed** for reporting by evaluating originals-only `n=60`. |
| F2 — validation/test leak | `python/main.py` and `python/qat_export.py` previously used test data for validation/early stopping. | `cf02a95` | **Fixed** with train-derived validation split and a baseline retrain mitigation. |

## Citing this work

If this repository helps your course project or embedded-ML experiment, please cite it using [`CITATION.cff`](CITATION.cff). GitHub can render that file into BibTeX or APA-style citations from the repository sidebar.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

## AI usage

AI tools were used for planning, review, methodology suggestions, evaluation-harness scaffolding, and documentation drafting. They were not used to make the final design decision, collect or label the dataset, or replace human course interpretation. See [`docs/report/ai-usage.md`](docs/report/ai-usage.md) for the full disclosure.

## Contributors

- Amine
- Rifki
- Jakub Piotrowski

## Open-source release notes

The face images in `data/` are private biometric data and are deliberately excluded from git. Reproducing the exact numbers requires the team's private dataset or a consented substitute dataset in the same folder layout.

Generated artefacts in `python/gen/` are ignored by default, with selected release models tracked only when needed for reproducibility. The published evaluation outputs under `bench/results/` are kept so that readers can audit the reported numbers without rerunning the full training pipeline.

Before using this as a template for another face-recognition project, replace the private dataset, rerun the originals-only evaluation, and re-measure latency on the target hardware. Do not publish biometric data without explicit consent and a clear retention policy.

Report artefacts are intentionally stored in small, auditable formats where possible. Large local scratch arrays can be regenerated from the scripts and are not required for a normal clone.
