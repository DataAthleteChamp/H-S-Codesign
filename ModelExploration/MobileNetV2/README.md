# Face Recognition for the XIAO ESP32-S3 Sense

> A 3-class face-recognition pipeline that fits in 662 KB of INT8 TFLite,
> targets the **XIAO ESP32-S3 Sense** development board, and ships with a
> reproducible Python training pipeline, a statistically-grounded evaluation
> harness, and the corresponding ESP-IDF firmware.
> Built as the **DTU 02214 — ML for Embedded Systems** course project
> (Spring 2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](python/requirements.txt)
[![TensorFlow: 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](python/requirements.txt)
[![ESP-IDF: 5.x](https://img.shields.io/badge/ESP--IDF-5.x-red.svg)](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)

## What this is

A complete edge-ML codesign exercise: collect images of three team members,
train a quantized MobileNetV2 classifier on a laptop, deploy it to a
dual-core 240 MHz ESP32-S3 with 2 MB SRAM and 8 MB PSRAM, and measure honest
on-device performance. The repo includes everything the course asks for:
training and conversion pipeline (Python), application code (C++ on
ESP-IDF), and a written design-space / verification report (`docs/report/`).

| Stat | Value |
|---|---|
| Architecture | MobileNetV2 alpha = 0.35 + small dense head |
| Input | 96 × 96 RGB, MobileNetV2 [-1, 1] preprocessing |
| Quantization | INT8 QAT (full-integer) |
| TFLite size | **662 KB** |
| Total MACs | ~10.7 M (Conv 9.07 M, DW-Conv 1.58 M, FC 41 K) |
| Honest accuracy (n = 60 captures) | **98.33 %** for the deployed model, **93.33 %** for the F2-clean baseline |
| Operating rejection threshold | softmax q ≥ 0.77 (100 % accepted accuracy, 96.7 % accept rate) |
| ESP32-S3 latency | not yet measured on hardware (see `docs/report/outline.md`) |

> **What "honest" means:** we re-evaluate on the 60 *original* captures only
> (the test split was previously contaminated with augmented variants of
> those same captures — finding F1; the on-disk pollution has since been
> removed and audited via
> [`bench/results/test_pollution_inventory.md`](bench/results/test_pollution_inventory.md)
> and [`data/_quarantine/test_augmented/manifest.json`](../../data/_quarantine/test_augmented/manifest.json),
> see [`docs/report/methods_test_hygiene.md`](docs/report/methods_test_hygiene.md)).
> Both candidate models, the McNemar head-to-head, and the bootstrap CIs
> are computed at the capture level.

## Repository layout

```
.
├── README.md                          # this file
├── PROJECT.md                         # course project specification
├── LICENSE                            # MIT
├── CITATION.cff                       # how to cite
├── CONTRIBUTING.md                    # contribution + commit rules
├── Makefile                           # one-line developer commands
├── python/
│   ├── augment.py                     # Albumentations 2.0, 12 augmentations
│   ├── preprocess.py                  # MediaPipe Tasks API face detection + crop
│   ├── main.py                        # design-space training + QAT + INT8 export
│   ├── qat_export.py                  # legacy-Keras QAT fallback
│   ├── tune.py                        # Keras-Tuner Hyperband
│   ├── deploy.py                      # write esp32/main/model.{c,h}
│   ├── utils/
│   │   ├── train_val_split.py         # F2 fix: held-out val from train
│   │   ├── eval_utils.py
│   │   └── export_tflite.py
│   ├── bench/                         # evaluation harness (independent of training)
│   │   ├── eval_branches.py           # INT8 TFLite eval driver
│   │   ├── build_originals_test.py    # cleaned originals-only test set
│   │   ├── compare_models.py          # paired McNemar head-to-head
│   │   ├── stats.py                   # Wilson CI, cluster bootstrap, exact McNemar
│   │   ├── run_stats.py               # rejection sweep + bootstrap driver
│   │   ├── run_baseline_retrain.py    # F2-clean retrain (insurance)
│   │   ├── distill_tuner.py           # parse Keras-Tuner trials
│   │   ├── mac_count.py               # MAC count from a TFLite file
│   │   ├── footprint.py               # flash/params/arena summary
│   │   ├── make_figures.py            # report figures
│   │   └── firmware_preprocess_check.py  # F3 regression test
│   ├── tools/
│   │   └── serial_latency_logger.py   # parse on-device latency_ms log lines
│   ├── realworld_webcam_test.py       # desktop webcam proxy test
│   └── gen/                           # generated artefacts (model.tflite, etc.)
├── esp32/
│   ├── sdkconfig.defaults             # ESP32-S3, PSRAM Octal, large partition
│   └── main/
│       ├── main.cpp                   # capture → preprocess → infer → log
│       ├── inference.cpp              # TFLite Micro + MobileNetV2 [-1,1]
│       ├── camera.cpp                 # OV2640 driver for XIAO Sense
│       ├── model.{c,h}                # INT8 model as a C array
│       ├── CMakeLists.txt
│       └── idf_component.yml
├── bench/results/                     # markdown + CSV + JSON evaluation outputs
├── docs/
│   ├── decision.md                    # team-signed consolidation decision
│   ├── branch-audit.md                # per-branch inventory + history
│   ├── architecture.md                # pipeline + firmware diagrams
│   ├── figures/                       # report figures (PNG)
│   └── report/                        # full report markdown sources
└── data/                              # captured images (git-ignored, private)
```

## Hardware setup

- **Board:** [Seeed XIAO ESP32-S3 Sense](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)
- **MCU:** ESP32-S3, dual Xtensa LX7 @ 240 MHz, 512 KB SRAM + 8 MB PSRAM (Octal)
- **Camera:** OV2640 on the Sense board's expansion ribbon
- **Flash:** 8 MB (`CONFIG_PARTITION_TABLE_SINGLE_APP_LARGE`)
- **Power:** USB-C (no external supply needed for tethered demo)

The default firmware captures 320 × 240 RGB565, center-crops, scales to 96 × 96,
quantizes with the model's input scale and zero-point, and runs inference once
per second. Send the byte `S` over USB-CDC to toggle a per-frame RGB888 dump
that the host-side `python/preview_pred.py` can render.

## Reproduce: training pipeline

Requires Python 3.12 and ~6 GB of RAM. CUDA is *not* required; the Phase 2
baseline retrain runs CPU-only with `tf.config.experimental.enable_op_determinism()`
under `CUDA_VISIBLE_DEVICES=""`.

```bash
# 1. one-time setup
make venv                       # create ./venv and install requirements.txt

# 2. drop captured images under data/<class>/{train,test}/

# 3. augment (12 variants per original; Albumentations 2.0)
make augment

# 4. preprocess: MediaPipe face detect, crop, resize 96×96, normalize [-1,1]
#    (cached to python/gen/x_*.npy and y_*.npy)
source venv/bin/activate
python python/preprocess.py

# 5. train + QAT + INT8 export with the leak-free validation split
make train                      # design-space main.py
# OR fallback for environments where TF Keras 3.x QAT is finicky:
make qat                        # legacy tf-keras path

# 6. deploy: regenerate esp32/main/model.{c,h}
python python/deploy.py
```

Generated artefacts are written to `python/gen/` and the C arrays end up in
`esp32/main/`. The `python/gen/baseline_model.tflite` is the F2-clean
insurance retrain; `python/gen/model.tflite` is the deployed candidate.

## Reproduce: firmware build

Install [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/),
then:

```bash
. $IDF_PATH/export.sh           # once per shell
cd esp32
idf.py set-target esp32s3
idf.py menuconfig               # optional: tweak partition / log level
make firmware-build             # equivalent to: idf.py build
make firmware-flash             # idf.py -p $PORT flash monitor
```

`managed_components/` (esp-tflite-micro, esp32-camera) are downloaded
automatically on the first build.

## Evaluation methodology

Full numbers are in `bench/results/`. Headline:

```bash
make eval                                       # INT8 TFLite on originals-only
make bench-firmware-check                       # F3 regression test
python python/bench/compare_models.py \
    --baseline   python/gen/baseline_model.tflite \
    --challenger python/gen/model.tflite \
    --report     bench/results/mcnemar_comparison.md
```

| Step | Output |
|---|---|
| Calibration vs biased test | `bench/results/calibration_report.md` |
| Wilson CI / cluster-bootstrap / rejection sweep | `bench/results/stats_summary.md` |
| Paired head-to-head | `bench/results/mcnemar_comparison.md` |
| Tuner distillation (30 trials) | `bench/results/tuner_summary.md` |
| MAC count from TFLite | `bench/results/mac_count.csv` |
| Embedded footprint (flash/params/arena) | `bench/results/footprint.md` |
| F3 regression test | `bench/results/firmware_preprocess_check.md` |
| Report figures | `docs/figures/` |
| One-page numbers cheat-sheet | `docs/report/results-tables.md` |
| ML model card | `docs/report/model-card.md` |
| Live demo / on-device measurement checklist | `docs/report/live-demo-checklist.md` |

The cleaned test set has **60 captures** (20 per class). Asymptotic chi-square
McNemar is unsafe at this n, so we use the exact binomial form, and all CIs
are computed at the capture level. The pre-registered lexicographic decision
rule is documented in `docs/decision.md`.

## Design space and trade-offs

The Keras-Tuner ran 30 Hyperband trials over `alpha`, dense width, dropout,
learning rate, and label smoothing. The full sweep is in
`bench/results/tuner_{all,top10,pareto,summary}.csv/md`. For a fixed
target-size budget we picked the smallest `alpha` within 1 pp of the best
trial:

| Knob | Choice | Reason |
|---|---|---|
| `alpha` | 0.35 | smallest within 1 pp of the best alpha=0.5 trial |
| dense head | 32 units | best-in-class trial 0029, also smallest |
| `IMG_SIZE` | 96 | tightest fit; arena ≈ 1 MB on PSRAM |
| dropout | 0.4 / 0.1 | regularises the small head |
| label smoothing | 0.05 | improves rejection calibration |
| quantization | INT8 QAT | recovers ~1.5 pp over PTQ at the same size |
| rejection q | 0.77 (softmax) | 100 % accepted accuracy at 96.7 % accept rate |

## Known issues / fixed bugs

| Tag | Summary | Status | Commit |
|---|---|---|---|
| **F1** | Test split was contaminated with augmented variants of the same originals (260 files but only 20 independent captures per class). | Fixed by `python/bench/build_originals_test.py`; honest n = 60. | `c3174dd` |
| **F2** | `python/main.py` and `python/qat_export.py` passed `x_test` as `validation_data`, leaking the test set into early stopping and model selection. | Fixed by `python/utils/train_val_split.py`; held-out val drawn from train, grouped by capture id. | `cf02a95` |
| **F3** | Firmware preprocessor normalised RGB to `[0, 1]` but training used MobileNetV2 `[-1, 1]`, silently shifting the input distribution. | Fixed by replacing `inference.cpp` with the `mobilenet_v2_preprocess` helper; regression test in `python/bench/firmware_preprocess_check.py`. | `ff18dcd` |

A full discussion of how each was caught and fixed is in
`docs/report/firmware-bug-note.md`.

## Citing this work

See [`CITATION.cff`](CITATION.cff). GitHub renders a "Cite this repository"
button using that file.

## License

[MIT](LICENSE). The course project material in `PROJECT.md` belongs to DTU
and is reproduced for pedagogical context only.

## Use of generative AI

Per the course rules, AI was used as an evidence-and-review collaborator
(rubber-duck reviews, harness scaffolding, report figures). All design
decisions were made by the team. Full disclosure: [`docs/report/ai-usage.md`](docs/report/ai-usage.md).

## Contributors

- Amine
- Rifki
- Jakub Piotrowski

Contribution guidelines: [`CONTRIBUTING.md`](CONTRIBUTING.md).