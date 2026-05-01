# Contributing

## Development environment

Target toolchain: Python 3.12, TensorFlow 2.21, Keras 3.14, and ESP-IDF 5.x. Python dependencies are listed in `python/requirements.txt`.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r python/requirements.txt
```

For firmware work, install ESP-IDF 5.x and run `. $IDF_PATH/export.sh` before building under `esp32/`.

## Repository layout

- `python/` — data augmentation, preprocessing, training, tuning, export, and evaluation helpers.
- `esp32/` — ESP-IDF application for XIAO ESP32-S3 Sense camera capture and TFLite Micro inference.
- `data/` — local, git-ignored face-image dataset arranged by person and split.
- `docs/` — report support material, design notes, figures, and decision records.
- `bench/` — generated benchmark outputs, plots, and result tables.

## Workflow

Branch from `main` for every change and open a pull request for review. Do not force-push shared branches; prefer follow-up commits or a new branch when history needs correction.

Run the existing pre-commit hooks before opening a PR, and keep notebook outputs stripped unless the report explicitly needs them.

## Running tests / benchmarks

```bash
python -m python.bench.eval_branches --help
make bench  # when the project Makefile exists
```

Use benchmark outputs under `bench/results/` for report figures and tables. Hardware measurements should be clearly labeled separately from desktop or analytical proxy results.

## Course context

This repository is an academic project for DTU 02214 — Hardware/Software Codesign, Spring 2026. Contributions outside the course team are not expected before the May 7, 2026 submission deadline; after submission, the project may be opened for community improvements.
