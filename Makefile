# Top-level developer commands for the XIAO ESP32-S3 Sense face-recognition
# project. See README.md for a full walk-through. All targets are idempotent
# and assume you run them from the repository root.

PYTHON      ?= python3
VENV        ?= venv
VENV_PY     := $(VENV)/bin/python
VENV_PIP    := $(VENV)/bin/pip
ESP_PORT    ?= /dev/cu.usbmodem*

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ Setup ----

$(VENV)/bin/activate: python/requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r python/requirements.txt

.PHONY: venv
venv: $(VENV)/bin/activate ## Create venv and install python/requirements.txt

# ------------------------------------------------------------- Data --------

.PHONY: augment
augment: venv ## Generate 12 augmented variants per original (Albumentations 2.0)
	$(VENV_PY) python/augment.py

.PHONY: preprocess
preprocess: venv ## Detect faces, crop, resize 96x96, cache to python/gen/x_*.npy
	cd python && ../$(VENV_PY) preprocess.py

# ----------------------------------------------------------- Training ------

.PHONY: train
train: venv ## Train MobileNetV2 with QAT and INT8 export (design-space main.py)
	cd python && ../$(VENV_PY) main.py

.PHONY: qat
qat: venv ## Legacy-Keras QAT fallback (sets TF_USE_LEGACY_KERAS=1)
	cd python && TF_USE_LEGACY_KERAS=1 ../$(VENV_PY) qat_export.py

.PHONY: tune
tune: venv ## Run Keras-Tuner Hyperband search
	cd python && ../$(VENV_PY) tune.py

.PHONY: baseline-retrain
baseline-retrain: venv ## F2-clean baseline retrain (saves baseline_model.{tflite,h,c})
	$(VENV_PY) python/bench/run_baseline_retrain.py

# ----------------------------------------------------------- Evaluation ----

.PHONY: originals
originals: venv ## (Re)build the cleaned originals-only test arrays in bench/results/
	$(VENV_PY) python/bench/build_originals_test.py \
		--data-dir data --out-dir bench/results

.PHONY: eval
eval: venv ## Evaluate INT8 TFLite on the originals-only test set (n=60)
	$(VENV_PY) python/bench/eval_branches.py \
		--model python/gen/model.tflite \
		--x bench/results/x_test_originals_96_pm1.npy \
		--y bench/results/y_test_originals.npy \
		--capture-ids bench/results/capture_ids_originals.npy \
		--norm pm1 \
		--out bench/results/model_originals_test.npz

.PHONY: compare
compare: venv ## Paired McNemar head-to-head: model.tflite vs baseline_model.tflite
	$(VENV_PY) python/bench/compare_models.py \
		--baseline   python/gen/baseline_model.tflite \
		--challenger python/gen/model.tflite \
		--report     bench/results/mcnemar_comparison.md

.PHONY: stats
stats: venv ## Wilson CI, cluster bootstrap, rejection sweep
	$(VENV_PY) python/bench/run_stats.py

.PHONY: figures
figures: venv ## Regenerate report figures into docs/figures/
	$(VENV_PY) python/bench/make_figures.py

.PHONY: bench-firmware-check
bench-firmware-check: venv ## F3 regression test: firmware preprocess matches training
	$(VENV_PY) python/bench/firmware_preprocess_check.py

# ----------------------------------------------------------- Deploy --------

.PHONY: deploy
deploy: ## Copy python/gen/model.{c,h} into esp32/main/
	$(VENV_PY) python/deploy.py

.PHONY: firmware-build
firmware-build: ## idf.py build (requires ESP-IDF v5.x and `. $$IDF_PATH/export.sh`)
	cd esp32 && idf.py set-target esp32s3 && idf.py build

.PHONY: firmware-flash
firmware-flash: ## idf.py -p $(ESP_PORT) flash monitor
	cd esp32 && idf.py -p $(ESP_PORT) flash monitor

# ----------------------------------------------------------- Hygiene -------

.PHONY: lint
lint: venv ## Run ruff over the python/ tree
	$(VENV_PY) -m ruff check python || ($(VENV_PIP) install ruff && $(VENV_PY) -m ruff check python)

.PHONY: clean-gen
clean-gen: ## Remove generated python/gen/ artefacts (keeps tracked .h/.c/.tflite)
	rm -rf python/gen/tuner python/gen/__pycache__
	find python/gen -maxdepth 1 -type f \
		! -name 'model.tflite' ! -name 'model.h' ! -name 'model.c' \
		! -name 'baseline_model.tflite' ! -name 'baseline_model.h' ! -name 'baseline_model.c' \
		! -name 'val_split_seed42.json' \
		-delete

.PHONY: clean
clean: clean-gen ## Clean generated artefacts
	rm -rf $(VENV) build dist .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
