# Common developer commands for the XIAO ESP32-S3 Sense face-recognition project.
# Run `make help` from the repository root to list available targets.

PYTHON ?= python3.12
VENV ?= venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
REQ := $(if $(wildcard requirements.txt),requirements.txt,python/requirements.txt)
MODEL ?= python/gen/model.tflite
X_TEST ?= bench/results/x_test_originals_96_pm1.npy
Y_TEST ?= bench/results/y_test_originals.npy
CAPTURE_IDS ?= bench/results/capture_ids_originals.npy
EVAL_OUT ?= bench/results/jakubs_qat_originals_test.npz
ESP_PORT ?=

.DEFAULT_GOAL := help

.PHONY: help venv augment preprocess train qat eval bench-firmware-check clean-gen firmware-build firmware-flash report-pdf

help: ## Show available Makefile targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtual environment and install project requirements
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PIP) install -r $(REQ)

augment: ## Generate Albumentations variants for images in data/<class>/{train,test}
	$(VENV_PY) python/augment.py

preprocess: ## Build python/gen/*.npy caches with MediaPipe crop and [-1,1] normalization
	$(VENV_PY) python/preprocess.py

train: ## Train the MobileNetV2 transfer-learning model and export artefacts
	$(VENV_PY) python/main.py

qat: ## Run the legacy-Keras QAT exporter for the final INT8 TFLite model
	TF_USE_LEGACY_KERAS=1 $(VENV_PY) python/qat_export.py

eval: ## Evaluate the final TFLite model on the cleaned originals-only test set
	$(VENV_PY) -m python.bench.eval_branches \
		--model $(MODEL) \
		--x $(X_TEST) \
		--y $(Y_TEST) \
		--capture-ids $(CAPTURE_IDS) \
		--norm pm1 \
		--out $(EVAL_OUT)

bench-firmware-check: ## Run the F3 firmware-preprocess regression check
	$(VENV_PY) -m python.bench.firmware_preprocess_check --tflite $(MODEL)

clean-gen: ## Remove generated cache and tuner directories under python/gen/
	rm -rf python/gen/{cache,tuner}/

firmware-build: ## Build ESP-IDF firmware (requires ESP-IDF v5.x environment)
	cd esp32 && idf.py build

firmware-flash: ## Flash firmware and open the serial monitor
	cd esp32 && idf.py $(if $(ESP_PORT),-p $(ESP_PORT),) flash monitor

report-pdf: ## Build report PDF with pandoc, or print a TODO if pandoc is unavailable
	@if command -v pandoc >/dev/null 2>&1; then \
		pandoc docs/report/outline.md -o docs/report/report.pdf; \
	else \
		echo "TODO: install pandoc and wire the final report sources before building docs/report/report.pdf"; \
	fi
