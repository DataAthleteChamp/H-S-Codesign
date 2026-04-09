# Face Recognition for ESP32-S3

DTU 02214 — Hardware/Software Codesign

ML pipeline for dormitory access control: data augmentation, MediaPipe face detection, MobileNetV2 transfer learning, and INT8 TFLite export targeting the ESP32-S3 microcontroller.

## Repository Structure

```
.
├── README.md
├── PROJECT.md          # Course project specification
├── python/
│   ├── requirements.txt
│   ├── augment.py          # Stage 1 — data augmentation (12 variants per image)
│   ├── preprocess.py       # Stage 2 — face detection, crop, resize, normalize
│   ├── main.py             # Stage 3 — train MobileNetV2 + QAT + export INT8 TFLite
│   ├── tune.py             # Hyperparameter tuning with Keras Tuner
│   ├── compare.py          # Multi-config experiment comparison for report
│   └── utils/
│       ├── __init__.py
│       ├── eval_utils.py   # precision/recall/F1, confusion matrix
│       └── export_tflite.py# write model.h / model.c for ESP32
└── data/                   # not tracked — see Data Format below
```

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `augment.py` | Applies 12 augmentations (8 single + 4 combined) per original image. Idempotent. |
| 2 | `preprocess.py` | Detects faces with MediaPipe, crops with 15 % padding, resizes to 96x96, normalizes to [-1,1]. Saves `.npy` arrays. |
| 3 | `main.py` | Two-phase transfer learning on MobileNetV2 (frozen head → fine-tune), optional QAT, exports INT8-quantized TFLite model and generates `model.c` / `model.h` for ESP32. |

## Design Space Parameters

Configurable at the top of `main.py`:

| Parameter | Default | Options | Trade-off |
|-----------|---------|---------|-----------|
| `ALPHA` | 0.35 | 0.25 / 0.35 / 0.5 / 1.0 | Model size vs accuracy |
| `DENSE_UNITS` | 64 | 32 / 64 / 128 | Head capacity vs size |
| `LABEL_SMOOTHING` | 0.1 | 0.0 – 0.15 | Confidence calibration |
| `FINE_TUNE_LAYERS` | 20 | 10 / 15 / 20 / 30 | Overfitting vs adaptation |
| `REJECTION_THRESHOLD` | 0.90 | 0.80 – 0.95 | False accept vs false reject |
| `USE_QAT` | True | True / False | QAT vs PTQ quantization |

## Usage

```bash
# Install dependencies
pip install -r python/requirements.txt

# 1. Augment training and test images
python python/augment.py

# 2. Preprocess (face detect + crop + normalize → .npy)
python python/preprocess.py

# 3. Train model and export to TFLite / C
python python/main.py

# Optional: Hyperparameter tuning
python python/tune.py

# Optional: Run experiment comparison for report
python python/compare.py
```

Outputs are written to `python/gen/` (model files, NumPy arrays, C source).

## Data Format

Place images in the following layout before running the pipeline:

```
data/
├── Amine/
│   ├── train/   # training images (.png, .jpg, .jpeg, .bmp)
│   └── test/    # test images
├── Rifki/
│   ├── train/
│   └── test/
└── Jakub/
    ├── train/
    └── test/
```

The `data/` directory is git-ignored because it contains personal photos.

## Model Details

- **Architecture:** MobileNetV2 (ImageNet pretrained, `alpha=0.35`) with custom classification head
- **Input:** 96x96x3 RGB, normalized to [-1, 1]
- **Training:** Two-phase transfer learning — frozen feature extraction then fine-tuning last 20 layers
- **Regularization:** Label smoothing (ε=0.1), dropout, early stopping
- **Quantization:** Full INT8 via QAT (Quantization-Aware Training) or PTQ
- **Unknown rejection:** Softmax confidence threshold (configurable, default 0.90)
- **Output:** `model.tflite`, `model.c`, and `model.h` ready for ESP32-S3 deployment

## Dependencies

- TensorFlow / Keras
- TensorFlow Model Optimization (QAT)
- Keras Tuner (hyperparameter search)
- NumPy
- scikit-learn
- Albumentations
- OpenCV (headless)
- MediaPipe

See `python/requirements.txt` for the full list.

## Authors

- Amine
- Rifki
- Jakub
