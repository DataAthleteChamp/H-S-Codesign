# Face Recognition for ESP32-S3

DTU 02214 — Hardware/Software Codesign

ML pipeline for dormitory access control: data augmentation, MediaPipe face detection, MobileNetV2 transfer learning, and INT8 TFLite export targeting the ESP32-S3 microcontroller.

## Repository Structure

```
.
├── README.md
├── python/
│   ├── requirements.txt
│   ├── augment.py          # Stage 1 — data augmentation (8 variants per image)
│   ├── preprocess.py       # Stage 2 — face detection, crop, resize, normalize
│   ├── main.py             # Stage 3 — train MobileNetV2 + export INT8 TFLite
│   └── utils/
│       ├── __init__.py
│       ├── eval_utils.py   # precision/recall/F1, confusion matrix
│       └── export_tflite.py# write model.h / model.c for ESP32
└── data/                   # not tracked — see Data Format below
```

## Pipeline Overview

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `augment.py` | Applies 8 augmentations (flip, rotate, shift/scale, brightness, blur, compression, occlusion, grayscale) per original image. Idempotent. |
| 2 | `preprocess.py` | Detects faces with MediaPipe, crops with 15 % padding, resizes to 96x96, normalizes to [0,1]. Saves `.npy` arrays. |
| 3 | `main.py` | Two-phase transfer learning on MobileNetV2 (frozen head → fine-tune last 30 layers). Exports INT8-quantized TFLite model and generates `model.c` / `model.h` for ESP32. |

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

- **Architecture:** MobileNetV2 (ImageNet pretrained) with custom classification head
- **Input:** 96x96x3 RGB
- **Training:** two-phase transfer learning — frozen feature extraction then fine-tuning last 30 layers
- **Quantization:** full INT8 (weights + activations) via TFLite representative-dataset calibration
- **Output:** `model.tflite`, `model.c`, and `model.h` ready for ESP32-S3 deployment

## Dependencies

- TensorFlow / Keras
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
