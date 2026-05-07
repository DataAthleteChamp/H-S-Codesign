# Face Recognition on XIAO ESP32-S3 Sense
## Authors
Rifki Firdaus - s250169
Jakub Piotrowski - s253074
Mohamed Amine Cheikh - s252221


DTU 02214 - Hardware/Software Codesign

This repository contains a small hardware/software codesign project for running face recognition on the XIAO ESP32-S3 Sense. The project trains and evaluates lightweight image-classification models in Python, converts the selected model to TensorFlow Lite Micro format, and deploys it to an ESP32-S3 camera application.

## Repository Layout

- `esp32/` - ESP-IDF firmware for camera capture, preprocessing, and on-device inference.
- `ModelExploration/` - model experiments and prototype firmware.
- `Preview.py` - Python program to test and capture image.

## Main Workflow

1. Collect and prepare dataset face images for each class.
2. Train and compare candidate models in `ModelExploration`.
3. Select the Xception-based model based on the trade-off between accuracy, model size, memory usage, and inference latency.
4. Convert the selected model to TensorFlow Lite format and export it as `.c` and `.h` files for firmware embedding.
5. Build and flash the ESP32-S3 firmware in `esp32/`.
6. Run the deployed pipeline on the device: camera capture, face detection, Xception inference, and prediction thresholding.
7. Evaluate real-world results.

## Chosen Model

The deployed face-classification model is based on the Xception architecture. Xception was selected because it uses depthwise separable convolutions, which significantly reduce the number of parameters and computational operations compared to standard convolutions. This makes the architecture lightweight and efficient while still preserving important image features for classification.

## Build

From an ESP-IDF shell:

```powershell
cd esp32
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash
```

## Run and Test

From an ESP-IDF shell:

```powershell
cd esp32
idf.py -p <PORT> flash
cd ..
python Preview.py --port <PORT>
```

The test image will be automatically saved into a folder.
