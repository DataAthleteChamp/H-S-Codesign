# Face Recognition on XIAO ESP32-S3 Sense

DTU 02214 - Hardware/Software Codesign

This repository contains a small hardware/software codesign project for running face recognition on the XIAO ESP32-S3 Sense. The project trains and evaluates lightweight image-classification models in Python, converts the selected model to TensorFlow Lite Micro format, and deploys it to an ESP32-S3 camera application.

## Repository Layout

- `esp32/` - ESP-IDF firmware for camera capture, preprocessing, and on-device inference.
- `ModelExploration/` - earlier model experiments and prototype firmware.

## Main Workflow

1. Train and compare candidate models in `ModelExploration`.
2. Export the selected model as `.tflite`, `.c`, and `.h` files.
3. Build and flash the ESP32-S3 firmware in `esp32/`.
4. Test real-time camera predictions on the device.

## Build

From an ESP-IDF shell:

```powershell
cd Project\H-S-Codesign\esp32
idf.py set-target esp32s3
idf.py build
idf.py -p <PORT> flash
```
