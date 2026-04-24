# Face Recognition for ESP32-S3

DTU 02214 - Hardware/Software Codesign

This project has three main stages: training a 3-class MobileNetV2 model in Keras, quantizing that model to INT8 TensorFlow Lite, and deploying the exported model on an ESP32-S3 camera application.

## MobileNet Notebook

`python/MobileNetV2_3ClassKeras.ipynb` is the main training notebook.

- It reads the 3-class dataset from `data/` using the `train/` and `test/` folder split for `Amine`, `Rifki`, and `Jakub`.
- It trains a `MobileNetV2` transfer-learning model with input size `160x160x3`.
- It evaluates the trained model on the held-out test set.
- It saves the trained Keras model to `python/saved_model/mobilenetv2_3class_classifier.keras`.
- It also exports the deterministic preprocessed training subset to `python/saved_model/x_train.npy` and `python/saved_model/y_train.npy`, which are later used for quantization calibration.

## Quantization Notebook

`python/Quantization.ipynb` performs the post-training export step.

- It loads the trained Keras model from `python/saved_model/mobilenetv2_3class_classifier.keras`.
- It loads `x_train.npy` as the representative calibration dataset.
- It converts the model to fully INT8 TensorFlow Lite format.
- It exports:
  - `python/gen/model.tflite`
  - `python/gen/model.c`
  - `python/gen/model.h`
- It then evaluates the quantized TFLite model on the cached test data to verify the exported model still performs correctly.

## ESP32 Program

`esp32_camera_classifier/` is the embedded deployment target for the quantized model.

- `esp32_camera_classifier/main/main.cpp` captures camera frames on the ESP32-S3, center-crops the frame, resizes it to the model input size, applies MobileNetV2-style preprocessing, and sends the result to the inference code.
- `esp32_camera_classifier/main/inference.cpp` loads `model.c` and `model.h` with TensorFlow Lite Micro, quantizes the input to `int8`, runs inference, and converts the output back to floating-point scores.
- The ESP32 firmware expects the generated model files from `python/gen/`.
- The current embedded pipeline performs classification on the center crop of the camera frame. It does not run face detection on-device.
- PSRAM is used for the frame buffer and tensor arena, and the partition table is enlarged so the firmware can include the model binary.

To build and flash the ESP32-S3 app from an ESP-IDF shell:

```powershell
cd Project\H-S-Codesign\esp32_camera_classifier
idf.py set-target esp32s3
$env:IDF_CCACHE_ENABLE="0"
idf.py build
idf.py flash monitor
```

## Dataset Layout

```text
data/
|-- Amine/
|   |-- train/
|   `-- test/
|-- Rifki/
|   |-- train/
|   `-- test/
`-- Jakub/
    |-- train/
    `-- test/
```
