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
  - `python/quantized_model/model.tflite`
  - `python/quantized_model/model.c`
  - `python/`quantized_model `/model.h`
- It then evaluates the quantized TFLite model on the cached test data to verify the exported model still performs correctly.

## ESP32 Program

`esp32/` is the embedded deployment target for the quantized model.

- `esp32/main/main.cpp` captures camera frames on the ESP32-S3, center-crops the frame, resizes it to the model input size, applies MobileNetV2-style preprocessing, and sends the result to the inference code.
- `esp32/main/inference.cpp` loads `model.c` and `model.h` with TensorFlow Lite Micro, quantizes the input to `int8`, runs inference, and converts the output back to floating-point scores.
- The ESP32 firmware expects the generated model files from `python/quantized_model/`.

To build and flash the ESP32-S3 app from an ESP-IDF shell:

```powershell
cd Project\H-S-Codesign\esp32_camera_classifier
idf.py set-target esp32s3
idf.py fullclean
idf.py build
idf.py -p <PORT> flash
```

## ESP32 Preview and Prediction Capture

`python/preview_pred.py` previews the live RGB frame stream sent from the ESP32 over USB serial and overlays the latest embedded prediction on the image.

The preview window shows:

- `Best`: the class with the highest score and its percentage.
- `Prediction`: the final class name when the best score is at least `0.85`; otherwise it shows `Unknown`.

The script also saves one frame and its prediction data every 2 seconds by default. Saved images are written as PNG files, and `frames.csv` stores the metadata for each saved frame.

Install the required Python packages:

```powershell
python -m pip install -r python\requirements.txt
```

Run the preview with the default serial port `COM7`:

```powershell
python python\preview_pred.py
```

Run with a different port:

```powershell
python python\preview_pred.py --port COM5
```

Keyboard controls:

- `SPACE`: save the current frame immediately.
- `r`: toggle timed recording on or off.
- `s`: send the stream toggle command to the ESP32.
- `q` or `ESC`: quit the preview.

Output files:

```text
pred_capture/
|-- frame_YYYYMMDD_HHMMSS_microseconds.png
`-- frames.csv
```

`frames.csv` contains:

```text
timestamp, filename, label, index, confidence, Amine, Rifki, Jakub
```

The `label`, `index`, and `confidence` columns come from the ESP32 prediction message. The `Amine`, `Rifki`, and `Jakub` columns contain the raw class scores.

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
