# Clean F2-fixed baseline retrain

## Status

Completed single-seed insurance run on CPU with TensorFlow op determinism enabled.

## Configuration

| parameter | value |
| --- | --- |
| seed | 42 |
| IMG_SIZE | 96 |
| alpha | 0.35 |
| dense_units | 32 |
| dropout_1 / dropout_2 | 0.4 / 0.1 |
| learning_rate | 0.0005 |
| label_smoothing | 0.05 |
| train epochs | 20 |
| QAT epochs | 10 |
| batch size | 32 |
| validation split | 15% of train captures, stratified by class and grouped by capture prefix |

## F2 fix / validation split

Validation is held out from `data/*/train/` only; `x_test` is not used for early stopping or model selection. Manifest: `python/gen/val_split_seed42.json`.

- Amine: 68 train captures / 12 val captures; 884 train files / 156 val files.
- Rifki: 68 train captures / 12 val captures; 884 train files / 156 val files.
- Jakub: 68 train captures / 12 val captures; 884 train files / 156 val files.

## Train/validation curves summary

| phase | epochs run | final loss | final acc | final val loss | final val acc | best val loss | best val acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float feature-extraction | 20 | 0.2085 | 0.9966 | 0.2480 | 0.9786 | 0.2440 | 0.9808 |
| QAT | 10 | 0.2119 | 0.9955 | 0.2440 | 0.9786 | 0.2440 | 0.9786 |

Full history JSON: `bench/results/baseline_training_history_seed42.json`.

## TFLite export

- Output model: `python/gen/baseline_model.tflite`
- Header: `python/gen/baseline_model.h`
- Size: 662056 bytes (646.5 KiB)
- Input quantization: scale=0.0078431377, zero_point=0
- Output quantization: scale=0.00390625, zero_point=-128

## Originals-only test evaluation

Evaluation reuses `python/bench/build_originals_test.py` to filter augmented test files and `python/bench/eval_branches.py` for INT8 inference.

| model | correct/n | accuracy | Wilson 95% CI | macro-F1 | path |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline_retrain | 56/60 | 93.33% | [84.07%, 97.38%] | 0.9329 | `python/gen/baseline_model.tflite` |
| existing_model | 59/60 | 98.33% | [91.14%, 99.71%] | 0.9833 | `python/gen/model.tflite` |

Baseline retrain is -5.00 pp vs the evaluated existing model (98.33%). The planning baseline was 98.33%, so the retrain is -5.00 pp vs that reference.

## Known issue

None for this run. Multi-seed retraining is deferred; this is the requested seed-42 insurance baseline.
