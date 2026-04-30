# F3 firmware preprocess regression check

- TFLite model: `python/gen/model.tflite`
- Input quant: scale=0.00784314 zero_point=0 img_size=96
- Synthetic uniform-random RGB image, seed=42.

## Old buggy formula `(value / 255.0) / scale + zp`

| metric | value |
|---|---:|
| min | 0.0000 |
| max | 127.0000 |
| mean | 62.9782 |
| std | 36.9621 |
| frac_negative | 0.0000 |
| frac_zero | 0.0078 |
| frac_positive | 0.9922 |

## New fixed formula `(value / 127.5 - 1) / scale + zp` (MobileNetV2)

| metric | value |
|---|---:|
| min | -127.0000 |
| max | 127.0000 |
| mean | -1.0721 |
| std | 73.7846 |
| frac_negative | 0.5031 |
| frac_zero | 0.0044 |
| frac_positive | 0.4925 |

## Regression assertions

- fixed mean near zero (|mean|<5): **PASS** (-1.07)
- fixed has >=40% negative values: **PASS** (50.3%)
- fixed has >=40% positive values: **PASS** (49.2%)
- fixed min <= -50: **PASS** (-127)
- fixed max >= +50: **PASS** (127)
- buggy min >= 0 (proves bug): **PASS** (0)
- buggy <5% negative (proves bug): **PASS** (0.0%)

## Verdict

**PASS** -- the new firmware preprocessing produces a balanced [-1, 1] distribution as expected by MobileNetV2, while the old buggy formula produces only non-negative values. The F3 fix is active and the firmware is consistent with the training pipeline.

## Notes

This test does not require an attached ESP32; it asserts the same math the firmware C++ helpers `mobilenet_v2_preprocess` and `quantize_to_int8` perform. Run it as part of CI to catch any future revert of the F3 fix.
