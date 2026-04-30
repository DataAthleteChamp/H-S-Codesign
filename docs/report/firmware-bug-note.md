# Firmware bug note: F3 preprocessing mismatch

## Summary

Finding F3 was a deployment bug in the ESP32 firmware preprocessing path.

Training used the MobileNetV2 convention:

```text
pixel_float = pixel_uint8 / 127.5 - 1.0
```

That maps camera bytes from `[0,255]` to `[-1,1]` before INT8 quantization.

The firmware had instead normalized camera bytes to `[0,1]`.

For an INT8 model with input scale about `0.00784314` and zero point `0`, that meant the device used mostly `[0,127]` rather than the expected signed range around `[-127,127]`.

## How it was detected

The issue was found during a rubber-duck code review of the C++ firmware alongside the Python training pipeline.

The Python side was clear: `python/preprocess.py`, `python/main.py`, and `python/qat_export.py` all expect MobileNetV2-style `[-1,1]` inputs.

The firmware side did not match that contract.

Reviewing the math in `esp32/main/inference.cpp` showed a divide-by-255 path that produced `[0,1]` before quantization.

This was easy to miss because the model still received valid int8 values and could produce plausible predictions.

The bug was therefore not a compiler failure; it was an input-distribution mismatch.

## Fix

The fix was to cherry-pick/port the `mobilenet_v2_preprocess` helper from `origin/main` into the selected trunk.

That helper applies the same equation as training:

```text
pixel_float = pixel_uint8 / 127.5 - 1.0
pixel_int8  = round(pixel_float / INPUT_SCALE) + INPUT_ZERO_POINT
```

The helper then clips to the legal int8 interval.

The relevant fix commit is `ff18dcd`.

After the fix, firmware preprocessing and Python preprocessing are deliberately named and documented as the same MobileNetV2 preprocessing step.

## Regression test

The regression test is `python/bench/firmware_preprocess_check.py`.

It loads the TFLite model, reads the input quantization parameters, generates a synthetic RGB image, and compares the old formula against the fixed formula.

The old buggy formula produces no negative int8 values.

The fixed formula produces a balanced signed distribution.

The last recorded report is [`bench/results/firmware_preprocess_check.md`](../../bench/results/firmware_preprocess_check.md).

The recorded fixed-path statistics pass the expected checks:

- mean near zero: `-1.07`
- negative fraction: `50.3%`
- positive fraction: `49.2%`
- minimum: `-127`
- maximum: `127`

Run it locally with:

```bash
python -m python.bench.firmware_preprocess_check --tflite python/gen/model.tflite
```

The Makefile wrapper is:

```bash
make bench-firmware-check
```

## Lesson

Training preprocessing and firmware preprocessing must be kept in lockstep.

For embedded ML, a one-line normalization difference can silently invalidate an otherwise correct model conversion.

The safest pattern is to document the preprocessing equation, give the firmware helper a specific name, and keep an offline regression test that checks the quantized input distribution whenever the model or firmware changes.
