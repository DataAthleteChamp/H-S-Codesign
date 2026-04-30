# Finding F3 — firmware preprocessing bug ([0,1] vs [-1,1])

## Summary

The `jakubs-solution` branch had a silent input-distribution bug in the ESP32
firmware: `inference.cpp` normalised camera bytes to **[0, 1]** before
quantizing to INT8, while training (`python/preprocess.py`,
`python/main.py`, `python/qat_export.py`) used **MobileNetV2 [-1, 1]**
preprocessing. The bug halved the dynamic range available to the network on
device and is exactly the class of issue the course warns about under
"verification".

This note documents how the bug was discovered, the fix, and the regression
test that prevents it from coming back.

## Symptom

On-device confidence was systematically lower than desktop simulation
confidence for the same captures, even though the TFLite weights were
identical. Because both errors and correct calls shifted in the same
direction, accuracy on the (already-contaminated) test set looked plausible
and the bug went unnoticed in earlier evaluations.

## Root cause

`python/preprocess.py:88-90` (jakubs-solution) does

```python
images_array = images_array.astype('float32') / 127.5 - 1.0
```

i.e. maps `[0, 255]` -> `[-1.0, 1.0]`. The corresponding training-time INT8
quantization uses `INPUT_SCALE = 0.0078431...f` and `INPUT_ZERO_POINT = 0`,
so the model expects each pixel `v` to satisfy `q = round(v/INPUT_SCALE)` and
fall roughly evenly across the signed-int8 range.

`esp32/main/inference.cpp` (pre-fix) instead did

```cpp
float red = (float)((rgb_pixel >> 16) & 0xFF);
red /= 255.0f;                          // <-- wrong: maps to [0, 1]
input->data.int8[idx] = red / INPUT_SCALE + INPUT_ZERO_POINT;
```

So the firmware shipped pixels in `[0.0, 1.0]` to a quantizer expecting
`[-1.0, 1.0]`. After quantization, every pixel sat in the **non-negative**
half of the int8 range; the negative half was effectively dead.

## Fix

We replaced the inline normalisation with a `mobilenet_v2_preprocess` helper
ported from `origin/main`'s firmware:

```cpp
static inline int8_t mobilenet_v2_preprocess(uint8_t channel) {
    float v = (float)channel / 127.5f - 1.0f;     // <-- [-1, 1]
    int q = (int)lroundf(v / INPUT_SCALE) + INPUT_ZERO_POINT;
    if (q < -128) q = -128;
    if (q >  127) q = 127;
    return (int8_t)q;
}
```

The helper is now used by every call site in `esp32/main/inference.cpp`
(see commit `2c833e6`).

## Regression test

`python/bench/firmware_preprocess_check.py` simulates the firmware
quantization in NumPy on the cleaned originals test set and asserts:

1. The new formula produces **balanced** INT8 values: `mean ~= 0` and roughly
   half the pixels are negative (the *signature* of [-1, 1] preprocessing).
2. The old formula does **not** satisfy 1; it produces `mean ~= +63` with
   ~0% negative pixels.
3. INT8 inference on the firmware-style preprocessed input matches a
   reference desktop float reference within 1 ULP per logit.

The check is invoked by `make bench-firmware-check`. It is also enforced by
`.github/workflows/ci.yml`: a CI step greps for `/255.0f` in
`esp32/main/inference.cpp` and fails the build if the [0, 1] divide is ever
re-introduced.

## Lessons

- *Both halves of the input range matter for INT8 inference.* Halving the
  effective range silently degrades the model without producing obvious
  errors at desktop simulation time.
- *Document preprocessing as a function name, not a one-liner.* The fix is
  easier to audit because `mobilenet_v2_preprocess` is a single helper
  shared by every call site.
- *Encode the rule in CI.* A single-line `grep` in CI is enough to keep this
  class of regression out of the trunk.

## References

- Commit `2c833e6` — firmware preprocess fix
- `python/bench/firmware_preprocess_check.py` — regression test
- `bench/results/firmware_preprocess_check.md` — last passing run
- `.github/workflows/ci.yml` — CI guard
