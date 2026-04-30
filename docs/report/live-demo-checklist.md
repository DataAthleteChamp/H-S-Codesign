# Live demo & on-device measurement checklist

This checklist captures the exact steps to run the live demo and collect the on-device numbers (latency, real-world accuracy, demo recording) once the XIAO ESP32-S3 Sense is plugged in. Everything below uses scripts and firmware already in the repo.

## 0 · Prerequisites

| item | check |
|---|---|
| ESP-IDF ≥ v5.x activated (`. $IDF_PATH/export.sh`) | required |
| `esp32/main/model.c` matches `python/gen/model.c` | run `make deploy` |
| F3 firmware-preprocess regression PASS | run `make bench-firmware-check` |
| Python venv ready with `pyserial` installed | `make venv && venv/bin/pip install pyserial` |
| USB-C cable that supports data (not just power) | required |

## 1 · Build & flash

```bash
make deploy                      # syncs python/gen → esp32/main
cd esp32 && idf.py set-target esp32s3 && idf.py build
idf.py -p /dev/cu.usbmodem* flash monitor
```

The firmware now logs one machine-parseable latency line per inference:

```
I (1234) FaceRec: latency_ms=42.30 capture_ms=2.10 preprocess_ms=5.10 inference_ms=35.10
```

If the build fails on `esp_timer.h`, confirm `esp32/main/CMakeLists.txt` lists `REQUIRES esp_timer` (already added in this repo).

## 2 · On-device latency capture (50–200 frames per scene)

Two equivalent paths:

### A · live serial

```bash
# Captures p50 / p95 / p99 over 200 inferences, plots histogram.
make venv
venv/bin/python python/tools/serial_latency_logger.py \
    --port /dev/cu.usbmodem* --num-samples 200 --baud 115200
# Outputs:
#   bench/results/onboard_latency.csv
#   bench/results/onboard_latency.md
#   bench/results/onboard_latency.png
```

### B · log-file replay (preferred for the demo session)

```bash
# Capture with idf.py monitor while running the device:
cd esp32 && idf.py -p /dev/cu.usbmodem* monitor | tee ../bench/results/idf_monitor.log
# Ctrl-]  to stop, then offline:
cd ..
venv/bin/python python/tools/serial_latency_logger.py \
    --replay bench/results/idf_monitor.log --num-samples 500
```

The script ignores non-matching log lines, so any other `ESP_LOGI` output is safely passed through.

## 3 · Real-world accuracy capture (target: 30 frames per team member + 30 non-team)

Run the scripted webcam proxy *separately* from the on-device demo so the two number sets don't get confused. The on-device version is **not** automated yet — you mark each frame manually below.

### Desktop proxy (already automated)

```bash
venv/bin/python python/realworld_webcam_test.py --label Amine --num-frames 30
venv/bin/python python/realworld_webcam_test.py --label Rifki --num-frames 30
venv/bin/python python/realworld_webcam_test.py --label Jakub --num-frames 30
venv/bin/python python/realworld_webcam_test.py --label none  --num-frames 30
venv/bin/python python/realworld_webcam_test.py --finalize
# bench/results/realworld_webcam.{csv,md} updated.
```

### On-device manual tally

The firmware already prints `>>> <Class> (<conf>%)` per frame. With the device pointed at each subject:

1. Start the firmware with `idf.py monitor`.
2. Have each team member sit in front of the device for 30 stable inferences (≈ 30 s at ≈ 1 fps).
3. Tally per-subject correct vs incorrect predictions in a small worksheet. Suggested table to drop into the report:

| subject | frames | correct | accepted (≥0.77) | accepted-correct |
|---|---:|---:|---:|---:|
| Amine | 30 | _fill_ | _fill_ | _fill_ |
| Rifki | 30 | _fill_ | _fill_ | _fill_ |
| Jakub | 30 | _fill_ | _fill_ | _fill_ |
| Non-team person | 30 | _N/A_ | _fill_ (lower is better) | – |

4. Capture an `idf.py monitor` log while doing it; the latency-logger replay above can re-process the same log so you don't run twice.

## 4 · Recording / screenshots for the report

Pick one (or both):

- **Screen recording** of `idf.py monitor` so the live class predictions and `latency_ms` lines are visible. Use macOS `Cmd+Shift+5` (built-in screen-record).
- **Phone video** of the device + the laptop screen showing predictions for each team member. Trim to ≤ 90 s. Embed a still frame in the report.
- **Whatever live preview** `python/preview_pred.py` produces. That script already has frame-saving on Space + 's' for streaming.

Drop the artefacts under (do NOT commit large media files):

```
docs/figures/demo_screenshot_<member>.png   <-- still images OK to commit
```

For raw videos, host outside the repo (e.g., link in the report, optionally check into a `release/` GitHub release asset).

## 5 · Update results-tables.md after capture

Once the on-device runs are done, fill the "11 · Pending real-world" section in `docs/report/results-tables.md` with the resulting numbers (or just point at `bench/results/onboard_latency.md` and `bench/results/realworld_webcam.md` — both are already linkable artefacts).

## 6 · Sanity checks before showing the demo

- [ ] `python python/realworld_webcam_test.py --finalize` succeeds and reports headline numbers
- [ ] `python python/tools/serial_latency_logger.py --replay …` succeeds and `onboard_latency.png` is non-empty
- [ ] `make eval` still says `acc=0.9833 macro_f1=0.9833 n=60` (deployed model unchanged)
- [ ] `make bench-firmware-check` says PASS
- [ ] Firmware monitor log shows `latency_ms=…` lines (proves the timer instrumentation is live)
- [ ] Confidence threshold in firmware (`best_conf >= 0.9f` in `esp32/main/main.cpp` at the time of writing) matches the value the report claims; if you change it to 0.77 to match the calibration sweep, rebuild and re-flash before the demo
