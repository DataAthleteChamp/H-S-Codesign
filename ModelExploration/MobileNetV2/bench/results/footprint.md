# Embedded footprint

Numbers below describe the deployed model only and are derived from the artefacts in `python/gen/` plus the firmware constants in `esp32/main/inference.cpp`. No new measurements are performed.

## Deployed model (`python/gen/model.tflite`)

| metric | value |
|---|---:|
| file size (flash budget) | 662,056 B (646.54 KiB) |
| total parameter count | 430,184 |
| INT8 weights | 423,107 |
| INT32 biases | 7,077 |
| weight + bias bytes | 451,415 B (440.83 KiB) |
| input shape / dtype | (1, 96, 96, 3) / int8 |
| input quant scale / zp | 0.00784314 / 0 |
| output shape / dtype | (1, 3) / int8 |
| output quant scale / zp | 0.00390625 / -128 |

## Baseline reference (`python/gen/baseline_model.tflite`)

| metric | challenger | baseline | Δ |
|---|---:|---:|---:|
| file size | 662,056 B | 662,056 B | +0 B |
| parameters | 430,184 | 430,184 | +0 |

## Firmware-side budget

| component | value | source |
|---|---:|---|
| `TENSOR_ARENA_SIZE` (declared) | 1,048,576 B (1024.00 KiB) | `esp32/main/inference.cpp` |
| arena allocation region | PSRAM (`heap_caps_malloc(... MALLOC_CAP_SPIRAM)`) | `esp32/main/inference.cpp` |
| arena_used_bytes() (runtime) | _measured on device, see ESP_LOGI "Arena used: ... bytes"_ | live serial log |

## Board capability ceilings (XIAO ESP32-S3 Sense)

| resource | total | used by model |
|---|---:|---:|
| flash | 8.00 MiB | 646.54 KiB (7.89%) |
| PSRAM | 8.00 MiB | arena reserved 1024.00 KiB (12.50%) |
| internal SRAM | 512.00 KiB | _firmware code + stacks; not measured here_ |

## Notes

- Weight bytes are the on-flash representation: INT8 weights are 1 byte each and INT32 biases are 4 bytes each. Activation tensors are not parameters and do not contribute to flash; they live in the runtime arena.
- `TENSOR_ARENA_SIZE` is a worst-case allocation; the actual arena_used_bytes() reported by the runtime can be smaller. Read the on-device serial log to confirm and tighten the bound.
- The arena lives in PSRAM, so the 512 KiB internal SRAM ceiling is not the binding constraint for inference memory.
