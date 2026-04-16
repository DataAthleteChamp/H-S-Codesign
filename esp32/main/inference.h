#pragma once

#include <cstdint>

bool inference_init();
void inference_preprocess(const uint8_t *rgb565_frame);
bool inference_predict(float *prediction);
