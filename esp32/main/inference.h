#pragma once

#include <cstdint>

struct CropRect
{
    int x;
    int y;
    int w;
    int h;
};

bool inference_init();
void inference_preprocess(const uint8_t *rgb565_frame);
void inference_preprocess_crop(const uint8_t *rgb565_frame, const CropRect *crop_rect);
bool inference_predict(float *prediction);
