#include <cstdio>
#include <cstdint>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include "camera.h"
#include "model.h"
#include "inference.h"

static const char *TAG = "FaceRec";
static const char *LABELS[] = {"Amine", "Rifki", "Jakub"};

// Frame buffer allocated in PSRAM to avoid wasting internal RAM
static uint8_t *frame_buffer = nullptr;

void setup()
{
    // Allocate frame buffer in PSRAM
    frame_buffer = (uint8_t *)heap_caps_malloc(FRAME_W * FRAME_H * FRAME_C, MALLOC_CAP_SPIRAM);
    if (!frame_buffer)
    {
        ESP_LOGE(TAG, "Failed to allocate frame buffer in PSRAM!");
        abort();
    }

    // Initialize camera
    ESP_LOGI(TAG, "Initializing camera...");
    if (!camera_init())
    {
        ESP_LOGE(TAG, "Camera init failed!");
        abort();
    }

    // Initialize TFLite Micro inference
    ESP_LOGI(TAG, "Initializing inference engine...");
    if (!inference_init())
    {
        ESP_LOGE(TAG, "Inference init failed!");
        abort();
    }

    ESP_LOGI(TAG, "Ready. Starting inference loop.");
}

void loop()
{
    // Capture frame
    if (!camera_capture_frame(frame_buffer))
    {
        ESP_LOGW(TAG, "Frame capture failed, retrying...");
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    // Preprocess: 320×240 RGB565 → 96×96 INT8 RGB888
    inference_preprocess(frame_buffer);

    // Run inference
    float prediction[NUM_CLASSES];
    if (!inference_predict(prediction))
    {
        ESP_LOGE(TAG, "Inference failed!");
        return;
    }

    // Find argmax and check confidence
    int best_class = 0;
    float best_conf = prediction[0];
    for (int i = 1; i < NUM_CLASSES; i++)
    {
        if (prediction[i] > best_conf)
        {
            best_conf = prediction[i];
            best_class = i;
        }
    }

    // Apply rejection threshold
    if (best_conf >= 0.9f)
    {
        ESP_LOGI(TAG, ">>> %s (%.1f%%)", LABELS[best_class], best_conf * 100.0f);
    }
    else
    {
        ESP_LOGI(TAG, ">>> Unknown (best: %s %.1f%%)", LABELS[best_class], best_conf * 100.0f);
    }

    // Print all class probabilities
    ESP_LOGI(TAG, "    Amine=%.2f  Rifki=%.2f  Jakub=%.2f",
             prediction[0], prediction[1], prediction[2]);

    vTaskDelay(pdMS_TO_TICKS(1000));
}

extern "C" void app_main()
{
    setup();
    while (true)
    {
        loop();
    }
}
