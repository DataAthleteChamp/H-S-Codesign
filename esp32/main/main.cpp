#include <cstdio>
#include <cstdint>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "nvs_flash.h"

#include "camera.h"
#include "model.h"
#include "inference.h"

static constexpr size_t CHUNK_SIZE = 256;
static const char *TAG = "FaceRec";
static const char *FRAME_PREAMBLE = "\n===FRAME===\n";
static const char *PREDICTION_PREAMBLE = "\n===PRED===\n";
static const char *LABELS[] = {"Amine", "Rifki", "Jakub"};
static constexpr float PREDICTION_THRESHOLD = 0.9f;

static uint8_t *frame_buffer = nullptr;
static uint8_t *rgb888_buffer = nullptr;
static bool stream_enabled = false;

static bool ensure_rgb888_buffer()
{
    if (rgb888_buffer)
    {
        return true;
    }

    rgb888_buffer = static_cast<uint8_t *>(heap_caps_malloc(FRAME_W * FRAME_H * 3, MALLOC_CAP_SPIRAM));
    if (!rgb888_buffer)
    {
        ESP_LOGE(TAG, "Failed to allocate RGB888 streaming buffer in PSRAM!");
        return false;
    }
    return true;
}

static int best_prediction_index(const float *prediction)
{
    int best_class = 0;
    for (int i = 1; i < NUM_CLASSES; i++)
    {
        if (prediction[i] > prediction[best_class])
        {
            best_class = i;
        }
    }
    return best_class;
}

static void rgb565_frame_to_rgb888(const uint8_t *rgb565_frame, uint8_t *rgb888_frame)
{
    for (int pixel = 0; pixel < FRAME_W * FRAME_H; ++pixel)
    {
        const int src_idx = pixel * 2;
        const int dst_idx = pixel * 3;

        const uint8_t byte1 = rgb565_frame[src_idx];
        const uint8_t byte2 = rgb565_frame[src_idx + 1];

        rgb888_frame[dst_idx + 0] = byte1 & 0xF8;
        rgb888_frame[dst_idx + 1] = static_cast<uint8_t>(((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3));
        rgb888_frame[dst_idx + 2] = static_cast<uint8_t>((byte2 & 0x1F) << 3);
    }
}

static void maybe_handle_serial_command()
{
    char c = 0;
    const int r = usb_serial_jtag_read_bytes(&c, 1, 0);
    if (r == 1 && c == 'S')
    {
        if (!stream_enabled && !ensure_rgb888_buffer())
        {
            return;
        }
        stream_enabled = !stream_enabled;
        ESP_LOGI(TAG, "RGB888 frame streaming %s.", stream_enabled ? "enabled" : "disabled");
    }
}

static void maybe_stream_rgb888_frame(const float *prediction, int best_class, float best_conf)
{
    if (!stream_enabled)
    {
        return;
    }

    char prediction_line[128];
    const int prediction_len = snprintf(
        prediction_line,
        sizeof(prediction_line),
        "%s,%d,%.6f,%.6f,%.6f,%.6f\n",
        LABELS[best_class],
        best_class,
        best_conf,
        prediction[0],
        prediction[1],
        prediction[2]);

    usb_serial_jtag_write_bytes(PREDICTION_PREAMBLE, strlen(PREDICTION_PREAMBLE), pdMS_TO_TICKS(1000));
    usb_serial_jtag_write_bytes(prediction_line, prediction_len, pdMS_TO_TICKS(1000));

    rgb565_frame_to_rgb888(frame_buffer, rgb888_buffer);
    usb_serial_jtag_write_bytes(FRAME_PREAMBLE, strlen(FRAME_PREAMBLE), pdMS_TO_TICKS(1000));

    const size_t frame_size = FRAME_W * FRAME_H * 3;
    for (size_t offset = 0; offset < frame_size;)
    {
        const size_t to_write = offset + CHUNK_SIZE < frame_size ? CHUNK_SIZE : frame_size - offset;
        const int written = usb_serial_jtag_write_bytes(rgb888_buffer + offset, to_write, pdMS_TO_TICKS(1000));
        if (written < static_cast<int>(to_write))
        {
            vTaskDelay(1);
        }
        if (written > 0)
        {
            offset += written;
        }
    }
}

void setup()
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    frame_buffer = static_cast<uint8_t *>(heap_caps_malloc(FRAME_W * FRAME_H * FRAME_C, MALLOC_CAP_SPIRAM));
    if (!frame_buffer)
    {
        ESP_LOGE(TAG, "Failed to allocate frame buffer in PSRAM!");
        abort();
    }

    if (!camera_init())
    {
        ESP_LOGE(TAG, "Camera init failed!");
        abort();
    }

    ESP_LOGI(TAG, "Initializing inference engine...");
    if (!inference_init())
    {
        ESP_LOGE(TAG, "Inference init failed!");
        abort();
    }

    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 512,
        .rx_buffer_size = 512,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    printf("Send 'S' to toggle RGB888 frame streaming. Inference runs regardless.\n");
    ESP_LOGI(TAG, "Ready. Starting inference loop.");
}

void loop()
{
    maybe_handle_serial_command();

    if (!camera_capture_frame(frame_buffer))
    {
        ESP_LOGW(TAG, "Frame capture failed, retrying...");
        vTaskDelay(pdMS_TO_TICKS(100));
        return;
    }

    inference_preprocess(frame_buffer);

    float prediction[NUM_CLASSES];
    if (!inference_predict(prediction))
    {
        ESP_LOGE(TAG, "Inference failed!");
        return;
    }

    const int best_class = best_prediction_index(prediction);
    const float best_conf = prediction[best_class];

    if (best_conf >= PREDICTION_THRESHOLD)
    {
        ESP_LOGI(TAG, ">>> %s (%.1f%%)", LABELS[best_class], best_conf * 100.0f);
    }
    else
    {
        ESP_LOGI(TAG, ">>> Unknown (best: %s %.1f%%)", LABELS[best_class], best_conf * 100.0f);
    }

    ESP_LOGI(TAG, "    Amine=%.2f  Rifki=%.2f  Jakub=%.2f",
             prediction[0], prediction[1], prediction[2]);

    maybe_stream_rgb888_frame(prediction, best_class, best_conf);

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
