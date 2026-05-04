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

#if __has_include("human_face_detect.hpp")
#include "human_face_detect.hpp"
#include "dl_image_define.hpp"
#define HAVE_HUMAN_FACE_DETECT 1
#else
#define HAVE_HUMAN_FACE_DETECT 0
#endif

static constexpr size_t CHUNK_SIZE = 256;
static const char *TAG = "FaceRec";
static const char *FRAME_PREAMBLE = "\n===FRAME===\n";
static const char *PREDICTION_PREAMBLE = "\n===PRED===\n";
static const char *LABELS[] = {"Amine", "Rifki", "Jakub"};
static constexpr float PREDICTION_THRESHOLD = 0.8f;
static constexpr float FACE_DETECT_MIN_SCORE = 0.50f;
static constexpr float FACE_CROP_MARGIN = 0.20f;

struct DetectionResult
{
    bool found;
    float score;
    CropRect crop;
    int x1;
    int y1;
    int x2;
    int y2;
};

static uint8_t *frame_buffer = nullptr;
static uint8_t *rgb888_buffer = nullptr;
static bool stream_enabled = false;

#if HAVE_HUMAN_FACE_DETECT
static HumanFaceDetect *face_detector = nullptr;
#endif

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
    if (r != 1)
    {
        return;
    }

    if (c == 'S')
    {
        if (!stream_enabled && !ensure_rgb888_buffer())
        {
            return;
        }
        stream_enabled = !stream_enabled;
        ESP_LOGI(TAG, "RGB888 frame streaming %s.", stream_enabled ? "enabled" : "disabled");
    }
    else if (c == '1')
    {
        if (!ensure_rgb888_buffer())
        {
            return;
        }
        stream_enabled = true;
        ESP_LOGI(TAG, "RGB888 frame streaming enabled.");
    }
    else if (c == '0')
    {
        stream_enabled = false;
        ESP_LOGI(TAG, "RGB888 frame streaming disabled.");
    }
}

static void maybe_stream_rgb888_frame(const float *prediction,
                                      int best_class,
                                      float best_conf,
                                      const DetectionResult *detection)
{
    if (!stream_enabled)
    {
        return;
    }

    char prediction_line[128];
    int prediction_len = 0;
    if (detection && detection->found)
    {
        if (!prediction || best_class < 0)
        {
            return;
        }
        prediction_len = snprintf(
            prediction_line,
            sizeof(prediction_line),
            "%s,%d,%.6f,%.6f,%.6f,%.6f,FACE,%.6f,%d,%d,%d,%d,%d,%d,%d,%d\n",
            LABELS[best_class],
            best_class,
            best_conf,
            prediction[0],
            prediction[1],
            prediction[2],
            detection->score,
            detection->x1,
            detection->y1,
            detection->x2,
            detection->y2,
            detection->crop.x,
            detection->crop.y,
            detection->crop.w,
            detection->crop.h);
    }
    else
    {
        prediction_len = snprintf(
            prediction_line,
            sizeof(prediction_line),
            "NO_FACE,-1,0.000000,0.000000,0.000000,0.000000,NONE,0.000000,0,0,0,0,0,0,0,0\n");
    }

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

static CropRect make_square_crop(int x1, int y1, int x2, int y2)
{
    int side = (x2 - x1) > (y2 - y1) ? (x2 - x1) : (y2 - y1);
    side = static_cast<int>(side * (1.0f + FACE_CROP_MARGIN * 2.0f));
    if (side < 1)
    {
        side = 1;
    }
    if (side > FRAME_W)
    {
        side = FRAME_W;
    }
    if (side > FRAME_H)
    {
        side = FRAME_H;
    }

    int crop_x = ((x1 + x2) / 2) - side / 2;
    int crop_y = ((y1 + y2) / 2) - side / 2;

    if (crop_x < 0)
    {
        crop_x = 0;
    }
    if (crop_y < 0)
    {
        crop_y = 0;
    }
    if (crop_x + side > FRAME_W)
    {
        crop_x = FRAME_W - side;
    }
    if (crop_y + side > FRAME_H)
    {
        crop_y = FRAME_H - side;
    }

    return CropRect{crop_x, crop_y, side, side};
}

static bool detect_face_crop(const uint8_t *rgb565_frame, DetectionResult *result)
{
#if HAVE_HUMAN_FACE_DETECT
    if (!face_detector)
    {
        return false;
    }

    dl::image::img_t img = {};
    img.data = const_cast<uint8_t *>(rgb565_frame);
    img.width = FRAME_W;
    img.height = FRAME_H;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565;

    auto &results = face_detector->run(img);
    float best_score = FACE_DETECT_MIN_SCORE;
    bool found = false;

    for (const auto &res : results)
    {
        if (res.score < best_score)
        {
            continue;
        }
        best_score = res.score;
        result->found = true;
        result->score = res.score;
        result->x1 = res.box[0];
        result->y1 = res.box[1];
        result->x2 = res.box[2];
        result->y2 = res.box[3];
        result->crop = make_square_crop(res.box[0], res.box[1], res.box[2], res.box[3]);
        found = true;
    }

    return found;
#else
    (void)rgb565_frame;
    (void)result;
    return false;
#endif
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

#if HAVE_HUMAN_FACE_DETECT
    face_detector = new HumanFaceDetect();
    if (!face_detector)
    {
        ESP_LOGE(TAG, "Face detector init failed!");
        abort();
    }
    ESP_LOGI(TAG, "Human face detector initialized.");
#else
    ESP_LOGE(TAG, "Human face detector component not installed; face-crop inference is required.");
    abort();
#endif

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

    DetectionResult detection = {};
#if HAVE_HUMAN_FACE_DETECT
    if (detect_face_crop(frame_buffer, &detection))
    {
        inference_preprocess_crop(frame_buffer, &detection.crop);
    }
    else
    {
        ESP_LOGI(TAG, ">>> No face detected");
        maybe_stream_rgb888_frame(nullptr, -1, 0.0f, &detection);
        vTaskDelay(pdMS_TO_TICKS(200));
        return;
    }
#else
    ESP_LOGE(TAG, "Human face detector component not installed; skipping inference.");
    vTaskDelay(pdMS_TO_TICKS(200));
    return;
#endif

    float prediction[NUM_CLASSES];
    if (!inference_predict(prediction))
    {
        ESP_LOGE(TAG, "Inference failed!");
        vTaskDelay(pdMS_TO_TICKS(100));
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

    ESP_LOGI(TAG, "    Amine=%.6f  Rifki=%.6f  Jakub=%.6f",
             prediction[0], prediction[1], prediction[2]);

    maybe_stream_rgb888_frame(prediction, best_class, best_conf, &detection);

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
