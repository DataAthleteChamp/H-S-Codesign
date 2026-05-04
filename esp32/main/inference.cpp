#include <cmath>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "camera.h"
#include "inference.h"
#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Custom Xception is small enough that a 2 MB arena is a reasonable starting
// point and leaves room in PSRAM for a face detector.
#define TENSOR_ARENA_SIZE (2 * 1024 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t *tensor_arena = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG = "Inference";
static bool logged_input_stats = false;
static int output_debug_logs_left = 5;

static inline int8_t quantize_to_int8(float value, float scale, int32_t zero_point)
{
    const float q = roundf(value / scale) + static_cast<float>(zero_point);
    return static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, q)));
}

static inline uint16_t read_rgb565_pixel(const uint8_t *rgb565_frame, int pixel_index)
{
    const int src_idx = pixel_index * 2;
    return (static_cast<uint16_t>(rgb565_frame[src_idx]) << 8)
           | static_cast<uint16_t>(rgb565_frame[src_idx + 1]);
}

static inline void rgb565_to_rgb888(uint16_t px, uint8_t &red, uint8_t &green, uint8_t &blue)
{
    red = static_cast<uint8_t>(((px >> 11) & 0x1F) * 255 / 31);
    green = static_cast<uint8_t>(((px >> 5) & 0x3F) * 255 / 63);
    blue = static_cast<uint8_t>((px & 0x1F) * 255 / 31);
}

static inline void read_rgb888_pixel(const uint8_t *rgb565_frame, int row, int col,
                                     uint8_t &red, uint8_t &green, uint8_t &blue)
{
    const uint16_t px = read_rgb565_pixel(rgb565_frame, row * FRAME_W + col);
    rgb565_to_rgb888(px, red, green, blue);
}

static inline float mobilenet_v2_preprocess(uint8_t pixel)
{
    return (static_cast<float>(pixel) / 127.5f) - 1.0f;
}

static inline float preprocess_pixel(uint8_t pixel, float scale, int32_t zero_point)
{
    (void)scale;
    (void)zero_point;
    return mobilenet_v2_preprocess(pixel);
}

static inline int input_height()
{
    return input->dims->data[1];
}

static inline int input_width()
{
    return input->dims->data[2];
}

bool inference_init()
{
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model schema mismatch: got %lu, expected %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM));
    if (!tensor_arena)
    {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM!");
        return false;
    }
    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes in PSRAM", TENSOR_ARENA_SIZE);

    static tflite::MicroMutableOpResolver<11> resolver;
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddRelu();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Failed to allocate tensors!");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    ESP_LOGI(TAG, "Input: type=%s, shape=[%d,%d,%d,%d]",
             TfLiteTypeGetName(input->type),
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Output: type=%s, shape=[%d,%d]",
             TfLiteTypeGetName(output->type),
             output->dims->data[0], output->dims->data[1]);
    ESP_LOGI(TAG, "Arena used: %zu bytes", interpreter->arena_used_bytes());

    return true;
}

void inference_preprocess(const uint8_t *rgb565_frame)
{
    inference_preprocess_crop(rgb565_frame, nullptr);
}

void inference_preprocess_crop(const uint8_t *rgb565_frame, const CropRect *crop_rect)
{
    if (input->type != kTfLiteInt8)
    {
        ESP_LOGE(TAG, "Expected int8 input tensor, got %s", TfLiteTypeGetName(input->type));
        return;
    }

    int8_t *dst = input->data.int8;
    const float scale = input->params.scale;
    const int32_t zp = input->params.zero_point;

    const int height = input_height();
    const int width = input_width();
    if (!crop_rect || crop_rect->w <= 0 || crop_rect->h <= 0)
    {
        ESP_LOGE(TAG, "Face crop is required before preprocessing.");
        return;
    }

    int crop_x = crop_rect->x;
    int crop_y = crop_rect->y;
    int crop_side = crop_rect->w < crop_rect->h ? crop_rect->w : crop_rect->h;

    if (crop_x < 0)
    {
        crop_x = 0;
    }
    if (crop_y < 0)
    {
        crop_y = 0;
    }
    if (crop_x + crop_side > FRAME_W)
    {
        crop_side = FRAME_W - crop_x;
    }
    if (crop_y + crop_side > FRAME_H)
    {
        crop_side = FRAME_H - crop_y;
    }
    if (crop_side <= 0)
    {
        ESP_LOGE(TAG, "Invalid face crop after bounds clamp.");
        return;
    }

    uint32_t red_sum = 0;
    uint32_t green_sum = 0;
    uint32_t blue_sum = 0;

    for (int r = 0; r < height; r++)
    {
        const float src_y = static_cast<float>(crop_y) +
                            ((static_cast<float>(r) + 0.5f) * crop_side / height) - 0.5f;
        int y0 = static_cast<int>(floorf(src_y));
        float y_lerp = src_y - y0;
        if (y0 < crop_y)
        {
            y0 = crop_y;
            y_lerp = 0.0f;
        }
        const int y1 = y0 + 1 < crop_y + crop_side ? y0 + 1 : y0;

        for (int c = 0; c < width; c++)
        {
            const float src_x = static_cast<float>(crop_x) +
                                ((static_cast<float>(c) + 0.5f) * crop_side / width) - 0.5f;
            int x0 = static_cast<int>(floorf(src_x));
            float x_lerp = src_x - x0;
            if (x0 < crop_x)
            {
                x0 = crop_x;
                x_lerp = 0.0f;
            }
            const int x1 = x0 + 1 < crop_x + crop_side ? x0 + 1 : x0;

            uint8_t r00, g00, b00;
            uint8_t r01, g01, b01;
            uint8_t r10, g10, b10;
            uint8_t r11, g11, b11;
            read_rgb888_pixel(rgb565_frame, y0, x0, r00, g00, b00);
            read_rgb888_pixel(rgb565_frame, y0, x1, r01, g01, b01);
            read_rgb888_pixel(rgb565_frame, y1, x0, r10, g10, b10);
            read_rgb888_pixel(rgb565_frame, y1, x1, r11, g11, b11);

            const float top_r = r00 + (r01 - r00) * x_lerp;
            const float top_g = g00 + (g01 - g00) * x_lerp;
            const float top_b = b00 + (b01 - b00) * x_lerp;
            const float bottom_r = r10 + (r11 - r10) * x_lerp;
            const float bottom_g = g10 + (g11 - g10) * x_lerp;
            const float bottom_b = b10 + (b11 - b10) * x_lerp;

            const uint8_t red = static_cast<uint8_t>(roundf(top_r + (bottom_r - top_r) * y_lerp));
            const uint8_t green = static_cast<uint8_t>(roundf(top_g + (bottom_g - top_g) * y_lerp));
            const uint8_t blue = static_cast<uint8_t>(roundf(top_b + (bottom_b - top_b) * y_lerp));

            const int dst_idx = (r * width + c) * 3;
            dst[dst_idx + 0] = quantize_to_int8(preprocess_pixel(red, scale, zp), scale, zp);
            dst[dst_idx + 1] = quantize_to_int8(preprocess_pixel(green, scale, zp), scale, zp);
            dst[dst_idx + 2] = quantize_to_int8(preprocess_pixel(blue, scale, zp), scale, zp);

            red_sum += red;
            green_sum += green;
            blue_sum += blue;
        }
    }

    if (!logged_input_stats)
    {
        const uint32_t pixels = static_cast<uint32_t>(height * width);
        ESP_LOGI(TAG, "Preprocess: crop=%dx%d at (%d,%d), input_scale=%.6f, input_zp=%ld, mean_rgb=(%lu,%lu,%lu)",
                 crop_side, crop_side, crop_x, crop_y, scale, static_cast<long>(zp),
                 static_cast<unsigned long>(red_sum / pixels),
                 static_cast<unsigned long>(green_sum / pixels),
                 static_cast<unsigned long>(blue_sum / pixels));
        logged_input_stats = true;
    }
}

bool inference_predict(float *prediction)
{
    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Invoke failed!");
        return false;
    }

    if (output->type == kTfLiteInt8)
    {
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point)
                             * output->params.scale;
        }
        if (output_debug_logs_left > 0 && NUM_CLASSES == 3)
        {
            ESP_LOGI(TAG, "Output raw=[%d,%d,%d], scale=%.8f, zp=%ld, prob=[%.6f,%.6f,%.6f], sum=%.6f",
                     output->data.int8[0], output->data.int8[1], output->data.int8[2],
                     output->params.scale, static_cast<long>(output->params.zero_point),
                     prediction[0], prediction[1], prediction[2],
                     prediction[0] + prediction[1] + prediction[2]);
            output_debug_logs_left--;
        }
        return true;
    }

    if (output->type == kTfLiteFloat32)
    {
        for (int i = 0; i < NUM_CLASSES; i++)
        {
            prediction[i] = output->data.f[i];
        }
        return true;
    }

    ESP_LOGE(TAG, "Unsupported output tensor type: %s", TfLiteTypeGetName(output->type));
    return false;
}
