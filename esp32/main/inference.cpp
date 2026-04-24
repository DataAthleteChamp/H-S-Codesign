#include <cmath>
#include <cstdint>

#include "esp_heap_caps.h"
#include "esp_log.h"

#include "camera.h"
#include "model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define TENSOR_ARENA_SIZE (1024 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t *tensor_arena = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG = "Inference";

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

static inline float mobilenet_v2_preprocess(uint8_t pixel)
{
    return (static_cast<float>(pixel) / 127.5f) - 1.0f;
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

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMean();
    resolver.AddPad();
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

    for (int r = 0; r < height; r++)
    {
        const int src_r = r * FRAME_H / height;

        for (int c = 0; c < width; c++)
        {
            const int src_c = c * FRAME_W / width;
            const uint16_t px = read_rgb565_pixel(rgb565_frame, src_r * FRAME_W + src_c);

            uint8_t red;
            uint8_t green;
            uint8_t blue;
            rgb565_to_rgb888(px, red, green, blue);

            const int dst_idx = (r * width + c) * 3;
            dst[dst_idx + 0] = quantize_to_int8(mobilenet_v2_preprocess(red), scale, zp);
            dst[dst_idx + 1] = quantize_to_int8(mobilenet_v2_preprocess(green), scale, zp);
            dst[dst_idx + 2] = quantize_to_int8(mobilenet_v2_preprocess(blue), scale, zp);
        }
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
