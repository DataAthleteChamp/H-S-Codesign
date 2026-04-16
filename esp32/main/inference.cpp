#include <cstdint>
#include <cmath>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "model.h"
#include "camera.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

// Tensor arena size — MobileNetV2 alpha=0.35 at 96×96 needs ~1 MB
#define TENSOR_ARENA_SIZE (1024 * 1024)

static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static uint8_t *tensor_arena = nullptr;
static TfLiteTensor *input = nullptr;
static TfLiteTensor *output = nullptr;
static const char *TAG = "Inference";

bool inference_init()
{
    // Load TFLite model from flash
    model = tflite::GetModel(model_binary);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        ESP_LOGE(TAG, "Model schema mismatch: got %lu, expected %d",
                 (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Allocate tensor arena in PSRAM
    tensor_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena)
    {
        ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM!");
        return false;
    }
    ESP_LOGI(TAG, "Tensor arena allocated: %d bytes in PSRAM", TENSOR_ARENA_SIZE);

    // MobileNetV2 op resolver
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

    // Create interpreter
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

/**
 * Preprocess a 320×240 RGB565 frame into the model's 96×96 INT8 input tensor.
 * Steps: center-crop to 240×240, nearest-neighbor scale to 96×96,
 * convert RGB565→RGB888, normalize to [0,1], quantize to INT8.
 */
void inference_preprocess(const uint8_t *rgb565_frame)
{
    int8_t *dst = input->data.int8;
    const int crop_x = (FRAME_W - FRAME_H) / 2; // 40 pixels from each side
    const float scale = input->params.scale;
    const int32_t zp = input->params.zero_point;

    for (int r = 0; r < IMG_SIZE; r++)
    {
        // Source row in 240×240 crop
        int src_r = r * FRAME_H / IMG_SIZE;

        for (int c = 0; c < IMG_SIZE; c++)
        {
            // Source col (offset by crop)
            int src_c = c * FRAME_H / IMG_SIZE + crop_x;

            // Read RGB565 pixel (2 bytes, big-endian from camera)
            int src_idx = (src_r * FRAME_W + src_c) * 2;
            uint16_t px = ((uint16_t)rgb565_frame[src_idx] << 8) | rgb565_frame[src_idx + 1];

            // Extract RGB components and scale to 0-255
            uint8_t red   = (px >> 11) << 3;
            uint8_t green = ((px >> 5) & 0x3F) << 2;
            uint8_t blue  = (px & 0x1F) << 3;

            // Normalize to [0,1] then quantize to INT8 using runtime params
            int dst_idx = (r * IMG_SIZE + c) * 3;
            dst[dst_idx + 0] = (int8_t)fminf(127.0f, fmaxf(-128.0f,
                roundf((red   / 255.0f) / scale + zp)));
            dst[dst_idx + 1] = (int8_t)fminf(127.0f, fmaxf(-128.0f,
                roundf((green / 255.0f) / scale + zp)));
            dst[dst_idx + 2] = (int8_t)fminf(127.0f, fmaxf(-128.0f,
                roundf((blue  / 255.0f) / scale + zp)));
        }
    }
}

/**
 * Run inference and dequantize the output to float probabilities.
 */
bool inference_predict(float *prediction)
{
    if (interpreter->Invoke() != kTfLiteOk)
    {
        ESP_LOGE(TAG, "Invoke failed!");
        return false;
    }

    for (int i = 0; i < NUM_CLASSES; i++)
    {
        prediction[i] = (static_cast<float>(output->data.int8[i]) - output->params.zero_point)
                         * output->params.scale;
    }

    return true;
}
