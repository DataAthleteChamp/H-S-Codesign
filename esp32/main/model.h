#ifndef MODEL_H
#define MODEL_H

#define IMG_SIZE 96
#define NUM_CLASSES 3
#define INPUT_SCALE 0.007843137718737125f
#define INPUT_ZERO_POINT -1
#define OUTPUT_SCALE 0.00390625f
#define OUTPUT_ZERO_POINT -128

// Labels: Amine=0, Rifki=1, Jakub=2

extern const unsigned char model_binary[];

#endif
