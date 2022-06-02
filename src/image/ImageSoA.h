//
// Created by giova on 02/06/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_CUDA_IMAGESOA_H
#define KERNEL_IMAGE_PROCESSING_CUDA_IMAGESOA_H

#include <memory>

typedef struct {
    int width;
    int height;
    int channels;
    float *r;
    float *g;
    float *b;
} ImageSoA_t;

#define IMAGE_CHANNELS 3

#define image_getWidth(img) ((img)->width)
#define image_getHeight(img) ((img)->height)
#define image_getChannels(img) ((img)->channels)
#define image_getR(img) ((img)->r)
#define image_getG(img) ((img)->g)
#define image_getB(img) ((img)->b)

#define image_setWidth(img, val) (image_getWidth(img) = val)
#define image_setHeight(img, val) (image_getHeight(img) = val)
#define image_setChannels(img, val) (image_getChannels(img) = val)
#define image_setR(img, val) (image_getR(img) = val)
#define image_setG(img, val) (image_getG(img) = val)
#define image_setB(img, val) (image_getB(img) = val)

ImageSoA_t* new_imageSoA(int width, int height, int channels, float *data);
ImageSoA_t* new_imageSoA(int width, int height, int channels);
float image_getPixel(ImageSoA_t* img, int x, int y, int c);
void image_setPixel(ImageSoA_t* img, int x, int y, int c, float val);
void image_delete(ImageSoA_t* img);

#endif //KERNEL_IMAGE_PROCESSING_CUDA_IMAGESOA_H
