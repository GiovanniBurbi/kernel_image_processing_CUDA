//
// Created by giova on 27/05/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_CUDA_IMAGE_H
#define KERNEL_IMAGE_PROCESSING_CUDA_IMAGE_H

#include <memory>

/*
 * Struct that represent an image,
 * Data contains the values of the pixels store in an Array of Structures layout
 * */
typedef struct {
    int width;
    int height;
    int channels;
    int pitch;
    float *data;
} Image_t;

#define IMAGE_CHANNELS 3

#define image_getWidth(img) ((img)->width)
#define image_getHeight(img) ((img)->height)
#define image_getChannels(img) ((img)->channels)
#define image_getPitch(img) ((img)->pitch)
#define image_getData(img) ((img)->data)

#define image_setWidth(img, val) (image_getWidth(img) = val)
#define image_setHeight(img, val) (image_getHeight(img) = val)
#define image_setChannels(img, val) (image_getChannels(img) = val)
#define image_setPitch(img, val) (image_getPitch(img) = val)
#define image_setData(img, val) (image_getData(img) = val)

Image_t* new_image(int width, int height, int channels, float *data);
Image_t* new_image(int width, int height, int channels);
Image_t* new_image(int width, int height);
float image_getPixel(Image_t* img, int x, int y, int c);
void image_setPixel(Image_t* img, int x, int y, int c, float val);
void image_delete(Image_t* img);
bool image_is_same(Image_t* a, Image_t* b);


#endif //KERNEL_IMAGE_PROCESSING_CUDA_IMAGE_H
