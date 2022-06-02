//
// Created by giova on 02/06/2022.
//

#include "ImageSoA.h"
#include <iostream>

ImageSoA_t* new_ImageSoA(int width, int height, int channels, float *r, float *g, float *b) {
    ImageSoA_t* img;

    img = (ImageSoA_t*) malloc(sizeof(ImageSoA_t));

    image_setWidth(img, width);
    image_setHeight(img, height);
    image_setChannels(img, channels);

    image_setR(img, r);
    image_setG(img, g);
    image_setB(img, b);
    return img;
}

ImageSoA_t* new_imageSoA(int width, int height, int channels) {
    if (channels != 3) {
        std::cerr << "This data struct works only for image with three channels" << std::endl;
        return NULL;
    }

    auto *r = (float*) malloc(sizeof(float) * width * height);
    auto *g = (float*) malloc(sizeof(float) * width * height);
    auto *b = (float*) malloc(sizeof(float) * width * height);
    return new_ImageSoA(width, height, channels, r, g, b);
}

void image_delete(ImageSoA_t* img) {
    if (img != NULL) {
        if (image_getR(img) != NULL) {
            free(image_getR(img));
        }
        if (image_getG(img) != NULL) {
            free(image_getG(img));
        }
        if (image_getB(img) != NULL) {
            free(image_getB(img));
        }
        free(img);
    }
}


void image_setPixel(ImageSoA_t* img, int x, int y, int c, float val) {
    int width = image_getWidth(img);

    float *r = image_getR(img);
    float *g = image_getG(img);
    float *b = image_getB(img);

    switch (c) {
        case 0: {
            r[y * width + x] = val;
            break;
        }
        case 1: {
            g[y * width + x] = val;
            break;
        }
        case 2: {
            b[y * width + x] = val;
            break;
        }
        default:
            std::cerr << "Wrong channel, it must be 0, 1 or 2" << std::endl;
    }
}

float image_getPixel(ImageSoA_t* img, int x, int y, int c) {
    int width = image_getWidth(img);

    float *r = image_getR(img);
    float *g = image_getG(img);
    float *b = image_getB(img);

    switch (c) {
        case 0: {
            return r[y * width + x];
        }
        case 1: {
            return g[y * width + x];
        }
        case 2: {
            return b[y * width + x];
        }
        default:
            std::cerr << "Wrong channel, it must be 0, 1 or 2" << std::endl;
    }
    return 0;
}