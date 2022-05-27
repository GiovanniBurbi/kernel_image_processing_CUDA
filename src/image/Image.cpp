//
// Created by giova on 27/05/2022.
//

#include "Image.h"
#include "utils/Utils.h"
#include <iostream>
#include <cassert>

Image_t* new_image(int width, int height, int channels, float *data) {
    Image_t* img;

    img = (Image_t*) malloc(sizeof(Image_t));

    image_setWidth(img, width);
    image_setHeight(img, height);
    image_setChannels(img, channels);
    image_setPitch(img, width * channels);

    image_setData(img, data);
    return img;
}

Image_t* new_image(int width, int height, int channels) {
    float *data = (float*) malloc(sizeof(float) * width * height * channels);
    return new_image(width, height, channels, data);
}

Image_t* new_image(int width, int height) {
    return new_image(width, height, IMAGE_CHANNELS);
}

void image_delete(Image_t* img) {
    if (img != NULL) {
        if (image_getData(img) != NULL) {
            free(image_getData(img));
        }
        free(img);
    }
}

void image_setPixel(Image_t* img, int x, int y, int c, float val) {
    float *data = image_getData(img);
    int channels = image_getChannels(img);
    int pitch = image_getPitch(img);

    data[y * pitch + x * channels + c] = val;

    return;
}

float image_getPixel(Image_t* img, int x, int y, int c) {
    float *data = image_getData(img);
    int channels = image_getChannels(img);
    int pitch = image_getPitch(img);

    return data[y * pitch + x * channels + c];
}

bool image_is_same(Image_t* a, Image_t* b) {
    if (a == NULL || b == NULL) {
        std::cerr << "Comparing null images." << std::endl;
        return false;
    } else if (a == b) {
        return true;
    } else if (image_getWidth(a) != image_getWidth(b)) {
        std::cerr << "Image widths do not match." << std::endl;
        return false;
    } else if (image_getHeight(a) != image_getHeight(b)) {
        std::cerr << "Image heights do not match." << std::endl;
        return false;
    } else if (image_getChannels(a) != image_getChannels(b)) {
        std::cerr << "Image channels do not match." << std::endl;
        return false;
    } else {
        float *aData, *bData;
        int width, height, channels;
        int ii, jj, kk;

        aData = image_getData(a);
        bData = image_getData(b);

        assert(aData != NULL);
        assert(bData != NULL);

        width = image_getWidth(a);
        height = image_getHeight(a);
        channels = image_getChannels(a);

        for (ii = 0; ii < height; ii++) {
            for (jj = 0; jj < width; jj++) {
                for (kk = 0; kk < channels; kk++) {
                    float x, y;
                    if (channels <= 3) {
                        x = clamp(*aData++, 0, 1);
                        y = clamp(*bData++, 0, 1);
                    } else {
                        x = *aData++;
                        y = *bData++;
                    }
                    if (almostUnequalFloat(x, y)) {
                        std::cerr
                                << "Image pixels do not match at position ( row = "
                                << ii << ", col = " << jj << ", channel = "
                                << kk << ") expecting a value of " << y
                                << " but got a value of " << x << std::endl;

                        return false;
                    }
                }
            }
        }
        return true;
    }
}