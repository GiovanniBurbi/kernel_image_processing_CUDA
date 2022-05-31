//
// Created by giova on 31/05/2022.
//

#include "Convolution.cuh"
#include "kernel/Kernel.h"

extern __constant__ float MASK[MASK_WIDTH * MASK_WIDTH];

#define PIXEL_LOST MASK_RADIUS * 2

__global__ void convolutionNaive(const float* data, const float* mask, float* result,
                                 int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < width - PIXEL_LOST && row < height - PIXEL_LOST) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    accum += data[((row + y) * width + col + x) * channels + k] * mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                }
            }
            result[(row * (width - 2) + col) * channels + k] = accum;
        }
    }
}

__global__ void convolutionNaive3DThreadsCoverage(const float* data, const float* mask, float* result,
                                                  int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < width - PIXEL_LOST && row < height - PIXEL_LOST) {
        float accum = 0;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                accum += data[((row + y) * width + col + x) * channels + threadIdx.z] * mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
            }
        }
        result[(row * (width - 2) + col) * channels + threadIdx.z] = accum;
    }
}

__global__ void convolutionConstantMemory(const float* data, float* result,
                                 int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < width - PIXEL_LOST && row < height - PIXEL_LOST) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    accum += data[((row + y) * width + col + x) * channels + k] * MASK[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                }
            }
            result[(row * (width - 2) + col) * channels + k] = accum;
        }
    }
}