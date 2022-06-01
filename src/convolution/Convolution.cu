//
// Created by giova on 31/05/2022.
//

#include "Convolution.cuh"
#include "kernel/Kernel.h"

extern __constant__ float MASK[MASK_WIDTH * MASK_WIDTH];

// number of input elements per block
#define w (TILE_WIDTH + MASK_WIDTH - 1)


__global__ void convolutionNaive(const float* __restrict__ data, const float* __restrict__ mask, float* result,
                                 int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                        accum += data[((row + y) * width + col + x) * channels + k] *
                                 mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    }
                }
            }
            result[(row * width + col) * channels + k] = accum;
        }
    }
}

__global__ void convolutionConstantMemory(const float* __restrict__ data, float* result,
                                 int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                        accum += data[((row + y) * width + col + x) * channels + k] * MASK[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    }
                }
            }
            result[(row * width + col) * channels + k] = accum;
        }
    }
}

__global__ void convolutionTiling(const float* __restrict__ data, float* result,
                                          int width, int height, int channels) {
    __shared__ float data_ds[w][w];

    for (int k = 0; k < channels; k++) {
        // First batch loading
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        int src = (srcY * width + srcX) * channels + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            data_ds[destY][destX] = data[src];
        } else {
            data_ds[destY][destX] = 0;
        }

        // Second batch loading
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w;
        destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
        srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
        src = (srcY * width + srcX) * channels + k;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
                data_ds[destY][destX] = data[src];
            } else {
                data_ds[destY][destX] = 0;
            }
        }
        __syncthreads();

        float accum = 0;
        for (int y = 0; y < MASK_WIDTH; y++) {
            for (int x = 0; x < MASK_WIDTH; x++) {
                accum += data_ds[threadIdx.y + y][threadIdx.x + x]
                         * MASK[y * MASK_WIDTH + x];
            }
        }
        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            result[(y * width + x) * channels + k] = accum;
        __syncthreads();
    }
}