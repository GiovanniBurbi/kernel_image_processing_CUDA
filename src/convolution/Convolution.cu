//
// Created by giova on 31/05/2022.
//

#include "Convolution.cuh"
#include "../kernel/Kernel.h"

extern __constant__ float MASK[MASK_WIDTH * MASK_WIDTH];

// number of input elements per block
#define w (TILE_WIDTH + MASK_WIDTH - 1)

#define PIXEL_LOST (MASK_RADIUS * 2)


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

__global__ void convolutionNaiveSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                    const float* __restrict__ mask, float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB,
                                    int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accumR = 0;
        float accumG = 0;
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
            resultR[row * width + col] = accumR;
            resultG[row * width + col] = accumG;
            resultB[row * width + col] = accumB;
    }
}

__global__ void convolutionNaiveNoPadding(const float* __restrict__ data, const float* __restrict__ mask, float* result,
                                          int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    accum += data[((row + y) * width + col + x) * channels + k] *
                             mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                }
            }
            result[(row * (width - PIXEL_LOST) + col) * channels + k] = accum;
        }
    }
}

__global__ void convolutionNaiveNoPaddingSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                    const float* __restrict__ mask, float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB,
                                    int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accumR = 0;
        float accumG = 0;
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultR[row * (width - PIXEL_LOST) + col] = accumR;
        resultG[row * (width - PIXEL_LOST) + col] = accumG;
        resultB[row * (width - PIXEL_LOST) + col] = accumB;
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

__global__ void convolutionConstantMemoryNoPadding(const float* __restrict__ data, float* result,
                                                   int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accum;

        for (int k = 0; k < channels; k++){
            accum = 0;
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    accum += data[((row + y) * width + col + x) * channels + k] * MASK[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                }
            }
            result[(row * (width - PIXEL_LOST) + col) * channels + k] = accum;
        }
    }
}

__global__ void convolutionConstantMemorySoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                    float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accumR = 0;
        float accumG = 0;
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = MASK[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultR[row * width + col] = accumR;
        resultG[row * width + col] = accumG;
        resultB[row * width + col] = accumB;
    }
}

__global__ void convolutionConstantMemoryNoPaddingSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                             float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accumR = 0;
        float accumG = 0;
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = MASK[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultR[row * (width - PIXEL_LOST) + col] = accumR;
        resultG[row * (width - PIXEL_LOST) + col] = accumG;
        resultB[row * (width - PIXEL_LOST) + col] = accumB;
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
                accum += data_ds[threadIdx.y + y][threadIdx.x + x] * MASK[y * MASK_WIDTH + x];
            }
        }
        int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            result[(y * width + x) * channels + k] = accum;

        __syncthreads();
    }
}

__global__ void convolutionTilingNoPadding(const float* __restrict__ data, float* result,
                                           int width, int height, int channels) {
    __shared__ float data_ds[w][w];

    for (int k = 0; k < channels; k++) {
        // First batch loading
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
        int destY = dest / w;
        int destX = dest % w;
        int srcY = blockIdx.y * TILE_WIDTH + destY;
        int srcX = blockIdx.x * TILE_WIDTH + destX;
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
        srcY = blockIdx.y * TILE_WIDTH + destY;
        srcX = blockIdx.x * TILE_WIDTH + destX;
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
                accum += data_ds[threadIdx.y + y][threadIdx.x + x] * MASK[y * MASK_WIDTH + x];
            }
        }
        int y = blockIdx.y * TILE_WIDTH + threadIdx.y + MASK_RADIUS;
        int x = blockIdx.x * TILE_WIDTH + threadIdx.x + MASK_RADIUS;
        if (y < (height - PIXEL_LOST) && x < (width - PIXEL_LOST))
            result[(y * (width - PIXEL_LOST) + x) * channels + k] = accum;
        __syncthreads();
    }
}

__global__ void convolutionTilingSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                     float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height){
    __shared__ float dataR_ds[w][w];
    __shared__ float dataG_ds[w][w];
    __shared__ float dataB_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataR_ds[destY][destX] = dataR[src];
        dataG_ds[destY][destX] = dataG[src];
        dataB_ds[destY][destX] = dataB[src];
    } else {
        dataR_ds[destY][destX] = 0;
        dataG_ds[destY][destX] = 0;
        dataB_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataR_ds[destY][destX] = dataR[src];
            dataG_ds[destY][destX] = dataG[src];
            dataB_ds[destY][destX] = dataB[src];
        } else {
            dataR_ds[destY][destX] = 0;
            dataG_ds[destY][destX] = 0;
            dataB_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumR = 0;
    float accumG = 0;
    float accumB = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumR += dataR_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
            accumG += dataG_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
            accumB += dataB_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width) {
        resultR[(y * width + x)] = accumR;
        resultG[(y * width + x)] = accumG;
        resultB[(y * width + x)] = accumB;
    }

    __syncthreads();
}

__global__ void convolutionTilingNoPaddingSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                     float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height){
    __shared__ float dataR_ds[w][w];
    __shared__ float dataG_ds[w][w];
    __shared__ float dataB_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY;
    int srcX = blockIdx.x * TILE_WIDTH + destX;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataR_ds[destY][destX] = dataR[src];
        dataG_ds[destY][destX] = dataG[src];
        dataB_ds[destY][destX] = dataB[src];
    } else {
        dataR_ds[destY][destX] = 0;
        dataG_ds[destY][destX] = 0;
        dataB_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY;
    srcX = blockIdx.x * TILE_WIDTH + destX;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataR_ds[destY][destX] = dataR[src];
            dataG_ds[destY][destX] = dataG[src];
            dataB_ds[destY][destX] = dataB[src];
        } else {
            dataR_ds[destY][destX] = 0;
            dataG_ds[destY][destX] = 0;
            dataB_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumR = 0;
    float accumG = 0;
    float accumB = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumR += dataR_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
            accumG += dataG_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
            accumB += dataB_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y + MASK_RADIUS;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x + MASK_RADIUS;
    if (y < height - PIXEL_LOST && x < width - PIXEL_LOST) {
        resultR[(y * (width - PIXEL_LOST) + x)] = accumR;
        resultG[(y * (width - PIXEL_LOST) + x)] = accumG;
        resultB[(y * (width - PIXEL_LOST) + x)] = accumB;
    }

    __syncthreads();
}

__global__ void convolutionNaiveSoAChannelR(const float* __restrict__ dataR, const float* __restrict__ mask, float* __restrict__ resultR,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accumR = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultR[row * width + col] = accumR;
    }
}

__global__ void convolutionNaiveSoAChannelG(const float* __restrict__ dataG, const float* __restrict__ mask, float* __restrict__ resultG,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accumG = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultG[row * width + col] = accumG;
    }
}

__global__ void convolutionNaiveSoAChannelB(const float* __restrict__ dataB, const float* __restrict__ mask, float* __restrict__ resultB,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultB[row * width + col] = accumB;
    }
}

__global__ void convolutionNaiveNoPaddingSoAChannelR(const float* __restrict__ dataR, const float* __restrict__ mask, float* __restrict__ resultR,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accumR = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumR += dataR[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultR[row * (width - PIXEL_LOST) + col] = accumR;
    }
}


__global__ void convolutionNaiveNoPaddingSoAChannelG(const float* __restrict__ dataG, const float* __restrict__ mask, float* __restrict__ resultG,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accumG = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumG += dataG[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultG[row * (width - PIXEL_LOST) + col] = accumG;
    }
}
__global__ void convolutionNaiveNoPaddingSoAChannelB(const float* __restrict__ dataB, const float* __restrict__ mask, float* __restrict__ resultB,
                                            int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x + MASK_RADIUS;
    int row = blockIdx.y * blockDim.y + threadIdx.y + MASK_RADIUS;

    if (col < (width - PIXEL_LOST) && row < (height - PIXEL_LOST)) {
        float accumB = 0;
        float maskValue;

        for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
            for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                if((row + y) > -1 && (row + y) < height && (col + x) > -1 && (col + x) < width) {
                    maskValue = mask[(y + MASK_RADIUS) * MASK_WIDTH + x + MASK_RADIUS];
                    accumB += dataB[(row + y) * width + col + x] * maskValue;
                }
            }
        }
        resultB[row * (width - PIXEL_LOST) + col] = accumB;
    }
}

__global__ void convolutionTilingSoAChannelR(const float* __restrict__ dataR, float* __restrict__ resultR, int width, int height){
    __shared__ float dataR_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataR_ds[destY][destX] = dataR[src];
    } else {
        dataR_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataR_ds[destY][destX] = dataR[src];
        } else {
            dataR_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumR = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumR += dataR_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width) {
        resultR[(y * width + x)] = accumR;
    }

    __syncthreads();
}

__global__ void convolutionTilingSoAChannelG(const float* __restrict__ dataG, float* __restrict__ resultG, int width, int height){
    __shared__ float dataG_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataG_ds[destY][destX] = dataG[src];
    } else {
        dataG_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataG_ds[destY][destX] = dataG[src];
        } else {
            dataG_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumG = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumG += dataG_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width) {
        resultG[(y * width + x)] = accumG;
    }

    __syncthreads();
}

__global__ void convolutionTilingSoAChannelB(const float* __restrict__ dataB, float* __restrict__ resultB, int width, int height){
    __shared__ float dataB_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    int srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataB_ds[destY][destX] = dataB[src];
    } else {
        dataB_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY - MASK_RADIUS;
    srcX = blockIdx.x * TILE_WIDTH + destX - MASK_RADIUS;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataB_ds[destY][destX] = dataB[src];
        } else {
            dataB_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumB = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumB += dataB_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (y < height && x < width) {
        resultB[(y * width + x)] = accumB;
    }

    __syncthreads();
}

__global__ void convolutionTilingNoPaddingSoAChannelR(const float* __restrict__ dataR, float* __restrict__ resultR, int width, int height){
    __shared__ float dataR_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY;
    int srcX = blockIdx.x * TILE_WIDTH + destX;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataR_ds[destY][destX] = dataR[src];
    } else {
        dataR_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY;
    srcX = blockIdx.x * TILE_WIDTH + destX;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataR_ds[destY][destX] = dataR[src];
        } else {
            dataR_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumR = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumR += dataR_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y + MASK_RADIUS;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x + MASK_RADIUS;
    if (y < height - PIXEL_LOST && x < width - PIXEL_LOST) {
        resultR[(y * (width - PIXEL_LOST) + x)] = accumR;
    }

//    __syncthreads();
}

__global__ void convolutionTilingNoPaddingSoAChannelG(const float* __restrict__ dataG, float* __restrict__ resultG, int width, int height){
    __shared__ float dataG_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY;
    int srcX = blockIdx.x * TILE_WIDTH + destX;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataG_ds[destY][destX] = dataG[src];
    } else {
        dataG_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY;
    srcX = blockIdx.x * TILE_WIDTH + destX;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataG_ds[destY][destX] = dataG[src];
        } else {
            dataG_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumG = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumG += dataG_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y + MASK_RADIUS;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x + MASK_RADIUS;
    if (y < height - PIXEL_LOST && x < width - PIXEL_LOST) {
        resultG[(y * (width - PIXEL_LOST) + x)] = accumG;
    }

//    __syncthreads();
}

__global__ void convolutionTilingNoPaddingSoAChannelB(const float* __restrict__ dataB, float* __restrict__ resultB, int width, int height){
    __shared__ float dataB_ds[w][w];

    // First batch loading
    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY;
    int srcX = blockIdx.x * TILE_WIDTH + destX;
    int src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
        dataB_ds[destY][destX] = dataB[src];
    } else {
        dataB_ds[destY][destX] = 0;
    }

    // Second batch loading
    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY;
    srcX = blockIdx.x * TILE_WIDTH + destX;
    src = (srcY * width + srcX);
    if (destY < w) {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            dataB_ds[destY][destX] = dataB[src];
        } else {
            dataB_ds[destY][destX] = 0;
        }
    }
    __syncthreads();

    float accumB = 0;

    float maskValue;

    for (int y = 0; y < MASK_WIDTH; y++) {
        for (int x = 0; x < MASK_WIDTH; x++) {
            maskValue = MASK[y * MASK_WIDTH + x];
            accumB += dataB_ds[threadIdx.y + y][threadIdx.x + x] * maskValue;
        }
    }
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y + MASK_RADIUS;
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x + MASK_RADIUS;
    if (y < height - PIXEL_LOST && x < width - PIXEL_LOST) {
        resultB[(y * (width - PIXEL_LOST) + x)] = accumB;
    }

//    __syncthreads();
}