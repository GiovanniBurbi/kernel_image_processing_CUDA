//
// Created by giova on 31/05/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH
#define KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH

#include <cuda_runtime_api.h>

//#define TILE_WIDTH 30
#define TILE_WIDTH 32

__global__ void convolutionNaive(const float* __restrict__ data, const float* __restrict__ mask, float* result,
                                 int width, int height, int channels);
__global__ void convolutionConstantMemory(const float* __restrict__ data, float* result,
                                          int width, int height, int channels);
__global__ void convolutionTiling(const float* __restrict__ data, float* result,
                                          int width, int height, int channels);

#endif //KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH
