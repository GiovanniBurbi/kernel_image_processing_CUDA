//
// Created by giova on 31/05/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH
#define KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH

#include <cuda_runtime_api.h>

#define TILE_WIDTH 32
/*
 * Implementation with "padding"
 * */
__global__ void convolutionNaivePadding(const float* __restrict__ data, const float* __restrict__ mask, float* result,
                                        int width, int height, int channels);
__global__ void convolutionNaiveSoAPadding(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                           const float* __restrict__ mask, float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB,
                                           int width, int height);
__global__ void convolutionNaiveSoAChannelRPadding(const float* __restrict__ dataR, const float* __restrict__ mask, float* __restrict__ resultR,
                                                   int width, int height);
__global__ void convolutionNaiveSoAChannelGPadding(const float* __restrict__ dataG, const float* __restrict__ mask, float* __restrict__ resultG,
                                                   int width, int height);
__global__ void convolutionNaiveSoAChannelBPadding(const float* __restrict__ dataB, const float* __restrict__ mask, float* __restrict__ resultB,
                                                   int width, int height);
__global__ void convolutionConstantMemoryPadding(const float* __restrict__ data, float* result,
                                                 int width, int height, int channels);
__global__ void convolutionConstantMemorySoAPadding(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                                    float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height);
__global__ void convolutionTilingPadding(const float* __restrict__ data, float* result,
                                         int width, int height, int channels);
__global__ void convolutionTilingSoAPadding(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                            float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height);
__global__ void convolutionTilingSoAChannelRPadding(const float* __restrict__ dataR, float* __restrict__ resultR, int width, int height);
__global__ void convolutionTilingSoAChannelGPadding(const float* __restrict__ dataG, float* __restrict__ resultG, int width, int height);
__global__ void convolutionTilingSoAChannelBPadding(const float* __restrict__ dataB, float* __restrict__ resultB, int width, int height);


/*
 * Implementations that crop the edges of the image
 * */

__global__ void convolutionNaive(const float* __restrict__ data, const float* __restrict__ mask, float* result,
                                 int width, int height, int channels);
__global__ void convolutionNaiveSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                    const float* __restrict__ mask, float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB,
                                    int width, int height);


__global__ void convolutionNaiveSoAChannelR(const float* __restrict__ dataR, const float* __restrict__ mask, float* __restrict__ resultR,
                                            int width, int height);
__global__ void convolutionNaiveSoAChannelG(const float* __restrict__ dataG, const float* __restrict__ mask, float* __restrict__ resultG,
                                            int width, int height);
__global__ void convolutionNaiveSoAChannelB(const float* __restrict__ dataB, const float* __restrict__ mask, float* __restrict__ resultB,
                                            int width, int height);


__global__ void convolutionConstantMemory(const float* __restrict__ data, float* result,
                                          int width, int height, int channels);
__global__ void convolutionConstantMemorySoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                             float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height);


__global__ void convolutionTiling(const float* __restrict__ data, float* result,
                                  int width, int height, int channels);

__global__ void convolutionTilingSoA(const float* __restrict__ dataR, const float* __restrict__ dataG, const float* __restrict__ dataB,
                                     float* __restrict__ resultR, float* __restrict__ resultG, float* __restrict__ resultB, int width, int height);


__global__ void convolutionTilingSoAChannelR(const float* __restrict__ dataR, float* __restrict__ resultR, int width, int height);
__global__ void convolutionTilingSoAChannelG(const float* __restrict__ dataG, float* __restrict__ resultG, int width, int height);
__global__ void convolutionTilingSoAChannelB(const float* __restrict__ dataB, float* __restrict__ resultB, int width, int height);


#endif //KERNEL_IMAGE_PROCESSING_CUDA_CONVOLUTION_CUH
