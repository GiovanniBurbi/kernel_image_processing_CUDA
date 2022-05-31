#include <iostream>
#include <chrono>
#include <string>

#include "image/PpmParser.h"
#include "kernel/Kernel.h"
#include "convolution/Convolution.cuh"


#define IMPORT_PATH "../resources/source/"
#define EXPORT_PATH "../resources/results/"
#define IMAGE "lake"
#define BLOCK_WIDTH 32
#define CHANNELS 3

#define ITER 1
#define THREADS3D true

#define CONSTANT_MEM true

__constant__ float MASK[MASK_WIDTH * MASK_WIDTH];

static void CheckCudaErrorAux(const char *, unsigned, const char *,
                              cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
                              const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

int main() {
    std::string filename;
    std::string output_name;

    std::string log;
    log.append("Cuda version ");

    filename.append(IMPORT_PATH).append(IMAGE).append(".ppm");
    output_name.append(EXPORT_PATH).append(IMAGE).append("Cuda");

    float* kernel = createKernel(kernelsType::outline);

    Image_t* image = PPM_import(filename.c_str());

    int width = image_getWidth(image);
    int height = image_getHeight(image);
    int channels = image_getChannels(image);

    int outputWidth = width - MASK_RADIUS * 2;
    int outputHeight = height - MASK_RADIUS * 2;
    Image_t *output;

    float time = 0;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;

    for (int i = 0; i < ITER; i++) {

        if (i != 0) image_delete(output);

        output = new_image(outputWidth, outputHeight, channels);

        float *host_imageData = image_getData(image);
        float *host_outputData = image_getData(output);

        float *device_imageData;
        float *device_outputData;
        float *device_maskData;

        startTime = std::chrono::high_resolution_clock::now();

        if(CONSTANT_MEM) {
            CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MASK, kernel,
                                                 MASK_WIDTH * MASK_WIDTH * sizeof(float)));
        }

        CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageData,
                                     width * height * channels * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputData,
                                     outputWidth * outputHeight * channels * sizeof(float)));
        if(!CONSTANT_MEM) {
            CUDA_CHECK_RETURN(cudaMalloc((void **) &device_maskData,
                                         MASK_WIDTH * MASK_WIDTH * sizeof(float)));
        }

        CUDA_CHECK_RETURN(cudaMemcpy(device_imageData, host_imageData,
                                     width * height * channels * sizeof(float),
                                     cudaMemcpyHostToDevice));
        if(!CONSTANT_MEM) {
            CUDA_CHECK_RETURN(cudaMemcpy(device_maskData, kernel,
                                         MASK_WIDTH * MASK_WIDTH * sizeof(float),
                                         cudaMemcpyHostToDevice));
        }

        if(!THREADS3D){
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
            dim3 dimGrid(ceil((float) outputWidth / BLOCK_WIDTH), ceil((float) outputHeight / BLOCK_WIDTH));
            if (CONSTANT_MEM) {
                convolutionConstantMemory<<<dimGrid, dimBlock>>>(device_imageData,
                                                                 device_outputData, width, height, channels);
                output_name.append("ConstMemory");
            } else {
                convolutionNaive<<<dimGrid, dimBlock>>>(device_imageData, device_maskData,
                                                        device_outputData, width, height, channels);
                output_name.append("Naive");
            }
        }
        if(THREADS3D){
            dim3 dimBlock(18, 18, CHANNELS);
            dim3 dimGrid(ceil((float) outputWidth / dimBlock.x), ceil((float) outputHeight / dimBlock.y));

            output_name.append("3DCoverage");
            if(CONSTANT_MEM){
                convolutionNaive3DThreadsCoverageConstantMemory<<<dimGrid, dimBlock>>>(device_imageData,
                                                                         device_outputData, width, height, channels);
                output_name.append("ConstMemory");
            }

            if(!CONSTANT_MEM) {
                convolutionNaive3DThreadsCoverage<<<dimGrid, dimBlock>>>(device_imageData, device_maskData,
                                                                         device_outputData, width, height, channels);
                output_name.append("Naive");
            }
        }

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        CUDA_CHECK_RETURN(cudaMemcpy(host_outputData, device_outputData,
                                     outputWidth * outputHeight * channels * sizeof(float),
                                     cudaMemcpyDeviceToHost));

        endTime = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

        cudaFree(device_imageData);
        cudaFree(device_outputData);
        if(!CONSTANT_MEM) cudaFree(device_maskData);

    }

    log.append("took ").append(std::to_string(time/ITER)).append(" seconds");
    printf("%s\n", log.c_str());

    output_name.append(".ppm");

    PPM_export(output_name.c_str(), output);

    image_delete(image);
    image_delete(output);
    free(kernel);

    return 0;
}
