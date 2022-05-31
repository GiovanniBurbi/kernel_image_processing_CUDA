#include <iostream>
#include "image/PpmParser.h"
#include "kernel/Kernel.h"
//#include "convolution/Convolution.cu"

#define IMPORT_PATH "../resources/source/"
#define EXPORT_PATH "../resources/results/"
#define IMAGE "lake"

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


#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define PIXEL_LOST 2

__global__ void convolutionNaive(float* data, float* mask, float* result,
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


int main() {
    std::string filename;
    std::string output_name;

    filename.append(IMPORT_PATH).append(IMAGE).append(".ppm");
    output_name.append(EXPORT_PATH).append(IMAGE);

    float* kernel = createKernel(kernelsType::outline);

    Image_t* image = PPM_import(filename.c_str());

    int width = image_getWidth(image);
    int height = image_getHeight(image);
    int channels = image_getChannels(image);

    int outputWidth = width - MASK_RADIUS * 2;
    int outputHeight = height - MASK_RADIUS * 2;

    Image_t* output = new_image(outputWidth, outputHeight, channels);

    float* host_imageData = image_getData(image);
    float * host_outputData = image_getData(output);

    float *device_imageData;
    float *device_outputData;
    float *device_maskData;

    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageData,
               width * height * channels * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputData,
               outputWidth * outputHeight * channels * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &device_maskData,
               MASK_WIDTH * MASK_WIDTH * sizeof(float)));

    CUDA_CHECK_RETURN(cudaMemcpy(device_imageData, host_imageData,
               width * height * channels * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(device_maskData, kernel,
               MASK_WIDTH * MASK_WIDTH * sizeof(float),
               cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil((float)outputWidth / dimBlock.x), ceil((float)outputHeight / dimBlock.y));

    convolutionNaive<<<dimGrid, dimBlock>>>(device_imageData, device_maskData,
                                            device_outputData, width, height, channels);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(host_outputData, device_outputData,
               outputWidth * outputHeight * channels * sizeof(float),
               cudaMemcpyDeviceToHost));

    output_name.append(".ppm");

    PPM_export(output_name.c_str(), output);

    cudaFree(device_imageData);
    cudaFree(device_outputData);
    cudaFree(device_maskData);

    image_delete(image);
    image_delete(output);
    free(kernel);

    return 0;
}
