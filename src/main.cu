#include <iostream>
#include <chrono>
#include <string>
#include <cassert>

#include "image/PpmParser.h"
#include "kernel/Kernel.h"
#include "convolution/Convolution.cuh"

// From terminal
//#define IMPORT_PATH "resources/source/"
//#define EXPORT_PATH "resources/results/"

// From IDE
#define IMPORT_PATH "../resources/source/"
#define EXPORT_PATH "../resources/results/"

#define IMAGE "lake"

#define BLOCK_WIDTH_NAIVE 8

#define BLOCK_WIDTH (TILE_WIDTH)
static_assert(BLOCK_WIDTH * BLOCK_WIDTH <= 1024, "max number of threads per block exceeded");

#define ITER 1

#define SOA true
#define AOS false

#define ASYNC true

#define NAIVE true
#define TILING false

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
    assert(MASK_WIDTH == 3);

    std::string filename;
    std::string output_name;

    std::string log;
    log.append("Cuda version ");

    filename.append(IMPORT_PATH).append(IMAGE).append(".ppm");
    output_name.append(EXPORT_PATH).append(IMAGE).append("Cuda");

    float *kernel = createKernel(kernelsType::outline);

    float time = 0;

    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;

    if (AOS) {
        output_name.append("AoS");
        log.append("AoS ");

        Image_t* inputImage = PPM_import(filename.c_str());

        int imageWidth = image_getWidth(inputImage);
        int imageHeight = image_getHeight(inputImage);
        int imageChannels = image_getChannels(inputImage);

        Image_t* outputImage;

        float *imageData;
        float *outputData;

        int outputWidth = imageWidth - 2;
        int outputHeight = imageHeight - 2;

        if (NAIVE) {
            log.append("naive ");

            float *device_imageData;
            float *device_outputData;
            float *device_maskData;

            for (int i = 0; i < ITER; i++) {
                if (i != 0) image_delete(outputImage);

                outputImage = new_image(outputWidth, outputHeight, imageChannels);

                imageData = image_getData(inputImage);
                outputData = image_getData(outputImage);

                startTime = std::chrono::high_resolution_clock::now();

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageData,
                                             imageWidth * imageHeight * imageChannels * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputData,
                                             outputWidth * outputHeight * imageChannels * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_maskData,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMemcpy(device_imageData, imageData,
                                             imageWidth * imageHeight * imageChannels * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_maskData, kernel,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float),
                                             cudaMemcpyHostToDevice));

                dim3 dimGrid(ceil((float) imageWidth / BLOCK_WIDTH_NAIVE),
                             ceil((float) imageHeight / BLOCK_WIDTH_NAIVE));
                dim3 dimBlock(BLOCK_WIDTH_NAIVE, BLOCK_WIDTH_NAIVE);

                convolutionNaiveNoPadding<<<dimGrid, dimBlock>>>(device_imageData, device_maskData,
                                                                 device_outputData, imageWidth, imageHeight,
                                                                 imageChannels);

                CUDA_CHECK_RETURN(cudaDeviceSynchronize());

                CUDA_CHECK_RETURN(cudaMemcpy(outputData, device_outputData,
                                             outputWidth * outputHeight * imageChannels * sizeof(float),
                                             cudaMemcpyDeviceToHost));

                endTime = std::chrono::high_resolution_clock::now();
                time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

                cudaFree(device_imageData);
                cudaFree(device_outputData);
                cudaFree(device_maskData);
            }
        }

        if (TILING) {
            log.append("with tiling ");
            output_name.append("Tiling");

            float *device_imageData;
            float *device_outputData;

            for (int i = 0; i < ITER; i++) {
                if (i != 0) image_delete(outputImage);

                outputImage = new_image(outputWidth, outputHeight, imageChannels);

                imageData = image_getData(inputImage);
                outputData = image_getData(outputImage);

                startTime = std::chrono::high_resolution_clock::now();

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageData,
                                             imageWidth * imageHeight * imageChannels * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputData,
                                             outputWidth * outputHeight * imageChannels * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageData, imageData,
                                             imageWidth * imageHeight * imageChannels * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MASK, kernel, MASK_WIDTH * MASK_WIDTH * sizeof(float)));

                dim3 dimGrid(ceil((float) imageWidth / BLOCK_WIDTH),
                             ceil((float) imageHeight / BLOCK_WIDTH));
                dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

                convolutionTilingNoPadding<<<dimGrid, dimBlock>>>(device_imageData,
                                                                  device_outputData, imageWidth, imageHeight,
                                                                  imageChannels);

                CUDA_CHECK_RETURN(cudaMemcpy(outputData, device_outputData,
                                             outputWidth * outputHeight * imageChannels * sizeof(float),
                                             cudaMemcpyDeviceToHost));

                endTime = std::chrono::high_resolution_clock::now();
                time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

                cudaFree(device_imageData);
                cudaFree(device_outputData);
            }
        }
        log.append("took ").append(std::to_string(time/ITER)).append(" seconds");
        printf("%s\n", log.c_str());

        output_name.append(".ppm");

        PPM_export(output_name.c_str(), outputImage);

        image_delete(outputImage);
        image_delete(inputImage);
    }

    if(SOA) {
        output_name.append("SoA");
        log.append("SoA ");

        ImageSoA_t* inputImage = PPM_importSoA(filename.c_str());

        int imageWidth = image_getWidth(inputImage);
        int imageHeight = image_getHeight(inputImage);
        int imageChannels = image_getChannels(inputImage);

        assert(imageChannels == 3);

        float *imageDataR;
        float *imageDataG;
        float *imageDataB;

        ImageSoA_t* outputImage;

        float *outputDataR;
        float *outputDataG;
        float *outputDataB;

        float *device_imageDataR;
        float *device_imageDataG;
        float *device_imageDataB;

        float *device_outputDataR;
        float *device_outputDataG;
        float *device_outputDataB;

        int outputWidth = imageWidth - 2;
        int outputHeight = imageHeight - 2;

        if (NAIVE && !ASYNC) {
            log.append("naive ");
            output_name.append("Naive");

            float *device_maskData;

            for (int i = 0; i < ITER; i++) {
                if (i != 0) image_delete(outputImage);

                outputImage = new_imageSoA(outputWidth, outputHeight, imageChannels);

                imageDataR = image_getR(inputImage);
                imageDataG = image_getG(inputImage);
                imageDataB = image_getB(inputImage);

                outputDataR = image_getR(outputImage);
                outputDataG = image_getG(outputImage);
                outputDataB = image_getB(outputImage);

                startTime = std::chrono::high_resolution_clock::now();

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataR,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataG,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataB,
                                             imageWidth * imageHeight * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataR,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataG,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataB,
                                             outputWidth * outputHeight * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_maskData,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataR, imageDataR,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataG, imageDataG,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataB, imageDataB,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));

                CUDA_CHECK_RETURN(cudaMemcpy(device_maskData, kernel,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float),
                                             cudaMemcpyHostToDevice));

                dim3 dimGrid(ceil((float) imageWidth / BLOCK_WIDTH_NAIVE),
                             ceil((float) imageHeight / BLOCK_WIDTH_NAIVE));
                dim3 dimBlock(BLOCK_WIDTH_NAIVE, BLOCK_WIDTH_NAIVE);

                convolutionNaiveNoPaddingSoA<<<dimGrid, dimBlock>>>(device_imageDataR, device_imageDataG, device_imageDataB,
                                                           device_maskData, device_outputDataR, device_outputDataG,
                                                           device_outputDataB,
                                                           imageWidth, imageHeight);

                CUDA_CHECK_RETURN(cudaDeviceSynchronize());

                CUDA_CHECK_RETURN(cudaMemcpy(outputDataR, device_outputDataR,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(outputDataG, device_outputDataG,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(outputDataB, device_outputDataB,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));

                endTime = std::chrono::high_resolution_clock::now();
                time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

                cudaFree(device_imageDataR);
                cudaFree(device_imageDataG);
                cudaFree(device_imageDataB);
                cudaFree(device_outputDataR);
                cudaFree(device_outputDataG);
                cudaFree(device_outputDataB);

                cudaFree(device_maskData);
            }
        }

        if(NAIVE && ASYNC) {
            log.append("naive with async loading ");
            output_name.append("NaiveAsync");

            float *device_maskData;

            for (int i = 0; i < ITER; i++) {
                if (i != 0) image_delete(outputImage);

                outputImage = new_imageSoA(outputWidth, outputHeight, imageChannels);

                imageDataR = image_getR(inputImage);
                imageDataG = image_getG(inputImage);
                imageDataB = image_getB(inputImage);

                outputDataR = image_getR(outputImage);
                outputDataG = image_getG(outputImage);
                outputDataB = image_getB(outputImage);

                cudaStream_t stream1;
                cudaStream_t stream2;
                cudaStream_t stream3;

                cudaStreamCreate(&stream1);
                cudaStreamCreate(&stream2);
                cudaStreamCreate(&stream3);

                startTime = std::chrono::high_resolution_clock::now();

                dim3 dimGrid(ceil((float) imageWidth / BLOCK_WIDTH_NAIVE),
                             ceil((float) imageHeight / BLOCK_WIDTH_NAIVE));
                dim3 dimBlock(BLOCK_WIDTH_NAIVE, BLOCK_WIDTH_NAIVE);

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_maskData,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMemcpy(device_maskData, kernel,
                                             MASK_WIDTH * MASK_WIDTH * sizeof(float),
                                             cudaMemcpyHostToDevice));

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataR,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataG,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataB,
                                             imageWidth * imageHeight * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataR,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataG,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataB,
                                             outputWidth * outputHeight * sizeof(float)));

//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &imageDataR,
//                                                 imageWidth * imageHeight * sizeof(float)));
//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &imageDataG,
//                                                 imageWidth * imageHeight * sizeof(float)));
//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &imageDataB,
//                                                 imageWidth * imageHeight * sizeof(float)));
//
//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &outputDataR,
//                                                 outputWidth * outputHeight * sizeof(float)));
//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &outputDataG,
//                                                 outputWidth * outputHeight * sizeof(float)));
//                CUDA_CHECK_RETURN(cudaMallocHost((void **) &outputDataB,
//                                                 outputWidth * outputHeight * sizeof(float)));

//                CUDA_CHECK_RETURN(cudaMemcpyAsync(device_imageDataR, imageDataR, imageWidth * imageHeight * sizeof(float),
//                                                  cudaMemcpyHostToDevice, stream1));
//                CUDA_CHECK_RETURN(cudaMemcpyAsync(device_imageDataG, imageDataG, imageWidth * imageHeight * sizeof(float),
//                                                  cudaMemcpyHostToDevice, stream2));
//                CUDA_CHECK_RETURN(cudaMemcpyAsync(device_imageDataB, imageDataB, imageWidth * imageHeight * sizeof(float),
//                                                  cudaMemcpyHostToDevice, stream3));

                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataR, imageDataR, imageWidth * imageHeight * sizeof(float),
                                                  cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataG, imageDataG, imageWidth * imageHeight * sizeof(float),
                                                  cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataB, imageDataB, imageWidth * imageHeight * sizeof(float),
                                                  cudaMemcpyHostToDevice));

                convolutionNaiveNoPaddingSoAChannelR<<<dimGrid, dimBlock, 0, stream1>>>(device_imageDataR, device_maskData, device_outputDataR,
                                                                                        imageWidth, imageHeight);
                convolutionNaiveNoPaddingSoAChannelG<<<dimGrid, dimBlock, 0, stream2>>>(device_imageDataG, device_maskData, device_outputDataG,
                                                                                        imageWidth, imageHeight);
                convolutionNaiveNoPaddingSoAChannelB<<<dimGrid, dimBlock, 0, stream3>>>(device_imageDataB, device_maskData, device_outputDataB,
                                                                                        imageWidth, imageHeight);

                CUDA_CHECK_RETURN(cudaStreamSynchronize(stream1));
                CUDA_CHECK_RETURN(cudaMemcpyAsync(outputDataR, device_outputDataR,
                                                  outputWidth * outputHeight * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream1));

                CUDA_CHECK_RETURN(cudaStreamSynchronize(stream2));
                CUDA_CHECK_RETURN(cudaMemcpyAsync(outputDataG, device_outputDataG,
                                                  outputWidth * outputHeight * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream2));

                CUDA_CHECK_RETURN(cudaStreamSynchronize(stream3));
                CUDA_CHECK_RETURN(cudaMemcpyAsync(outputDataB, device_outputDataB,
                                                  outputWidth * outputHeight * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream3));

                CUDA_CHECK_RETURN(cudaDeviceSynchronize());

                endTime = std::chrono::high_resolution_clock::now();
                time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

                cudaStreamDestroy(stream1);
                cudaStreamDestroy(stream2);
                cudaStreamDestroy(stream3);

                cudaFree(device_imageDataR);
                cudaFree(device_imageDataG);
                cudaFree(device_imageDataB);
                cudaFree(device_outputDataR);
                cudaFree(device_outputDataG);
                cudaFree(device_outputDataB);

                cudaFree(device_maskData);
            }
        }


        if (TILING && !ASYNC) {
            log.append("tiling ");
            output_name.append("Tiling");

            for (int i = 0; i < ITER; i++) {
                if (i != 0) image_delete(outputImage);

                outputImage = new_imageSoA(outputWidth, outputHeight, imageChannels);

                imageDataR = image_getR(inputImage);
                imageDataG = image_getG(inputImage);
                imageDataB = image_getB(inputImage);

                outputDataR = image_getR(outputImage);
                outputDataG = image_getG(outputImage);
                outputDataB = image_getB(outputImage);

                startTime = std::chrono::high_resolution_clock::now();

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataR,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataG,
                                             imageWidth * imageHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_imageDataB,
                                             imageWidth * imageHeight * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataR,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataG,
                                             outputWidth * outputHeight * sizeof(float)));
                CUDA_CHECK_RETURN(cudaMalloc((void **) &device_outputDataB,
                                             outputWidth * outputHeight * sizeof(float)));

                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataR, imageDataR,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataG, imageDataG,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));
                CUDA_CHECK_RETURN(cudaMemcpy(device_imageDataB, imageDataB,
                                             imageWidth * imageHeight * sizeof(float),
                                             cudaMemcpyHostToDevice));

                CUDA_CHECK_RETURN(cudaMemcpyToSymbol(MASK, kernel, MASK_WIDTH * MASK_WIDTH * sizeof(float)));

                dim3 dimGrid(ceil((float) imageWidth / BLOCK_WIDTH_NAIVE),
                             ceil((float) imageHeight / BLOCK_WIDTH_NAIVE));
                dim3 dimBlock(BLOCK_WIDTH_NAIVE, BLOCK_WIDTH_NAIVE);

                convolutionTilingNoPaddingSoA<<<dimGrid, dimBlock>>>(device_imageDataR, device_imageDataG,
                                                                     device_imageDataB,
                                                                     device_outputDataR, device_outputDataG,
                                                                     device_outputDataB,
                                                                     imageWidth, imageHeight);

                CUDA_CHECK_RETURN(cudaMemcpy(outputDataR, device_outputDataR,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(outputDataG, device_outputDataG,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));
                CUDA_CHECK_RETURN(cudaMemcpy(outputDataB, device_outputDataB,
                                             outputWidth * outputHeight * sizeof(float),
                                             cudaMemcpyDeviceToHost));

                endTime = std::chrono::high_resolution_clock::now();
                time += std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime).count();

                cudaFree(device_imageDataR);
                cudaFree(device_imageDataG);
                cudaFree(device_imageDataB);
                cudaFree(device_outputDataR);
                cudaFree(device_outputDataG);
                cudaFree(device_outputDataB);
            }
        }

        log.append("took ").append(std::to_string(time/ITER)).append(" seconds");
        printf("%s\n", log.c_str());

        output_name.append(".ppm");

        PPM_exportSoA(output_name.c_str(), outputImage);

        image_delete(outputImage);
        image_delete(inputImage);
    }

    free(kernel);
    return 0;
}