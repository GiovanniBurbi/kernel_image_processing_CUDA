cmake_minimum_required(VERSION 3.22)
project(kernel_image_processing_CUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(kernel_image_processing_CUDA src/main.cu src/utils/Utils.h src/image/PpmParser.cpp src/image/PpmParser.h src/image/Image.cpp src/image/Image.h src/convolution/Convolution.cu src/kernel/Kernel.cpp src/kernel/Kernel.h src/convolution/Convolution.cuh)
include_directories(./src)

set_target_properties(kernel_image_processing_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
