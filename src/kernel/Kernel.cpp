//
// Created by giova on 31/05/2022.
//

#include "Kernel.h"

#define KERNEL_SIZE 9

/*
 * Return a 3x3 kernel based on an int value from the enum type defined in kernel.h
 * */
float* createKernel(int type) {
    switch (type) {
        case kernelsType::boxBlur:
            return createBoxBlurKernel();
        case kernelsType::gaussianBlur:
            return createGaussianBlurKernel();
        case kernelsType::emboss:
            return createEmbossKernel();
        case kernelsType::outline:
            return createOutlineKernel();
        case kernelsType::sharpen:
            return createSharpenKernel();
        default:
            std::cerr << "Filter type not available." << std::endl;
            return nullptr;
    }
}

std::string kernelName(int type){
    switch (type) {
        case kernelsType::boxBlur:
            return "BoxBlur";
        case kernelsType::gaussianBlur:
            return "GaussianBlur";
        case kernelsType::emboss:
            return "Emboss";
        case kernelsType::outline:
            return "Outline";
        case kernelsType::sharpen:
            return "Sharpen";
        default:
            std::cerr << "Filter type not available." << std::endl;
            return "";
    }
}

/*
 * Allocate space for a 3x3 kernel and return the pointer
 * */
float* allocateEmptyKernel() {
    auto* kernel = (float*) malloc(sizeof(float) * KERNEL_SIZE);
    return kernel;
}

float* createBoxBlurKernel() {
    auto* kernel = allocateEmptyKernel();
    for (int i = 0; i < KERNEL_SIZE; i++) {
        kernel[i] = 0.111111;
    }

    return kernel;
}

float* createEmbossKernel(){
    auto* kernel = allocateEmptyKernel();

    for (int i = 0; i < KERNEL_SIZE; i++) {
        if (i == 0)
            kernel[i] = -2.f;
        else if (i == KERNEL_SIZE - 1)
            kernel[i] = 2.f;
        else if (i == 2 || i == 6)
            kernel[i] = 0.f;
        else if (i == 1 || i == 3)
            kernel[i] = -1.f;
        else kernel[i] = 1;

    }

    return kernel;
}

float* createGaussianBlurKernel() {
    auto* kernel = allocateEmptyKernel();

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        if (i == KERNEL_SIZE/2)
            kernel[i] = 1/4.f;
        else if (i % 2 == 0)
            kernel[i] = 1/16.f;
        else kernel[i] = 1/8.f;
    }

    return kernel;
}

float* createOutlineKernel() {
    auto* kernel = allocateEmptyKernel();

    for (int i = 0; i < KERNEL_SIZE; i++){
        if(i == KERNEL_SIZE/2)
            kernel[i] = 8.f;
        else
            kernel[i] = -1.f;
    }
    return kernel;
}

float* createSharpenKernel() {
    auto* kernel = allocateEmptyKernel();

    for (int i = 0; i < KERNEL_SIZE; ++i) {
        if (i == KERNEL_SIZE/2)
            kernel[i] = 5.f;
        else if (i % 2 == 0)
            kernel[i] = -1.f;
        else kernel[i] = 0.f;
    }

    return kernel;
}
