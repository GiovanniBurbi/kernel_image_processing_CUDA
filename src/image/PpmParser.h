//
// Created by giova on 27/05/2022.
//

#ifndef KERNEL_IMAGE_PROCESSING_CUDA_PPMPARSER_H
#define KERNEL_IMAGE_PROCESSING_CUDA_PPMPARSER_H

#include "Image.h"
#include "ImageSoA.h"

Image_t* PPM_import(const char *filename);
ImageSoA_t* PPM_importSoA(const char *filename);
bool PPM_export(const char *filename, Image_t* img);
bool PPM_exportSoA(const char *filename, ImageSoA_t* img);

void test_images();

#endif //KERNEL_IMAGE_PROCESSING_CUDA_PPMPARSER_H
