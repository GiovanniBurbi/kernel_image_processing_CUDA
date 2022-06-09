//
// Created by giova on 27/05/2022.
//

#include "PpmParser.h"
#include "../utils/Utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define PPMREADBUFLEN 256

static const char *skipSpaces(const char *line) {
    while (*line == ' ' || *line == '\t') {
        line++;
        if (*line == '\0') {
            break;
        }
    }
    return line;
}

static char nextNonSpaceChar(const char *line0) {
    const char *line = skipSpaces(line0);
    return *line;
}

static bool isComment(const char *line) {
    char nextChar = nextNonSpaceChar(line);
    if (nextChar == '\0') {
        return true;
    } else {
        return nextChar == '#';
    }
}

static void parseDimensions(const char *line0, int *width, int *height) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d %d", width, height);
}

static void parseDimensions(const char *line0, int *width, int *height,
                            int *channels) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d %d %d", width, height, channels);
}

static void parseDepth(const char *line0, int *depth) {
    const char *line = skipSpaces(line0);
    sscanf(line, "%d", depth);
}

static char *File_readLine(FILE* file) {
    static char buffer[PPMREADBUFLEN];
    if (file == NULL) {
        return NULL;
    }
    memset(buffer, 0, PPMREADBUFLEN);

    if (fgets(buffer, PPMREADBUFLEN - 1, file)) {
        return buffer;
    } else {
        return NULL;
    }
}

static char *nextLine(FILE* file) {
    char *line = NULL;
    while ((line = File_readLine(file)) != NULL) {
        if (!isComment(line)) {
            break;
        }
    }
    return line;
}

char* File_read(FILE* file, size_t size, size_t count) {
    size_t res;
    char *buffer;
    size_t bufferLen;

    if (file == NULL) {
        return NULL;
    }

    bufferLen = size * count + 1;
    buffer = (char*) malloc(sizeof(char) * bufferLen);

    res = fread(buffer, size, count, file);
    // make valid C string
    buffer[size * res] = '\0';

    return buffer;
}

bool File_write(FILE* file, const void *buffer, size_t size, size_t count) {
    if (file == NULL) {
        return false;
    }

    size_t res = fwrite(buffer, size, count, file);
    if (res != count) {
        printf("ERROR: Failed to write data to PPM file");
    }

    return true;
}

Image_t* PPM_import(const char *filename) {
    Image_t* img;
    FILE* file;
    char *header;
    char *line;
    int ii, jj, kk, channels;
    int width, height, depth;
    unsigned char *charData, *charIter;
    float *imgData, *floatIter;
    float scale;

    img = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        goto cleanup;
    }
    header = File_readLine(file);
    if (header == NULL) {
        printf("Could not read from %s\n", filename);
        goto cleanup;
    } else if (strcmp(header, "P6") != 0 && strcmp(header, "P6\n") != 0
               && strcmp(header, "P5") != 0 && strcmp(header, "P5\n") != 0
               && strcmp(header, "S6") != 0 && strcmp(header, "S6\n") != 0) {
        printf("Could not find magic number for %s\n", filename);
        goto cleanup;
    }

    // P5 are monochrome while P6/S6 are RGB
    // S6 needs to parse number of channels out of file
    if (strcmp(header, "P5") == 0 || strcmp(header, "P5\n") == 0) {
        channels = 1;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
        channels = 3;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else {
        line = nextLine(file);
        parseDimensions(line, &width, &height, &channels);
    }

    // the line now contains the depth information
    line = nextLine(file);
    parseDepth(line, &depth);

    // the rest of the lines contain the data in binary format
    charData = (unsigned char *) File_read(file,
                                           width * channels * sizeof(unsigned char), height);

    img = new_image(width, height, channels);

    imgData = image_getData(img);

    charIter = charData;
    floatIter = imgData;

    scale = 1.0f / ((float) depth);

    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            for (kk = 0; kk < channels; kk++) {
                *floatIter = ((float) *charIter) * scale;
                floatIter++;
                charIter++;
            }
        }
    }

    free(charData);
    cleanup: fclose(file);
    return img;
}

/*
 * Import a ppm image and return a pointer to an ImageSoA_t struct
 * The data layout is Structure of Arrays
 * */
ImageSoA_t* PPM_importSoA(const char *filename) {
    ImageSoA_t* img;
    FILE* file;
    char *header;
    char *line;
    int ii, jj, kk, channels;
    int width, height, depth;
    unsigned char *charData, *charIter;
    float *imgR, *imgG, *imgB, *floatRIter, *floatGIter, *floatBIter;
    float scale;

    img = NULL;

    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Could not open %s\n", filename);
        goto cleanup;
    }
    header = File_readLine(file);
    if (header == NULL) {
        printf("Could not read from %s\n", filename);
        goto cleanup;
    } else if (strcmp(header, "P6") != 0 && strcmp(header, "P6\n") != 0
               && strcmp(header, "P5") != 0 && strcmp(header, "P5\n") != 0
               && strcmp(header, "S6") != 0 && strcmp(header, "S6\n") != 0) {
        printf("Could not find magic number for %s\n", filename);
        goto cleanup;
    }

    // P5 are monochrome while P6/S6 are RGB
    // S6 needs to parse number of channels out of file
    if (strcmp(header, "P5") == 0 || strcmp(header, "P5\n") == 0) {
        channels = 1;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else if (strcmp(header, "P6") == 0 || strcmp(header, "P6\n") == 0) {
        channels = 3;
        line = nextLine(file);
        parseDimensions(line, &width, &height);
    } else {
        line = nextLine(file);
        parseDimensions(line, &width, &height, &channels);
    }

    // the line now contains the depth information
    line = nextLine(file);
    parseDepth(line, &depth);

    // the rest of the lines contain the data in binary format
    charData = (unsigned char *) File_read(file,
                                           width * channels * sizeof(unsigned char), height);

    img = new_imageSoA(width, height, channels);

    imgR = image_getR(img);
    imgG = image_getG(img);
    imgB = image_getB(img);

    charIter = charData;
    floatRIter = imgR;
    floatGIter = imgG;
    floatBIter = imgB;

    scale = 1.0f / ((float) depth);
    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            floatRIter[ii * width + jj] = ((float) *charIter) * scale;
            charIter++;
            floatGIter[ii * width + jj] = ((float) *charIter) * scale;
            charIter++;
            floatBIter[ii * width + jj] = ((float) *charIter) * scale;
            charIter++;
        }
    }

    free(charData);
    cleanup: fclose(file);
    return img;
}

bool PPM_export(const char *filename, Image_t* img) {
    int ii;
    int jj;
    int kk;
    int depth;
    int width;
    int height;
    int channels;
    FILE* file;
    float *floatIter;
    unsigned char *charData;
    unsigned char *charIter;

    file = fopen(filename, "wb+");
    if (file == NULL) {
        printf("Could not open %s in mode %s\n", filename, "wb+");
        return false;
    }

    width = image_getWidth(img);
    height = image_getHeight(img);
    channels = image_getChannels(img);
    depth = 255;

    if (channels == 1) {
        fprintf(file, "P5\n");
    } else {
        fprintf(file, "P6\n");
    }
    fprintf(file, "#Created via PPM Export\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", depth);

    charData = (unsigned char*) malloc(
            sizeof(unsigned char) * width * height * channels);

    charIter = charData;
    floatIter = image_getData(img);

    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            for (kk = 0; kk < channels; kk++) {
                *charIter = (unsigned char) ceil(
                        clamp(*floatIter, 0, 1) * depth);
                floatIter++;
                charIter++;
            }
        }
    }

    bool writeResult = File_write(file, charData,
                                  width * channels * sizeof(unsigned char), height);

    free(charData);
    fflush(file);
    fclose(file);

    return true;
}

/*
 * It takes as input a reference to an ImageSoA_t struct and export a ppm image from it
 * */
bool PPM_exportSoA(const char *filename, ImageSoA_t* img) {
    int ii;
    int jj;
    int kk;
    int depth;
    int width;
    int height;
    int channels;
    FILE* file;
    float *floatRIter;
    float *floatGIter;
    float *floatBIter;
    unsigned char *charData;
    unsigned char *charIter;

    file = fopen(filename, "wb+");
    if (file == NULL) {
        printf("Could not open %s in mode %s\n", filename, "wb+");
        return false;
    }

    width = image_getWidth(img);
    height = image_getHeight(img);
    channels = 3;
    depth = 255;

    if (channels == 1) {
        fprintf(file, "P5\n");
    } else {
        fprintf(file, "P6\n");
    }
    fprintf(file, "#Created via PPM Export\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "%d\n", depth);

    charData = (unsigned char*) malloc(
            sizeof(unsigned char) * width * height * channels);

    charIter = charData;
    floatRIter = image_getR(img);
    floatGIter = image_getG(img);
    floatBIter = image_getB(img);

    for (ii = 0; ii < height; ii++) {
        for (jj = 0; jj < width; jj++) {
            *charIter = (unsigned char) ceil(
                    clamp(*floatRIter, 0, 1) * depth);
            charIter++;
            *charIter = (unsigned char) ceil(
                    clamp(*floatGIter, 0, 1) * depth);
            charIter++;
            *charIter = (unsigned char) ceil(
                    clamp(*floatBIter, 0, 1) * depth);
            charIter++;

            floatRIter++;
            floatGIter++;
            floatBIter++;
        }
    }

    bool writeResult = File_write(file, charData,
                                  width * channels * sizeof(unsigned char), height);

    free(charData);
    fflush(file);
    fclose(file);

    return true;
}

void test_images() {
    Image_t* inputImg = PPM_import("../resources/source/computer_programming.ppm");

    for (int i = 0; i < 300; i++) {
        image_setPixel(inputImg, i, 100, 0, float(i) / 300);
        image_setPixel(inputImg, i, 100, 1, float(i) / 300);
        image_setPixel(inputImg, i, 100, 2, float(i) / 200);
    }
    PPM_export("../resources/results/test_output.ppm", inputImg);
    image_delete(inputImg);

    Image_t* newImg = PPM_import("../resources/results/test_output.ppm");
    inputImg = PPM_import("../resources/source/computer_programming.ppm");
    if (image_is_same(inputImg, newImg))
        printf("Img uguali\n");
    else
        printf("Img diverse\n");

    image_delete(newImg);
    image_delete(inputImg);
}

//    Convert RGB image to grayscale
Image_t* PPMtoGrayscale(Image_t* inputImg){
    Image_t* outImg = new_image(image_getWidth(inputImg), image_getHeight(inputImg), 1);
    int width = image_getWidth(inputImg);
    int height = image_getHeight(inputImg);
    float r_val ;
    float g_val;
    float b_val;
    float gray_val;
    for (int col = 0; col < width; col++) {
        for (int row = 0; row < height; row++) {
            r_val = image_getPixel(inputImg, col, row, 0);
            g_val = image_getPixel(inputImg, col, row, 1);
            b_val = image_getPixel(inputImg, col, row, 2);
            gray_val = 0.3*r_val + 0.59*g_val + 0.11*b_val;
            image_setPixel(outImg, col, row, 0, gray_val);
        }

    }
//    PPM_export("../resources/source/grayImage.ppm", outImg);
    return outImg;
}