#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

__global__ void rgbToGrayscaleKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
);

__global__ void gaussianBlurKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
);

__global__ void sobelEdgeKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
);

void launchGrayscaleKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
);

void launchGaussianBlurKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
);

void launchSobelEdgeKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
);

#endif
