#include "kernels.h"
#include <math.h>

__global__ void rgbToGrayscaleKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int grayIndex = y * width + x;
        int rgbIndex = grayIndex * 3;

        unsigned char b = input[rgbIndex];
        unsigned char g = input[rgbIndex + 1];
        unsigned char r = input[rgbIndex + 2];

        output[grayIndex] = static_cast<unsigned char>(
            0.299f * r +
            0.587f * g +
            0.114f * b
        );
    }
}

__global__ void gaussianBlurKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        output[y * width + x] = input[y * width + x];
        return;
    }

    int sum = 0;

    sum += input[(y - 1) * width + (x - 1)] * 1;
    sum += input[(y - 1) * width + x] * 2;
    sum += input[(y - 1) * width + (x + 1)] * 1;

    sum += input[y * width + (x - 1)] * 2;
    sum += input[y * width + x] * 4;
    sum += input[y * width + (x + 1)] * 2;

    sum += input[(y + 1) * width + (x - 1)] * 1;
    sum += input[(y + 1) * width + x] * 2;
    sum += input[(y + 1) * width + (x + 1)] * 1;

    output[y * width + x] =
        static_cast<unsigned char>(sum / 16);
}

__global__ void sobelEdgeKernel(
    const unsigned char *input,
    unsigned char *output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        output[y * width + x] = 0;
        return;
    }

    int gx = 0;
    int gy = 0;

    gx += -1 * input[(y - 1) * width + (x - 1)];
    gx +=  0 * input[(y - 1) * width + x];
    gx +=  1 * input[(y - 1) * width + (x + 1)];

    gx += -2 * input[y * width + (x - 1)];
    gx +=  0 * input[y * width + x];
    gx +=  2 * input[y * width + (x + 1)];

    gx += -1 * input[(y + 1) * width + (x - 1)];
    gx +=  0 * input[(y + 1) * width + x];
    gx +=  1 * input[(y + 1) * width + (x + 1)];

    gy += -1 * input[(y - 1) * width + (x - 1)];
    gy += -2 * input[(y - 1) * width + x];
    gy += -1 * input[(y - 1) * width + (x + 1)];

    gy +=  0 * input[y * width + (x - 1)];
    gy +=  0 * input[y * width + x];
    gy +=  0 * input[y * width + (x + 1)];

    gy +=  1 * input[(y + 1) * width + (x - 1)];
    gy +=  2 * input[(y + 1) * width + x];
    gy +=  1 * input[(y + 1) * width + (x + 1)];

    float magnitude = sqrtf(
        static_cast<float>(gx * gx + gy * gy)
    );

    if (magnitude > 255.0f)
    {
        magnitude = 255.0f;
    }

    output[y * width + x] =
        static_cast<unsigned char>(magnitude);
}

void launchGrayscaleKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
)
{
    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    rgbToGrayscaleKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}

void launchGaussianBlurKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
)
{
    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    gaussianBlurKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}

void launchSobelEdgeKernel(
    const unsigned char *d_input,
    unsigned char *d_output,
    int width,
    int height
)
{
    dim3 block(16, 16);

    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    sobelEdgeKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}
