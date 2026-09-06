#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "kernels.h"

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                      \
do                                                                            \
{                                                                             \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
        return 1;                                                             \
    }                                                                         \
} while (0)

int main()
{
    std::string inputDir = "data/input";
    std::string grayDir = "data/output/grayscale";
    std::string blurDir = "data/output/blurred";
    std::string edgeDir = "data/output/edges";

    fs::create_directories(grayDir);
    fs::create_directories(blurDir);
    fs::create_directories(edgeDir);
    fs::create_directories("results");

    std::ofstream csv("results/execution_times.csv");

    if (!csv.is_open())
    {
        std::cerr << "Could not open results/execution_times.csv\n";
        return 1;
    }

    csv << "image,width,height,grayscale_ms,blur_ms,sobel_ms,total_gpu_ms\n";

    int processedImages = 0;

    for (const auto &entry : fs::directory_iterator(inputDir))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string extension = entry.path().extension().string();

        if (extension != ".tiff" &&
            extension != ".tif" &&
            extension != ".jpg" &&
            extension != ".jpeg" &&
            extension != ".png")
        {
            continue;
        }

        std::string inputPath = entry.path().string();
        std::string filename = entry.path().stem().string();

        cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);

        if (image.empty())
        {
            std::cerr << "Could not read: " << inputPath << "\n";
            continue;
        }

        if (!image.isContinuous())
        {
            image = image.clone();
        }

        int width = image.cols;
        int height = image.rows;

        size_t rgbSize =
            static_cast<size_t>(width) *
            static_cast<size_t>(height) *
            3 *
            sizeof(unsigned char);

        size_t graySize =
            static_cast<size_t>(width) *
            static_cast<size_t>(height) *
            sizeof(unsigned char);

        unsigned char *d_rgb = nullptr;
        unsigned char *d_gray = nullptr;
        unsigned char *d_blur = nullptr;
        unsigned char *d_edges = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_rgb),
            rgbSize
        ));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_gray),
            graySize
        ));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_blur),
            graySize
        ));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_edges),
            graySize
        ));

        CUDA_CHECK(cudaMemcpy(
            d_rgb,
            image.data,
            rgbSize,
            cudaMemcpyHostToDevice
        ));

        cudaEvent_t start;
        cudaEvent_t stop;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float grayscaleTime = 0.0f;
        float blurTime = 0.0f;
        float sobelTime = 0.0f;

        CUDA_CHECK(cudaEventRecord(start));

        launchGrayscaleKernel(
            d_rgb,
            d_gray,
            width,
            height
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(
            &grayscaleTime,
            start,
            stop
        ));

        CUDA_CHECK(cudaEventRecord(start));

        launchGaussianBlurKernel(
            d_gray,
            d_blur,
            width,
            height
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(
            &blurTime,
            start,
            stop
        ));

        CUDA_CHECK(cudaEventRecord(start));

        launchSobelEdgeKernel(
            d_blur,
            d_edges,
            width,
            height
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(
            &sobelTime,
            start,
            stop
        ));

        cv::Mat grayImage(
            height,
            width,
            CV_8UC1
        );

        cv::Mat blurImage(
            height,
            width,
            CV_8UC1
        );

        cv::Mat edgeImage(
            height,
            width,
            CV_8UC1
        );

        CUDA_CHECK(cudaMemcpy(
            grayImage.data,
            d_gray,
            graySize,
            cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            blurImage.data,
            d_blur,
            graySize,
            cudaMemcpyDeviceToHost
        ));

        CUDA_CHECK(cudaMemcpy(
            edgeImage.data,
            d_edges,
            graySize,
            cudaMemcpyDeviceToHost
        ));

        cv::imwrite(
            grayDir + "/" + filename + "_gray.png",
            grayImage
        );

        cv::imwrite(
            blurDir + "/" + filename + "_blur.png",
            blurImage
        );

        cv::imwrite(
            edgeDir + "/" + filename + "_edges.png",
            edgeImage
        );

        float totalTime =
            grayscaleTime +
            blurTime +
            sobelTime;

        csv
            << filename << ","
            << width << ","
            << height << ","
            << grayscaleTime << ","
            << blurTime << ","
            << sobelTime << ","
            << totalTime
            << "\n";

        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));
        CUDA_CHECK(cudaFree(d_blur));
        CUDA_CHECK(cudaFree(d_edges));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        processedImages++;

        std::cout
            << "Processed "
            << processedImages
            << ": "
            << filename
            << " ("
            << width
            << "x"
            << height
            << ")\n";
    }

    csv.close();

    std::cout
        << "Finished processing "
        << processedImages
        << " images.\n";

    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
