# CUDA-Accelerated Batch Image Processing Pipeline

 Overview

This project implements a GPU-accelerated batch image-processing pipeline using CUDA C++.

The program processes a collection of large images using custom CUDA kernels for:

- RGB/BGR to grayscale conversion
- Gaussian blur
- Sobel edge detection

The project is designed to demonstrate large-scale GPU image processing using CUDA and to measure the execution time of the individual GPU kernels.

The dataset used in this project contains 38 large aerial images from the USC-SIPI Image Database.



 Project Objective

The objective of this project is to process a dataset containing tens of large images using GPU computation.

Each image is:

1. Loaded from disk using OpenCV
2. Copied from host memory to GPU memory
3. Converted to grayscale using a CUDA kernel
4. Blurred using a Gaussian filter CUDA kernel
5. Processed using a Sobel edge-detection CUDA kernel
6. Copied back to host memory
7. Saved as output images
8. Timed using CUDA events

The program also records GPU kernel execution times in a CSV file.



 Dataset

This project uses the USC-SIPI Image Database Aerials dataset.

The dataset contains 38 aerial images of relatively large resolution.
