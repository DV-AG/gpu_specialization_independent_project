NVCC = nvcc

TARGET = cuda_image_processor

SRC = src/main.cu src/kernels.cu

CXXFLAGS = -std=c++17 -O2

OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f results/execution_times.csv
	rm -f data/output/grayscale/*
	rm -f data/output/blurred/*
	rm -f data/output/edges/*

.PHONY: all run clean
