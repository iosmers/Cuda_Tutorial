NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17
# 本机 GPU：nvcc -arch=native；无 GPU 或交叉编译时改为具体 sm_XX
ARCH ?= native
NVCC_FLAGS += -arch=$(ARCH)

TARGET := stage2_bench
SRCS := main.cu kernels.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) kernels.cuh cuda_check.h
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
