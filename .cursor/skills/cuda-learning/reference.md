# CUDA 学习 — 速查

## 编译（示例）

```bash
nvcc -O2 -std=c++17 -arch=native main.cu -o main
```

按目标 GPU 将 `-arch=native` 改为具体 `sm_XX`，或与项目 Makefile 保持一致。

## CUDA_CHECK 宏（示例）

```cpp
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)
```

对 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel` 后均可包一层；kernel 后用 `CUDA_CHECK(cudaDeviceSynchronize())` 捕获异步错误。

## 常见笔误（教学时主动核对）

- 线程索引：`blockIdx.x * blockDim.x + threadIdx.x`（注意是 `threadIdx`，不是 `threadId`）。
- 设备函数内不能调用 host API；`__global__` 由 host 启动。

## 官方与样本

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
