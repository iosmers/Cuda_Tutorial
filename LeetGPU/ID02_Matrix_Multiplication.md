# LeetGPU Matrix Multiplication 解题思路

> **难度**：Easy  
> **题型**：Naive GEMM / 二维线程映射  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);
> ```
>
> `A` 是 `M × N`，`B` 是 `N × K`，输出 `C` 是 `M × K`。

---

## 1. 数学公式

```text
C[row, col] = sum_t A[row, t] * B[t, col]
```

其中：

```text
row = 0..M-1
col = 0..K-1
t   = 0..N-1
```

---

## 2. CUDA 并行思路

最简单的 baseline：

```text
一个线程负责 C 的一个元素
```

使用二维 block：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

矩阵都是 row-major：

```cpp
A[row * N + t]
B[t * K + col]
C[row * K + col]
```

---

## 3. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int t = 0; t < N; ++t) {
            sum += A[row * N + t] * B[t * K + col];
        }
        C[row * K + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

---

## 4. 讲课重点

- 这是 GEMM 的 naive 版，为后面的 shared memory tiled GEMM 铺垫。
- `grid.x` 对应列方向，`grid.y` 对应行方向。
- 最容易错的是 `B[t * K + col]`，因为 B 的列数是 `K`。
