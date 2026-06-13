# LeetGPU Matrix Addition 解题思路

> **难度**：Easy  
> **题型**：二维矩阵展平 / Elementwise  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* B, float* C, int N);
> ```

---

## 1. 题目要求

两个 `N × N` 矩阵逐元素相加：

```text
C[row, col] = A[row, col] + B[row, col]
```

由于矩阵是 row-major，一维下标：

```cpp
idx = row * N + col
```

也可以直接把整个矩阵看成长度 `N*N` 的一维数组。

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

- 二维矩阵很多时候可以先展平成一维处理。
- 这题和 Vector Addition 代码几乎一样，只是总元素数变成 `N*N`。
