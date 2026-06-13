# LeetGPU Matrix Copy 解题思路

> **难度**：Easy  
> **题型**：Memory copy / Elementwise  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, float* B, int N);
> ```

---

## 1. 题目要求

复制一个 `N × N` 矩阵：

```text
B[i] = A[i]
```

总元素数：

```text
total = N * N
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* A, float* B, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        B[idx] = A[idx];
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

- Matrix Copy 是最简单的 global memory read/write 练习。
- 也可以用 `cudaMemcpy`，但题目要求写 GPU program，所以用 kernel 展示并行复制。
