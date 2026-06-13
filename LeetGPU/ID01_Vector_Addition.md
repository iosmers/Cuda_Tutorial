# LeetGPU Vector Addition 解题思路

> **难度**：Easy  
> **题型**：Elementwise / 一线程一元素  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* B, float* C, int N);
> ```

---

## 1. 题目要求

给定两个长度为 `N` 的 float 向量：

```text
A = [a0, a1, ...]
B = [b0, b1, ...]
```

输出：

```text
C[i] = A[i] + B[i]
```

这是 CUDA 入门最经典的一线程一元素题。

---

## 2. CUDA 并行思路

把 CPU for 循环：

```cpp
for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
```

改成 GPU 并行：

```text
thread i 负责 C[i]
```

关键是线性线程编号：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

并且一定要加边界判断：

```cpp
if (i < N) { ... }
```

---

## 3. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

---

## 4. 讲课重点

- 这是第一个 CUDA kernel 的标准模板。
- `blocksPerGrid = (N + threads - 1) / threads` 是向上取整。
- 即使 block 数向上取整，也必须用 `if (i < N)` 防越界。
