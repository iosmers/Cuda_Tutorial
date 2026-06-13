# LeetGPU Interleave Arrays 解题思路

> **难度**：Easy  
> **题型**：索引变换 / 一个线程写两个位置  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* B, float* output, int N);
> ```

---

## 1. 题目要求

```text
output = [A0, B0, A1, B1, A2, B2, ...]
```

公式：

```text
output[2*i]     = A[i]
output[2*i + 1] = B[i]
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[2 * i] = A[i];
        output[2 * i + 1] = B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 常见错误

- 输出长度是 `2N`。
- 不要让第 `idx` 个线程直接写 `output[idx]`，除非额外判断奇偶。
- 一个线程写两个连续位置最简单。
