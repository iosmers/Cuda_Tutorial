# LeetGPU ReLU 解题思路

> **难度**：Easy  
> **题型**：Activation / Elementwise  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N);
> ```

---

## 1. 公式

```text
ReLU(x) = max(0, x)
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.0f;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

这是最典型的 elementwise activation：每个元素互不依赖。
