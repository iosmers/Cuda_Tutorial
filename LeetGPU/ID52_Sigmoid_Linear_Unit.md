# LeetGPU Sigmoid Linear Unit (SiLU) 解题思路

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
sigmoid(x) = 1 / (1 + exp(-x))
SiLU(x) = x * sigmoid(x)
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        output[i] = x * sig;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

SiLU 是很多现代网络中的 activation，也是 SwiGLU 的组成部分。
