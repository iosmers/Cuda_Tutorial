# LeetGPU Sigmoid Activation 解题思路

> **难度**：Easy  
> **题型**：Activation / Elementwise  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* X, float* Y, int N);
> ```

---

## 1. 公式

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = X[i];
        Y[i] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

- `expf` 是 float 版本指数函数。
- Sigmoid 是 SiLU、门控激活、二分类模型的基础。
