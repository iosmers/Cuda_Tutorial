# LeetGPU Leaky ReLU 解题思路

> **难度**：Easy  
> **题型**：Activation / Elementwise with branch  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N);
> ```

---

## 1. 公式

题目固定：

```text
alpha = 0.01
```

```text
f(x) = x,            if x > 0
f(x) = 0.01 * x,    if x <= 0
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.01f * x;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 常见错误

- 忘记 alpha 固定为 `0.01`。
- 把 `x == 0` 单独处理没有必要，结果仍为 0。
