# LeetGPU Swish-Gated Linear Unit (SwiGLU) 解题思路

> **难度**：Easy  
> **题型**：Gated Activation / 输入拆半  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N);
> ```

---

## 1. 公式

输入长度 `N` 为偶数，拆成两半：

```text
x1 = input[0 .. N/2-1]
x2 = input[N/2 .. N-1]
```

题目定义：

```text
SiLU(x1) = x1 * sigmoid(x1)
SwiGLU(x1, x2) = SiLU(x1) * x2
```

输出长度：

```text
N / 2
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[halfN + i];
        float sig = 1.0f / (1.0f + expf(-x1));
        float silu = x1 * sig;
        output[i] = silu * x2;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

---

## 3. 常见错误

- 输出长度是 `N/2`，不是 `N`。
- 本题是对第一半 `x1` 做 SiLU，再乘第二半 `x2`。
- 输入拆半索引：`input[i]` 和 `input[halfN+i]`。
