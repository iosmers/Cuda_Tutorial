# LeetGPU Gaussian Error Gated Linear Unit (GEGLU) 解题思路

> **难度**：Easy  
> **题型**：Gated Activation / GELU  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N);
> ```

---

## 1. 公式

输入长度 `N` 为偶数，拆成：

```text
x1 = input[0 .. N/2-1]
x2 = input[N/2 .. N-1]
```

GELU：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

GEGLU：

```text
output[i] = x1[i] * GELU(x2[i])
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[halfN + i];
        float gelu = 0.5f * x2 * (1.0f + erff(x2 * 0.7071067811865476f));
        output[i] = x1 * gelu;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

---

## 3. 常见错误

- GELU 作用在第二半 `x2`，不是第一半。
- 输出长度是 `N/2`。
- `erff` 是 float 版本，适合 CUDA float。
