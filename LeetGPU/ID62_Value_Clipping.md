# LeetGPU Value Clipping 解题思路

> **难度**：Easy  
> **题型**：Elementwise clamp  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, float lo, float hi, int N);
> ```

---

## 1. 公式

```text
output[i] = lo,        if input[i] < lo
output[i] = hi,        if input[i] > hi
output[i] = input[i],  otherwise
```

也就是：

```text
output[i] = min(max(input[i], lo), hi)
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        if (x < lo) x = lo;
        if (x > hi) x = hi;
        output[i] = x;
    }
}

extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```

---

## 3. 讲课重点

- 这是带上下界的 elementwise map。
- 可以用 `fminf(fmaxf(x, lo), hi)` 写得更短。
