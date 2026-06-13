# LeetGPU 1D Convolution 解题思路

> **难度**：Easy  
> **题型**：一维邻域计算 / valid convolution  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, const float* kernel, float* output,
>                       int input_size, int kernel_size);
> ```

---

## 1. 题目要求

valid 边界条件：kernel 只在完全覆盖 input 的位置计算。

输出长度：

```text
output_size = input_size - kernel_size + 1
```

题目示例说明这里实际采用直接相关形式：

```text
output[i] = sum_j input[i+j] * kernel[j]
```

不需要翻转 kernel。

---

## 2. CUDA 并行思路

一个线程负责一个输出位置 `i`：

```text
thread i 计算 output[i]
```

每个线程内部循环 `kernel_size` 次。

---

## 3. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (i < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output,
                                                              input_size, kernel_size);
    cudaDeviceSynchronize();
}
```

---

## 4. 常见错误

- 把输出长度写成 `input_size`。
- 错误翻转 kernel，导致和题目示例不一致。
- 忘记 valid 模式下只计算完全覆盖的位置。
