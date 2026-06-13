# LeetGPU RGB to Grayscale 解题思路

> **难度**：Easy  
> **题型**：图像 Elementwise / RGB layout  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int width, int height);
> ```

---

## 1. 题目要求

输入是一维 RGB 数组：

```text
R, G, B, R, G, B, ...
```

每个像素输出一个灰度值：

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

---

## 2. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel < total_pixels) {
        int base = pixel * 3;
        float r = input[base + 0];
        float g = input[base + 1];
        float b = input[base + 2];
        output[pixel] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
```

---

## 3. 常见错误

- RGB 每个像素 3 个 float，不是 4 个。
- 输出长度是 `width * height`，不是 `width * height * 3`。
- 权重顺序是 R/G/B：`0.299, 0.587, 0.114`。
