# LeetGPU Color Inversion 解题思路

> **难度**：Easy  
> **题型**：图像 Elementwise / RGBA 索引  
> **接口**：
>
> ```cpp
> extern "C" void solve(unsigned char* image, int width, int height);
> ```

---

## 1. 题目要求

图像以一维数组存储，每个像素 4 个 `unsigned char`：

```text
R, G, B, A
```

颜色反转：

```text
R = 255 - R
G = 255 - G
B = 255 - B
A 不变
```

---

## 2. CUDA 并行思路

一个线程负责一个像素：

```cpp
int pixel = blockIdx.x * blockDim.x + threadIdx.x;
int base = pixel * 4;
```

然后修改：

```cpp
image[base + 0]
image[base + 1]
image[base + 2]
```

不动 alpha：

```cpp
image[base + 3]
```

---

## 3. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel < total_pixels) {
        int base = pixel * 4;
        image[base + 0] = 255 - image[base + 0];
        image[base + 1] = 255 - image[base + 1];
        image[base + 2] = 255 - image[base + 2];
        // image[base + 3] alpha unchanged
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

---

## 4. 常见错误

- 把线程数量开成 `width*height*4` 后又按 pixel 处理，会重复处理。
- 错误地反转 alpha 通道。
- 忘记 `pixel < width * height` 边界判断。
