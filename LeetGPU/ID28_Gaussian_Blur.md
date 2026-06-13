# LeetGPU Gaussian Blur 解题思路

> **难度**：medium  
> **题号**：28  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Gaussian Blur`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_rows,
                      int input_cols, int kernel_rows, int kernel_cols) {}
```

---

## 2. 题目摘要

```text
Implement a program that applies a Gaussian blur filter to a 2D image. Given an input image represented as a floating-point array and a Gaussian kernel, the program should compute the convolution of the image with the kernel.
 All inputs and outputs are stored in row-major order.

 The Gaussian blur is performed by convolving each pixel with a weighted average of its neighbors, where the weights are determined by the Gaussian kernel. For each output pixel at position (i, j), the value is calculated as:

 \[ output[i, j] = \sum_{m=-k_h/2}^{k_h/2} \sum_{n=-k_w/2}^{k_w/2} input[i+m, j+n] \times kernel[m+k_h/2, n+k_w/2] \]

 where \(k_h\) and \(k_w\) are the kernel height and width.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in the output array

 Handle boundary conditions by using zero-padding (treat values outside the image boundary as zeros)
```

---

## 3. 核心数学/算法公式

Gaussian blur 是 2D convolution，kernel 通常已给定。

---

## 4. CUDA 并行划分

- 一个线程负责一个输出像素。
- 每个线程遍历 kernel_rows × kernel_cols。
- 优化时把输入 tile 和 halo 放 shared memory。

---

## 5. 推荐解法步骤

1. 计算 output[row,col]。
2. 遍历高斯核邻域。
3. 越界位置按 0-padding/跳过处理。
4. 累加 input * kernel。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 output[row,col]。
  - 遍历高斯核邻域。
  - 越界位置按 0-padding/跳过处理。
  - 累加 input * kernel。
```

---

## 7. 复杂度分析

O(H×W×KH×KW)。

---

## 8. 常见错误

- Gaussian kernel 是否已归一化由题目输入决定，不要重复归一。
- kernel 中心 offset = rows/2, cols/2。
- 边界处理要和示例一致。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

