# LeetGPU 2D Convolution 解题思路

> **难度**：medium  
> **题号**：10  
> **目标**：根据题目给定输入，在 CUDA 中实现 `2D Convolution`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

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
Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved
 output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied
 where it fully overlaps with the input.

 Input (4x4)

 1

 2

 3

 4

 5

 6
```

---

## 3. 核心数学/算法公式

对二维输入做卷积/相关：每个 output[row,col] 累加邻域 input * kernel。

---

## 4. CUDA 并行划分

- 一个线程负责一个输出像素。
- 用 2D grid/block 映射 row/col。
- kernel 较小时直接 global memory；追求性能时用 shared memory tile 加 halo。

---

## 5. 推荐解法步骤

1. 计算输出尺寸或按题目约定保持同尺寸。
2. 每个线程遍历 kernel_rows × kernel_cols。
3. 对越界邻域做 0-padding 或跳过。
4. 写 output[row*out_cols+col]。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算输出尺寸或按题目约定保持同尺寸。
  - 每个线程遍历 kernel_rows × kernel_cols。
  - 对越界邻域做 0-padding 或跳过。
  - 写 output[row*out_cols+col]。
```

---

## 7. 复杂度分析

O(output_rows × output_cols × kernel_rows × kernel_cols)。

---

## 8. 常见错误

- 卷积是否翻转 kernel 要按题目定义；多数 LeetGPU 题按直接相关实现也可从示例确认。
- 边界 padding 索引容易错。
- input 是 row-major：input[r*input_cols+c]。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

