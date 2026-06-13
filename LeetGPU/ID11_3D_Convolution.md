# LeetGPU 3D Convolution 解题思路

> **难度**：medium  
> **题号**：11  
> **目标**：根据题目给定输入，在 CUDA 中实现 `3D Convolution`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) {}
```

---

## 2. 题目摘要

```text
Implement a program that performs a 3D convolution operation. Given a 3D input volume and a 3D kernel (filter), compute the convolved
 output. The convolution should use a "valid" boundary condition (no padding).

 For a 3D convolution, the output at position \((i,j,k)\) is given by:

 \[
 output(i,j,k) = \sum_{d=0}^{K_d-1} \sum_{r=0}^{K_r-1} \sum_{c=0}^{K_c-1} input(i+d,j+r,k+c) \cdot kernel(d,r,c)
 \]

 The input consists of:

 input: A 3D volume of 32-bit floats, as a 1D array (row-major, then depth).

 kernel: A 3D kernel of 32-bit floats, as a 1D array (row-major, then depth).

 input_depth,
 input_rows,
 input_cols: Dimensions of the input.
```

---

## 3. 核心数学/算法公式

三维卷积：每个输出体素累加 depth/row/col 三个方向的邻域。

---

## 4. CUDA 并行划分

- 使用 3D grid 或把线性 id 解码成 z,y,x。
- 一个线程负责一个 output[z,y,x]。
- 小 kernel 直接循环，大 kernel 可用 shared memory tile。

---

## 5. 推荐解法步骤

1. 计算当前输出体素坐标。
2. 三重循环遍历 kernel_depth/kernel_rows/kernel_cols。
3. 检查输入体素是否越界。
4. 累加并写回 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算当前输出体素坐标。
  - 三重循环遍历 kernel_depth/kernel_rows/kernel_cols。
  - 检查输入体素是否越界。
  - 累加并写回 output。
```

---

## 7. 复杂度分析

O(D×H×W×KD×KH×KW)。

---

## 8. 常见错误

- 三维 row-major 索引：((z*rows)+r)*cols+c。
- padding/边界约定要统一。
- kernel 索引维度顺序容易写反。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

