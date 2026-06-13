# LeetGPU Max Subarray Sum 解题思路

> **难度**：medium  
> **题号**：51  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Max Subarray Sum`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int window_size) {}
```

---

## 2. 题目摘要

```text
Implement a program that computes the maximum sum of any contiguous subarray of length exactly window_size. You are given an array input of length N consisting of 32-bit signed integers, and an integer window_size.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input = [1, 2, 4, 2, 3], window_size = 2
Output: output = 6

Example 2:

Input: input = [-1, -4, -2, 1], window_size = 3
```

---

## 3. 核心数学/算法公式

题目接口带 window_size，通常求每个固定窗口的最大子数组/窗口和，可用滑动窗口或 prefix sum。

---

## 4. CUDA 并行划分

- 如果是固定窗口和最大值：先 prefix sum，再每个线程算一个窗口 sum。
- 如果输出每个窗口最大，可一个线程负责一个起点，窗口内循环。
- N 大时 prefix sum + reduction 找最大更高效。

---

## 5. 推荐解法步骤

1. 构建 prefix sum。
2. window_sum[i]=prefix[i+w]-prefix[i]。
3. 并行 reduce max window_sum。
4. 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 构建 prefix sum。
  - window_sum[i]=prefix[i+w]-prefix[i]。
  - 并行 reduce max window_sum。
  - 写 output。
```

---

## 7. 复杂度分析

prefix 方法 O(N)。

---

## 8. 常见错误

- 确认 output 是一个标量还是数组，按题目样例决定。
- window_size 边界 N-window_size+1。
- 整数和可能较大，必要时用更宽类型中间值。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

