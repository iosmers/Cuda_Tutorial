# LeetGPU 2D Subarray Sum 解题思路

> **难度**：medium  
> **题号**：48  
> **目标**：根据题目给定输入，在 CUDA 中实现 `2D Subarray Sum`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL) {}
```

---

## 2. 题目摘要

```text
Implement a program that computes the sum of a 2D subarray of 32-bit integers.
 You are given an input 2D array input of length N x M, and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL.
 S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input = [[1, 2, 3],
 [4, 5, 1]]

 S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2
```

---

## 3. 核心数学/算法公式

求二维子矩形 [S_ROW,E_ROW]×[S_COL,E_COL] 的和。

---

## 4. CUDA 并行划分

- 把子矩形展平成 total 个元素做 reduction。
- 每个线程把线性 idx 映射到 row/col。
- block reduction + atomicAdd。

---

## 5. 推荐解法步骤

1. height=E_ROW-S_ROW+1, width=E_COL-S_COL+1。
2. idx -> r=idx/width, c=idx%width。
3. 访问 input[(S_ROW+r)*M + (S_COL+c)]。
4. 归约。

---

## 6. 伪代码骨架

```text
solve(...):
  - height=E_ROW-S_ROW+1, width=E_COL-S_COL+1。
  - idx -> r=idx/width, c=idx%width。
  - 访问 input[(S_ROW+r)*M + (S_COL+c)]。
  - 归约。
```

---

## 7. 复杂度分析

O(height×width)。

---

## 8. 常见错误

- 二维数组列数是 M。
- 闭区间长度要 +1。
- 输出清零。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

