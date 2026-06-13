# LeetGPU 3D Subarray Sum 解题思路

> **难度**：medium  
> **题号**：49  
> **目标**：根据题目给定输入，在 CUDA 中实现 `3D Subarray Sum`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {}
```

---

## 2. 题目摘要

```text
Implement a program that computes the sum of a 3D subarray of 32-bit integers.
 You are given an input 3D array input of length N x M x K, and two depth indices S_DEP and E_DEP and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL.
 S_DEP, E_DEP, S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input = [[[1, 2, 3],
 [4, 5, 1]],
 [[1, 1, 1],
 [2, 2, 2]]]
```

---

## 3. 核心数学/算法公式

求三维子块的元素和。

---

## 4. CUDA 并行划分

- 把子块线性化做 reduction。
- idx 解码 dep,row,col。
- block reduction + atomicAdd。

---

## 5. 推荐解法步骤

1. 计算子块 D/H/W。
2. idx -> dep,row,col。
3. 访问 input[((S_DEP+d)*M + (S_ROW+r))*K + (S_COL+c)]。
4. 归约写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算子块 D/H/W。
  - idx -> dep,row,col。
  - 访问 input[((S_DEP+d)*M + (S_ROW+r))*K + (S_COL+c)]。
  - 归约写 output。
```

---

## 7. 复杂度分析

O(subD×subH×subW)。

---

## 8. 常见错误

- N/M/K 分别是 depth/rows/cols。
- 闭区间 +1。
- 输出清零。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

