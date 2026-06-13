# LeetGPU Parallel Merge 解题思路

> **难度**：medium  
> **题号**：71  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Parallel Merge`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {}
```

---

## 2. 题目摘要

```text
Given two sorted arrays A of length M and B of length
 N, both containing 32-bit floating-point values in non-decreasing order, produce a
 single sorted array C of length M + N containing all elements of
 A and B in non-decreasing order.

Implementation Requirements

 Use only GPU native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final merged result must be stored in C

Example

Input:
 A = [1.0, 3.0, 5.0, 7.0], M = 4
 B = [2.0, 4.0, 6.0, 8.0], N = 4
```

---

## 3. 核心数学/算法公式

合并两个有序数组 A 和 B 到有序 C。

---

## 4. CUDA 并行划分

- 每个输出位置 r 独立通过二分找到来自 A/B 的划分。
- 或每个 A/B 元素二分计算 rank 后 scatter。
- merge path 是更高性能的分块方法。

---

## 5. 推荐解法步骤

1. 对每个 output rank r。
2. 二分 a_count，使 a_count + b_count = r。
3. 满足 A[a_count-1] <= B[b_count] 且 B[b_count-1] <= A[a_count]。
4. C[r]=max(left boundary)。

---

## 6. 伪代码骨架

```text
solve(...):
  - 对每个 output rank r。
  - 二分 a_count，使 a_count + b_count = r。
  - 满足 A[a_count-1] <= B[b_count] 且 B[b_count-1] <= A[a_count]。
  - C[r]=max(left boundary)。
```

---

## 7. 复杂度分析

O((M+N) log min(M,N)) baseline，merge-path 可近 O(M+N)。

---

## 8. 常见错误

- 重复值 tie-break 要稳定。
- 边界 a_count=0/M 和 b_count=0/N。
- 输入已经分别有序。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

