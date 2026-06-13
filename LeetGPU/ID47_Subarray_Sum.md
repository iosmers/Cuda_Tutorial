# LeetGPU Subarray Sum 解题思路

> **难度**：medium  
> **题号**：47  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Subarray Sum`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {}
```

---

## 2. 题目摘要

```text
Implement a program that computes the sum of a subarray of 32-bit integers.
 You are given an input array input of length N, and two indices S and E.
 S and E are inclusive, 0-based start and end indices — compute the sum of input[S..E].

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6

Example 2:
```

---

## 3. 核心数学/算法公式

求一维区间 [S,E] 的元素和。

---

## 4. CUDA 并行划分

- 这是 reduction over range。
- 线程只遍历 S..E。
- block reduction 后 atomicAdd 到 output。

---

## 5. 推荐解法步骤

1. cudaMemset output。
2. total=E-S+1。
3. grid-stride loop 访问 input[S+idx]。
4. 归约写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - cudaMemset output。
  - total=E-S+1。
  - grid-stride loop 访问 input[S+idx]。
  - 归约写 output。
```

---

## 7. 复杂度分析

O(E-S+1)。

---

## 8. 常见错误

- 确认 E 是否 inclusive，LeetGPU 通常按闭区间。
- 输出清零。
- 避免访问 S/E 越界。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

