# LeetGPU Count 2D Array Element 解题思路

> **难度**：medium  
> **题号**：44  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Count 2D Array Element`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {}
```

---

## 2. 题目摘要

```text
Write a GPU program that counts the number of elements with the integer value k in an 2D array of 32-bit integers.
 The program should count the number of elements with k in an 2D array.
 You are given an input 2D array input of length N x M and integer k.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input [[1, 2, 3],
 [4, 5, 1]]
 k = 1
Output: output = 2
```

---

## 3. 核心数学/算法公式

统计二维数组中各整数值出现次数，本质仍是 histogram。

---

## 4. CUDA 并行划分

- 把二维数组展平成 N×M 个元素处理。
- shared memory local histogram。
- 最后合并到 output[0..K-1]。

---

## 5. 推荐解法步骤

1. total=N*M。
2. 每个线程遍历线性 idx。
3. value=input[idx]。
4. atomicAdd local[value]。
5. 合并 local 到 global。

---

## 6. 伪代码骨架

```text
solve(...):
  - total=N*M。
  - 每个线程遍历线性 idx。
  - value=input[idx]。
  - atomicAdd local[value]。
  - 合并 local 到 global。
```

---

## 7. 复杂度分析

O(N×M + blocks×K)。

---

## 8. 常见错误

- 二维输入仍然 row-major，线性 idx 可直接访问。
- K 与维度 M 不要混淆。
- 输出清零。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

