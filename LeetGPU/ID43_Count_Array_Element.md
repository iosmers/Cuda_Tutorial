# LeetGPU Count Array Element 解题思路

> **难度**：medium  
> **题号**：43  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Count Array Element`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int K) {}
```

---

## 2. 题目摘要

```text
Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers.
 The program should count the number of elements with k in an array.
 You are given an input array input of length N and integer k.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: [1, 2, 3, 4, 1], k = 1
Output: 2

Example 2:
```

---

## 3. 核心数学/算法公式

统计一维数组中 0..K-1 每个值出现次数，等价 histogram。

---

## 4. CUDA 并行划分

- 使用 shared memory 局部 histogram。
- 每个 block 先统计 local count，再合并到 global output。
- K 小时非常适合 shared memory。

---

## 5. 推荐解法步骤

1. cudaMemset output。
2. 初始化 shared hist。
3. grid-stride loop atomicAdd local[input[i]]。
4. block 内同步后 atomicAdd 到 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - cudaMemset output。
  - 初始化 shared hist。
  - grid-stride loop atomicAdd local[input[i]]。
  - block 内同步后 atomicAdd 到 output。
```

---

## 7. 复杂度分析

O(N + blocks×K)。

---

## 8. 常见错误

- 输出必须清零。
- input 值范围要按题目保证。
- global atomic 次数应尽量降到 blocks×K。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

