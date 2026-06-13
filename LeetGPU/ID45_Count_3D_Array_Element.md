# LeetGPU Count 3D Array Element 解题思路

> **难度**：medium  
> **题号**：45  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Count 3D Array Element`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {}
```

---

## 2. 题目摘要

```text
Write a GPU program that counts the number of elements with the integer value p in an 3D array of 32-bit integers.
 The program should count the number of elements with p in an 3D array.
 You are given an input 3D array input of length N x M x K and integer p.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: input [[[1, 2, 3],
 [4, 5, 1]],
 [[1, 1, 1],
 [2, 2, 2]]]
```

---

## 3. 核心数学/算法公式

统计三维数组中各整数值出现次数，仍然是 histogram。

---

## 4. CUDA 并行划分

- 把 N×M×K 个元素线性化。
- shared memory 局部 histogram。
- 合并到 output[0..P-1]。

---

## 5. 推荐解法步骤

1. total=N*M*K。
2. grid-stride loop 扫 input[idx]。
3. local histogram 累加。
4. global histogram 合并。

---

## 6. 伪代码骨架

```text
solve(...):
  - total=N*M*K。
  - grid-stride loop 扫 input[idx]。
  - local histogram 累加。
  - global histogram 合并。
```

---

## 7. 复杂度分析

O(N×M×K + blocks×P)。

---

## 8. 常见错误

- 参数 K 既是维度名，P 是类别数，别写混。
- 输出必须清零。
- local hist size 是 P。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

