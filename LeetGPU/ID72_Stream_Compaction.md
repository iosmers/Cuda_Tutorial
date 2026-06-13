# LeetGPU Stream Compaction 解题思路

> **难度**：medium  
> **题号**：72  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Stream Compaction`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// A, out are device pointers
extern "C" void solve(const float* A, int N, float* out) {}
```

---

## 2. 题目摘要

```text
Given a 1D array A of N 32-bit floating point numbers, compact all
 positive elements (A[i] > 0) to the front of the output array out,
 preserving their original relative order. Fill any remaining positions with 0.0.
 Stream compaction is a fundamental GPU primitive used throughout rendering, sparse computation,
 and collision detection.

Implementation Requirements

 Use only native GPU features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The first k positions of out must contain the k elements of
 A where A[i] > 0, in their original order

 Positions k through N−1 of out must be 0.0

 Elements where A[i] = 0.0 are not selected
```

---

## 3. 核心数学/算法公式

过滤数组，保留满足 predicate 的元素并连续写到 out。

---

## 4. CUDA 并行划分

- 典型流程：flag -> prefix sum -> scatter。
- flag[i]=predicate(A[i])。
- 位置 pos=exclusive_scan(flag)[i]，若 flag 写 out[pos]=A[i]。

---

## 5. 推荐解法步骤

1. 计算 flags。
2. 对 flags 做 exclusive prefix sum。
3. scatter 有效元素。
4. 如果需要数量，最后 count=scan_last+flag_last。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 flags。
  - 对 flags 做 exclusive prefix sum。
  - scatter 有效元素。
  - 如果需要数量，最后 count=scan_last+flag_last。
```

---

## 7. 复杂度分析

O(N)。

---

## 8. 常见错误

- predicate 要按题目定义，常见是 A[i] != 0 或 A[i] > 0。
- exclusive/inclusive scan 位置不要混。
- 输出长度可能小于 N。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

