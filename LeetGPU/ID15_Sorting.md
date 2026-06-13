# LeetGPU Sorting 解题思路

> **难度**：hard  
> **题号**：15  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Sorting`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// data is device pointer
extern "C" void solve(float* data, int N) {}
```

---

## 2. 题目摘要

```text
Write a program that sorts an array of 32-bit floating-point numbers in ascending order. You are free to choose any sorting algorithm.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The sorted result must be stored back in the input data array

Example

Input: data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0], N = 6
Output: data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]

Constraints

 1 ≤ N ≤ 1,000,000
```

---

## 3. 核心数学/算法公式

对 float 数组升序或按题目要求排序；通用 GPU 排序可用 bitonic sort 或 radix sort。

---

## 4. CUDA 并行划分

- N 较小/教学实现可用 bitonic sort。
- 每一层 compare-and-swap 由线程并行处理一对元素。
- N 非 2 的幂时补 +inf 或在比较时做边界判断。

---

## 5. 推荐解法步骤

1. 按 nextPow2(N) 组织 bitonic 网络。
2. 外层 size 从 2 翻倍到 P。
3. 内层 stride 从 size/2 递减。
4. 每轮 kernel 对 pair 做 compare-swap。

---

## 6. 伪代码骨架

```text
solve(...):
  - 按 nextPow2(N) 组织 bitonic 网络。
  - 外层 size 从 2 翻倍到 P。
  - 内层 stride 从 size/2 递减。
  - 每轮 kernel 对 pair 做 compare-swap。
```

---

## 7. 复杂度分析

bitonic O(N log²N)。

---

## 8. 常见错误

- float 排序方向要和题目一致。
- 原地 compare-swap 需要保证每对只被一个线程处理。
- bitonic 是 O(N log²N)，大 N 追求性能应改 radix/merge sort。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

