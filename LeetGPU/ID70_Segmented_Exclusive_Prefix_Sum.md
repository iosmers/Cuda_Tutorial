# LeetGPU Segmented Exclusive Prefix Sum 解题思路

> **难度**：medium  
> **题号**：70  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Segmented Exclusive Prefix Sum`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// values, flags, output are device pointers
extern "C" void solve(const float* values, const int* flags, float* output, int N) {}
```

---

## 2. 题目摘要

```text
Given an array of N 32-bit floating point values and an integer array
 flags of the same length, where flags[i] = 1 marks the start of a new
 segment and flags[i] = 0 continues the current segment, compute the
 exclusive prefix sum within each segment and store the result in
 output. The first element is always a segment start
 (flags[0] = 1). Within each segment, output[i] equals the sum of all
 values elements in the same segment that appear before index i, so the
 first element of every segment is always 0.0.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in output

 Read from values and flags; write to output
```

---

## 3. 核心数学/算法公式

flags 标记新 segment，output[i] 是同一 segment 内 i 之前 values 的和。

---

## 4. CUDA 并行划分

- 可以用并行 scan 扩展 pair(value, flag)。
- 组合规则：如果右元素 flag=1，则断开前缀；否则累加左值。
- 简单 baseline 可每段/每元素回扫，优化用 block scan + 跨 block 修正。

---

## 5. 推荐解法步骤

1. 把每个元素表示为 (sum=values[i], head=flags[i])。
2. 做 segmented inclusive scan。
3. exclusive 输出 = inclusive - values[i]，若 flags[i]=1 输出 0。
4. 处理跨 block segment 延续。

---

## 6. 伪代码骨架

```text
solve(...):
  - 把每个元素表示为 (sum=values[i], head=flags[i])。
  - 做 segmented inclusive scan。
  - exclusive 输出 = inclusive - values[i]，若 flags[i]=1 输出 0。
  - 处理跨 block segment 延续。
```

---

## 7. 复杂度分析

优化版 O(N)，baseline 可能更慢。

---

## 8. 常见错误

- 普通 prefix sum 会跨 segment 错误累加。
- flags[i]=1 的位置 exclusive 应为 0。
- 跨 block 的 segment carry 是难点。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

