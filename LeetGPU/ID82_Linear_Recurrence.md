# LeetGPU Linear Recurrence 解题思路

> **难度**：medium  
> **题号**：82  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Linear Recurrence`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// a, x, h are device pointers
extern "C" void solve(const float* a, const float* x, float* h, int B, int L) {}
```

---

## 2. 题目摘要

```text
Given two matrices a and x, each of shape [B, L] (batch size × sequence length),
 compute the linear recurrence h of shape [B, L] defined by:
 h[b, 0] = x[b, 0] and h[b, t] = a[b, t] × h[b, t−1] + x[b, t] for t ≥ 1.
 All values are float32. This operation is the core computational primitive of
 State Space Models (SSMs) such as Mamba, S4, and H3.

 Linear Recurrence: h[t] = a[t] · h[t-1] + x[t]

 h[0]
 h[1]
 h[2]
 h[3]

 ×a[1]

 ×a[2]

 ×a[3]
```

---

## 3. 核心数学/算法公式

常见形式 h[t] = a[t] * h[t-1] + x[t]，对每个 batch 独立。

---

## 4. CUDA 并行划分

- batch 维并行，每个 block 处理一个序列。
- L 不大时单线程/单 block 串行扫描最简单。
- 高性能可用 parallel scan over affine transforms。

---

## 5. 推荐解法步骤

1. 对每个 batch b。
2. h_prev=0。
3. for t in 0..L-1: h=a*b? 按公式更新。
4. 写 h[b,t]。

---

## 6. 伪代码骨架

```text
solve(...):
  - 对每个 batch b。
  - h_prev=0。
  - for t in 0..L-1: h=a*b? 按公式更新。
  - 写 h[b,t]。
```

---

## 7. 复杂度分析

baseline O(B×L)，并行 scan 深度 O(logL)。

---

## 8. 常见错误

- 递推有时间依赖，不能普通逐元素并行。
- 每个 batch 独立。
- scan 组合规则是 (A,B) 复合：h=A*h0+B。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

