# LeetGPU Rotary Positional Embedding 解题思路

> **难度**：medium  
> **题号**：61  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Rotary Positional Embedding`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, cos, sin, output are device pointers
extern "C" void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query vectors.
 RoPE is a method for encoding positional information in transformer models by rotating the query and key vectors using precomputed cosine and sine components.

 Mathematically, given a query vector \(x\) and corresponding cosine and sine vectors, the operation is defined as:
 \[
 \text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin
 \]

 Where \(\odot\) denotes element-wise multiplication. The \(\text{rotate\_half}(x)\) operation swaps the first and second halves of the vector and negates the first half. For a vector of dimension \(d\):
 \[
 \text{rotate\_half}([x_1, \dots, x_{d/2}, x_{d/2+1}, \dots, x_d]) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}]
 \]

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged
```

---

## 3. 核心数学/算法公式

RoPE 对每对维度旋转：out_even=q_even*cos - q_odd*sin，out_odd=q_even*sin + q_odd*cos。

---

## 4. CUDA 并行划分

- 一个线程处理一个元素对 (row, pair)。
- M 是位置/序列长度，D 是维度。
- cos/sin 按位置和 pair 索引读取。

---

## 5. 推荐解法步骤

1. pair = dim/2。
2. 读取 x0=Q[m,2p], x1=Q[m,2p+1]。
3. 读取 cos[m,p], sin[m,p] 或按 spec 的布局。
4. 写两个输出维度。

---

## 6. 伪代码骨架

```text
solve(...):
  - pair = dim/2。
  - 读取 x0=Q[m,2p], x1=Q[m,2p+1]。
  - 读取 cos[m,p], sin[m,p] 或按 spec 的布局。
  - 写两个输出维度。
```

---

## 7. 复杂度分析

O(M×D)。

---

## 8. 常见错误

- D 必须按偶数对处理。
- cos/sin 的布局要按题目 starter/说明。
- 不要原地覆盖导致 odd/even 读错，使用 output 更安全。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

