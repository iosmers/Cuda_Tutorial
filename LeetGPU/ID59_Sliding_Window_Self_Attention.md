# LeetGPU Sliding Window Self-Attention 解题思路

> **难度**：hard  
> **题号**：59  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Sliding Window Self-Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d,
                      int window_size) {}
```

---

## 2. 题目摘要

```text
Implement Sliding Window Self-Attention for a given set of matrices.
 Before introducing the sliding window version, let's first recall standard Self-Attention.

1. Standard Softmax Attention

 Given query matrix Q, key matrix K, and value matrix V, each position i attends to all positions j using a softmax-weighted sum:

 \( \text{score}_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d}} \)

 \( \text{output}_i = \sum_{j=1}^{M} \text{softmax}(\text{score}_{i,*})_j \cdot V_j \)

 In other words, each query computes similarity with all keys, applies a softmax to get attention weights, and then computes a weighted sum of values.

2. Sliding Window Self-Attention

 Sliding Window Attention modifies standard attention by restricting each query to attend only to a local window around its position.

 For each position i, only consider the keys and values within a window of size window_size around i (positions [i-window_size, ..., i+window_size]).
```

---

## 3. 核心数学/算法公式

每个 query 只关注窗口内 key，例如 [i-window_size, i] 或中心窗口。

---

## 4. CUDA 并行划分

- 一个 block 负责一个 query。
- 遍历窗口范围内 key，复杂度从 O(M²d) 降到 O(M×window×d)。
- shared memory 保存窗口 scores。

---

## 5. 推荐解法步骤

1. 确定 start/end window。
2. 计算窗口内 score。
3. 稳定 softmax。
4. 对窗口内 V 加权求和。

---

## 6. 伪代码骨架

```text
solve(...):
  - 确定 start/end window。
  - 计算窗口内 score。
  - 稳定 softmax。
  - 对窗口内 V 加权求和。
```

---

## 7. 复杂度分析

O(M×window_size×d)。

---

## 8. 常见错误

- 窗口边界 clamp 到 [0,M)。
- 如果是 causal sliding window，end=i。
- softmax 分母只包含窗口内元素。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

