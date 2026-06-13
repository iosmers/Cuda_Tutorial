# LeetGPU Decaying Causal Attention 解题思路

> **难度**：medium  
> **题号**：92  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Decaying Causal Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int seq_len,
                      int d_model, float gamma) {}
```

---

## 2. 题目摘要

```text
Implement decaying causal attention. Given query matrix Q, key matrix K,
 and value matrix V, each of shape seq_len × d_model, and a scalar
 decay factor gamma ∈ (0, 1], compute the unnormalized causal attention output
 where position n attends to all past positions m ≤ n with weight
 gamman−m:

 \[
 \text{output}[n] = \sum_{m=0}^{n} \gamma^{n-m} \cdot \frac{Q[n] \cdot K[m]}{\sqrt{d_{\text{model}}}} \cdot V[m]
 \]

 Unlike standard softmax attention, there is no normalization — the weights decay geometrically from
 the current position backward. This is the parallel form of the Retention mechanism (RetNet), used
 as a recurrence-friendly alternative to attention in sequence models.

 Causal Decay Mask D[n,m] = γ^(n−m)

 m=0
 m=1
```

---

## 3. 核心数学/算法公式

causal attention 加时间衰减，score(i,j)=dot/sqrt(d)+decay(i-j)，j<=i。

---

## 4. CUDA 并行划分

- 一个 block 负责一个 query position。
- 只遍历过去 key。
- score 中加入 gamma 衰减项。

---

## 5. 推荐解法步骤

1. for j<=i 计算 dot。
2. score=dot*scale + decay，常见 decay=gamma*(i-j) 或 -gamma*(i-j)。
3. 稳定 softmax。
4. 加权 V。

---

## 6. 伪代码骨架

```text
solve(...):
  - for j<=i 计算 dot。
  - score=dot*scale + decay，常见 decay=gamma*(i-j) 或 -gamma*(i-j)。
  - 稳定 softmax。
  - 加权 V。
```

---

## 7. 复杂度分析

O(seq_len²×d_model/2)。

---

## 8. 常见错误

- gamma 的符号按题目公式。
- mask 未来位置。
- softmax 分母只包含 causal 范围。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

