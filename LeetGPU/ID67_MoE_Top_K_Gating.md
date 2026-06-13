# LeetGPU MoE Top-K Gating 解题思路

> **难度**：medium  
> **题号**：67  
> **目标**：根据题目给定输入，在 CUDA 中实现 `MoE Top-K Gating`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E,
                      int k) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape [M, E] where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

 For each row i, the operation computes:
 \[
 \begin{align}
 \text{indices}_i, \text{vals}_i &= \text{TopK}(\text{logits}_i, k) \\
 \text{vals}_i &= \text{logits}_i[\text{indices}_i] \\
 \text{weights}_i &= \text{Softmax}(\text{vals}_i)
 \end{align}
 \]

 The selected experts must remain ordered by descending logit value, matching the order returned by
 topk. The topk_weights array must correspond positionally to
 topk_indices in that same order.

Implementation Requirements

 External libraries are not permitted
```

---

## 3. 核心数学/算法公式

对每个 token 的 E 个 expert logits，选 top-k expert，并对 top-k 做 softmax 得 weights。

---

## 4. CUDA 并行划分

- 一个 block 负责一行 token。
- 在 E 上做 k 次 selection，保存 topk_indices。
- 对选出的 k 个 logits 做稳定 softmax。

---

## 5. 推荐解法步骤

1. for each row m。
2. 选择 k 个最大 logits，可用 pair(value,index) 处理重复。
3. max_top = max selected logits。
4. weights[r]=exp(logit-max)/sum。
5. 写 topk_weights/topk_indices。

---

## 6. 伪代码骨架

```text
solve(...):
  - for each row m。
  - 选择 k 个最大 logits，可用 pair(value,index) 处理重复。
  - max_top = max selected logits。
  - weights[r]=exp(logit-max)/sum。
  - 写 topk_weights/topk_indices。
```

---

## 7. 复杂度分析

O(M×E×k) baseline。

---

## 8. 常见错误

- 权重只在 top-k 内归一化，不是所有 E。
- 重复值需要稳定 tie-break。
- 输出 indices 是 expert id。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

