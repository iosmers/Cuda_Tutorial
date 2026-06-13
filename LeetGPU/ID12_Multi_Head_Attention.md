# LeetGPU Multi-Head Attention 解题思路

> **难度**：hard  
> **题号**：12  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Multi-Head Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) {}
```

---

## 2. 题目摘要

```text
Implement a program for multi-head self-attention. Given three input matrices \(Q\) (queries), \(K\) (keys), and \(V\) (values) of size \(N \times d_{\text{model}}\), compute:
 \[ \text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h) \]
 where each head computes:
 \[ \text{head}_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i \]
 with \(d_k = d_{\text{model}}/h\) and \(Q_i, K_i, V_i\) being the i-th head's partition of the input matrices.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output array

Example 1:

Input:
\[
```

---

## 3. 核心数学/算法公式

把 d_model 拆成 h 个 head，每个 head 做 scaled dot-product attention，再拼回 output。

---

## 4. CUDA 并行划分

- grid 维度可设为 (head, query_position)。
- 一个 block 负责一个 head 的一个 query，shared memory 保存 scores。
- 对每个 head 使用 head_dim=d_model/h，分别计算 softmax(QK^T/sqrt(head_dim))V。

---

## 5. 推荐解法步骤

1. 计算 head_dim。
2. 对每个 head/query 计算所有 key 的 score。
3. 稳定 softmax：max、sum。
4. 对 V 加权求和，写回对应 head 的 output slice。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 head_dim。
  - 对每个 head/query 计算所有 key 的 score。
  - 稳定 softmax：max、sum。
  - 对 V 加权求和，写回对应 head 的 output slice。
```

---

## 7. 复杂度分析

O(h × N² × head_dim) = O(N² × d_model)。

---

## 8. 常见错误

- 缩放因子是 sqrt(head_dim)，不是 sqrt(d_model)。
- Q/K/V 的 head offset 计算要正确。
- softmax 是每个 head、每个 query 独立做。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

