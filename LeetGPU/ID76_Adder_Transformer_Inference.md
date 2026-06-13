# LeetGPU Adder Transformer Inference 解题思路

> **难度**：medium  
> **题号**：76  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Adder Transformer Inference`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// prompts, output, weights are device pointers
extern "C" void solve(const int* prompts, float* output, const float* weights, int batch_size) {}
```

---

## 2. 题目摘要

```text
Run batched autoregressive inference for a 10-parameter transformer that adds two 10-digit
numbers. Given prompts of shape [batch_size, 31] (int32) and a 10-float weight
buffer, write output logits of shape [batch_size, 11, 10] — one logit
row per decode step over the 10-digit vocabulary (0–9). All tensors are float32 except
the int32 prompts.

The model comes from the
AdderBoard
competition for the smallest autoregressive transformer that adds 10-digit numbers at
≥99% accuracy. It encodes carry propagation in 10 learned parameters via RoPE geometry,
tied embeddings, and SwiGLU gating.

 Token Prompt [B,31]

 Embed: [w0-w1*d², -d]

 Unit RMSNorm
```

---

## 3. 核心数学/算法公式

小型固定 transformer 推理，把 prompts 经过 embedding/attention/MLP/输出头得到结果。

---

## 4. CUDA 并行划分

- 按 batch 和 token 并行。
- 权重固定布局，分阶段 kernel 实现。
- 小模型可用一个 block 处理一个样本，重点是正确切分 weights。

---

## 5. 推荐解法步骤

1. 解析 prompts 为 token embedding。
2. 执行题目指定层数的 transformer block。
3. 对最终 hidden 做输出投影。
4. 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 解析 prompts 为 token embedding。
  - 执行题目指定层数的 transformer block。
  - 对最终 hidden 做输出投影。
  - 写 output。
```

---

## 7. 复杂度分析

随模型结构，通常 attention O(batch×seq²×d)。

---

## 8. 常见错误

- 这是端到端模型题，weights offset 是核心。
- 推理通常 autoregressive，注意 token 位置。
- softmax/attention 要数值稳定。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

