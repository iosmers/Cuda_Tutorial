# LeetGPU GPT-2 Transformer Block 解题思路

> **难度**：hard  
> **题号**：74  
> **目标**：根据题目给定输入，在 CUDA 中实现 `GPT-2 Transformer Block`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {}
```

---

## 2. 题目摘要

```text
Implement a single GPT-2 transformer decoder block. Given an input tensor
 \(x\) of shape (seq_len, 768) and a packed weight buffer containing
 all block parameters, compute the output using pre-norm architecture with
 multi-head self-attention and a feed-forward network with GELU activation.

 x (seq_len, 768)

 LN1 -->

 residual

 LayerNorm 1

 QKV Projection

 Multi-Head Attention

 Output Projection
```

---

## 3. 核心数学/算法公式

GPT-2 block：LayerNorm -> causal MHA -> residual -> LayerNorm -> MLP(GELU) -> residual。

---

## 4. CUDA 并行划分

- 分解成多个 kernels：layernorm、QKV projection、attention、output projection、MLP。
- 矩阵乘法部分用 tiled GEMM。
- attention 按 head/query 分块。

---

## 5. 推荐解法步骤

1. 读取 weights 按题目布局切分。
2. LN1(x)。
3. 计算 Q,K,V。
4. causal attention。
5. proj + residual。
6. LN2。
7. MLP: fc1 + GELU + fc2。
8. residual 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 读取 weights 按题目布局切分。
  - LN1(x)。
  - 计算 Q,K,V。
  - causal attention。
  - proj + residual。
  - LN2。
  - MLP: fc1 + GELU + fc2。
  - residual 写 output。
```

---

## 7. 复杂度分析

主要是 O(seq_len²×d_model + seq_len×d_model×d_ff)。

---

## 8. 常见错误

- weights 布局必须完全按 spec 偏移。
- LayerNorm 是每个 token 独立归一化 hidden 维。
- causal mask 不能看未来。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

