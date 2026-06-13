# LeetGPU Llama Transformer Block 解题思路

> **难度**：hard  
> **题号**：93  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Llama Transformer Block`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// x, output, weights, cos, sin are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, const float* cos,
                      const float* sin, int seq_len) {}
```

---

## 2. 题目摘要

```text
Implement a single Llama-style transformer decoder block. Given an input tensor \(x\) of shape
 (seq_len, 512), a packed weight buffer, and precomputed RoPE tables, compute the
 output using pre-norm architecture with Grouped Query Attention (GQA), Rotary Position Embeddings
 (RoPE), and a SwiGLU feed-forward network.

 x (seq_len, 512)

 RMSNorm1 -->

 residual

 RMSNorm 1

 QKV Projection (GQA)

 RoPE (Q and K)

 Causal Attention
```

---

## 3. 核心数学/算法公式

Llama block：RMSNorm -> RoPE attention/GQA -> residual -> RMSNorm -> SwiGLU MLP -> residual。

---

## 4. CUDA 并行划分

- 分解为 RMSNorm、QKV projection、RoPE、attention、projection、SwiGLU MLP。
- GEMM/attention 是主要计算。
- 权重布局按题目 spec 偏移解析。

---

## 5. 推荐解法步骤

1. RMSNorm 输入。
2. 计算 Q/K/V 并应用 RoPE。
3. 执行 causal attention。
4. 输出投影并 residual。
5. RMSNorm。
6. SwiGLU MLP。
7. 第二次 residual 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - RMSNorm 输入。
  - 计算 Q/K/V 并应用 RoPE。
  - 执行 causal attention。
  - 输出投影并 residual。
  - RMSNorm。
  - SwiGLU MLP。
  - 第二次 residual 写 output。
```

---

## 7. 复杂度分析

主要 O(seq²×d + seq×d×d_ff)。

---

## 8. 常见错误

- Llama 用 RMSNorm，不是 LayerNorm。
- RoPE 只作用于 Q/K。
- SwiGLU gate/up/down 维度和权重偏移要正确。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

