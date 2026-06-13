# LeetGPU SwiGLU MLP Block 解题思路

> **难度**：medium  
> **题号**：84  
> **目标**：根据题目给定输入，在 CUDA 中实现 `SwiGLU MLP Block`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// x, W_gate, W_up, W_down, output are device pointers
extern "C" void solve(const float* x, const float* W_gate, const float* W_up, const float* W_down,
                      float* output, int M, int d_model, int d_ffn) {}
```

---

## 2. 题目摘要

```text
Implement the SwiGLU MLP block — the feedforward network used in LLaMA, Mistral, Gemma, and most
 modern large language models. Given an input matrix x of shape
 [M, d_model] and three weight matrices W_gate, W_up
 (each [d_model, d_ffn]), and W_down ([d_ffn, d_model]),
 compute:
 output = (SiLU(x × W_gate) ⊙ (x × W_up)) × W_down,
 where SiLU(z) = z × sigmoid(z) and ⊙ denotes element-wise
 multiplication. All tensors are float32.

 x
 [M, d_model]

 x · W_gate
 gate projection

 x · W_up
 up projection
```

---

## 3. 核心数学/算法公式

SwiGLU：hidden = silu(xW_gate) * (xW_up)，output = hidden W_down。

---

## 4. CUDA 并行划分

- 三个矩阵乘法是主体：X×W_gate，X×W_up，hidden×W_down。
- 逐元素 kernel 计算 silu(gate)*up。
- GEMM 可用 tiled kernel。

---

## 5. 推荐解法步骤

1. 计算 gate[M,d_ffn]。
2. 计算 up[M,d_ffn]。
3. hidden=silu(gate)*up。
4. output=hidden×W_down。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 gate[M,d_ffn]。
  - 计算 up[M,d_ffn]。
  - hidden=silu(gate)*up。
  - output=hidden×W_down。
```

---

## 7. 复杂度分析

O(M×d_model×d_ffn)。

---

## 8. 常见错误

- silu(x)=x/(1+exp(-x))。
- W_down 维度是 d_ffn×d_model。
- 中间矩阵需要临时 buffer。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

