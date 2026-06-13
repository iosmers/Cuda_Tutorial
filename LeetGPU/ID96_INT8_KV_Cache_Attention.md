# LeetGPU INT8 KV-Cache Attention 解题思路

> **难度**：medium  
> **题号**：96  
> **目标**：根据题目给定输入，在 CUDA 中实现 `INT8 KV-Cache Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
extern "C" void solve(const float* Q, const int8_t* K_int8, const int8_t* V_int8,
                      const float* k_scale, const float* v_scale, float* output, int num_heads,
                      int seq_len, int head_dim) {}
```

---

## 2. 题目摘要

```text
Implement decode-phase multi-head attention where the key and value caches are stored as
int8 with per-token scale factors. This memory layout halves KV-cache bandwidth
versus float32 and is used in production LLM serving systems such as TensorRT-LLM
and vLLM. Given a query tensor Q for a single new token, int8 key cache
K_int8, int8 value cache V_int8, and per-token scales
k_scale and v_scale, dequantize the caches and compute scaled
dot-product attention to produce output. All non-integer tensors use
float32.

Implementation Requirements

 Implement the function solve(Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim).

 Do not change the function signature or use external libraries beyond the standard GPU frameworks.

 Write the result into the provided output buffer.

 Dequantize using per-token scales: K_float[h, s, d] = K_int8[h, s, d] × k_scale[h, s] (and analogously for V).
```

---

## 3. 核心数学/算法公式

decode attention：Q 是 float，K/V cache 是 int8，需要按 scale 反量化后做 attention。

---

## 4. CUDA 并行划分

- 一个 block 负责一个 head。
- 遍历 seq_len 个 cache token。
- K_int8/V_int8 读取后乘对应 scale。
- 稳定 softmax 后加权 V。

---

## 5. 推荐解法步骤

1. 对每个 head 计算 score_j=dot(Q,K_deq_j)/sqrt(head_dim)。
2. 求 row_max。
3. 求 denom 和加权 V_deq。
4. 写 output[head,dim]。

---

## 6. 伪代码骨架

```text
solve(...):
  - 对每个 head 计算 score_j=dot(Q,K_deq_j)/sqrt(head_dim)。
  - 求 row_max。
  - 求 denom 和加权 V_deq。
  - 写 output[head,dim]。
```

---

## 7. 复杂度分析

O(num_heads×seq_len×head_dim)。

---

## 8. 常见错误

- K 和 V 的 scale 可能按 head/token/维度不同布局。
- int8 反量化后再参与 dot。
- decode 阶段通常只有一个 query。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

