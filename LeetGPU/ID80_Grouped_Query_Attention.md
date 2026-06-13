# LeetGPU Grouped Query Attention 解题思路

> **难度**：medium  
> **题号**：80  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Grouped Query Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {}
```

---

## 2. 题目摘要

```text
Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language
models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during
inference by sharing key and value heads across groups of query heads. Given query tensor
Q with num_q_heads heads and key/value tensors K,
V each with num_kv_heads heads, compute scaled dot-product attention
where every group of num_q_heads / num_kv_heads consecutive query heads attends to
the same key and value head. All tensors use float32.

 Grouped Query Attention (num_q_heads=4, num_kv_heads=2, groups=2)

 Q heads

 Q[0]

 Q[1]

 Q[2]
```

---

## 3. 核心数学/算法公式

GQA：num_q_heads 个 query head 共享 num_kv_heads 个 K/V head。

---

## 4. CUDA 并行划分

- grid 维度包含 q_head 和 query position。
- kv_head = q_head / (num_q_heads/num_kv_heads)。
- 每个 q_head 对对应 kv_head 做 attention。

---

## 5. 推荐解法步骤

1. 计算 head_dim。
2. 对每个 q_head,row 遍历所有 key。
3. 使用映射到的 kv_head 读取 K/V。
4. 稳定 softmax 并加权求和。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 head_dim。
  - 对每个 q_head,row 遍历所有 key。
  - 使用映射到的 kv_head 读取 K/V。
  - 稳定 softmax 并加权求和。
```

---

## 7. 复杂度分析

O(num_q_heads×seq_len²×head_dim)。

---

## 8. 常见错误

- q_head 到 kv_head 的映射是关键。
- 缩放因子 sqrt(head_dim)。
- 输出仍按 q_head 存储。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

