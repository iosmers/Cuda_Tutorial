# LeetGPU Top-p Sampling 解题思路

> **难度**：medium  
> **题号**：60  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Top-p Sampling`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

extern "C" void solve(const float* logits, const float* p, const int* seed, int* sampled_token,
                      int vocab_size) {}
```

---

## 2. 题目摘要

```text
Write a GPU program that implements top-p (nucleus) sampling for LLM inference.

 Top-p sampling is a text generation technique where you sample from the smallest set of tokens whose cumulative probability exceeds threshold p.
 This balances randomness and quality better than pure top-k or greedy sampling.

 Given logits (unnormalized scores) from a language model:

 Convert logits to probabilities using softmax

 Sort tokens by probability (descending)

 Find the smallest set where cumulative probability ≥ p (the "nucleus")

 Renormalize the nucleus probabilities to sum to 1

 Sample a token from the nucleus using the provided random seed

Implementation Requirements
```

---

## 3. 核心数学/算法公式

对 logits softmax 后，按概率降序累积到阈值 p，截断后重归一化并采样。

---

## 4. CUDA 并行划分

- vocab_size 通常较大：先求 max/sum 得概率。
- 需要排序或 top-k/top-p 候选；简单版可单 block 排序/选择。
- 使用 seed/uniform_samples 进行采样。

---

## 5. 推荐解法步骤

1. 稳定 softmax logits。
2. 按概率降序排序 token。
3. 累积概率直到 >= p。
4. 在截断集合内按 uniform sample 选择 token。

---

## 6. 伪代码骨架

```text
solve(...):
  - 稳定 softmax logits。
  - 按概率降序排序 token。
  - 累积概率直到 >= p。
  - 在截断集合内按 uniform sample 选择 token。
```

---

## 7. 复杂度分析

排序版 O(V logV)，选择/分桶可优化。

---

## 8. 常见错误

- top-p 是 nucleus，不是 top-k。
- 排序依据是概率/softmax 后与 logits 单调等价，但累积要用概率。
- 随机数要使用输入 seed/样本，保证可复现。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

