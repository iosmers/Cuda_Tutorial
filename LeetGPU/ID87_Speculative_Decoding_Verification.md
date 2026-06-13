# LeetGPU Speculative Decoding Verification 解题思路

> **难度**：medium  
> **题号**：87  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Speculative Decoding Verification`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
extern "C" void solve(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) {}
```

---

## 2. 题目摘要

```text
Implement the token verification step of speculative decoding. A draft model proposes \(T\) tokens;
 the target model evaluates them in one forward pass and accepts or rejects each. Given \(B\)
 sequences, produce the verified output tokens. Probability tensors are float32;
 token tensors are int32.

 Notation for each sequence \(b\), at each draft position \(i = 0, \ldots, T{-}1\):

 \(t_i = \texttt{draft_tokens}[b, i]\) — the token proposed by the draft model

 \(p_i(v) = \texttt{draft_probs}[b, i, v]\) — draft model's probability for token \(v\)

 \(q_i(v) = \texttt{target_probs}[b, i, v]\) — target model's probability for token \(v\)

 \(u_i = \texttt{uniform_samples}[b, i]\) — pre-generated \(U[0,1)\) sample for position \(i\)

 pos 0
 pos 1
 pos 2
```

---

## 3. 核心数学/算法公式

验证 draft token：接受概率 min(1, target_prob/draft_prob)，否则从修正分布采样。

---

## 4. CUDA 并行划分

- 一个 block/线程组处理一个 batch。
- 沿 T 顺序有依赖：一旦拒绝，后续 draft 通常停止。
- Vocab 维度可并行做采样/归一化。

---

## 5. 推荐解法步骤

1. 对每个 batch b 顺序遍历 draft step t。
2. token=draft_tokens[b,t]。
3. accept_prob=min(1,target_probs/draft_probs)。
4. uniform_samples 判断接受。
5. 拒绝时按题目规则输出替代 token。

---

## 6. 伪代码骨架

```text
solve(...):
  - 对每个 batch b 顺序遍历 draft step t。
  - token=draft_tokens[b,t]。
  - accept_prob=min(1,target_probs/draft_probs)。
  - uniform_samples 判断接受。
  - 拒绝时按题目规则输出替代 token。
```

---

## 7. 复杂度分析

O(B×T×V) 若拒绝采样需扫 vocab。

---

## 8. 常见错误

- 时间步 T 有顺序依赖，不要完全并行破坏逻辑。
- 概率除法要处理 draft_prob=0。
- 输出 token 数/拒绝后的行为按 spec。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

