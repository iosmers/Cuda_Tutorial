# LeetGPU Causal Depthwise Conv1d 解题思路

> **难度**：medium  
> **题号**：90  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Causal Depthwise Conv1d`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// x, weight, bias, output are device pointers
extern "C" void solve(const float* x, const float* weight, const float* bias, float* output, int B,
                      int L, int D, int K) {}
```

---

## 2. 题目摘要

```text
Implement a causal depthwise 1D convolution over a batched sequence tensor
 x of shape (B, L, D), producing an output of the same shape.
 In a depthwise convolution, each channel d is convolved independently using its
 own kernel weight[d, :] — there is no mixing across channels.
 The convolution is causal: output position l may only depend on
 input positions 0, 1, …, l (past and present), never future positions.
 This operation is a key component of state-space models such as Mamba, where it is applied
 before the selective scan to mix local context within each feature channel.

 Causal Depthwise Conv1d (K=3, one channel shown)

 x[d]

 x₀

 x₁

 x₂
```

---

## 3. 核心数学/算法公式

对每个通道独立做 causal 1D convolution：out[b,t,d]=bias[d]+sum_k x[b,t-k,d]*w[d,k]。

---

## 4. CUDA 并行划分

- 一个线程负责一个 (B,L,D) 输出元素。
- 只沿 K 做小循环。
- depthwise 表示通道之间不混合。

---

## 5. 推荐解法步骤

1. 解码 b,t,d。
2. acc=bias[d]。
3. for kk in 0..K-1: src=t-kk，如果 >=0 累加。
4. 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 解码 b,t,d。
  - acc=bias[d]。
  - for kk in 0..K-1: src=t-kk，如果 >=0 累加。
  - 写 output。
```

---

## 7. 复杂度分析

O(B×L×D×K)。

---

## 8. 常见错误

- causal 只能看当前和过去，不能看未来。
- weight 索引通常 d*K+kk。
- x 索引 ((b*L+t)*D+d)。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

