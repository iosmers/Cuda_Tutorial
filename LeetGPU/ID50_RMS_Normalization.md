# LeetGPU RMS Normalization 解题思路

> **难度**：medium  
> **题号**：50  
> **目标**：根据题目给定输入，在 CUDA 中实现 `RMS Normalization`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {}
```

---

## 2. 题目摘要

```text
Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the normalized output using a scalar scale (gamma) and shift (beta) parameter.

 RMS Normalization computes:
 \[
 \begin{align}
 \text{rms} &= \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon} \\
 \hat{x}_i &= \frac{x_i}{\text{rms}} \\
 y_i &= \gamma \hat{x}_i + \beta
 \end{align}
 \]

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output tensor
```

---

## 3. 核心数学/算法公式

rms=sqrt(mean(x_i²)+eps)，output_i=x_i/rms*gamma+beta。

---

## 4. CUDA 并行划分

- 先 reduce sum of squares。
- 再逐元素 normalize。
- 小 N 可一个 block，两阶段可处理大 N。

---

## 5. 推荐解法步骤

1. sum_sq = sum input[i]^2。
2. rms = sqrt(sum_sq/N + eps)。
3. output[i] = input[i]/rms*gamma + beta。

---

## 6. 伪代码骨架

```text
solve(...):
  - sum_sq = sum input[i]^2。
  - rms = sqrt(sum_sq/N + eps)。
  - output[i] = input[i]/rms*gamma + beta。
```

---

## 7. 复杂度分析

O(N)。

---

## 8. 常见错误

- RMSNorm 不减 mean，和 LayerNorm 不同。
- eps 在 sqrt 内。
- gamma/beta 本题是标量。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

