# LeetGPU Batch Normalization 解题思路

> **难度**：medium  
> **题号**：40  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Batch Normalization`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output,
                      int N, int C, float eps) {}
```

---

## 2. 题目摘要

```text
Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape [N, C] where N is the batch size and C is the number of features, compute the normalized output using learnable scale (gamma) and shift (beta) parameters.

 For each feature channel j, batch normalization computes:
 \[
 \begin{align}
 \mu_j &= \frac{1}{N} \sum_{i=1}^{N} x_{i,j} \\
 \sigma_j^2 &= \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2 \\
 \hat{x}_{i,j} &= \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} \\
 y_{i,j} &= \gamma_j \hat{x}_{i,j} + \beta_j
 \end{align}
 \]

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged
```

---

## 3. 核心数学/算法公式

对每个 channel：y=(x-mean)/sqrt(var+eps)*gamma+beta。

---

## 4. CUDA 并行划分

- 先对 N 维度为每个 channel 求 mean。
- 再求 variance。
- 第三个 kernel 做 normalize。
- 如果 N×C 中 C 较小，可一个 block 负责一个 channel。

---

## 5. 推荐解法步骤

1. reduce sum input[n,c] 得 mean[c]。
2. reduce sum (x-mean)^2 得 var[c]。
3. 逐元素写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - reduce sum input[n,c] 得 mean[c]。
  - reduce sum (x-mean)^2 得 var[c]。
  - 逐元素写 output。
```

---

## 7. 复杂度分析

O(N×C)。

---

## 8. 常见错误

- mean/var 是按 channel 统计，不是全局一个值。
- eps 放在 sqrt 里面。
- gamma/beta 按 channel 索引。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

