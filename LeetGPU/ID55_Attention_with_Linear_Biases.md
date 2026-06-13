# LeetGPU Attention with Linear Biases 解题思路

> **难度**：medium  
> **题号**：55  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Attention with Linear Biases`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) {}
```

---

## 2. 题目摘要

```text
Implement Attention with Linear Biases (ALiBi), following the method described in

 "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
 , for a given set of matrices.
 Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix
 V of size N×d, your program should compute the output matrix using the formula:

 $$
 \text{Attention}_{ALiBi}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} + \alpha \cdot \Delta \Bigr)V
 $$

 where α is a slope controlling the linear bias and Δ = i - j represents the relative position between query i and key j.
 The softmax function is applied row-wise. Q, K, V, output, and α are all of data type float32;
 M, N, d are of data type int32.

Implementation Requirements

 Use only native features (external libraries are not permitted)
```

---

## 3. 核心数学/算法公式

ALiBi attention：score(i,j)=dot(Q_i,K_j)/sqrt(d)+bias，bias 通常与距离成线性关系。

---

## 4. CUDA 并行划分

- 一个 block 负责一个 query row。
- 计算 score 时加 alpha 相关的线性距离项。
- 之后与普通 attention 相同：max/sum/weighted V。

---

## 5. 推荐解法步骤

1. score=dot*scale + alpha*(i-j) 或题目定义的 bias。
2. 稳定 softmax。
3. 加权求和 V。

---

## 6. 伪代码骨架

```text
solve(...):
  - score=dot*scale + alpha*(i-j) 或题目定义的 bias。
  - 稳定 softmax。
  - 加权求和 V。
```

---

## 7. 复杂度分析

O(M×N×d)。

---

## 8. 常见错误

- bias 符号要按题目公式确认。
- softmax 仍然按 query row 做。
- 若是 causal ALiBi，需要同时 mask 未来。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

