# LeetGPU Causal Self-Attention 解题思路

> **难度**：hard  
> **题号**：53  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Causal Self-Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d) {

}
```

---

## 2. 题目摘要

```text
Implement Causal (masked) Self-Attention for a given set of matrices.
 Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix
 V of size M×d, your program should compute the output matrix using the formula:
 $$\text{Attention}_{\text{causal}}(Q, K, V) = \text{softmax}\Bigl(\text{masked}\Bigl( \frac{QK^T}{\sqrt{d}} \Bigr)\Bigr)V$$

 where mask is a causal mask that sets all positions corresponding to keys after the current query to \(-\infty\).
 $$$$
 i.e., for query i and key j:
 $$
 \text{masked}(a_{ij}) =
 \begin{cases}
 a_{ij}, & j \le i \\
 -\infty, & j > i
 \end{cases}
 $$
 The softmax function is applied row-wise. Q, K, V, and output are all of data type float32;
 M, and d are of data type int32.
```

---

## 3. 核心数学/算法公式

自注意力但只能看当前位置及之前：j <= i。

---

## 4. CUDA 并行划分

- 一个 block 负责一个 query position i。
- 只遍历 key j=0..i。
- shared memory 保存 scores，稳定 softmax 后加权 V。

---

## 5. 推荐解法步骤

1. score_j=dot(Q_i,K_j)/sqrt(d), j<=i。
2. row_max=max score。
3. denom=sum exp(score-row_max)。
4. output_i=sum weight_j V_j。

---

## 6. 伪代码骨架

```text
solve(...):
  - score_j=dot(Q_i,K_j)/sqrt(d), j<=i。
  - row_max=max score。
  - denom=sum exp(score-row_max)。
  - output_i=sum weight_j V_j。
```

---

## 7. 复杂度分析

O(M²×d/2)。

---

## 8. 常见错误

- mask 方向：不能看未来 j>i。
- softmax 分母只包含 j<=i。
- 缩放因子 sqrt(d)。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

