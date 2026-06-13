# LeetGPU Categorical Cross Entropy Loss 解题思路

> **难度**：medium  
> **题号**：25  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Categorical Cross Entropy Loss`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions.
 Given a matrix of predicted logits \(Z\) of size \(N \times C\) and a vector of true class labels true_labels of size \(N\), compute the average cross-entropy loss over the batch.
 The loss for a single sample \(j\) with logits \(z_j = [z_{j1}, \ldots, z_{jC}]\) and true label \(y_j\) is calculated using the numerically stable formula:
 \[ \text{Loss}_j = \log\left(\sum_{k=1}^{C} e^{z_{jk}}\right) - z_{j, y_j} \]
 The final output stored in the loss variable should be the average loss over the \(N\) samples:
 \[ L = \frac{1}{N} \sum_{j=1}^{N} \text{Loss}_j \]
 The input parameters are logits, true_labels, N (number of samples), and C (number of classes). The result should be stored in loss (a pointer to a single float).

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result (average loss) must be stored in loss

Example 1:
```

---

## 3. 核心数学/算法公式

对每个样本：loss_i = -log(softmax(logits_i)[label_i])，总 loss 通常取平均。

---

## 4. CUDA 并行划分

- 一个 block 处理一个样本的一行 logits，做 max/sum reduction。
- 或一个线程处理一个样本并串行遍历 C，C 小时足够。
- 最后对 N 个 loss 做 reduction。

---

## 5. 推荐解法步骤

1. 对每行 logits 求 max。
2. sum_exp = sum_c exp(logit_c - max)。
3. loss_i = -(logit_label - max - log(sum_exp))。
4. 归约所有 loss_i / N。

---

## 6. 伪代码骨架

```text
solve(...):
  - 对每行 logits 求 max。
  - sum_exp = sum_c exp(logit_c - max)。
  - loss_i = -(logit_label - max - log(sum_exp))。
  - 归约所有 loss_i / N。
```

---

## 7. 复杂度分析

O(N×C)。

---

## 8. 常见错误

- 不要先显式 softmax 再 log，数值不稳定。
- label 索引是 true_labels[i]。
- 确认输出 loss 是 sum 还是 mean，通常是 mean。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

