# LeetGPU Linear Self-Attention 解题思路

> **难度**：hard  
> **题号**：56  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Linear Self-Attention`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

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
Implement Linear Attention for a given set of matrices, following the method described in

 "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
 .
 Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix
 V of size M×d, your program should compute the output matrix using the formula:
 $$
 \text{LinearAttention}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^T V \right)}{\phi(Q) \left(\sum_j \phi(K_j) \right)}
 $$

 where \( \phi(x) \) is a feature map applied element-wise, for example:
 $$
 \phi(x) = \text{ELU}(x) + 1 =
 \begin{cases}
 x + 1, & x > 0 \\
 e^x, & x \le 0
 \end{cases}
 $$
```

---

## 3. 核心数学/算法公式

用特征映射 phi 替代 softmax：output_i = phi(Q_i)^T sum_j phi(K_j)V_j / phi(Q_i)^T sum_j phi(K_j)。

---

## 4. CUDA 并行划分

- 先聚合 KV = sum_j phi(K_j) outer V_j 和 Ksum=sum_j phi(K_j)。
- 再每个 query 用聚合矩阵计算输出。
- 如果题目定义更简单，按 spec 的 kernel feature 实现。

---

## 5. 推荐解法步骤

1. 计算/使用 phi(x)，常见是 elu(x)+1。
2. reduce 得到全局 Ksum 和 KV。
3. 每个 query 计算 numerator/denominator。
4. 写 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算/使用 phi(x)，常见是 elu(x)+1。
  - reduce 得到全局 Ksum 和 KV。
  - 每个 query 计算 numerator/denominator。
  - 写 output。
```

---

## 7. 复杂度分析

O(M×d² + M×d) 或按具体 spec。

---

## 8. 常见错误

- Linear attention 不是普通 softmax attention。
- denominator 要加小 eps 防止除 0。
- KV 聚合维度是 d×d。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

