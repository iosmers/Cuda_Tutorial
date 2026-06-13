# LeetGPU Logistic Regression 解题思路

> **难度**：medium  
> **题号**：34  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Logistic Regression`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {}
```

---

## 2. 题目摘要

```text
Solve the logistic regression problem on a GPU. Given a feature matrix \(X\) of size \(n\_samples \times n\_features\) and a binary target vector \(y\) of size \(n\_samples\) (containing only 0s and 1s), compute the coefficient vector \(\beta\) that maximizes the log-likelihood:
 \[ \max_{\beta} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right] \]

 where \(p_i = \sigma(X_i^T \beta)\) and \(\sigma(z) = \frac{1}{1 + e^{-z}}\) is the sigmoid function.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final coefficients must be stored in the beta vector

 The target vector y contains only binary values (0 and 1)

Example:

Input:
```

---

## 3. 核心数学/算法公式

通常做若干步梯度下降：sigmoid(X beta)，梯度 X^T(p-y)/N。

---

## 4. CUDA 并行划分

- prediction/gradient kernel：样本维度并行。
- 对每个 feature 的梯度做 reduction。
- 更新 beta。

---

## 5. 推荐解法步骤

1. 初始化或读取 beta。
2. 循环固定迭代次数。
3. 计算 z_i=dot(X_i,beta)，p_i=sigmoid(z_i)。
4. 累加 grad_j = sum_i (p_i-y_i)X_ij/N。
5. beta_j -= lr*grad_j。

---

## 6. 伪代码骨架

```text
solve(...):
  - 初始化或读取 beta。
  - 循环固定迭代次数。
  - 计算 z_i=dot(X_i,beta)，p_i=sigmoid(z_i)。
  - 累加 grad_j = sum_i (p_i-y_i)X_ij/N。
  - beta_j -= lr*grad_j。
```

---

## 7. 复杂度分析

O(iterations×n_samples×n_features)。

---

## 8. 常见错误

- sigmoid 要避免 exp 溢出。
- 每轮 gradient buffer 要清零。
- 题目若只要求一轮/固定超参数，要按 spec 实现。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

