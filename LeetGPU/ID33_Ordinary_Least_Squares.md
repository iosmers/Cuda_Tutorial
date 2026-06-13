# LeetGPU Ordinary Least Squares 解题思路

> **难度**：medium  
> **题号**：33  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Ordinary Least Squares`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

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
Solve the Ordinary Least Squares (OLS) regression problem on a GPU. Given a feature matrix \(X\) of size \(n\_samples \times n\_features\) and a target vector \(y\) of size \(n\_samples\), compute the coefficient vector \(\beta\) that minimizes the sum of squared residuals:
 \[ \min_{\beta} ||X\beta - y||^2 \]

 The closed-form solution to OLS is:
 \[ \beta = (X^TX)^{-1}X^Ty \]

Implementation Requirements

 External libraries are not permitted.

 The solve function signature must remain unchanged.

 The final coefficients must be stored in the beta vector.

 Assume that the feature matrix \(X\) is full rank (i.e., \(X^TX\) is invertible).

Example:
```

---

## 3. 核心数学/算法公式

最小二乘 beta = (X^T X)^{-1} X^T y。

---

## 4. CUDA 并行划分

- 先并行计算 Gram 矩阵 G=X^T X 和向量 b=X^T y。
- n_features 通常较小，可用一个小 kernel/单 block 做 Gaussian elimination。
- 最后解线性方程 G beta = b。

---

## 5. 推荐解法步骤

1. 计算 G[p,q] = sum_i X[i,p]X[i,q]。
2. 计算 b[p] = sum_i X[i,p]y[i]。
3. 对 G|b 做 Gauss-Jordan 或 Cholesky。
4. 写 beta。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 G[p,q] = sum_i X[i,p]X[i,q]。
  - 计算 b[p] = sum_i X[i,p]y[i]。
  - 对 G|b 做 Gauss-Jordan 或 Cholesky。
  - 写 beta。
```

---

## 7. 复杂度分析

O(n_samples×n_features² + n_features³)。

---

## 8. 常见错误

- G 是 n_features×n_features，不是 n_samples×n_samples。
- 求逆不如直接解线性方程稳定。
- 特征数较小时单 block 解方程更简单。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

