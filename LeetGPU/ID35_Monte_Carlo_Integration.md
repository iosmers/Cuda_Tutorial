# LeetGPU Monte Carlo Integration 解题思路

> **难度**：medium  
> **题号**：35  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Monte Carlo Integration`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {}
```

---

## 2. 题目摘要

```text
Implement Monte Carlo integration on a GPU. Given a set of function values \(y_i = f(x_i)\) sampled at random points \(x_i\) uniformly distributed in the interval \([a, b]\), estimate the definite integral:
 \[ \int_a^b f(x) \, dx \approx (b - a) \cdot \frac{1}{n} \sum_{i=1}^{n} y_i \]

 The Monte Carlo method approximates the integral by computing the average of the function values and multiplying by the interval width.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in the result variable

 Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2

Example:

Input: a = 0, b = 2, n_samples = 8
```

---

## 3. 核心数学/算法公式

积分近似 result = (b-a) * average(y_samples)。

---

## 4. CUDA 并行划分

- 本题已给 y_samples，所以只需要 reduce sum。
- 每个线程累加多个样本，block 内 reduction。
- 最终乘以 (b-a)/n_samples。

---

## 5. 推荐解法步骤

1. cudaMemset(result,0)。
2. grid-stride loop 累加 y_samples。
3. block reduction。
4. atomicAdd(result, block_sum*(b-a)/n_samples)。

---

## 6. 伪代码骨架

```text
solve(...):
  - cudaMemset(result,0)。
  - grid-stride loop 累加 y_samples。
  - block reduction。
  - atomicAdd(result, block_sum*(b-a)/n_samples)。
```

---

## 7. 复杂度分析

O(n_samples)。

---

## 8. 常见错误

- 不要再次生成随机数，输入已经是采样值。
- 最后要乘区间长度 b-a。
- 平均值除以 n_samples。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

