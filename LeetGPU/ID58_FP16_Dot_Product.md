# LeetGPU FP16 Dot Product 解题思路

> **难度**：medium  
> **题号**：58  
> **目标**：根据题目给定输入，在 CUDA 中实现 `FP16 Dot Product`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, result are device pointers
extern "C" void solve(const half* A, const half* B, half* result, int N) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that computes the dot product of two vectors containing 16-bit floating point numbers (FP16/half).
 The dot product is the sum of the products of the corresponding elements of two vectors.

 Mathematically, the dot product of two vectors \(A\) and \(B\) of length \(n\) is defined as:
 \[
 A \cdot B = \sum_{i=0}^{n-1} A_i \cdot B_i = A_0 \cdot B_0 + A_1 \cdot B_1 + \ldots + A_{n-1} \cdot B_{n-1}
 \]

 All inputs are stored as 16-bit floating point numbers (FP16/half). For best precision, accumulation during multiplication should use FP32 before converting the final result to FP16.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
```

---

## 3. 核心数学/算法公式

sum_i half(A[i])*half(B[i])，输出 half。

---

## 4. CUDA 并行划分

- 每线程处理多个 half，转换为 float 累加。
- block reduction 用 float。
- 最终 result 可写 half。

---

## 5. 推荐解法步骤

1. cudaMemset result 或临时 float result。
2. grid-stride loop: sum += __half2float(A[i])*__half2float(B[i])。
3. block reduction。
4. 最终转换 __float2half。

---

## 6. 伪代码骨架

```text
solve(...):
  - cudaMemset result 或临时 float result。
  - grid-stride loop: sum += __half2float(A[i])*__half2float(B[i])。
  - block reduction。
  - 最终转换 __float2half。
```

---

## 7. 复杂度分析

O(N)。

---

## 8. 常见错误

- 不要用 half 做长归约。
- 如果直接 atomicAdd half 支持性有限，建议 float 临时。
- 最后输出类型是 half。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

