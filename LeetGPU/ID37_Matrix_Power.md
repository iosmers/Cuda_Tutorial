# LeetGPU Matrix Power 解题思路

> **难度**：medium  
> **题号**：37  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Matrix Power`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int P) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that raises a square matrix \(A\) of size \(N \times N\) to an integer power \(P\).

 The solve function receives a flattened input matrix input (row-major order), an empty output matrix output of the same size, the dimension N, and the exponent P.

 You must compute \(\text{output} = A^{P}\) where matrix multiplication is standard dense multiplication over 32-bit floating point numbers.

 Implementation Requirements

 External libraries are not permitted.

 The solve function signature must remain unchanged.

 The final result must be written to the output array in row-major order.

 Example 1:

 Input:
 input = [[1.0, 2.0],
```

---

## 3. 核心数学/算法公式

计算 output = input^P。矩阵乘法反复应用；P 是正整数。

---

## 4. CUDA 并行划分

- 使用 tiled GEMM 作为基础算子。
- P 小可重复乘 P-1 次；P 大可用 exponentiation by squaring。
- 需要临时矩阵 ping-pong。

---

## 5. 推荐解法步骤

1. 初始化 result 为 identity。
2. base=input。
3. while P>0：如果 P&1，result=result×base；base=base×base；P>>=1。
4. 拷贝 result 到 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - 初始化 result 为 identity。
  - base=input。
  - while P>0：如果 P&1，result=result×base；base=base×base；P>>=1。
  - 拷贝 result 到 output。
```

---

## 7. 复杂度分析

快速幂 O(logP × N³)，朴素 O(P×N³)。

---

## 8. 常见错误

- 需要临时 device buffer。
- 矩阵是 N×N row-major。
- 浮点误差随乘法次数增长。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

