# LeetGPU Batched Matrix Multiplication 解题思路

> **难度**：medium  
> **题号**：30  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Batched Matrix Multiplication`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {}
```

---

## 2. 题目摘要

```text
Implement a batched matrix multiplication in FP32. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
 \[
 C_b = A_b \times B_b
 \]
 All matrices are stored in row-major order and use 32-bit floating point numbers (FP32).

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in the C array

Example 1:

Input:
B = 2, M = 2, K = 3, N = 2
```

---

## 3. 核心数学/算法公式

对每个 batch 独立计算 C_b = A_b × B_b。

---

## 4. CUDA 并行划分

- grid.z 表示 batch，grid.x/y 表示 C tile。
- 可复用普通 tiled GEMM，只是索引加 batch offset。
- 一个 block 计算一个 TILE×TILE 的 C 子块。

---

## 5. 推荐解法步骤

1. batch = blockIdx.z。
2. row/col 定位 C 中元素。
3. 沿 K 维分块加载 A/B 到 shared memory。
4. 累加后写 C[batch,row,col]。

---

## 6. 伪代码骨架

```text
solve(...):
  - batch = blockIdx.z。
  - row/col 定位 C 中元素。
  - 沿 K 维分块加载 A/B 到 shared memory。
  - 累加后写 C[batch,row,col]。
```

---

## 7. 复杂度分析

O(BATCH×M×N×K)。

---

## 8. 常见错误

- A/B/C 的 batch offset 分别是 batch*M*K、batch*K*N、batch*M*N。
- 矩阵维度 M,N,K 不要写反。
- 边界 tile 需要补 0。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

