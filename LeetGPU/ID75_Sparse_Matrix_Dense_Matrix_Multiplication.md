# LeetGPU Sparse Matrix-Dense Matrix Multiplication 解题思路

> **难度**：medium  
> **题号**：75  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Sparse Matrix-Dense Matrix Multiplication`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K, int nnz) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that multiplies a sparse matrix A of dimensions M × N
 by a dense matrix B of dimensions N × K, producing a dense output matrix
 C of dimensions M × K.
 All matrices are stored in row-major order using 32-bit floats.
 The matrix A is approximately 60–70% sparse (i.e., 60–70% of elements are zero),
 and nnz gives the number of non-zero elements in A.

 Mathematically, the operation is defined as:
 \[
 C_{ij} = \sum_{k=0}^{N-1} A_{ik} \cdot B_{kj} \quad \text{for} \quad i = 0, \ldots, M-1,\; j = 0, \ldots, K-1
 \]

Implementation Requirements

 Use only GPU native features (external libraries are not permitted)

 The solve function signature must remain unchanged
```

---

## 3. 核心数学/算法公式

C = A × B，其中 A 以 dense row-major 给出但多数为 0，B 是 dense。

---

## 4. CUDA 并行划分

- 一个 block/线程块负责 C 的 tile。
- 因为没有 CSR 索引，baseline 仍扫描 K 维并跳过 A 中 0。
- 可一个 block 负责 A 的一行和 B 的多个列。

---

## 5. 推荐解法步骤

1. row,col 定位 C。
2. for t in 0..K-1：a=A[row*K+t]，如果非零则 acc += a*B[t*N+col]。
3. 写 C[row*N+col]。

---

## 6. 伪代码骨架

```text
solve(...):
  - row,col 定位 C。
  - for t in 0..K-1：a=A[row*K+t]，如果非零则 acc += a*B[t*N+col]。
  - 写 C[row*N+col]。
```

---

## 7. 复杂度分析

接口限制下 baseline O(M×N×K)。

---

## 8. 常见错误

- nnz 不能直接跳过，因为没有非零索引。
- B 索引是 t*N+col。
- 如果多线程累加同一个 C，需要 reduction/atomic；一线程一元素最简单。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

