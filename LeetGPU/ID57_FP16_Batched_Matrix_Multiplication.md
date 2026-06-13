# LeetGPU FP16 Batched Matrix Multiplication 解题思路

> **难度**：medium  
> **题号**：57  
> **目标**：根据题目给定输入，在 CUDA 中实现 `FP16 Batched Matrix Multiplication`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {}
```

---

## 2. 题目摘要

```text
Implement a batched matrix multiplication in FP16. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
 \[
 C_b = A_b \times B_b
 \]
 All matrices are stored in row-major order and use 16-bit floating point numbers (FP16/half). Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16

 The final result must be stored in the C array as half

Example 1:
```

---

## 3. 核心数学/算法公式

批量 FP16 GEMM：C_b=A_b×B_b。

---

## 4. CUDA 并行划分

- 与 Batched GEMM 相同，但输入输出 half。
- 累加建议使用 float。
- grid.z 表示 batch。

---

## 5. 推荐解法步骤

1. 每个 block 计算一个 batch 的 TILE×TILE C。
2. shared memory 加载 half A/B tile。
3. float acc 累加。
4. 写回 half C。

---

## 6. 伪代码骨架

```text
solve(...):
  - 每个 block 计算一个 batch 的 TILE×TILE C。
  - shared memory 加载 half A/B tile。
  - float acc 累加。
  - 写回 half C。
```

---

## 7. 复杂度分析

O(BATCH×M×N×K)。

---

## 8. 常见错误

- half 累加精度差，使用 float acc。
- batch offset 正确。
- 可进一步用 WMMA/Tensor Core。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

