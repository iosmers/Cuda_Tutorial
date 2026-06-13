# LeetGPU Weight Dequantization 解题思路

> **难度**：medium  
> **题号**：64  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Weight Dequantization`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix X of shape [M, N] containing quantized values and a scale matrix S of shape [ceil(M/T), ceil(N/T)], where T is the tile size.

 For each element \(X_{i,j}\), the corresponding scale factor is \(S_{row, col}\) where \(row = \lfloor i / T \rfloor\) and \(col = \lfloor j / T \rfloor\).
 The output \(Y_{i,j}\) should be computed as:
 \[
 Y_{i,j} = X_{i,j} \times S_{row, col}
 \]

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in the output buffer Y

Example 1:
```

---

## 3. 核心数学/算法公式

把量化权重 X 按 tile/group scale S 反量化到 Y。

---

## 4. CUDA 并行划分

- 一个线程负责一个 Y[m,n]。
- 根据 n 或 tile id 找对应 scale。
- Y = X * scale。

---

## 5. 推荐解法步骤

1. idx -> row,col。
2. tile = col / TILE_SIZE 或按题目定义。
3. Y[idx] = X[idx] * S[row,num_tiles + tile]。

---

## 6. 伪代码骨架

```text
solve(...):
  - idx -> row,col。
  - tile = col / TILE_SIZE 或按题目定义。
  - Y[idx] = X[idx] * S[row,num_tiles + tile]。
```

---

## 7. 复杂度分析

O(M×N)。

---

## 8. 常见错误

- scale 的布局要确认是 per-row per-tile 还是全局 per-tile。
- TILE_SIZE 参数不要写死。
- 边界 tile 可能不足 TILE_SIZE。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

