# LeetGPU 2D Jacobi Stencil 解题思路

> **难度**：medium  
> **题号**：69  
> **目标**：根据题目给定输入，在 CUDA 中实现 `2D Jacobi Stencil`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int rows, int cols) {}
```

---

## 2. 题目摘要

```text
Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil:
 each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom,
 left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged
 from the input to the output.

 5-Point Jacobi Stencil

 T
 L
 (2,2)
 R
 B

 Center cell

 Neighbors

 Boundary
```

---

## 3. 核心数学/算法公式

五点 Jacobi：output[r,c] 通常为中心和上下左右的平均/加权。

---

## 4. CUDA 并行划分

- 一个线程负责一个网格点。
- 内部点读取 up/down/left/right/center。
- 边界按题目约定保持原值或特殊处理。

---

## 5. 推荐解法步骤

1. 计算 r,c。
2. 如果是边界，复制 input 或按 spec。
3. 否则 output = 0.25*(neighbors) 或题目公式。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 r,c。
  - 如果是边界，复制 input 或按 spec。
  - 否则 output = 0.25*(neighbors) 或题目公式。
```

---

## 7. 复杂度分析

O(rows×cols)。

---

## 8. 常见错误

- 边界条件最容易错。
- 输入输出必须分离，不能原地更新。
- row-major 索引 r*cols+c。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

