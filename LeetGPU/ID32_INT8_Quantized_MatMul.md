# LeetGPU INT8 Quantized MatMul 解题思路

> **难度**：medium  
> **题号**：32  
> **目标**：根据题目给定输入，在 CUDA 中实现 `INT8 Quantized MatMul`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// A, B, C are device pointers
extern "C" void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K,
                      float scale_A, float scale_B, float scale_C, int zero_point_A,
                      int zero_point_B, int zero_point_C) {}
```

---

## 2. 题目摘要

```text
Implement a quantized matrix multiplication program for 8-bit signed integer matrices. Given two input matrices A of dimensions \(M \times K\) and B of dimensions \(K \times N\), quantization scales scale_A, scale_B, output scale scale_C, zero-points zero_point_A, zero_point_B, zero_point_C, compute:
 \[
 C_{\text{quant}}(i, j) = \mathrm{clamp}\left(
 \mathrm{round}\left(
 \frac{
 \sum_{k=0}^{K-1} (A_{ik} - z_A)(B_{kj} - z_B) \cdot s_A s_B
 }{s_C}
 \right) + z_C,\ -128,\ 127
 \right)
 \]
 where s_A = scale_A, z_A = zero_point_A, etc.

 Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged
```

---

## 3. 核心数学/算法公式

反量化计算：C_float = scale_A*scale_B*sum((A-zpA)*(B-zpB))，再量化到 int8。

---

## 4. CUDA 并行划分

- 一个线程或一个 block 负责一个 C[row,col]。
- 累加使用 int32，最后乘 scale 并加 zero_point_C。
- 可用 tiled shared memory 优化 A/B 读取。

---

## 5. 推荐解法步骤

1. 读取 int8 A/B 并减 zero point。
2. int32 acc += a*b。
3. float y = acc * scale_A * scale_B / scale_C + zero_point_C。
4. round + clamp 到 [-128,127] 写 C。

---

## 6. 伪代码骨架

```text
solve(...):
  - 读取 int8 A/B 并减 zero point。
  - int32 acc += a*b。
  - float y = acc * scale_A * scale_B / scale_C + zero_point_C。
  - round + clamp 到 [-128,127] 写 C。
```

---

## 7. 复杂度分析

O(M×N×K)。

---

## 8. 常见错误

- 累加不能用 int8，必须 int32。
- 量化缩放方向要确认：输出 int8 需要除以 scale_C。
- 最后一定要 clamp。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

