# LeetGPU INT4 Weight-Only Quantized MatMul 解题思路

> **难度**：medium  
> **题号**：81  
> **目标**：根据题目给定输入，在 CUDA 中实现 `INT4 Weight-Only Quantized MatMul`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// x, w_q, scales, y are device pointers
extern "C" void solve(const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
                      int N, int K, int group_size) {}
```

---

## 2. 题目摘要

```text
Implement a weight-only INT4 quantized matrix multiplication (W4A16), a core kernel used in
 modern LLM inference. Given a float16 activation matrix x of shape
 M × K and a weight matrix stored in packed INT4 format, compute the output
 matrix y = x × WT of shape M × N, where
 W is the dequantized float16 weight matrix of shape N × K.

 Packing format: Each byte of w_q stores two INT4 weights. The
 high nibble (bits 7–4) holds weight w[n, 2i] and the low nibble (bits
 3–0) holds w[n, 2i+1]. INT4 values are stored unsigned in the range
 [0, 15] with an offset of 8, so the signed weight is nibble − 8,
 giving values in [−8, 7].

 Dequantization: Weights are dequantized group-wise. Each contiguous block of
 group_size weights along the K dimension shares one float16 scale:

W[n, k] = (w_q_nibble[n, k] - 8) * scales[n, k // group_size]

Implementation Requirements
```

---

## 3. 核心数学/算法公式

W 是 int4 打包权重，x 是 half；反量化 W 后做 y=xW。

---

## 4. CUDA 并行划分

- 一个线程/warp 负责 y[m,n] 或 tile。
- 从 uint8_t 中解包两个 4-bit 权重。
- 按 group_size 读取 scale，half/float 累加。

---

## 5. 推荐解法步骤

1. 定位 K 维对应的 packed byte。
2. 解出 low/high nibble，并还原 signed/zero-point 形式。
3. w = dequant(q, scale[group])。
4. acc += x[m,k]*w。
5. 写 half y。

---

## 6. 伪代码骨架

```text
solve(...):
  - 定位 K 维对应的 packed byte。
  - 解出 low/high nibble，并还原 signed/zero-point 形式。
  - w = dequant(q, scale[group])。
  - acc += x[m,k]*w。
  - 写 half y。
```

---

## 7. 复杂度分析

O(M×N×K)，访存比 FP16 GEMM 更小。

---

## 8. 常见错误

- int4 nibble 顺序最容易错。
- scale 按 group_size 分组。
- weight-only：只有 W 量化，activation x 是 half。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

