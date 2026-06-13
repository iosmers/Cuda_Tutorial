# LeetGPU Fast Fourier Transform 解题思路

> **难度**：hard  
> **题号**：39  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Fast Fourier Transform`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// signal and spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int N) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that computes the Fast Fourier Transform (FFT) of a
 complex-valued 1-D signal. Given an input signal array containing
 N complex numbers stored as interleaved real/imaginary pairs,
 compute the discrete Fourier transform and store the result in the
 spectrum array. The FFT converts a time-domain signal into its
 frequency-domain representation using the formula: \[ X_k = \sum_{n=0}^{N-1}
 x_n \cdot e^{-j 2\pi kn / N} \quad \text{for } k = 0, 1, \ldots, N-1 \] The
 FFT algorithm reduces the computational complexity from O(N²) to O(N log N) by
 exploiting symmetries in the twiddle factors.

Implementation Requirements

 External libraries (cuFFT etc.) are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in the spectrum array
```

---

## 3. 核心数学/算法公式

一维 FFT：Cooley-Tukey radix-2，把 DFT 分解成 logN 层 butterfly。

---

## 4. CUDA 并行划分

- 每层 butterfly 可以并行；一个线程处理一对元素。
- 输入/输出复数通常用 interleaved float：real, imag。
- N 非 2 的幂时 fallback 到 DFT 或按题目约束处理。

---

## 5. 推荐解法步骤

1. bit-reversal 重排。
2. for stage=2..N 翻倍。
3. 计算 twiddle e^{-2πik/stage}。
4. 并行 butterfly 更新。
5. 写 spectrum。

---

## 6. 伪代码骨架

```text
solve(...):
  - bit-reversal 重排。
  - for stage=2..N 翻倍。
  - 计算 twiddle e^{-2πik/stage}。
  - 并行 butterfly 更新。
  - 写 spectrum。
```

---

## 7. 复杂度分析

FFT O(N logN)，朴素 DFT O(N²)。

---

## 8. 常见错误

- 复数布局要确认。
- 符号方向 FFT/DFT 要按题目定义。
- 每一 stage 之间需要 kernel 同步或在单 kernel 内只处理小 N。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

