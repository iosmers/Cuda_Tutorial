# LeetGPU 2D FFT 解题思路

> **难度**：medium  
> **题号**：78  
> **目标**：根据题目给定输入，在 CUDA 中实现 `2D FFT`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// signal, spectrum are device pointers
extern "C" void solve(const float* signal, float* spectrum, int M, int N) {}
```

---

## 2. 题目摘要

```text
Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU.
 Given a 2D complex input signal of shape M × N, compute its 2D DFT spectrum
 using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each
 column of the result. All values are 32-bit floating point.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in spectrum

 The input and output are stored as 1D arrays of interleaved real and imaginary parts in
 row-major order: element x[m, n] has its real part at index
 2*(m*N + n) and imaginary part at index 2*(m*N + n) + 1

Example
```

---

## 3. 核心数学/算法公式

二维 DFT/FFT 可分离：先对每一行做 1D FFT，再对每一列做 1D FFT。

---

## 4. CUDA 并行划分

- 如果 M,N 小，可直接每个频点做朴素 DFT。
- FFT 版对行/列分别做 butterfly。
- 需要中间矩阵保存行变换结果。

---

## 5. 推荐解法步骤

1. row FFT: 对每一行 signal 做 1D FFT。
2. column FFT: 对每一列中间结果做 1D FFT。
3. 写 spectrum。

---

## 6. 伪代码骨架

```text
solve(...):
  - row FFT: 对每一行 signal 做 1D FFT。
  - column FFT: 对每一列中间结果做 1D FFT。
  - 写 spectrum。
```

---

## 7. 复杂度分析

FFT O(MN(logM+logN))，朴素 DFT O(M²N²)。

---

## 8. 常见错误

- 复数布局要确认。
- 二维频率索引符号和归一化按题目。
- 行列两阶段之间需要同步/单独 kernel。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

