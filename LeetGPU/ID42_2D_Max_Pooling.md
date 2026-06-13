# LeetGPU 2D Max Pooling 解题思路

> **难度**：medium  
> **题号**：42  
> **目标**：根据题目给定输入，在 CUDA 中实现 `2D Max Pooling`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {}
```

---

## 2. 题目摘要

```text
Implement a 2D max pooling operation for image/feature map downsampling.
 The program should take an input tensor and produce an output tensor by applying max pooling with specified kernel size, stride, and padding.

 Input (4x4)

 1
 3
 2
 4

 5
 8
 6
 7

 9
 2
 4
```

---

## 3. 核心数学/算法公式

对 NCHW 输入做窗口最大池化。

---

## 4. CUDA 并行划分

- 一个线程负责一个输出元素 n,c,oh,ow。
- 遍历 kernel_size×kernel_size 窗口。
- 考虑 stride 和 padding。

---

## 5. 推荐解法步骤

1. 根据 H,W,kernel,stride,padding 计算 outH/outW。
2. 解码输出线性 id。
3. 计算输入窗口起点。
4. 遍历窗口，越界跳过或当 -inf。
5. 写最大值。

---

## 6. 伪代码骨架

```text
solve(...):
  - 根据 H,W,kernel,stride,padding 计算 outH/outW。
  - 解码输出线性 id。
  - 计算输入窗口起点。
  - 遍历窗口，越界跳过或当 -inf。
  - 写最大值。
```

---

## 7. 复杂度分析

O(N×C×outH×outW×kernel²)。

---

## 8. 常见错误

- NCHW 索引：((n*C+c)*H+h)*W+w。
- padding 后窗口坐标可能越界。
- 输出尺寸公式要正确。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

