# LeetGPU Softmax 解题思路

> **难度**：medium  
> **题号**：5  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Softmax`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 2. 题目摘要

```text
Write a program that computes the softmax function for an array of 32-bit floating-point numbers on a GPU. The softmax function is defined as follows:

 For an input array \(x\) of length \(n\), the softmax of \(x\), denoted \(\sigma(x)\), is an array of length \(n\) where the \(i\)-th element is:

 \(\sigma(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}\)

 Your solution should handle potential overflow issues by using the "max trick". Subtract the maximum value of the input array from each element before exponentiation.

Implementation Requirements

 Use only native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the array output

Example 1:
```

---

## 3. 核心数学/算法公式

对一维数组做 softmax：output[i] = exp(input[i]-max) / sum_j exp(input[j]-max)。

---

## 4. CUDA 并行划分

- 第一阶段求全局最大值 max，第二阶段求 exp 和 denominator，第三阶段写 output。
- N 较小时可一个 block 完成；N 很大时用多 block 两级 reduction。
- 稳定 softmax 必须先减最大值。

---

## 5. 推荐解法步骤

1. reduce max(input)。
2. reduce sum(exp(input-max))。
3. 每个线程写 output[i] = exp(input[i]-max)/sum。

---

## 6. 伪代码骨架

```text
solve(...):
  - reduce max(input)。
  - reduce sum(exp(input-max))。
  - 每个线程写 output[i] = exp(input[i]-max)/sum。
```

---

## 7. 复杂度分析

O(N)，通常需要 2~3 次遍历。

---

## 8. 常见错误

- 直接 exp(input[i]) 可能溢出。
- softmax 的分母必须是所有元素的和。
- 多 block 求 max/sum 需要跨 kernel 合并。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

