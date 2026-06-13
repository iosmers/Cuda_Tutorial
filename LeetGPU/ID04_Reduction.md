# LeetGPU Reduction 解题思路

> **难度**：medium  
> **题号**：4  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Reduction`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {}
```

---

## 2. 题目摘要

```text
Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum.
 The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements

 Use only GPU native features (external libraries are not permitted)

 The solve function signature must remain unchanged

 The final result must be stored in the output variable

Example 1:

Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0

Example 2:
```

---

## 3. 核心数学/算法公式

计算 output[0] = sum(input[i])。这是最经典的一维并行归约。

---

## 4. CUDA 并行划分

- 每个线程用 grid-stride loop 累加多个元素，得到 local_sum。
- 每个 block 用 shared memory 把 local_sum 归约成 block_sum。
- block_sum 可以 atomicAdd 到 output，也可以写入中间数组后第二阶段归约。

---

## 5. 推荐解法步骤

1. cudaMemset(output, 0)。
2. 启动若干 block，每个线程累加 input[i]。
3. block 内 reduction。
4. 每个 block 的 thread0 将 block_sum atomicAdd 到 output。

---

## 6. 伪代码骨架

```text
solve(...):
  - cudaMemset(output, 0)。
  - 启动若干 block，每个线程累加 input[i]。
  - block 内 reduction。
  - 每个 block 的 thread0 将 block_sum atomicAdd 到 output。
```

---

## 7. 复杂度分析

O(N)，global atomic 次数约等于 block 数。

---

## 8. 常见错误

- 忘记清零 output 会在旧值上累加。
- 每个元素都 atomicAdd 会很慢。
- N 很大时用 grid-stride loop，不要只处理一个元素。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

