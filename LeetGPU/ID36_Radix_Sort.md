# LeetGPU Radix Sort 解题思路

> **难度**：hard  
> **题号**：36  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Radix Sort`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// input, output are device pointers
extern "C" void solve(const unsigned int* input, unsigned int* output, int N) {}
```

---

## 2. 题目摘要

```text
Implement a radix sort algorithm that sorts an array of 32-bit unsigned integers on a GPU.
 The program should take an input array of unsigned integers and sort them in ascending order using the radix sort algorithm.
 The input parameter contains the unsorted array, and the sorted result should be stored in the output array.

 Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final sorted result must be stored in the output array

 Use radix sort algorithm (not other sorting algorithms)

 Sort in ascending order

 Example 1:
```

---

## 3. 核心数学/算法公式

对 unsigned int 做 LSD radix sort，每轮按若干 bit 分桶。

---

## 4. CUDA 并行划分

- 简单版每次处理 1 bit：计算 bit=0/1，prefix sum 得到目标位置。
- 高性能版每轮 4/8 bit，做 histogram + prefix + scatter。
- 需要 ping-pong input/output buffer。

---

## 5. 推荐解法步骤

1. for shift in 0..31。
2. 标记每个元素当前 bit 是否为 0。
3. exclusive scan zeros。
4. 根据 zeros_count 计算 0/1 元素的新位置。
5. scatter 到输出并交换缓冲。

---

## 6. 伪代码骨架

```text
solve(...):
  - for shift in 0..31。
  - 标记每个元素当前 bit 是否为 0。
  - exclusive scan zeros。
  - 根据 zeros_count 计算 0/1 元素的新位置。
  - scatter 到输出并交换缓冲。
```

---

## 7. 复杂度分析

1-bit 版本 O(32N)，多 bit 版本 O((32/r)N)。

---

## 8. 常见错误

- 排序必须稳定，否则高位轮会破坏低位顺序。
- 每轮都需要全局 prefix sum。
- 最后如果轮数为奇数，需要拷回 output。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

