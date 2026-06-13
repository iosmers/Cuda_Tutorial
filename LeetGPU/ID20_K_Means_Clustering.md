# LeetGPU K-Means Clustering 解题思路

> **难度**：hard  
> **题号**：20  
> **目标**：根据题目给定输入，在 CUDA 中实现 `K-Means Clustering`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) {}
```

---

## 2. 题目摘要

```text
Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for data points, initial centroids, and other parameters, assign each point to the nearest centroid and update the centroids iteratively. The final centroids and labels should be stored in the output arrays.

Implementation Requirements

 External libraries are not permitted

 The solve function signature must remain unchanged

 The final result must be stored in labels, final_centroid_x, and final_centroid_y

Example 1:

Input:
sample_size = 4, k = 2, max_iterations = 10
data_x = [1.0, 2.0, 8.0, 9.0]
data_y = [1.0, 2.0, 8.0, 9.0]
initial_centroid_x = [1.0, 8.0]
initial_centroid_y = [1.0, 8.0]
```

---

## 3. 核心数学/算法公式

迭代执行：每个点分配到最近质心，然后按 label 重新计算质心。

---

## 4. CUDA 并行划分

- assign kernel：一个线程处理一个样本，遍历 k 个质心。
- update kernel：对每个 cluster 累加 x/y 和 count，可用 atomicAdd。
- 再用 normalize kernel 得到新质心。

---

## 5. 推荐解法步骤

1. 初始化质心来自 initial_centroid。
2. 循环 max_iterations 次。
3. 为每个样本找最近 centroid，写 labels。
4. 按 labels 聚合 sum_x/sum_y/count。
5. sum/count 得到 final_centroid。

---

## 6. 伪代码骨架

```text
solve(...):
  - 初始化质心来自 initial_centroid。
  - 循环 max_iterations 次。
  - 为每个样本找最近 centroid，写 labels。
  - 按 labels 聚合 sum_x/sum_y/count。
  - sum/count 得到 final_centroid。
```

---

## 7. 复杂度分析

O(iterations × sample_size × k)。

---

## 8. 常见错误

- 每轮更新前要清零聚合数组。
- 空 cluster 要保留旧质心或按题目约定处理。
- atomicAdd 是简单正确 baseline，优化可做 block 局部聚合。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

