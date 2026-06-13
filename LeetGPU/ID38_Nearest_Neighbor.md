# LeetGPU Nearest Neighbor 解题思路

> **难度**：medium  
> **题号**：38  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Nearest Neighbor`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// points and indices are device pointers
extern "C" void solve(const float* points, int* indices, int N) {}
```

---

## 2. 题目摘要

```text
Implement a GPU program that, for N three-dimensional points stored on the device, fills indices[i] with the index j ≠ i of the point closest to points[i]. Comparing squared Euclidean distance is sufficient—you do not need to compute square-roots.

Implementation Requirements

 The solve function signature must remain unchanged

 External libraries are not permitted

 The final result must be stored in the indices array

Example 1:

Input: points = [(0,0,0), (1,0,0), (5,5,5)]
 indices = [-1, -1, -1]
 N = 3
Output: indices = [1, 0, 1] # 0⇆1 are nearest, 2 is closest to 1

Constraints
```

---

## 3. 核心数学/算法公式

对每个点 i，寻找距离最近的另一个点 j。

---

## 4. CUDA 并行划分

- 一个 block 负责一个点 i，block 内线程并行扫描候选 j。
- 每个线程维护局部 best distance/index。
- block 内 reduction 选最小距离。

---

## 5. 推荐解法步骤

1. 读取 point i。
2. 遍历所有 j，跳过 i。
3. 计算 squared distance。
4. 按 distance 最小、index 最小打破平局。
5. 写 indices[i]。

---

## 6. 伪代码骨架

```text
solve(...):
  - 读取 point i。
  - 遍历所有 j，跳过 i。
  - 计算 squared distance。
  - 按 distance 最小、index 最小打破平局。
  - 写 indices[i]。
```

---

## 7. 复杂度分析

O(N²)。

---

## 8. 常见错误

- 不要对距离开 sqrt，比较平方距离即可。
- 跳过自身 j==i。
- points 可能是二维/多维，按 spec 的布局解码。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

