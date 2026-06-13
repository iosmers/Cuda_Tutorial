# LeetGPU All-Pairs Shortest Paths 解题思路

> **难度**：hard  
> **题号**：73  
> **目标**：根据题目给定输入，在 CUDA 中实现 `All-Pairs Shortest Paths`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// dist, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* dist, float* output, int N) {}
```

---

## 2. 题目摘要

```text
Given a weighted directed graph of N vertices represented as an
 N × N distance matrix, compute the shortest path distance between
 every pair of vertices using the Floyd-Warshall algorithm. The matrix is stored as a flat array in
 row-major order: dist[i * N + j] is the weight of the directed edge from vertex
 i to vertex j. A value of +infinity means no direct edge
 exists. The diagonal is always zero. For each intermediate vertex k from 0 to N - 1
 (in order), update all pairs:

 \[
 \text{output}[i][j] = \min\!\bigl(\text{output}[i][j],\;
 \text{output}[i][k] + \text{output}[k][j]\bigr)
 \quad \forall\, i, j
 \]

Implementation Requirements

 Use only native features (external libraries are not permitted)
```

---

## 3. 核心数学/算法公式

Floyd-Warshall：for k, dist[i,j]=min(dist[i,j],dist[i,k]+dist[k,j])。

---

## 4. CUDA 并行划分

- 每个 k 是一轮全局同步。
- 每轮 kernel 中一个线程负责一个 (i,j)。
- 更快版本使用 blocked Floyd-Warshall。

---

## 5. 推荐解法步骤

1. 拷贝 dist 到 output 或原地更新。
2. for k in 0..N-1 launch kernel。
3. kernel 更新所有 i,j。
4. 处理 INF 避免溢出。

---

## 6. 伪代码骨架

```text
solve(...):
  - 拷贝 dist 到 output 或原地更新。
  - for k in 0..N-1 launch kernel。
  - kernel 更新所有 i,j。
  - 处理 INF 避免溢出。
```

---

## 7. 复杂度分析

O(N³)。

---

## 8. 常见错误

- k 轮之间必须同步，不能一个 kernel 内跨 grid 同步。
- INF + value 要避免产生错误。
- 输出矩阵 row-major。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

