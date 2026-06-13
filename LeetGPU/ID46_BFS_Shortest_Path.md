# LeetGPU BFS Shortest Path 解题思路

> **难度**：hard  
> **题号**：46  
> **目标**：根据题目给定输入，在 CUDA 中实现 `BFS Shortest Path`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, int start_row,
                      int start_col, int end_row, int end_col) {}
```

---

## 2. 题目摘要

```text
Implement a program that finds the shortest path in an unweighted 2D grid using Breadth-First Search (BFS). Given a grid with obstacles and start/end positions, return the minimum number of steps needed to reach the destination.

 #
 #
 #
 #
 #
 #
 #

 S
 Start

 E
 End

 1
 2
```

---

## 3. 核心数学/算法公式

在 grid 图上求从 start 到 end 的最短路径长度。

---

## 4. CUDA 并行划分

- 使用 frontier BFS：每轮并行扩展当前 frontier。
- dist/result 数组初始化为 -1。
- atomicCAS 设置未访问邻居距离。
- 循环直到 frontier 为空或到达终点。

---

## 5. 推荐解法步骤

1. 初始化 start 的距离为 0，frontier[start]=1。
2. 每一层 kernel 扩展四邻居。
3. 新访问节点写 next_frontier。
4. 交换 frontier，层数加一。
5. 读 end 的距离写 result。

---

## 6. 伪代码骨架

```text
solve(...):
  - 初始化 start 的距离为 0，frontier[start]=1。
  - 每一层 kernel 扩展四邻居。
  - 新访问节点写 next_frontier。
  - 交换 frontier，层数加一。
  - 读 end 的距离写 result。
```

---

## 7. 复杂度分析

O(rows×cols) 节点级别。

---

## 8. 常见错误

- BFS 每一层之间需要 kernel launch 同步。
- 障碍物 grid 值要按题目定义跳过。
- atomicCAS 防止多个线程重复访问同一节点。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

