# LeetGPU Multi-Agent Simulation 解题思路

> **难度**：hard  
> **题号**：14  
> **目标**：根据题目给定输入，在 CUDA 中实现 `Multi-Agent Simulation`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {}
```

---

## 2. 题目摘要

```text
Implement a program for a multi-agent flocking simulation (boids). The input consists of:

 An array agents containing N agents, where N is the total number of agents

 Each agent occupies 4 consecutive 32-bit floating point numbers in the array: \([x, y, v_x, v_y]\), where:

 \((x, y)\) represents the agent's position in 2D space

 \((v_x, v_y)\) represents the agent's velocity vector

 The total array size is 4 * N floats, with agent \(i\)'s data stored at indices [4i, 4i+1, 4i+2, 4i+3]

Simulation Rules

 For each agent \(i\), identify all neighbors \(j\) (where \(i \neq j\)) within radius \(r = 5.0\) using:
 \[
 \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} Compute average velocity of neighboring agents:
 \[
```

---

## 3. 核心数学/算法公式

对 N 个 agent 更新下一步状态；每个 agent 的新状态由自身和邻居/其他 agent 决定。

---

## 4. CUDA 并行划分

- 一个线程负责一个 agent。
- 如果每个 agent 需要检查所有其他 agent，则用 O(N²) baseline；N 较小时足够。
- 如果有空间半径，可用 grid/binning 降低邻居搜索成本。

---

## 5. 推荐解法步骤

1. 读取 agent i 的位置/速度等状态。
2. 遍历可能影响它的 agent j，累积作用力/规则。
3. 根据题目规则更新状态。
4. 写到 agents_next，避免原地覆盖。

---

## 6. 伪代码骨架

```text
solve(...):
  - 读取 agent i 的位置/速度等状态。
  - 遍历可能影响它的 agent j，累积作用力/规则。
  - 根据题目规则更新状态。
  - 写到 agents_next，避免原地覆盖。
```

---

## 7. 复杂度分析

baseline O(N²)，空间分桶可接近 O(N)。

---

## 8. 常见错误

- 不能直接写回 agents，否则其他线程读到混合的新旧状态。
- float 数组通常是 AoS，需要按 stride 解码字段。
- 双缓冲 agents -> agents_next 是关键。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

