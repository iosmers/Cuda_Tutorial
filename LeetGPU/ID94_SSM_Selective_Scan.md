# LeetGPU SSM Selective Scan 解题思路

> **难度**：medium  
> **题号**：94  
> **目标**：根据题目给定输入，在 CUDA 中实现 `SSM Selective Scan`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>
#include <math.h>

// u, delta, A, B, C, skip, y are device pointers
extern "C" void solve(const float* u, const float* delta, const float* A, const float* B,
                      const float* C, const float* skip, float* y, int batch, int seq_len,
                      int d_model, int d_state) {}
```

---

## 2. 题目摘要

```text
Implement the forward pass of a State Space Model (SSM) selective scan, the core operation in
 Mamba-style sequence models. Given an input sequence u, time-step parameters
 delta, state-transition matrix A, input projection B,
 output projection C, and skip-connection weights skip, compute the
 output sequence y in float32.

 h₀

 h₁

 h₂

 h₃

 Ā
 Ā
 Ā
```

---

## 3. 核心数学/算法公式

选择性状态空间扫描：h_t = f(delta_t,A) * h_{t-1} + g(delta_t,B_t,u_t)，y_t = C_t h_t + skip*u_t。

---

## 4. CUDA 并行划分

- batch 和 channel/d_model 维度并行。
- seq_len 方向有递推依赖，baseline 每个序列串行扫描。
- 高性能可把递推写成 associative scan。

---

## 5. 推荐解法步骤

1. 为每个 batch, d_model 初始化状态 h[d_state]=0。
2. 沿 t 从 0 到 seq_len-1。
3. 根据 delta/A/B/u 更新 d_state 状态。
4. 用 C 投影状态并加 skip。
5. 写 y。

---

## 6. 伪代码骨架

```text
solve(...):
  - 为每个 batch, d_model 初始化状态 h[d_state]=0。
  - 沿 t 从 0 到 seq_len-1。
  - 根据 delta/A/B/u 更新 d_state 状态。
  - 用 C 投影状态并加 skip。
  - 写 y。
```

---

## 7. 复杂度分析

baseline O(batch×seq_len×d_model×d_state)。

---

## 8. 常见错误

- 时间维不能普通逐元素并行。
- A/B/C/delta 的张量布局要按 spec。
- exp(delta*A) 可能需要数值稳定。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

