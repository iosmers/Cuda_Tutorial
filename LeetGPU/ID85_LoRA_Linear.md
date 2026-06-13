# LeetGPU LoRA Linear 解题思路

> **难度**：medium  
> **题号**：85  
> **目标**：根据题目给定输入，在 CUDA 中实现 `LoRA Linear`。本文侧重可提交 baseline 的算法设计、并行划分和常见坑。

---

## 1. CUDA 提交接口

```cpp
#include <cuda_runtime.h>

// x, W, A, B, output are device pointers
extern "C" void solve(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {}
```

---

## 2. 题目摘要

```text
Implement a LoRA (Low-Rank Adaptation) linear layer forward pass. Given an input matrix
 x of shape batch × d_in, a base weight matrix W of
 shape d_out × d_in, a LoRA down-projection matrix A of shape
 rank × d_in, and a LoRA up-projection matrix B of shape
 d_out × rank, compute
 output = x × WT + lora_scale × (x × AT) × BT.
 All tensors are float32.

 x
 B×D_in

 W
 D_out×D_in

 x@Wᵗ
 B×D_out

 A
```

---

## 3. 核心数学/算法公式

output = xW + lora_scale * (xA)B。

---

## 4. CUDA 并行划分

- 主分支 xW 是 GEMM。
- 低秩分支先 tmp=xA，再 tmpB。
- rank 通常较小，可直接 tiled/naive GEMM。

---

## 5. 推荐解法步骤

1. 计算 base = x × W。
2. 计算 tmp = x × A，形状 batch×rank。
3. 计算 lora = tmp × B，形状 batch×d_out。
4. output = base + lora_scale*lora。

---

## 6. 伪代码骨架

```text
solve(...):
  - 计算 base = x × W。
  - 计算 tmp = x × A，形状 batch×rank。
  - 计算 lora = tmp × B，形状 batch×d_out。
  - output = base + lora_scale*lora。
```

---

## 7. 复杂度分析

O(batch×d_in×d_out + batch×d_in×rank + batch×rank×d_out)。

---

## 8. 常见错误

- A/B 维度：A(d_in×rank)，B(rank×d_out)。
- lora_scale 要乘在低秩分支。
- 需要临时 tmp。

---

## 9. 优化方向

- 先写正确 baseline，再用 Nsight/计时定位瓶颈。
- 优先减少 global memory 重复读取；能复用的数据放 shared memory。
- 涉及归约/扫描/排序时，把跨 block 同步拆成多个 kernel。

