---
name: cuda-learning
description: >-
  Guides CUDA study, exercises, and code review in this repository using the
  four-stage curriculum in README.md and stage markdowns. Emphasizes correct
  kernel launches, memory hierarchy, CPU baseline checks, timing with cudaEvent,
  and profiling. Use when the user learns CUDA, writes or debugs CUDA C++ here,
  or asks about grids, blocks, shared memory, streams, reduction, or kernel
  optimization.
---

# CUDA 学习（本仓库）

## 课程地图

按顺序引用仓库内材料，不要跳过「可运行代码 + 验证」：

1. **总览与路线**：[README.md](../../../README.md)
2. **分阶段正文**：`Stage1_入门基础.md`、`stage2_内存模型与调试.md`、`stage3_性能优化.md`、`stage4_进阶与实战.md`

讲解或出题时对齐当前阶段目标；跨阶段内容只作预告，避免一次灌入过多概念。

## 助手行为准则

- **先正确再快**：先与 CPU 或朴素实现逐元素对比；再谈 GFLOPS、Occupancy。
- **每个 kernel 说清楚三件事**：线程如何映射到数据（索引公式）、边界条件、需要哪种同步（`__syncthreads` 等）。
- **启动配置**：`<<<gridDim, blockDim, sharedMemBytes, stream>>>`；一维问题时写清 `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock` 这类取整。
- **错误与调试**：鼓励封装 `CUDA_CHECK`（或等价宏）；复杂问题提到 `compute-sanitizer`、以及阶段三起的 Nsight Compute / Systems。
- **性能习惯**：阶段二起默认提 `cudaEvent_t` 计时；优化前先说清瓶颈类型（memory-bound / compute-bound）。

## 写代码时的默认约定

- 设备指针与 host 数据分清；`cudaMalloc` / `cudaMemcpy` / `cudaFree` 成对出现。
- 全局内存访问尽量合并（coalesced）；涉及 shared memory 时提醒 bank conflict。
- 矩阵类题目优先一维线性存储 + `row * stride + col` 索引，与 README 中二维块示例一致。

## 输出结构建议

回答学习问题时可用固定小节（按需删减）：

1. **概念**：一两句话对应 GPU 执行模型或内存层次。
2. **索引 / 并行映射**：写出 `blockIdx` / `threadIdx` 如何对应数据下标。
3. **代码**：完整可编译片段或明确补丁位置。
4. **如何验证**：CPU 对比或小规模用例。
5. **下一步**：指向 README 中下一道练习或下一阶段。

## 扩展参考

构建命令、检查宏示例与常见陷阱见 [reference.md](reference.md)。
