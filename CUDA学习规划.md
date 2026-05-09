# CUDA 编程学习规划

## 阶段一：入门阶段

**目标**：理解 GPU 并行模型，能写并运行第一个 kernel。

### 学习内容
- GPU vs CPU 架构差异
- 编程模型：Host/Device、Grid/Block/Thread、SM/Warp
- 关键语法：`__global__`、`__device__`、`<<<grid, block>>>`
- 内存管理：`cudaMalloc` / `cudaMemcpy` / `cudaFree`
- 线程索引：`int i = blockIdx.x * blockDim.x + threadIdx.x;`

### 代码练习
1. **Hello World from GPU**：熟悉 kernel 启动语法

    ```cpp
    __global__ void hello() {
        printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
    }
    int main() { hello<<<2,4>>>(); cudaDeviceSynchronize(); }
    ```

2. **向量加法** `C = A + B`（N=1<<20），对比 CPU 版本
3. **SAXPY**：`y = a*x + y`
4. **矩阵加法**（2D grid/block）

### 阶段产出
GitHub 仓库搭建，整理以上 4 个示例 + README。

---

## 阶段二：内存模型与调试

**目标**：掌握 CUDA 内存层次，会用 shared memory 优化，会调试。

### 学习内容
- 内存层次：Global / Shared / Constant / Register / Local
- `__shared__` 声明、`__syncthreads()` 同步
- Bank conflict 概念
- 错误处理：`CUDA_CHECK` 宏封装
- 调试工具：`compute-sanitizer`（越界检查）
- 计时：`cudaEvent_t` 测性能

### 代码练习
1. **矩阵乘法 - Naive 版**（仅 global memory）
2. **矩阵乘法 - Tiled 版**（shared memory 分块），对比加速比
3. **转置矩阵**（体会合并访存）

### 阶段产出
一份性能对比表格（Naive vs Tiled），输出每个 kernel 的 GFLOPS。

---

## 阶段三：性能优化

**目标**：能独立识别瓶颈并优化 kernel。

### 学习内容
- **访存优化**：Coalesced access、AoS vs SoA
- **Warp 级原语**：`__shfl_sync`、`__ballot_sync`、warp divergence
- **原子操作**：`atomicAdd` 及其性能陷阱
- **Stream 并发**：`cudaStream_t`，计算与拷贝重叠
- **Profiling**：Nsight Compute / Nsight Systems 入门

### 代码练习
1. **并行归约（Reduction）**：实现经典 7 版优化（参考 NVIDIA《Optimizing Parallel Reduction》）
   - v1: interleaved addressing
   - v2: 消除 bank conflict
   - v3: 首次加载时做加法
   - v4: warp shuffle
   - ...
2. **直方图统计（Histogram）**：先原子版，再 shared memory 分块版
3. **Stream 实战**：双 stream 实现拷贝与 kernel 重叠

### 阶段产出
用 Nsight Compute 对 reduction 各版本做分析报告，记录每版瓶颈（Memory Bound / Compute Bound）。

---

## 阶段四：进阶与实战项目

**目标**：会用 CUDA 生态库，完成一个综合项目。

### 学习内容
- **CUDA 库**：cuBLAS、cuRAND、Thrust
- **进阶主题**（选一深入）：
  - Unified Memory（`cudaMallocManaged`）
  - Cooperative Groups
  - Tensor Cores（WMMA API）入门
- **延伸阅读**：Triton、CUTLASS、FlashAttention 思想

### 代码练习
1. **cuBLAS GEMM**：对比自写 tiled 版本
2. **Thrust 实战**：sort / reduce / scan

### 综合项目（二选一）
- **选项 A**：小型 CNN 推理引擎（Conv2D + ReLU + MaxPool + FC），跑通 MNIST
- **选项 B**：K-means 聚类 GPU 版 或 N-body 模拟

### 阶段产出
- 完整项目代码 + 性能报告（vs CPU baseline）
- 学习笔记整理成博客/文档

---

## 贯穿始终的原则

1. **每天必须写代码**：理论看完立刻动手
2. **每个 kernel 必测性能**：`cudaEvent` 计时 + 加速比
3. **对比 CPU 基线**：验证正确性优先于性能
4. **Profiling 驱动优化**：用数据说话，不凭感觉

---

## 推荐资源

- 官方：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- 书籍：《Professional CUDA C Programming》(Cheng)、《Programming Massively Parallel Processors》(Kirk & Hwu)
- 代码：[NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- 中文视频：B 站"谭升"的 CUDA 系列
