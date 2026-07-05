# GPU 编程全栈学习资源整理

> 目标：从 GPU 架构基础、CUDA 编程、性能优化，一直到 Triton/CUTLASS、LLM 推理系统和 GPU 模拟器，整理一条比较完整的学习路线。

---

## 0. 总体学习路线

推荐按照下面顺序学习：

```text
GPU 架构与并行计算基础
  ↓
CUDA 编程模型
  ↓
CUDA 内存层级与性能优化
  ↓
经典 Kernel：reduce / scan / transpose / GEMM
  ↓
Profiling：Nsight Compute / Nsight Systems
  ↓
高性能库：cuBLAS / CUTLASS / cuDNN / FlashAttention
  ↓
Triton 与现代 AI Kernel
  ↓
LLM 推理系统：TensorRT-LLM / vLLM / llama.cpp / llm.c
  ↓
GPU 架构模拟：GPGPU-Sim / Accel-Sim
```

一句话理解：

| 层级 | 你需要掌握什么 |
|---|---|
| 架构层 | SM、Warp、Register、Shared Memory、L1/L2、HBM/GDDR、Tensor Core |
| 编程层 | CUDA thread/block/grid、memory copy、kernel launch、stream、event |
| 优化层 | memory coalescing、occupancy、bank conflict、tiling、pipeline |
| 算子层 | reduction、scan、transpose、GEMM、softmax、layernorm、attention |
| 框架层 | PyTorch extension、Triton、CUTLASS、TensorRT-LLM、vLLM |
| 系统层 | KV Cache、batching、parallelism、profiling、multi-GPU、通信 |
| 架构研究层 | GPGPU-Sim、Accel-Sim、trace、warp scheduler、memory subsystem |

---

## 1. 官方 CUDA 资料

这些是最权威、最应该优先收藏的资料。

| 资源 | 地址 | 适合学习 |
|---|---|---|
| CUDA C++ Programming Guide | <https://docs.nvidia.com/cuda/cuda-c-programming-guide/> | CUDA 编程模型、线程层级、内存模型、同步机制 |
| CUDA C++ Best Practices Guide | <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/> | CUDA 性能优化、访存优化、occupancy、profiling 思路 |
| CUDA Samples | <https://github.com/NVIDIA/cuda-samples> | 官方示例代码，适合边看边跑 |
| Nsight Compute 文档 | <https://docs.nvidia.com/nsight-compute/> | 单个 CUDA kernel 的性能分析 |
| Nsight Systems 文档 | <https://docs.nvidia.com/nsight-systems/> | 系统级时间线分析、CPU/GPU overlap、stream 并发 |

### 推荐阅读顺序

```text
CUDA Programming Guide 前几章
  ↓
CUDA Samples: vectorAdd / matrixMul / reduction / scan
  ↓
Best Practices Guide 中的 memory optimization
  ↓
Nsight Compute 看 kernel 指标
```

---

## 2. 系统课程与教学型资料

这些资料更适合作为系统学习路径，而不是零散查文档。

| 资源 | 地址 | 特点 |
|---|---|---|
| OLCF CUDA Training Series | <https://github.com/olcf/cuda-training-series> | Oak Ridge 国家实验室 CUDA 培训，系统、实用 |
| CUDA MODE Lectures | <https://github.com/cuda-mode/lectures> | 现代 CUDA/Triton/AI kernel 课程，非常适合 LLM Infra 方向 |
| Stanford CS149 Parallel Computing | <https://gfxcourses.stanford.edu/cs149/fall21> | 并行计算基础，包括 SIMD、多线程、GPU、并行架构 |
| PMPP 相关资料 | <https://github.com/R100001/Programming-Massively-Parallel-Processors> | 《Programming Massively Parallel Processors》相关练习/代码参考 |

### 建议

- 如果你刚开始学 CUDA：优先看 **OLCF CUDA Training Series**。
- 如果你目标是 AI Kernel / LLM Infra：优先看 **CUDA MODE Lectures**。
- 如果你并行计算基础不牢：补 **Stanford CS149**。

---

## 3. 练习型项目

这些项目适合动手写 kernel，训练 GPU 编程思维。

| 资源 | 地址 | 适合学习 |
|---|---|---|
| GPU Puzzles | <https://github.com/srush/GPU-Puzzles> | 用 puzzle 方式入门 GPU 并行思维 |
| Triton Puzzles | <https://github.com/srush/Triton-Puzzles> | 入门 Triton kernel，理解 block/program/mask |
| LeetGPU | <https://leetgpu.com/> | GPU 编程刷题平台，适合练 CUDA kernel |
| CUDA Samples | <https://github.com/NVIDIA/cuda-samples> | 官方练习，覆盖基础 CUDA API 和常见模式 |

### 推荐练习顺序

```text
vector add
  ↓
matrix transpose
  ↓
reduction
  ↓
prefix sum / scan
  ↓
histogram
  ↓
naive matmul
  ↓
tiled matmul
  ↓
softmax / layernorm
  ↓
attention
```

---

## 4. CUDA 性能优化资料

性能优化是 GPU 编程的核心。

| 资源 | 地址 | 重点 |
|---|---|---|
| How to Optimize a CUDA Matmul Kernel | <https://siboehm.com/articles/22/CUDA-MMM> | 从 naive GEMM 一步步优化到接近 cuBLAS |
| Lei Mao CUDA Blog | <https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/> | CUDA 访存、shared memory、bank conflict、tensor core 等 |
| NVIDIA Developer Blog | <https://developer.nvidia.com/blog/> | 官方博客，很多 CUDA 优化案例 |
| CUDA Best Practices Guide | <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/> | 官方性能优化方法论 |

### 性能优化关键词

| 关键词 | 解释 |
|---|---|
| Memory Coalescing | 一个 warp 的线程访问连续地址，让 global memory 访问合并 |
| Shared Memory Tiling | 把数据搬到 shared memory，减少 global memory 访问 |
| Bank Conflict | shared memory 多线程访问同一个 bank 导致串行化 |
| Occupancy | SM 上同时活跃 warp 的比例，不是越高越好，但过低通常有问题 |
| Register Pressure | 每个线程用太多寄存器会降低 occupancy，甚至 spill 到 local memory |
| Warp Divergence | 同一个 warp 内线程走不同分支，导致串行执行 |
| Arithmetic Intensity | 计算量 / 访存量，决定 kernel 是 compute-bound 还是 memory-bound |
| Roofline Model | 用算力峰值和带宽峰值分析性能上限 |

---

## 5. 高性能 GPU 库与 Kernel 框架

当你掌握 CUDA 基础后，需要看工业级库如何组织高性能 kernel。

| 资源 | 地址 | 适合学习 |
|---|---|---|
| CUTLASS | <https://github.com/NVIDIA/cutlass> | NVIDIA 官方 GEMM/Convolution 模板库，学习 Tensor Core、tiling、pipeline |
| Triton | <https://github.com/triton-lang/triton> | OpenAI Triton，适合写 AI kernel，门槛低于 CUDA 模板库 |
| FlashAttention | <https://github.com/Dao-AILab/flash-attention> | 高性能 Attention kernel，理解 IO-aware 算法 |
| ThunderKittens | <https://github.com/HazyResearch/ThunderKittens> | 面向现代 GPU 的轻量 kernel 编程 DSL/库 |
| cuDNN Frontend | <https://github.com/NVIDIA/cudnn-frontend> | cuDNN 图接口和深度学习算子后端 |

### 学习重点

```text
GEMM tiling
warp-level MMA
Tensor Core
shared memory pipeline
async copy / cp.async
double buffering
epilogue fusion
persistent kernel
IO-aware attention
```

---

## 6. Triton 与 AI Kernel 学习路线

Triton 很适合 AI Infra / LLM Kernel 方向，因为它比 CUDA 更接近 Python 生态，同时又能写高性能 GPU kernel。

| 资源 | 地址 | 说明 |
|---|---|---|
| Triton 官方仓库 | <https://github.com/triton-lang/triton> | Triton 编译器和示例 |
| Triton 官方教程 | <https://triton-lang.org/main/getting-started/tutorials/index.html> | matmul、softmax、layernorm 等教程 |
| Triton Puzzles | <https://github.com/srush/Triton-Puzzles> | 练习 Triton 编程模型 |

### 推荐实现的 Triton Kernel

```text
vector add
  ↓
row-wise sum
  ↓
softmax
  ↓
layernorm
  ↓
matmul
  ↓
fused matmul + activation
  ↓
attention
  ↓
quantized matmul
```

---

## 7. LLM 推理与训练系统资料

如果目标是 LLM Infra / AI 系统，需要从 kernel 继续往上看到推理框架、调度、KV Cache、多 GPU。

| 资源 | 地址 | 适合学习 |
|---|---|---|
| TensorRT-LLM | <https://github.com/NVIDIA/TensorRT-LLM> | NVIDIA LLM 推理框架，工业级优化 |
| vLLM | <https://github.com/vllm-project/vllm> | PagedAttention、连续 batching、LLM serving |
| llm.c | <https://github.com/karpathy/llm.c> | 用 C/CUDA 写 GPT 训练，适合理解底层训练流程 |
| llama.cpp | <https://github.com/ggml-org/llama.cpp> | 本地 LLM 推理，量化、CPU/GPU 后端、工程系统很好 |
| FlashAttention | <https://github.com/Dao-AILab/flash-attention> | LLM attention 优化的核心项目 |

### LLM Infra 关键词

| 方向 | 关键词 |
|---|---|
| Attention | FlashAttention、PagedAttention、MQA、GQA、KV Cache |
| 推理调度 | continuous batching、prefill、decode、speculative decoding |
| 并行 | tensor parallel、pipeline parallel、data parallel、expert parallel |
| 量化 | FP16、BF16、FP8、INT8、INT4、AWQ、GPTQ |
| 通信 | NCCL、all-reduce、all-gather、reduce-scatter |
| Kernel Fusion | layernorm fusion、bias+activation fusion、attention fusion |

---

## 8. GPU 架构与硬件模拟

这部分适合想深入 GPU 架构、硬件模拟、体系结构研究的人。

| 资源 | 地址 | 适合学习 |
|---|---|---|
| GPGPU-Sim | <https://github.com/gpgpu-sim/gpgpu-sim_distribution> | GPU microarchitecture 模拟，经典研究工具 |
| Accel-Sim | <https://github.com/accel-sim/accel-sim-framework> | 现代 GPU trace-based 模拟框架 |
| tiny-gpu | <https://github.com/adam-maj/tiny-gpu> | 小型 GPU 教学项目，适合理解 GPU 硬件结构 |
| MIAOW | <https://github.com/VerticalResearchGroup/miaow> | 开源 GPU 架构项目，偏硬件研究 |

你当前目录里已经有一些相关项目：

```text
gpgpu-sim_distribution/
accel-sim-framework/
tiny-gpu/
miaow/
```

可以优先利用本地已有代码学习。

### GPU 架构关键词

```text
SM / CU
warp / wavefront
warp scheduler
scoreboard
register file
shared memory
L1 / L2 cache
memory coalescing
DRAM controller
GDDR / HBM
interconnect
Tensor Core
SIMT execution model
```

---

## 9. Profiling 工具学习

GPU 编程不能只看代码，要会看 profile。

| 工具 | 用途 | 重点指标 |
|---|---|---|
| Nsight Systems | 系统级 timeline | kernel launch、CPU/GPU overlap、stream 并发、通信耗时 |
| Nsight Compute | 单 kernel 分析 | occupancy、memory throughput、SM utilization、warp stall reason |
| nvprof / nvvp | 老工具 | 新项目一般不优先使用 |
| PyTorch Profiler | PyTorch 模型分析 | operator 时间、CUDA kernel、memory、trace |

### 常见性能判断

```text
如果 memory throughput 接近峰值，可能是 memory-bound
如果 SM utilization 高但 memory throughput 不高，可能是 compute-bound
如果 warp stall memory dependency 高，说明访存等待严重
如果 shared memory bank conflict 高，需要调整 shared memory layout
如果 occupancy 很低，检查 register / shared memory 使用量
```

---

## 10. 建议的阶段性学习计划

### 阶段 1：CUDA 入门，1-2 周

目标：能写和运行基本 CUDA kernel。

必学：

```text
thread / block / grid
kernel launch
cudaMalloc / cudaMemcpy / cudaFree
__global__ / __device__ / __host__
同步与错误检查
```

推荐资源：

```text
CUDA Programming Guide
CUDA Samples
OLCF CUDA Training Series
GPU Puzzles
```

练习：

```text
vector add
matrix add
simple reduction
matrix transpose naive version
```

---

### 阶段 2：内存层级与优化，2-4 周

目标：理解为什么 GPU 程序快/慢。

必学：

```text
global memory coalescing
shared memory
bank conflict
register pressure
occupancy
warp divergence
```

推荐资源：

```text
CUDA Best Practices Guide
Lei Mao CUDA Blog
How to Optimize a CUDA Matmul Kernel
Nsight Compute 文档
```

练习：

```text
optimized reduction
coalesced matrix transpose
shared memory tiled matmul
benchmark different block size
用 Nsight Compute 分析 kernel
```

---

### 阶段 3：高性能矩阵乘和 Tensor Core，3-6 周

目标：理解现代 AI 算子的核心是 GEMM。

必学：

```text
tiling
thread tile / warp tile / block tile
shared memory pipeline
double buffering
Tensor Core / MMA
CUTLASS 结构
```

推荐资源：

```text
How to Optimize a CUDA Matmul Kernel
CUTLASS
CUDA MODE Lectures
NVIDIA Tensor Core 相关文档
```

练习：

```text
naive GEMM
shared memory GEMM
register tiled GEMM
Tensor Core GEMM
对比 cuBLAS 性能
```

---

### 阶段 4：Triton 与 AI Kernel，2-4 周

目标：能用 Triton 写常见深度学习算子。

必学：

```text
program id
block pointer
mask load/store
autotune
matmul
softmax
layernorm
```

推荐资源：

```text
Triton 官方教程
Triton Puzzles
CUDA MODE Lectures
```

练习：

```text
Triton vector add
Triton softmax
Triton layernorm
Triton matmul
fused matmul + activation
```

---

### 阶段 5：LLM 推理系统，长期

目标：理解 LLM 如何在 GPU 上高效运行。

必学：

```text
prefill / decode
KV Cache
PagedAttention
FlashAttention
continuous batching
quantization
tensor parallel
NCCL communication
```

推荐资源：

```text
TensorRT-LLM
vLLM
FlashAttention
llm.c
llama.cpp
```

练习：

```text
阅读 FlashAttention 论文和代码
理解 vLLM PagedAttention
跑 TensorRT-LLM demo
读 llm.c 的 forward/backward CUDA kernel
实现一个简化版 attention kernel
```

---

### 阶段 6：GPU 架构模拟与研究，长期

目标：理解 GPU 硬件如何执行 CUDA 程序。

必学：

```text
SIMT execution
warp scheduling
scoreboard
memory hierarchy
cache / DRAM behavior
interconnect
trace-based simulation
```

推荐资源：

```text
GPGPU-Sim
Accel-Sim
tiny-gpu
MIAOW
```

练习：

```text
跑通 GPGPU-Sim sample
跑通 Accel-Sim trace
修改 warp scheduler 策略
观察 cache miss / DRAM stall
比较不同 kernel 的模拟统计结果
```

---

## 11. 最小推荐收藏列表

如果只收藏 10 个，建议收藏这些：

1. CUDA Programming Guide  
   <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>

2. CUDA Best Practices Guide  
   <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

3. NVIDIA CUDA Samples  
   <https://github.com/NVIDIA/cuda-samples>

4. OLCF CUDA Training Series  
   <https://github.com/olcf/cuda-training-series>

5. CUDA MODE Lectures  
   <https://github.com/cuda-mode/lectures>

6. GPU Puzzles  
   <https://github.com/srush/GPU-Puzzles>

7. Triton Puzzles  
   <https://github.com/srush/Triton-Puzzles>

8. How to Optimize a CUDA Matmul Kernel  
   <https://siboehm.com/articles/22/CUDA-MMM>

9. CUTLASS  
   <https://github.com/NVIDIA/cutlass>

10. TensorRT-LLM  
    <https://github.com/NVIDIA/TensorRT-LLM>

---

## 12. 推荐实践项目

可以按照难度逐步做：

| 难度 | 项目 | 目标 |
|---|---|---|
| 入门 | CUDA vector add + reduction | 熟悉 kernel launch 和并行归约 |
| 入门 | matrix transpose 优化 | 理解 coalescing 和 shared memory |
| 中级 | tiled GEMM | 理解 shared memory tiling |
| 中级 | softmax / layernorm | 理解 AI 常见算子 |
| 中高级 | Triton matmul + autotune | 掌握 Triton 编程模型 |
| 高级 | FlashAttention 简化实现 | 理解 IO-aware attention |
| 高级 | PyTorch CUDA extension | 把自定义 kernel 接入 PyTorch |
| 高级 | 简化 LLM inference engine | 理解 KV Cache、batching、decode |
| 研究 | GPGPU-Sim 跑 CUDA kernel | 理解 GPU 硬件模拟 |

---

## 13. 与当前仓库的关联

当前目录中已经有 CUDA/GPU 相关笔记和项目，可以配合上面的资料使用：

```text
CUDA_Lesson.md
CUDA_GPU_Programming_Map.md
H100-Streaming-Multiprocessor-SM.md
Bank_Conflict.md
CoalesedMemoryAccess.md
GPU_Chips.md
GPU_Hardware_Supply.md
gpgpu-sim_distribution/
accel-sim-framework/
tiny-gpu/
miaow/
LeetGPU/
```

建议你后续可以把学习文档整理成三类：

```text
1. CUDA 编程笔记
2. GPU 架构笔记
3. LLM/GPU Kernel 优化笔记
```

---

## 14. 一句话总结

如果目标是 GPU 编程全栈，建议主线是：

```text
CUDA 基础 → 内存优化 → GEMM/Tensor Core → Triton/CUTLASS → LLM 推理系统 → GPU 架构模拟
```

其中最核心的能力是：

```text
理解 GPU 架构 + 写得出 kernel + 看得懂 profile + 能优化访存和并行度
```
