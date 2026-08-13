# 面向 CuTe DSL 的底层 GPU 编程博客精选（按模块整理）

> 整理日期：2026-08-13（Asia/Hong_Kong）  
> 本次在原清单基础上新增 **40 篇**；当前唯一文章数：**148 篇**；共 9 个模块。
> 目标读者：正在学习 NVIDIA **CuTe DSL**、希望补齐分块计算、线程映射、访存、Tensor Core、异步流水线和性能分析基础的读者。

## 先说明筛选口径

- 用户所说的 “CuteDSL” 按 NVIDIA **CuTe DSL** 理解。
- 不纳入 NVIDIA 官方文档、官方仓库 README、官方 API 参考和纯新闻转载；正文以个人博客/个人技术文章为主。
- **Colfax Research、PNF Software、eunomia** 等属于署名工程/技术博客，不是个人独立博客，已在“来源类型”中明确标注；它们的价值在于有作者、代码和硬件实测。
- 文章按“主学习目标”归类；少数文章天然跨模块，只在一个主模块中计数，避免重复。
- 日期来自页面元数据或页面显示；版本、GPU 架构和性能数字请结合当前环境复测。

## 阅读路线（建议）

1. **先建立硬件直觉**：硬件与执行模型 → 内存层次与访存。
2. **再学 CuTe 核心抽象**：Layout Algebra → Tilers/Local Partition → Tiled Copy。
3. **进入计算主线**：CUDA GEMM → Tensor Core/MMA → ldmatrix → CUTLASS/CuTe。
4. **补齐“喂饱计算单元”的方法**：异步拷贝、TMA、多阶段 pipeline、persistent kernel。
5. **最后做工程闭环**：常见融合 kernel → 编译器/PTX/SASS → Nsight/ Roofline/benchmark。

## 快速索引

| 模块 | 篇数 | 推荐起点 |
|---|---:|---|
| [GPU硬件与执行模型](#hardware) | 12 | [Making Deep Learning go Brrrr From First Principles](https://horace.io/brrr_intro.html) |
| [GPU内存层次与访存优化](#memory) | 16 | [CUDA Coalesced Memory Access](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) |
| [分块计算与 Layout Algebra](#layout) | 14 | [CuTe Layout Algebra](https://leimao.github.io/article/CuTe-Layout-Algebra/) |
| [线程映射、同步与并行原语](#parallel) | 11 | [CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/) |
| [GEMM、Tensor Core 与 CUTLASS/CuTe](#gemm) | 29 | [CUDA Matrix Multiplication](https://leimao.github.io/blog/CUDA-Matrix-Multiplication/) |
| [异步拷贝与软件流水线](#pipeline) | 11 | [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) |
| [常见高性能 Kernel 与融合](#kernels) | 20 | [A User’s Guide to FlexAttention in FlashAttention CuTe DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) |
| [编译器、DSL 与 PTX/SASS](#compiler) | 21 | [A Gentle Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction) |
| [Profiling、Roofline 与性能工程](#profiling) | 14 | [Roofline Performance Model](https://leimao.github.io/blog/Roofline-Performance-Model/) |

---

<a id="hardware"></a>
## GPU硬件与执行模型（12 篇）

> **本模块怎么读：** 关注 SM/warp/SIMT、占用率、分支发散、算术强度和 persistent kernel。先回答“硬件如何执行”，再谈 tile 参数。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [NVIDIA GPU Compute Capability](https://leimao.github.io/blog/NVIDIA-GPU-Compute-Capability/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-01-02 | 入门→中级 | SM, compute capability, hardware features | 按 compute capability 梳理 GPU 架构能力与可用指令/内存特性。 | 建立“硬件型号决定可用优化”的意识。 |
| 2 | [CUDA Concept: Block and Grid](https://leimao.github.io/blog/CUDA-Concept-Block-Grid/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2019-03-12 | 入门 | block, grid, thread, indexing | 从 block/grid/thread 的组织方式解释 CUDA kernel 的执行坐标。 | CuTe/CUDA 分层映射的起点。 |
| 3 | [Predicated Execution VS Conditional Execution](https://leimao.github.io/blog/Predicated-Execution-VS-Conditional-Execution/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-07-01 | 中级 | predication, divergence, control flow | 比较 predicated execution 与条件分支在 GPU 上的执行差异。 | 理解 warp divergence 与边界处理代价。 |
| 4 | [Math-Bound VS Memory-Bound Operations](https://leimao.github.io/blog/Math-Bound-VS-Memory-Bound-Operations/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2021-10-11 | 入门→中级 | roofline, arithmetic intensity, bandwidth | 用算术强度和带宽/算力上限区分计算瓶颈。 | 决定应先优化访存还是计算。 |
| 5 | [CUDA Occupancy Calculation](https://leimao.github.io/blog/CUDA-Occupancy-Calculation/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-06-25 | 中级 | occupancy, registers, shared memory, SM | 推导 occupancy 受寄存器、shared memory、线程数约束的计算方法。 | 理解 tile 变大后为何可能反而变慢。 |
| 6 | [CUDA Local Memory](https://leimao.github.io/blog/CUDA-Local-Memory/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-03-19 | 中级 | local memory, register spill, stack | 解释 CUDA local memory 何时出现以及它与寄存器/全局内存的关系。 | 排查寄存器溢出和隐藏访存。 |
| 7 | [CUDA Compatibility](https://leimao.github.io/blog/CUDA-Compatibility/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-02-04 | 入门→中级 | driver, runtime, architecture, binary compatibility | 梳理 CUDA driver/runtime、架构和二进制兼容性。 | 避免在不同 GPU/容器上误判性能问题。 |
| 8 | [Making Deep Learning go Brrrr From First Principles](https://horace.io/brrr_intro.html) | Horace He | Horace He’s Blog | 个人博客 | 2022 | 中级 | GPU, framework, kernel, performance | 从深度学习框架与 GPU kernel 的底层执行解释“为什么会变快”。 | 建立从框架调用到 kernel 性能的全链路直觉。 |
| 9 | [GPU Mode: CUTLASS and FlashAttention-3](https://research.colfax-intl.com/gpu-mode-cutlass-and-flashattention-3/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-11-18 | 中级→高级 | GPU mode, CUTLASS, FlashAttention, Hopper | 以 GPU Mode 课程为背景，串起 CUTLASS、FlashAttention-3 与 Hopper 执行模型。 | 把硬件概念连接到现代 ML kernel。 |
| 10 | [AI GPU Programming - Cuda Basics](https://kapilsh.github.io/posts/ai-gpu-programming-1/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-08-27 | 入门 | CUDA basics, thread hierarchy, memory | 以 AI kernel 为背景介绍 CUDA 基本编程模型。 | 适合从 Python/深度学习转到底层 CUDA。 |
| 11 | [GPU MODE Lecture 2: Ch.1-3 PMPP Book](https://christianjmills.com/posts/cuda-mode-notes/lecture-002/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-06-06 | 入门 | CUDA model, data parallelism, threads, blocks, memory | 根据 PMPP 前三章整理 CUDA 异构计算、数据并行、线程组织和内存管理基础。 | 补齐 CuTe 线程层级与 CTA 映射所依赖的 CUDA 执行模型。 |
| 12 | [Flash Attention from Scratch: Appendix B - Block Size Configuration](https://lubits.ch/flash/Appendix-B---Block-Size-Configuration) | Sonny | Sonny’s Blog | 个人博客 | 2025-11-08 | 高级 | block size, arithmetic intensity, persistent data, performance model | 专门分析 block 配置对算术强度、持久化数据和重复加载开销的影响。 | 帮助理解 CuTe tiler 参数与硬件资源之间的权衡。 |


<a id="memory"></a>
## GPU内存层次与访存优化（16 篇）

> **本模块怎么读：** 关注 global/shared/register/L2、coalescing、alignment、bank conflict、swizzle、缓存与内存池。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUDA Coalesced Memory Access](https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-03-19 | 入门→中级 | coalescing, global memory, transaction | 说明线程访问如何合并为高效 global-memory transaction。 | 是所有 tile/copy 优化的访存基础。 |
| 2 | [CUDA Vectorized Memory Access](https://leimao.github.io/blog/CUDA-Vectorized-Memory-Access/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-01-14 | 中级 | vectorized load, bandwidth, alignment | 讨论 float2/float4 等向量化加载对吞吐的影响。 | 理解 copy atom 和对齐要求。 |
| 3 | [CUDA Data Alignment](https://leimao.github.io/blog/CUDA-Data-Alignment/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-10-18 | 入门→中级 | alignment, vectorization, memory transaction | 解释数据对齐、类型宽度和向量化访问的关系。 | 排查向量化访存失效。 |
| 4 | [CUDA Shared Memory Bank](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-06-22 | 中级 | shared memory, bank conflict | 介绍 shared-memory bank 组织与冲突产生条件。 | 理解 CuTe swizzle 的硬件动机。 |
| 5 | [CUDA Shared Memory Bank Conflict-Free Vectorized Access](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank-Conflict-Free-Vectorized-Access/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-02-13 | 中级→高级 | bank conflict, vectorized access, shared memory | 分析共享内存 bank conflict 与向量化访问组合时的陷阱。 | 直接对应 Tiled Copy/ldmatrix 的布局设计。 |
| 6 | [CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-05-14 | 中级→高级 | swizzle, bank conflict, layout | 用 swizzling 改变地址映射以规避 bank conflict。 | CuTe Swizzle 的优秀先修材料。 |
| 7 | [CUDA Shared Memory Capacity](https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-07-04 | 入门→中级 | shared memory, capacity, occupancy | 梳理 shared memory 容量、配置和对 occupancy 的影响。 | 决定 CTA tile 和 pipeline stage 数。 |
| 8 | [CUDA Constant Memory](https://leimao.github.io/blog/CUDA-Constant-Memory/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-12-01 | 入门→中级 | constant memory, cache, broadcast | 介绍 constant memory 的缓存与 broadcast 行为。 | 适合理解只读参数/卷积 filter 的放置。 |
| 9 | [CUDA L2 Persistent Cache](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-09-12 | 高级 | L2 cache, persistence, locality | 分析如何利用 L2 persistence 控制数据驻留。 | 连接 cache locality 与高阶性能调优。 |
| 10 | [Page-Locked Host Memory Data Transfer](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2021-06-26 | 中级 | pinned memory, H2D/D2H, DMA | 解释 page-locked host memory 对异步数据传输的作用。 | 理解 copy/compute overlap 的前提。 |
| 11 | [CUDA Zero-Copy Mapped Memory](https://leimao.github.io/blog/CUDA-Zero-Copy-Mapped-Memory/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-12-16 | 中级 | zero-copy, mapped memory, PCIe | 介绍 host memory 映射到 GPU 后的访问路径与适用边界。 | 建立 PCIe/内存层次的真实直觉。 |
| 12 | [Memory Pool](https://leimao.github.io/blog/Memory-Pool/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-02-17 | 中级 | allocator, memory pool, fragmentation | 讨论 CUDA memory pool 与频繁分配的性能影响。 | 理解 workspace/KV/cache allocator 设计。 |
| 13 | [Tensor Physical Layout on Memory](https://leimao.github.io/blog/Tensor-Physical-Layout-on-Memory/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2019-04-17 | 中级 | stride, layout, physical memory, tensor | 从 stride 与物理地址解释张量布局。 | 为 CuTe layout/tensor 打基础。 |
| 14 | [CUDA Shared Memory Templated Kernel](https://leimao.github.io/blog/CUDA-Shared-Memory-Templated-Kernel/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2019-05-04 | 中级 | shared memory, templates, tile | 展示如何用模板参数化 shared-memory kernel。 | 理解静态 shape 与编译期优化。 |
| 15 | [GPU MODE Lecture 4: Compute and Memory Basics](https://christianjmills.com/posts/cuda-mode-notes/lecture-004/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-08-31 | 入门→中级 | compute capability, memory hierarchy, tiling, fusion | 概览 GPU 计算架构、内存层次，并介绍 kernel fusion、tiling 等性能优化策略。 | 把 CuTe 的分块抽象放回算力、带宽和数据复用的硬件背景。 |
| 16 | [Making matrix transpose really fast on Hopper GPUs](https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/) | Simon Veitner | Simon Veitner’s Blog | 个人博客 | 2025-05-02 | 高级 | Hopper, transpose, shared memory, swizzle | 用原生 CUDA 优化 Hopper 矩阵转置，讨论共享内存布局、swizzle 和搬运路径。 | 是观察 layout 设计如何影响实际访存冲突的好案例。 |


<a id="layout"></a>
## 分块计算与 Layout Algebra（14 篇）

> **本模块怎么读：** 这是 CuTe DSL 的核心：shape/stride、composition、tiler、local partition、thread/value layout 与 swizzle。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CuTe Layout Algebra](https://leimao.github.io/article/CuTe-Layout-Algebra/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-10-20 | 中级→高级 | layout algebra, shape, stride, composition | 系统介绍 CuTe Layout 的 shape/stride 表示、组合与坐标映射。 | CuTe DSL 最核心的数学/编程抽象。 |
| 2 | [CuTe Arithmetic Tuple Tensor](https://leimao.github.io/blog/CuTe-Arithmetic-Tuple-Tensor/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-10-20 | 中级 | CuTe, tuple, tensor, arithmetic | 解释 CuTe arithmetic、tuple 和 tensor 之间的类型/布局关系。 | 理解 DSL 中“数据 + layout”的组合方式。 |
| 3 | [CuTe Tilers](https://leimao.github.io/blog/CuTe-Tilers/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-09-15 | 中级→高级 | tiler, tile, shape, slice | 介绍 CuTe tiler 如何描述分块、切片和局部 tile。 | 对应分块计算的抽象表达。 |
| 4 | [CuTe Inverse Layout](https://leimao.github.io/blog/CuTe-Inverse-Layout/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-08-13 | 高级 | inverse layout, coordinate, bijection | 解释 layout inverse 与索引/坐标反解。 | 处理线程映射和地址反推时非常关键。 |
| 5 | [CuTe Blocked and Raked Products](https://leimao.github.io/blog/CuTe-Blocked-Raked-Products/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-08-07 | 高级 | blocked product, raked product, thread layout | 分析 blocked/raked product 如何构造线程和值布局。 | 理解 CTA/warp/thread/value 层级组合。 |
| 6 | [CuTe Local Tile](https://leimao.github.io/blog/CuTe-Local-Tile/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-08-01 | 中级 | local tile, slicing, tiling | 讲局部 tile 的提取与分块操作。 | 把全局 tensor 变成每个 CTA/warp 的工作集。 |
| 7 | [CuTe Local Partition](https://leimao.github.io/blog/CuTe-Local-Partition/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-07-25 | 高级 | local partition, thread mapping, partition | 讲 local partition 如何把 tensor 分配给线程。 | 直接对应 CuTe 的线程分工。 |
| 8 | [CuTe Index To Coordinate](https://leimao.github.io/blog/CuTe-Index-To-Coordinate/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-07-19 | 中级 | index, coordinate, layout mapping | 解释一维 index 如何通过 layout 映射成多维坐标。 | 排查 layout 映射错误的基础工具。 |
| 9 | [CuTe Thread-Value Layout](https://leimao.github.io/blog/CuTe-Thread-Value-Layout/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-10-13 | 高级 | thread-value layout, register tile, mapping | 从 thread/value 两级说明寄存器 tile 的布局。 | 连接 layout algebra 与 MMA operand。 |
| 10 | [CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-12-01 | 高级 | swizzle, shared memory, bank conflict | 解释 CuTe swizzle 如何重排地址位。 | 把抽象 layout 直接连接到 bank conflict 优化。 |
| 11 | [CuTe Matrix Transpose](https://leimao.github.io/article/CuTe-Matrix-Transpose/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-11-20 | 中级→高级 | transpose, layout, copy, shared memory | 用 CuTe 实现矩阵转置并分析 layout/copy。 | 一个非常好的 CuTe 综合练习。 |
| 12 | [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/) | Jay | Colfax Research | 署名工程博客 | 2023-12-15 | 高级 | CuTe, layout algebra, composition, cosize | 从代数视角补充 CuTe layout 的组合规则。 | 适合在 Lei Mao 入门后深化。 |
| 13 | [CuTe Tiled Copy](https://leimao.github.io/blog/CuTe-Tiled-Copy/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-10-16 | 高级 | CuTe, tiled copy, copy atom, layout | 深入解释 CuTe Tiled Copy、copy atom 与线程/值布局如何共同决定数据搬运。 | 补齐 CuTe 中“布局如何驱动拷贝”的关键环节，适合连接 layout 与 TMA/异步搬运。 |
| 14 | [Matrix Multiplication On GPU: Part 2, Tiling](https://indii.org/blog/gpu-matrix-multiply-tiling/) | Lawrence Murray | Lawrence Murray’s Blog | 个人博客 | 2024-10-01 | 中级 | tiling, matrix multiplication, shared memory, cache reuse | 专门拆解大矩阵乘的 tile 分解、数据复用和局部工作集。 | 可作为理解 CuTe tiler 与 CTA-level tile 的直观前置材料。 |


<a id="parallel"></a>
## 线程映射、同步与并行原语（11 篇）

> **本模块怎么读：** 关注 warp shuffle、reduction、scan、stream、cooperative groups、barrier 与阶段间同步。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-07-30 | 入门→中级 | reduction, warp, shared memory | 从朴素规约到层次化/warp 规约，分析同步与访存。 | 所有 block-level reduction 的先修。 |
| 2 | [CUDA Cooperative Groups](https://leimao.github.io/blog/CUDA-Cooperative-Groups/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-08-06 | 中级 | cooperative groups, synchronization, tiles | 介绍 cooperative groups 与更细粒度同步抽象。 | 理解现代 CUDA 同步组织。 |
| 3 | [CUDA Rendezvous Stream](https://leimao.github.io/blog/CUDA-Rendezvous-Stream/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-01-26 | 高级 | rendezvous, stream, synchronization | 讨论 rendezvous stream 的同步语义与使用场景。 | 理解跨阶段 pipeline 同步。 |
| 4 | [Online Safe Softmax](https://leimao.github.io/blog/Online-Safe-Softmax/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-06-23 | 中级 | online softmax, reduction, numerical stability | 推导 online safe softmax 的增量 max/sum 维护。 | FlashAttention/融合 softmax 的数学基础。 |
| 5 | [Radix Sort](https://leimao.github.io/blog/Radix-Sort/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-12-18 | 中级 | radix sort, scan, histogram, primitives | 从 radix sort 看 scan/histogram/分桶组合。 | 理解复杂并行 primitive 的拼装。 |
| 6 | [CUDA Kernel Execution Overlap](https://leimao.github.io/blog/CUDA-Kernel-Execution-Overlap/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-06-10 | 中级 | streams, overlap, concurrency | 分析多个 kernel 如何在 GPU 上重叠执行。 | 为异步 pipeline 做准备。 |
| 7 | [Multi-Thread Single-Stream VS Single-Thread Multi-Stream CUDA](https://leimao.github.io/blog/Multi-Thread-Single-Stream-VS-Single-Thread-Multi-Stream-CUDA/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2021-10-18 | 高级 | streams, host threads, concurrency | 比较 host thread 与 CUDA stream 组合对并发的影响。 | 避免把 host 并发误认为 device 并发。 |
| 8 | [CUDA Default Stream](https://leimao.github.io/blog/CUDA-Default-Stream/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-11-06 | 入门→中级 | default stream, ordering, synchronization | 解释 default stream 的隐式同步规则。 | 排查 stream pipeline 中的隐藏串行化。 |
| 9 | [C++ Latch and Barrier](https://leimao.github.io/blog/CPP-Latch-Barrier/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-02-06 | 中级 | barrier, latch, synchronization | 介绍 C++ barrier/latch 并联系并行任务同步。 | 补充 host/device pipeline 的同步思路。 |
| 10 | [Fast Reductions with Warp Shuffles](https://blog.melashri.net/micro/shfl-cuda/) | Mohamed Elashri | Mohamed Elashri’s Blog | 个人博客 | 2025-07-31 | 中级 | warp shuffle, reduction, atomic, shared memory | 用 warp shuffle 降低规约中原子操作和共享内存同步的开销。 | 补充线程级通信与 reduction 优化，便于理解 CuTe 的 warp-level partition。 |
| 11 | [GPU MODE Lecture 9: Reductions](https://christianjmills.com/posts/cuda-mode-notes/lecture-009/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-14 | 中级 | reduction, divergence, memory divergence, coarsening | 从并行规约的控制发散、内存发散和全局访存开销出发，逐步优化 reduction kernel。 | CuTe 的 thread/value 分配最终要落到这类 warp/block reduction 原语上。 |


<a id="gemm"></a>
## GEMM、Tensor Core 与 CUTLASS/CuTe（29 篇）

> **本模块怎么读：** 按 naive GEMM → shared-memory/register tiling → MMA/ldmatrix → WGMMA/CUTLASS/CuTe 递进。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUDA Matrix Multiplication](https://leimao.github.io/blog/CUDA-Matrix-Multiplication/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 未标注 | 入门→中级 | GEMM, tiling, shared memory | 从基础 CUDA GEMM 解释矩阵索引、tile 与计算流程。 | GEMM 优化路线的起点。 |
| 2 | [Build and Develop CUTLASS CUDA Kernels](https://leimao.github.io/blog/Build-Develop-CUTLASS-CUDA-Kernels/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-11-12 | 中级 | CUTLASS, build, kernel, CMake | 记录构建、修改和开发 CUTLASS kernel 的实践。 | 进入 CUTLASS 源码前的工程准备。 |
| 3 | [NVIDIA Tensor Core Programming](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-05-18 | 中级→高级 | Tensor Core, MMA, FP16, accumulation | 概览 Tensor Core 编程接口、数据类型与累加精度。 | 理解 MMA 指令前的地图。 |
| 4 | [NVIDIA Tensor Core TN Layout MMA Instruction](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-12-06 | 高级 | MMA, TN layout, fragments, Tensor Core | 拆解 Tensor Core MMA 指令的 TN layout 和 fragment 分布。 | 直接对应 CuTe MMA atom。 |
| 5 | [Benchmarking NVIDIA Tensor Core MMA Instruction Peak Performances](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-11-26 | 高级 | Tensor Core, benchmark, peak FLOPS | 实测 MMA 指令峰值并讨论测量方法。 | 学习如何把理论峰值变成可复现实验。 |
| 6 | [CuTe Tiled MMA](https://leimao.github.io/blog/CuTe-Tiled-MMA/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-01-09 | 高级 | CuTe, MMA, tiled MMA, atom | 用 CuTe 组合 MMA atom 与 tiled MMA。 | CuTe DSL Tensor Core 主线必读。 |
| 7 | [CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-10-03 | 高级 | ldmatrix, shared memory, warp, MMA | 解释 ldmatrix 从 shared memory 向寄存器加载矩阵片段。 | 理解 Tensor Core operand layout。 |
| 8 | [Developing CUDA Kernels for GEMM on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-gemm-cutlass/) | Jay | Colfax Research | 署名工程博客 | 2023-10-17 | 高级 | Hopper, CUTLASS, GEMM, warpgroup | 以 Hopper 为例拆解 CUTLASS GEMM kernel 层次。 | 把模板抽象和硬件执行对应起来。 |
| 9 | [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA Hopper GPUs](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-08-06 | 高级 | WGMMA, Hopper, warpgroup, Tensor Core | 讲 WGMMA 指令和 Hopper warpgroup GEMM。 | 学习从 MMA 到 WGMMA 的代际变化。 |
| 10 | [Tutorial: Matrix Transpose in CUTLASS](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-05-06 | 中级→高级 | CUTLASS, transpose, layout, copy | 用 CUTLASS 实现转置，展示 layout 与 copy 抽象。 | 将 CuTe layout 应用到非 GEMM kernel。 |
| 11 | [CUTLASS 3.x APIs: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design](https://research.colfax-intl.com/cutlass-3-x-apis-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design-external/) | Jay | Colfax Research | 署名工程博客 | 2025-07-20 | 高级 | CUTLASS 3.x, GEMM hierarchy, composability | 介绍 CUTLASS 3.x 的层级 API 与可组合设计。 | 理解 CuTe/CUTLASS 软件架构。 |
| 12 | [CUTLASS Tutorial: Sub-byte GEMM on NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/) | Colfax Research | Colfax Research | 署名工程博客 | 2025-06-07 | 高级 | Blackwell, sub-byte, GEMM, quantization | 讲 Blackwell sub-byte GEMM 的数据打包、对齐与执行。 | 了解低比特 tile 的额外 layout 复杂度。 |
| 13 | [CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/) | Ryo | Colfax Research | 署名工程博客 | 2025-04-19 | 高级 | Blackwell, tensor memory, GEMM, layout | 用 Blackwell Tensor Memory 编写 GEMM。 | 连接新硬件内存与 CuTe layout。 |
| 14 | [CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/) | Colfax Research | Colfax Research | 署名工程博客 | 2025-05-10 | 高级 | thread block cluster, Blackwell, GEMM | 介绍 thread block cluster 下的 GEMM 分工和通信。 | 从 CTA 扩展到 cluster 级分块。 |
| 15 | [CUTLASS Tutorial: NVFP4 Blockscaled GEMM on NVIDIA RTX Pro Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/) | Colfax Research | Colfax Research | 署名工程博客 | 2026-06-20 | 高级 | NVFP4, block scaling, GEMM, Blackwell | 介绍 NVFP4 block-scaled GEMM 的 scale/layout/计算流程。 | 学习低比特分块计算的真实约束。 |
| 16 | [Learn CUTLASS the hard way!](https://kapilsh.github.io/posts/learn-cutlass-the-hard-way/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-11-01 | 中级→高级 | CUTLASS, GEMM, bf16, optimization | 从 naive FP32 GEMM 逐步优化到 CUTLASS BF16 kernel。 | 非常适合把理论转成可运行实验。 |
| 17 | [Learn CUTLASS the hard way - part 2!](https://kapilsh.github.io/posts/learn-cutlass-the-hard-way-2/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-12-31 | 高级 | Hopper, CUTLASS, H100, GEMM | 在 H100 上继续探索 Hopper 优化并对比 PyTorch/cuBLAS。 | 适合已有 GEMM 基础后的实测进阶。 |
| 18 | [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) | Simon Boehm | Simon Boehm’s Blog | 个人博客 | 2022-12-31 | 中级→高级 | GEMM, register tiling, shared memory, cuBLAS | 以 worklog 方式记录从朴素 matmul 到接近 cuBLAS 的优化。 | 学习性能优化的迭代方法。 |
| 19 | [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/gemm-gpu) | Amanzhol Salykov | Amanzhol Salykov’s Blog | 个人博客 | 2025-01-12 | 高级 | GEMM, warp tiling, Tensor Core, performance | 深入 NVIDIA GPU 矩阵乘优化、warp tiling 和资源权衡。 | 适合系统复盘 GEMM 分层。 |
| 20 | [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul) | Aleksa Gordić | Aleksa Gordić’s Blog | 个人博客 | 2025-09-29 | 高级 | matmul, warp tiling, PTX, SASS, async pipeline | 从 GPU 架构、PTX/SASS、warp tiling 到异步 Tensor Core pipeline，完整拆解高性能 matmul。 | 把 CuTe 分块、线程映射和流水线优化放到一篇端到端案例中。 |
| 21 | [GPU MODE Lecture 5: Going Further with CUDA for Python Programmers](https://christianjmills.com/posts/cuda-mode-notes/lecture-005/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-01 | 中级 | GEMM, shared memory, tiling, Numba, CUDA Python | 从 Python、CUDA C 和 Numba 三种路径比较矩阵乘，重点展示 shared memory 和 tiling。 | 适合作为 CuTe GEMM 教程前的直观分块练习。 |
| 22 | [GPU MODE Lecture 7: Advanced Quantization](https://christianjmills.com/posts/cuda-mode-notes/lecture-007/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-10 | 中级→高级 | quantization, Triton, CUDA, weight-only, dynamic quantization | 讨论动态量化和 weight-only quantization，并比较 Triton/CUDA kernel 在低精度路径中的性能问题。 | 低比特数据的打包、scale 和 tile 对 CuTe layout 有直接影响。 |
| 23 | [GPU MODE Lecture 15: CUTLASS](https://christianjmills.com/posts/cuda-mode-notes/lecture-015/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-11-17 | 高级 | CUTLASS, CuTe, shape, stride, tiling | 以 CUTLASS 的 tensor layout algebra 为核心，解释 shape、stride、tiling 和可组合抽象。 | 直接对应 CuTe DSL 的核心概念，可作为系列化入门材料。 |
| 24 | [Load and store matrices efficiently with PTX instructions](https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/) | Simon Veitner | Simon Veitner’s Blog | 个人博客 | 2025-05-14 | 高级 | PTX, ldmatrix, matrix load, Tensor Core | 讲解 ldmatrix 等 PTX 指令怎样协作加载矩阵，为 MMA/Tensor Core 计算准备 operand。 | 可帮助理解 CuTe Copy Atom 与 MMA operand layout 的实际指令基础。 |
| 25 | [A gentle introduction to GEMM using MMA tensor cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/) | Aman | Aman’s Blog | 个人博客 | 2025-10-02 | 高级 | GEMM, MMA, ldmatrix, Tensor Core | 从最小 MMA microkernel 开始，逐步扩展到完整 GEMM，解释 Tensor Core 数据布局和线程协作。 | 非常适合作为 CuTe Tiled MMA 前的微内核练习。 |
| 26 | [Tiling Matrix Multiplication on the GPU](https://www.sethweidman.com/blog/cuda_matmul.html) | Seth Weidman | Seth Weidman’s Website | 个人博客 | 2025-12-03 | 中级 | CUDA GEMM, block tiling, shared memory, L4 | 从矩阵分块、shared memory 到线程映射，逐步构建 GPU 矩阵乘，并在 L4 上做性能比较。 | 适合把抽象 tiling 还原为容易运行和调试的 CUDA C++ kernel。 |
| 27 | [Worklog: Optimising GEMM on NVIDIA H100 for cuBLAS-like Performance (WIP)](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) | Hamza Elshafie | Hamza’s Blog | 个人博客 | 2026-01-12 | 高级 | H100, WGMMA, TMA, shared memory, GEMM | 以 H100 为目标，从 warp tile、bank conflict、TMA 到 WGMMA descriptor 逐步优化 GEMM。 | 是理解 Hopper CuTe/CUTLASS kernel 资源布局和性能迭代的实战 worklog。 |
| 28 | [Notes on reading “How to Optimize a CUDA Matmul Kernel”](https://stpn.bearblog.dev/notes-on-reading-how-to-optimize-a-cuda-matmul-kernel/) | Stephen Wan | Stephen Wan’s Blog | 个人博客 | 2025-06-28 | 中级→高级 | GEMM, register tiling, shared memory, optimization notes | 以读书笔记形式复盘 CUDA matmul 优化路线，串联 tile、寄存器和共享内存复用。 | 适合在阅读原始 worklog 后做二次总结和知识结构化。 |
| 29 | [Matrix Multiplication on GPU: Faster than Nvidia, Sometimes](https://indii.org/blog/gpu-matrix-multiply/) | Lawrence Murray | Lawrence Murray’s Blog | 个人博客 | 2024-10-01 | 中级→高级 | CUDA GEMM, benchmarking, numerical correctness, cuBLAS | 实现单精度 CUDA 矩阵乘，在特定形状上比较 cuBLAS，并关注 bit-for-bit 正确性。 | 展示如何用独立 kernel 对基线、性能和数值正确性做完整闭环。 |


<a id="pipeline"></a>
## 异步拷贝与软件流水线（11 篇）

> **本模块怎么读：** 关注 double buffer、多阶段 pipeline、TMA、异步执行、warp specialization、persistent scheduling。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-09-22 | 高级 | software pipeline, double buffer, GEMM | 解释 GEMM 中 global→shared→register→MMA 的多阶段流水。 | 回答“如何持续喂满 Tensor Core”。 |
| 2 | [CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-06-24 | 高级 | TMA, async copy, Hopper, pipeline | 介绍 TMA 的多维异步数据搬运和 barrier 协作。 | CuTe TMA copy 的最佳背景材料。 |
| 3 | [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using CUTLASS](https://research.colfax-intl.com/nvidia-hopper-flashattention-2/) | Jay | Colfax Research | 署名工程博客 | 2023-12-05 | 高级 | kernel fusion, Hopper, pipeline, FlashAttention | 以 FlashAttention-2 说明融合、异步搬运与计算重叠。 | 把 pipeline 放到真实 attention kernel。 |
| 4 | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-07-11 | 高级 | asynchrony, warp specialization, FP8, Hopper | 分析 Hopper 上异步执行、warp specialization 和低精度 attention。 | 理解现代 GPU pipeline 设计。 |
| 5 | [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-12-20 | 高级 | persistent kernel, Stream-K, scheduling | 讨论 persistent kernel 和 Stream-K 的 tile 调度。 | 学习如何减少 launch/调度空洞。 |
| 6 | [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/) | Jay | Colfax Research | 署名工程博客 | 2026-03-05 | 高级 | Blackwell, pipeline, co-design, softmax | 从算法与硬件不对称扩展角度分析新一代 pipeline。 | 了解硬件代际变化如何改变优化重点。 |
| 7 | [CUDA Stream](https://leimao.github.io/blog/CUDA-Stream/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2020-02-02 | 入门→中级 | stream, async execution, ordering | 介绍 stream 的顺序语义与异步执行。 | 所有 copy/compute overlap 的先修。 |
| 8 | [Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/) | Colfax Research | Colfax Research | 署名工程博客 | 2026-05-09 | 高级 | CLC, persistent scheduling, Blackwell, tiles | 介绍 Blackwell CLC 下动态 persistent tile 调度。 | 理解最新硬件调度原语。 |
| 9 | [GPU MODE Lecture 13: Ring Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-013/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-11-11 | 中级→高级 | Ring Attention, communication, long context, overlap | 介绍 Ring Attention 的分布式计算动机、分块交换和长上下文处理机制。 | 可把单卡 CuTe pipeline 进一步联系到跨设备通信/计算重叠。 |
| 10 | [TMA introduction](https://veitner.bearblog.dev/tma-introduction/) | Simon Veitner | Simon Veitner’s Blog | 个人博客 | 2025-04-27 | 高级 | Hopper TMA, tensor map, async copy, barrier | 介绍 Hopper Tensor Memory Accelerator、tensor map 和 global→shared 的异步搬运。 | 适合理解 CuTe TMA copy 与 mbarrier 的底层硬件对应关系。 |
| 11 | [Use TMA without CUDA](https://veitner.bearblog.dev/use-tma-without-cuda/) | Simon Veitner | Simon Veitner’s Blog | 个人博客 | 2025-06-04 | 高级 | TMA, low-level GPU interface, asynchronous transfer | 探索不依赖常规 CUDA API 使用 Hopper TMA，展示更接近硬件的异步内存传输。 | 适合在掌握 CuTe TMA 后进一步理解其底层接口边界。 |


<a id="kernels"></a>
## 常见高性能 Kernel 与融合（20 篇）

> **本模块怎么读：** 用 softmax、RMSNorm、卷积、稀疏矩阵、FlashAttention 等真实算子练习布局与融合。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUDA Convolution Tensor Layouts](https://leimao.github.io/blog/CUDA-Convolution-Tensor-Layouts/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-06-04 | 中级 | convolution, tensor layout, stride | 分析卷积输入/权重/输出的物理布局。 | 理解 layout 对卷积 kernel 的影响。 |
| 2 | [Deformable Convolution](https://leimao.github.io/blog/Deformable-Convolution/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2023-12-08 | 中级→高级 | deformable conv, irregular access, kernel | 讨论 deformable convolution 的不规则访存与实现。 | 学习规则 tile 失效时的优化思路。 |
| 3 | [Neural Network 1x1 Convolution Fusion](https://leimao.github.io/blog/Neural-Network-1x1-Convolution-Fusion/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2021-05-20 | 中级 | fusion, 1x1 convolution, memory traffic | 分析 1x1 convolution 与相邻算子的融合机会。 | 理解 fusion 如何减少中间张量。 |
| 4 | [CSR Sparse Matrix Multiplication](https://leimao.github.io/blog/CSR-Sparse-Matrix-Multiplication/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-12-21 | 中级→高级 | CSR, sparse, load balance, irregular memory | 介绍 CSR 稀疏矩阵乘及其负载/访存问题。 | 补齐稀疏 kernel 的现实复杂度。 |
| 5 | [Fast Fourier Transform Convolution](https://leimao.github.io/blog/Fast-Fourier-Transform-Convolution/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2021-10-04 | 中级 | FFT, convolution, algorithm tradeoff | 比较直接卷积和 FFT 卷积的复杂度与实现路径。 | 理解算法级替换与 kernel 优化的关系。 |
| 6 | [Triton Kernels - Fused Softmax](https://kapilsh.github.io/posts/triton-kernels-softmax/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-09-18 | 中级 | Triton, softmax, fusion, reduction | 用 Triton 实现 fused softmax 并分析 block 选择。 | 从 CUDA 过渡到 DSL kernel。 |
| 7 | [Triton Kernels - Fused Softmax - 2](https://kapilsh.github.io/posts/triton-kernels-softmax-2/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-09-20 | 中级→高级 | Triton, profiling, debugging, softmax | 以 worklog 记录 Triton softmax 的性能调试。 | 学习“看 profile 再改 kernel”。 |
| 8 | [Triton Kernels - RMS Norm](https://kapilsh.github.io/posts/triton-kernels-rms-norm/) | Kapil Sharma | Kapil Sharma’s Blog | 个人博客 | 2025-09-17 | 中级 | Triton, RMSNorm, reduction, LLM kernel | 实现并优化 RMSNorm Triton kernel。 | 适合练习 reduction + fusion。 |
| 9 | [A User’s Guide to FlexAttention in FlashAttention CuTe DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) | Reuben Stern | Colfax Research | 署名工程博客 | 2025-11-15 | 高级 | CuTe DSL, FlexAttention, FlashAttention, fusion | 介绍 CuTe DSL 中 FlexAttention 的结构与扩展方式。 | 把 DSL 学习连接到真实 attention 变体。 |
| 10 | [FlashAttention-3 for Inference: INT8 Quantization and Query Head Packing for MQA/GQA](https://research.colfax-intl.com/flashattention-3-for-inference-int8-quantization-and-query-head-packing-for-mqa-gqa-external/) | Jay | Colfax Research | 署名工程博客 | 2024-11-28 | 高级 | attention, MQA/GQA, INT8, packing | 分析推理 attention 中 query head packing 与低精度。 | 学习融合 kernel 的数据布局设计。 |
| 11 | [GPU MODE Lecture 6: Optimizing Optimizers in PyTorch](https://christianjmills.com/posts/cuda-mode-notes/lecture-006/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-02 | 中级 | kernel fusion, multi-tensor apply, launch overhead | 分析 PyTorch 优化器中的 kernel fusion 和 multi-tensor apply，说明如何减少 kernel launch 与中间张量开销。 | 帮助理解 epilogue fusion 和“小算子合并”为什么能改善端到端性能。 |
| 12 | [GPU MODE Lecture 11: Sparsity](https://christianjmills.com/posts/cuda-mode-notes/lecture-011/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-15 | 中级→高级 | semi-structured sparsity, block sparsity, sparse kernels | 介绍半结构化稀疏和块稀疏表示，以及利用专用 sparse kernel 加速训练/推理。 | 稀疏布局会改变 tile、负载均衡和 Tensor Core operand 的组织方式。 |
| 13 | [GPU MODE Lecture 12: Flash Attention](https://christianjmills.com/posts/cuda-mode-notes/lecture-012/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-15 | 中级→高级 | FlashAttention, tiling, online softmax, CUDA | 从算法和 kernel 两层介绍 FlashAttention 的 tiling、online softmax、片上数据复用和限制。 | 是学习 CuTe attention kernel 之前非常清晰的算法—硬件桥梁。 |
| 14 | [FlashAttention-2 in CuTe, from Scratch](https://blog.echen.io/p/flashattention-2-in-cute-from-scratch/) | echen（@below_ocean） | echen’s Blog | 个人博客 | 2026-05-18 | 高级 | CuTe, FlashAttention-2, Tiled Copy, Tiled MMA, pipeline | 逐行拆解 CuTe/C++ FlashAttention-2，实现并讨论 layout、Tiled Copy、Tiled MMA、异步拷贝和 softmax 累积。 | 与当前学习目标最贴近的端到端 CuTe kernel 案例之一。 |
| 15 | [Reimplementing FlashAttention for performance and giggles](https://aminediro.com/posts/flash_attn/) | Amine Dirhoussi | Amine Dirhoussi’s Blog | 个人博客 | 2025-12-04 | 高级 | FlashAttention, online softmax, Triton, CUDA, profiling | 从理论和 online softmax 出发重写 FlashAttention，并通过 profile 分析 shared memory、Tensor Core 和实现差异。 | 适合比较算法分块、kernel 资源和真实性能之间的关系。 |
| 16 | [Flash Attention from Scratch Part 6: FP Instruction Fusion and Auto-Tuning](https://lubits.ch/flash/Part-6) | Sonny | Sonny’s Blog | 个人博客 | 2025-11-01 | 高级 | instruction fusion, autotuning, FlashAttention, RTX 3090 | 在前序分块和双缓冲基础上加入浮点指令融合与自动调优，比较不同 kernel 配置。 | 展示如何把布局/分块选择转化为可搜索的性能参数。 |
| 17 | [Learning CUDA by optimizing softmax: A worklog](https://maharshi.bearblog.dev/optimizing-softmax-cuda/) | Maharshi Pandya | Maharshi’s Blog | 个人博客 | 2025-01-04 | 中级 | softmax, online softmax, shared memory, reduction | 从 PyTorch baseline 出发，逐步优化 softmax，涉及 online 计算、共享内存和 block 协作。 | 可将 softmax 的数学稳定性与 CuTe attention epilogue 联系起来。 |
| 18 | [Learning CUDA by optimizing matrix-vector multiplication (SGEMV) for cuBLAS-like performance](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/) | Maharshi Pandya | Maharshi’s Blog | 个人博客 | 2025-01-18 | 中级 | SGEMV, coalescing, warp reduction, bandwidth | 围绕 SGEMV 的算术强度、合并访存和 warp-level reduction 做迭代优化，并与 cuBLAS 对比。 | 适合练习 memory-bound kernel 的线程映射与归约设计。 |
| 19 | [Optimizing a Layer Normalization Kernel with CUDA: a Worklog](https://aryagxr.com/blogs/cuda-optimizing-layernorm) | Arya | Arya’s Blog | 个人博客 | 2025-02-17 | 中级 | LayerNorm, shared memory, reduction, float4 | 通过 shared memory reduction 和 float4 向量化访存优化 LayerNorm kernel。 | 对应 LLM 常见融合算子，可练习 reduction、alignment 和向量化布局。 |
| 20 | [Chipmunk: Deep Dive on GPU Kernel Optimizations and Systems (Part III)](https://sandyresearch.github.io/chipmunk-part-III/) | Sandy Research | Sandy Research / ML Systems Lab | 署名研究博客 | 2025-04-21 | 高级 | sparsity, GPU kernels, systems optimization, diffusion | 从动态列稀疏增量出发，深入讨论 GPU kernel 优化、稀疏表示和系统级加速。 | 补充规则 dense tile 之外的稀疏布局、负载均衡和 kernel 设计视角。 |


<a id="compiler"></a>
## 编译器、DSL 与 PTX/SASS（21 篇）

> **本模块怎么读：** 关注 CUDA/Triton/CuTe 到 PTX、SASS、JIT、PyTorch custom op 和 Python binding 的落地链路。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [CUDA Compilation](https://leimao.github.io/blog/CUDA-Compilation/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-04-28 | 中级 | nvcc, compilation, host/device | 解释 CUDA 源码从编译到 host/device 二进制的流程。 | 理解 DSL/kernel 编译链的前置知识。 |
| 2 | [CUDA Compilation Architecture Macro](https://leimao.github.io/blog/CUDA-Compilation-Architecture-Macro/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-05-01 | 中级 | architecture macro, compilation, conditional code | 介绍按 GPU 架构选择编译代码的宏机制。 | 理解多架构 kernel dispatch。 |
| 3 | [CUDA Driver VS CUDA Runtime](https://leimao.github.io/blog/CUDA-Driver-VS-CUDA-Runtime/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2020-10-01 | 中级 | driver API, runtime API, module | 比较 CUDA Driver API 与 Runtime API。 | 为 JIT/module loading 和 DSL backend 做准备。 |
| 4 | [Load CUDA Kernel at Runtime Using CUDA Driver APIs](https://leimao.github.io/blog/CUDA-Driver-Runtime-Load-Run-Kernel/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-06-30 | 高级 | PTX, driver API, runtime loading, JIT | 展示运行时加载和执行 CUDA kernel/module。 | 理解 codegen 后如何进入 GPU。 |
| 5 | [PyTorch Triton Kernel Transparent Tracing and Compilation](https://leimao.github.io/blog/PyTorch-Triton-Kernel-Transparent-Tracing-and-Compilation/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-05-22 | 高级 | Triton, tracing, compilation, PyTorch | 跟踪 PyTorch 中 Triton kernel 的编译/调用路径。 | 连接 Python DSL、编译器和运行时。 |
| 6 | [PyTorch AOTInductor Hybrid Lowering](https://leimao.github.io/blog/PyTorch-AOTInductor-Hybrid-Lowering/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-05-28 | 高级 | AOTInductor, lowering, codegen, compiler | 讨论 AOTInductor 的混合 lowering 与代码生成。 | 理解现代 ML compiler 的分层。 |
| 7 | [PyTorch Custom Operation](https://leimao.github.io/blog/PyTorch-Custom-Operation/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-05-10 | 中级 | custom op, PyTorch extension, dispatch | 介绍把自定义 CUDA kernel 接入 PyTorch。 | CuTe/CUDA kernel 工程落地必备。 |
| 8 | [Custom CUDA and Python](https://research.colfax-intl.com/custom-cuda-and-python/) | Andrey | Colfax Research | 署名工程博客 | 2024-03-02 | 中级 | CUDA Python, custom kernel, bindings | 介绍在 Python 工作流中编写/调用自定义 CUDA。 | 理解 DSL/绑定层与底层 kernel 的边界。 |
| 9 | [Tutorial: Python bindings for CUDA libraries in PyTorch](https://research.colfax-intl.com/tutorial-python-binding-for-cuda-libraries-in-pytorch/) | Colfax Research | Colfax Research | 署名工程博客 | 2024-03-13 | 中级 | Python bindings, CUDA libraries, PyTorch | 展示 CUDA library 与 PyTorch Python API 的绑定。 | 把底层库包装成可用算子。 |
| 10 | [A Gentle Introduction to CUDA PTX](https://philipfabianek.com/posts/cuda-ptx-introduction) | Philip Fabianek | Philip Fabianek’s Blog | 个人博客 | 2025-09-11 | 中级 | PTX, inline assembly, nvcc, SASS | 以可运行示例介绍 PTX 虚拟汇编、寄存器、谓词和 CUDA 编译流程。 | 帮助理解 DSL 生成的低层指令及其与硬件 ISA 的关系。 |
| 11 | [Tutorial: Understanding GPU Assembly with PTX](https://eunomia.dev/others/cuda-tutorial/02-ptx-assembly/) | yunwei37、officeyutong（eunomia） | eunomia | 署名技术博客 | 2025-05-23 | 中级 | PTX, GPU assembly, disassembly, inline PTX | 从 CUDA kernel 生成的 PTX 入手，说明虚拟汇编、指令读法和内联 PTX。 | 适合作为 CuTe DSL 生成代码后进行 PTX 层检查的入门材料。 |
| 12 | [Reversing Nvidia GPU’s SASS code – JEB in Action](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/) | Nicolas Falliere | PNF Software Blog | 署名工程博客 | 2025-08-15 | 高级 | SASS, cubin, disassembly, reverse engineering | 介绍 NVIDIA GPU SASS/cubin 的反汇编与分析方法，展示指令和内存空间的组织。 | 帮助从 SASS 侧验证编译器最终生成的指令与资源使用。 |
| 13 | [GPU MODE Lecture 3: Getting Started With CUDA for Python Programmers](https://christianjmills.com/posts/cuda-mode-notes/lecture-003/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-08-31 | 入门 | CUDA Python, PyTorch, kernel launch, indexing | 面向 Python/PyTorch 用户介绍如何编写、编译和调用 CUDA kernel，并用图像处理与矩阵乘做示例。 | 帮助理解 CuTe/CUDA kernel 如何从 Python 训练或推理代码进入运行时。 |
| 14 | [GPU MODE Lecture 10: Build a Prod Ready CUDA library](https://christianjmills.com/posts/cuda-mode-notes/lecture-010/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-15 | 中级 | CUDA library, C++, scheduling, abstraction, deployment | 讨论如何构建面向生产的 CUDA C++ 库，将通信、任务调度和错误处理封装成可复用接口。 | 帮助把 CuTe kernel 从实验代码推进到可集成、可维护的工程组件。 |
| 15 | [GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-11-15 | 中级 | Triton, block programming, masking, matmul, code generation | 通过 copy、图像处理和矩阵乘示例，讲解 Triton 的 block-level 编程和性能调优。 | Triton 与 CuTe DSL 都强调布局/分块，但抽象层次不同，适合对照学习。 |
| 16 | [Analyze CUDA programs by looking at GPU assembly.](https://veitner.bearblog.dev/analyze-cuda-programs-by-looking-at-gpu-assembly/) | Simon Veitner | Simon Veitner’s Blog | 个人博客 | 2025-04-21 | 中级→高级 | SASS, GPU assembly, vectorized load, bandwidth | 通过普通与向量化复制 kernel 的 SASS 对比，分析指令数量、访存宽度和带宽差异。 | 提供从 CuTe/CUDA 源码追到最终机器指令的具体方法。 |
| 17 | [PTX Mental Model](https://ita9naiwa.github.io/mlsys/2025/10/05/ptx-mental-model.html) | Hyunsung Lee | Hyunsung Lee’s Blog | 个人博客 | 2025-10-05 | 高级 | PTX, MMA, Tensor Core, register layout | 从最小矩阵乘示例建立 PTX 层面的 mma、寄存器和线程协作模型。 | 把 CuTe MMA 抽象与 PTX 指令语义逐项对应起来。 |
| 18 | [Flash Attention from Scratch Part 8: Instruction Reduction](https://lubits.ch/flash/Part-8) | Sonny | Sonny’s Blog | 个人博客 | 2025-11-08 | 高级 | instruction reduction, PTX, SASS, register pressure | 通过减少逻辑、位移和地址计算指令，降低寄存器压力并改善 attention kernel 性能。 | 提供从 SASS 指令数量反推 CuTe/CUDA 写法的优化思路。 |
| 19 | [PyTorch CUDA Extensions](https://dlewis.io/pytorch-cuda-extensions/) | Derek Lewis | Derek’s Blog | 个人博客 | 2024-08-08 | 中级 | PyTorch extension, C++/CUDA, binding, build | 介绍如何编写、编译并从 PyTorch 调用自定义 C++/CUDA 扩展。 | 帮助把 CuTe kernel 接入 Python/PyTorch 工程和测试基准。 |
| 20 | [Maybe consider putting “cutlass” in your CUDA/Triton kernels](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/) | Henry Zhu | Henry Zhu’s Blog | 个人博客 | 2025-12-15 | 高级 | CUTLASS, Triton, code generation, FP8 | 讨论在 CUDA/Triton kernel 中复用 CUTLASS 组件的价值，并分析低精度实现和代码生成。 | 帮助理解 CuTe/CUTLASS 原语如何作为 DSL kernel 的底层 building blocks。 |
| 21 | [pegainfer (3): From Launch Overhead to CUDA Graph (Part 1)](https://susun-blog.com/blog/pegainfer-3-cuda-graph/) | Jinyang Su | susun’s Blog | 个人博客 | 2026-02-26 | 中级→高级 | CUDA Graph, launch overhead, kernel fusion, mega kernel | 量化 kernel launch overhead，并比较 CUDA Graph、kernel fusion、mega-kernel 和 dynamic parallelism。 | 补充 CuTe kernel 之外的执行图优化，直接对应 CUDA Graph 学习模块。 |


<a id="profiling"></a>
## Profiling、Roofline 与性能工程（14 篇）

> **本模块怎么读：** 关注 roofline、hot/cold benchmark、Nsight Compute/Systems、trace、错误定位和可复现实验。

| # | 文章 | 作者 | 来源 | 类型 | 日期 | 难度 | 关键词 | 核心摘要 | 为什么适合 CuTe DSL |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | [Roofline Performance Model](https://leimao.github.io/blog/Roofline-Performance-Model/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-03-26 | 中级 | roofline, arithmetic intensity, FLOPS, bandwidth | 解释 roofline 模型与性能上限估算。 | 形成定量瓶颈分析能力。 |
| 2 | [CUDA Performance Hot VS Cold Measurement](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-03-12 | 入门→中级 | benchmark, warmup, cold start, timing | 区分 cold/hot 测量与 warmup、缓存、JIT 影响。 | 避免 benchmark 结论失真。 |
| 3 | [Nsight Streamer](https://leimao.github.io/blog/Nsight-Streamer/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-11-04 | 中级 | Nsight, profiling, remote GPU | 介绍 Nsight profiling 工作流/辅助工具。 | 提高远程 GPU 分析效率。 |
| 4 | [Docker Nsight Compute](https://leimao.github.io/blog/Docker-Nsight-Compute/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2024-01-02 | 中级 | Nsight Compute, Docker, profiling | 记录在容器中使用 Nsight Compute 的方法。 | 解决实际环境 profiling 难题。 |
| 5 | [Docker Nsight Systems](https://leimao.github.io/blog/Docker-Nsight-Systems/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-06-01 | 中级 | Nsight Systems, Docker, timeline | 记录容器化 Nsight Systems 工作流。 | 观察 stream/CPU/GPU overlap。 |
| 6 | [Perfetto GPU Flow Artifacts](https://leimao.github.io/blog/Perfetto-GPU-Flow-Artifacts/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-02-20 | 高级 | Perfetto, GPU trace, flow events | 分析 GPU trace 中的 flow artifacts 与事件关联。 | 深入解释系统级 timeline。 |
| 7 | [CUDA_LAUNCH_BLOCKING=1](https://leimao.github.io/blog/CUDA_LAUNCH_BLOCKING=1/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2026-03-20 | 入门→中级 | debugging, synchronization, launch blocking | 说明强制同步调试对错误定位和性能的影响。 | 快速定位异步错误的实用工具。 |
| 8 | [Proper CUDA Error Checking](https://leimao.github.io/blog/Proper-CUDA-Error-Checking/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2022-05-25 | 入门 | error checking, runtime API, debugging | 整理 CUDA API/kernel 错误检查模式。 | 所有底层实验的必备工程习惯。 |
| 9 | [Illegal Memory Access and Segmentation Fault](https://leimao.github.io/blog/Illegal-Memory-Access-Segmentation-Fault/) | Lei Mao | Lei Mao’s Log Book | 个人博客 | 2025-08-27 | 中级 | illegal access, sanitizer, debugging | 分析 illegal memory access 与 segmentation fault 的排查路径。 | 对 layout/copy 错误尤其有用。 |
| 10 | [How to set up Nsight Compute Locally to profile Remote GPUs](https://tspeterkim.github.io/posts/nsight-setup-on-ec2) | Taeksang Peter Kim | Taeksang Peter Kim’s Blog | 个人博客 | 2024-04-22 | 中级 | Nsight Compute, remote GPU, profiling | 介绍本地 Nsight Compute 连接远程 GPU 的配置。 | 适合云 GPU 学习环境。 |
| 11 | [The One Billion Row Challenge in CUDA: from 17m to 17s](https://tspeterkim.github.io/posts/cuda-1brc) | Taeksang Peter Kim | Taeksang Peter Kim’s Blog | 个人博客 | 2024-04-10 | 中级→高级 | CUDA, parsing, memory, benchmark | 通过 1BRC 任务展示 CUDA 优化、数据解析和 benchmark。 | 综合性能工程案例。 |
| 12 | [GPU MODE Lecture 1: How to profile CUDA kernels in PyTorch](https://christianjmills.com/posts/cuda-mode-notes/lecture-001/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-04-26 | 中级 | Nsight Compute, load_inline, Triton, custom CUDA | 以 PyTorch 自定义 CUDA kernel 为例，介绍 load_inline、Triton 和 Nsight Compute 的 profiling 工作流。 | CuTe kernel 写完后必须知道如何看指标；这篇适合作为 profile 入门。 |
| 13 | [GPU MODE Lecture 8: CUDA Performance Checklist](https://christianjmills.com/posts/cuda-mode-notes/lecture-008/index.html) | Christian Mills | Christian Mills’s Blog | 个人博客 | 2024-09-11 | 中级 | coalescing, occupancy, divergence, tiling, privatization, NCU | 以性能清单形式总结合并访存、occupancy、控制发散、tiling、privatization、thread coarsening 和 Nsight Compute。 | 是检查 CuTe kernel 是否踩常见性能坑的实用 checklist。 |
| 14 | [Flash Attention from Scratch Part 7: A100 Profiling](https://lubits.ch/flash/Part-7) | Sonny | Sonny’s Blog | 个人博客 | 2025-11-01 | 高级 | A100, profiling, block size, attention kernel | 将自实现 attention kernel 移植到 A100，分析 block size 限制和与参考实现的差距。 | 帮助建立跨 GPU 架构复测 CuTe kernel 的习惯。 |


## 去重与质量核验

- 唯一文章数：**148**；正文 URL 数：**148**。
- 候选池中原有 1 个 404 页面已删除；1 个复用 URL 的虚构标题已删除；已修正 CuTe Layout、CUDA Matrix Multiplication 等字段错位。
- 当前正文未使用 `nvidia.com`、CUDA 官方文档域名或 NVIDIA 官方仓库链接；Colfax 等条目按署名工程博客单独标注。
- “HTTP 200”只表示整理时页面可访问，不保证链接永久有效；部分站点可能因地区、反爬或登录策略变化而不可访问。

## 术语对照与学习提示

- **Layout**：不仅是“张量形状”，还包含从逻辑坐标到物理索引的映射；CuTe 的 shape/stride/composition 应与实际地址计算一起看。
- **Tiler / Local Partition**：把全局问题分解为 CTA、warp、thread、value 多级局部工作集，是分块 kernel 的主线。
- **Copy 与 MMA 是两条同等重要的流水线**：高性能 kernel 不只是算得快，还要持续、无冲突地把数据送到寄存器/Tensor Core。
- **不要孤立追求 occupancy**：寄存器、shared memory、指令级并行、访存延迟和 Tensor Core 利用率需要一起观察。
- **每篇文章的 benchmark 都有边界**：记录 GPU 型号、CUDA/编译器版本、数据类型、矩阵形状、batch、warmup 和统计方式。

## 最小实践清单

建议每读完一个模块，至少做一个可运行实验：

| 阶段 | 实验 | 应观察的指标 |
|---|---|---|
| 1 | 向量加法/矩阵转置 | coalescing、alignment、bank conflict |
| 2 | tiled GEMM | global transactions、shared-memory reuse、register 使用 |
| 3 | warp reduction/softmax/RMSNorm | shuffle、同步、数值稳定性、融合收益 |
| 4 | CuTe Tiled Copy + Tiled MMA | thread/value layout、copy atom、MMA atom |
| 5 | pipeline/TMA/persistent kernel | overlap、stall reason、occupancy、tensor pipe utilization |
| 6 | Nsight + Roofline 回归 | 算术强度、带宽、指令吞吐、端到端可复现性 |

