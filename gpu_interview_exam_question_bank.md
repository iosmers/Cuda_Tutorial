# GPU Hardware and Software Interview/Exam Question Bank (200)

> 用途：按层次学习 GPU 硬件、CUDA 编程、性能优化、软件栈、编译器和系统设计。建议每道题先独立回答，再查资料补充，最后用代码或工具验证关键结论。

## Level 1: 基础概念与硬件入门

| # | 问题 | 学习重点 |
|---|---|---|
| 1 | GPU 和 CPU 在架构目标上有什么核心区别？ | 吞吐量优先、延迟优先、SIMT、缓存层次 |
| 2 | 为什么 GPU 适合大规模数据并行任务？ | 海量线程、隐藏延迟、内存带宽、算术密度 |
| 3 | 什么是 GPU 的 SM、CUDA Core、Warp 和 Thread？ | NVIDIA 执行层次和术语 |
| 4 | Warp 通常包含多少个线程？为什么这个数字很重要？ | 32 线程、调度粒度、分支和访存行为 |
| 5 | SIMD、SIMT 和 SPMD 有什么区别？ | 执行模型、编程模型、硬件控制方式 |
| 6 | GPU 的 Global Memory、Shared Memory、Register 和 Cache 分别有什么作用？ | 内存层次、容量、延迟、带宽 |
| 7 | 什么是 HBM/GDDR 显存？它们和 CPU DDR 内存有什么区别？ | 显存技术、带宽、容量、功耗 |
| 8 | 什么是内存带宽？如何估算一个 kernel 的带宽需求？ | Bytes/s、读写量、Roofline 基础 |
| 9 | 什么是计算吞吐量？FLOPS、TFLOPS、TOPS 分别表示什么？ | 浮点和整数性能指标 |
| 10 | GPU 为什么需要大量寄存器？寄存器数量会影响什么？ | 线程上下文、occupancy、spill |
| 11 | SM 如何调度线程块？一个 block 能否跨多个 SM 执行？ | Block 到 SM 的调度和驻留限制 |
| 12 | 什么是 occupancy？它为什么不是越高越好？ | 活跃 warp、资源限制、性能相关性 |
| 13 | GPU 如何通过多 warp 隐藏访存延迟？ | Warp scheduling、latency hiding |
| 14 | 什么是 warp divergence？它会带来什么性能代价？ | 分支路径串行化、控制流一致性 |
| 15 | GPU 的 L1 Cache、L2 Cache 和 Texture Cache 各有什么作用？ | 缓存位置、访问模式、复用 |
| 16 | Shared Memory 和 L1 Cache 在硬件资源上有什么关系？ | 片上存储、配置、架构差异 |
| 17 | 什么是 bank conflict？为什么 shared memory 会发生 bank conflict？ | Bank 映射、访问冲突、并行度下降 |
| 18 | 什么是 coalesced memory access？ | 合并访存、连续地址、warp 访问 |
| 19 | 为什么非合并访存会显著降低 GPU 性能？ | Memory transaction、带宽浪费 |
| 20 | 什么是 memory transaction？它和线程访问地址有什么关系？ | 缓存线、事务粒度、对齐 |
| 21 | 什么是 PCIe？它对 CPU-GPU 数据传输有什么影响？ | Host-device 传输瓶颈 |
| 22 | NVLink 和 PCIe 相比解决了什么问题？ | GPU 间互联、高带宽、低延迟 |
| 23 | 什么是 NUMA？多 GPU 服务器里为什么要关注 CPU-GPU 拓扑？ | 亲和性、PCIe root complex、带宽 |
| 24 | 什么是 Tensor Core？它主要加速哪类运算？ | 矩阵乘加、混合精度、AI workloads |
| 25 | FP32、FP16、BF16、TF32、FP8 有什么区别？ | 精度、动态范围、吞吐量 |
| 26 | 什么是整数量化 INT8/INT4？为什么推理常用量化？ | 推理性能、显存占用、精度损失 |
| 27 | 什么是 ECC 显存？训练和科学计算为什么关注 ECC？ | 内存错误、可靠性、性能开销 |
| 28 | GPU 的功耗、温度和频率之间有什么关系？ | Thermal throttling、boost clock、TDP |
| 29 | GPU 的显存容量和显存带宽哪个更重要？ | 工作集大小、性能瓶颈分析 |
| 30 | 什么是 compute capability？它决定哪些 CUDA 特性？ | 架构能力、指令、API 支持 |
| 31 | NVIDIA、AMD 和 Intel GPU 在软件生态上有什么主要区别？ | CUDA、ROCm、oneAPI、兼容性 |
| 32 | 什么是集成 GPU 和独立 GPU？它们的内存访问模式有什么不同？ | UMA、独显显存、带宽和延迟 |
| 33 | GPU 中的 scheduler、dispatch unit 和 execution unit 大致负责什么？ | 指令发射和执行管线 |
| 34 | 什么是 latency-bound 和 throughput-bound workload？ | 瓶颈分类、优化方向 |
| 35 | 什么是 arithmetic intensity？它如何判断程序偏算力还是偏带宽？ | FLOPs/Byte、Roofline 模型 |
| 36 | 为什么矩阵乘法比向量加法更容易发挥 GPU 算力？ | 数据复用、算术密度 |
| 37 | 什么是 reduction？为什么 reduction 是 GPU 编程经典题？ | 并行归约、同步、访存优化 |
| 38 | 什么是 scan/prefix sum？它常用于哪些场景？ | 并行前缀和、压缩、排序 |
| 39 | 什么是 stencil computation？GPU 上优化 stencil 的关键是什么？ | 邻域访问、缓存复用、边界处理 |
| 40 | 如何用一句话描述 GPU 程序从 CPU 发起到硬件执行的流程？ | Host API、kernel launch、grid/block/warp |

## Level 2: CUDA 编程模型与内存体系

| # | 问题 | 学习重点 |
|---|---|---|
| 41 | CUDA 程序中的 host code 和 device code 分别运行在哪里？ | CPU/GPU 分工 |
| 42 | `__global__`、`__device__`、`__host__` 的含义和使用限制是什么？ | 函数限定符 |
| 43 | CUDA kernel 的 `<<<grid, block, sharedMem, stream>>>` 四个参数分别表示什么？ | kernel launch 配置 |
| 44 | `threadIdx`、`blockIdx`、`blockDim`、`gridDim` 如何计算全局线程索引？ | 索引公式 |
| 45 | 一维、二维、三维 grid/block 分别适合哪些数据布局？ | 数据映射策略 |
| 46 | 为什么 kernel 内经常需要边界判断 `if (idx < N)`？ | 越界保护、向上取整 |
| 47 | `cudaMalloc`、`cudaMemcpy`、`cudaFree` 的基本使用流程是什么？ | 显存管理 |
| 48 | `cudaMemcpyHostToDevice` 和 `cudaMemcpyDeviceToHost` 的方向如何判断？ | 数据传输方向 |
| 49 | CUDA API 调用为什么要做错误检查？ | 异步错误、调试可靠性 |
| 50 | `cudaGetLastError` 和 `cudaDeviceSynchronize` 在错误定位上有什么区别？ | launch error、runtime error |
| 51 | `cudaEvent` 如何用于 GPU 计时？它和 CPU 计时有什么区别？ | 异步执行、事件同步 |
| 52 | CUDA kernel 启动为什么通常是异步的？ | Host-device 并发 |
| 53 | 默认 stream 和非默认 stream 的行为有什么区别？ | stream 语义、隐式同步 |
| 54 | 什么是 pinned host memory？它为什么能加速拷贝？ | 页锁定内存、DMA |
| 55 | `cudaMemcpyAsync` 真正异步需要满足哪些条件？ | pinned memory、stream、硬件 copy engine |
| 56 | 什么是 unified memory？它和显式 `cudaMalloc` 有什么不同？ | UVM、页迁移、易用性 |
| 57 | Unified Memory 中 page fault 可能导致什么性能问题？ | 运行时迁移、预取、访问模式 |
| 58 | `cudaMemPrefetchAsync` 和 `cudaMemAdvise` 用来解决什么问题？ | UVM 优化 |
| 59 | 什么是 shared memory 静态分配和动态分配？ | `__shared__`、extern shared |
| 60 | `__syncthreads()` 的作用是什么？哪些场景下会死锁？ | block 级同步、分支一致性 |
| 61 | 为什么 `__syncthreads()` 不能同步不同 block？ | block 独立调度 |
| 62 | 什么时候应该使用 shared memory 而不是直接访问 global memory？ | 数据复用、局部通信 |
| 63 | 如何用 shared memory 实现 tiled matrix multiplication？ | 分块、同步、复用 |
| 64 | shared memory bank conflict 如何通过 padding 缓解？ | 对齐、步长、padding |
| 65 | 什么是 register spilling？如何发现和减少它？ | 本地内存、寄存器压力、编译信息 |
| 66 | 什么是 local memory？它一定在片上吗？ | 线程私有、global backing |
| 67 | constant memory 适合存放什么数据？ | 广播读取、只读小常量 |
| 68 | texture memory/read-only cache 适合哪些访问模式？ | 空间局部性、只读数据 |
| 69 | `__restrict__` 在 CUDA C++ 中可能带来什么优化？ | 指针别名分析 |
| 70 | `const` 指针和只读数据缓存有什么关系？ | 编译器优化、cache path |
| 71 | 什么是 atomic operation？CUDA 中常见原子操作有哪些？ | atomicAdd、atomicCAS、并发更新 |
| 72 | 原子操作为什么可能成为性能瓶颈？ | 争用、序列化、热点地址 |
| 73 | 如何用 block-level reduction 减少全局原子操作数量？ | 层次化归约 |
| 74 | Warp-level primitive `__shfl_sync` 能解决什么问题？ | warp 内数据交换 |
| 75 | `__ballot_sync`、`__any_sync`、`__all_sync` 适合哪些场景？ | warp 投票、掩码 |
| 76 | CUDA 中的 warp mask 为什么重要？ | 活跃线程、正确性、同步 |
| 77 | 什么是 cooperative groups？它比裸 `__syncthreads()` 多了什么抽象？ | 分组同步、可读性 |
| 78 | CUDA 动态并行是什么？为什么实际使用要谨慎？ | device launch、开销、适用场景 |
| 79 | CUDA graph 解决了什么问题？ | kernel launch overhead、图捕获和重放 |
| 80 | 如何把一个 CPU for 循环改写成最基础的 CUDA kernel？ | 数据并行拆分、索引、内存传输 |

## Level 3: 性能优化、调试与工程实践

| # | 问题 | 学习重点 |
|---|---|---|
| 81 | 优化 CUDA kernel 时应先看正确性还是性能？为什么？ | 工程流程、基准可信度 |
| 82 | 一个 CUDA 程序的性能基准应该如何设计？ | warmup、重复次数、同步、统计 |
| 83 | 如何计算 kernel 的有效带宽？ | 读写字节数、执行时间 |
| 84 | 如何计算矩阵乘法 kernel 的 GFLOPS？ | 运算量、时间、维度 |
| 85 | Roofline 模型如何指导 GPU 优化？ | 算术密度、带宽顶线、算力顶线 |
| 86 | Nsight Systems 和 Nsight Compute 分别解决什么问题？ | 系统时间线、kernel 细节 |
| 87 | `compute-sanitizer` 能检查哪些常见错误？ | memcheck、racecheck、synccheck |
| 88 | 如何定位 CUDA illegal memory access？ | 边界、同步、错误检查 |
| 89 | 如何判断一个 kernel 是访存瓶颈还是计算瓶颈？ | profiler metrics、Roofline |
| 90 | global memory load efficiency 低通常意味着什么？ | 非合并访存、事务浪费 |
| 91 | branch efficiency 低通常意味着什么？ | warp divergence |
| 92 | achieved occupancy 低一定是坏事吗？ | 瓶颈解释、资源权衡 |
| 93 | block size 如何选择？为什么常从 128、256、512 试起？ | warp 倍数、资源、调度 |
| 94 | grid-stride loop 有什么好处？ | 可扩展 kernel、复用线程 |
| 95 | 为什么向量化访存如 `float4` 可能提升性能？ | 宽加载、对齐、指令数 |
| 96 | 什么时候 `float4` 反而可能出问题？ | 对齐、边界、寄存器压力 |
| 97 | AoS 和 SoA 数据布局在 GPU 上有什么性能差异？ | 合并访存、缓存友好性 |
| 98 | 矩阵转置为什么容易出现非合并写入或读取？ | 行列访问模式 |
| 99 | 如何用 shared memory 优化矩阵转置？ | tile、padding、读写合并 |
| 100 | reduction 优化通常经历哪些版本？ | naive、shared、warp shuffle、多阶段 |
| 101 | 为什么 reduction 中要避免大量线程闲置？ | 并行效率、控制流 |
| 102 | prefix sum 的 work-efficient 和 step-efficient 版本有什么区别？ | 并行算法复杂度 |
| 103 | histogram 在 GPU 上为什么难优化？ | 原子冲突、数据分布 |
| 104 | 如何优化 histogram 的原子竞争？ | 私有化、分层合并、shared atomic |
| 105 | stencil kernel 如何利用 shared memory 做 halo 区域缓存？ | 邻域复用、边界 |
| 106 | 卷积 kernel 中 direct convolution、im2col、Winograd、FFT 方法各有什么取舍？ | 算法选择、数据变换 |
| 107 | 矩阵乘法 tile size 如何影响性能？ | 数据复用、寄存器、shared 容量 |
| 108 | 什么是 double buffering？它如何隐藏访存延迟？ | 预取、计算/加载重叠 |
| 109 | 什么是 asynchronous copy？它在新架构上解决什么问题？ | cp.async、global-to-shared pipeline |
| 110 | Loop unrolling 的收益和风险是什么？ | 指令调度、代码膨胀、寄存器 |
| 111 | Fast math 选项可能改变哪些数值行为？ | 近似函数、舍入、精度 |
| 112 | FMA 对性能和数值结果有什么影响？ | 融合乘加、舍入差异 |
| 113 | 浮点归约为什么可能非确定？ | 并行顺序、非结合性 |
| 114 | 如何提高 GPU 计算结果的数值稳定性？ | Kahan、pairwise、混合精度策略 |
| 115 | 如何减少 host-device 数据传输开销？ | 数据驻留、批处理、异步拷贝 |
| 116 | 如何用 stream 实现 H2D、kernel、D2H 的流水线？ | 多 stream、copy engine、事件依赖 |
| 117 | 什么是 overlap？如何确认计算和拷贝真的重叠了？ | timeline、pinned memory、stream |
| 118 | 多 kernel 拆分和融合各有什么优缺点？ | launch 开销、访存、可维护性 |
| 119 | 为什么小规模数据在 GPU 上可能比 CPU 慢？ | launch overhead、传输开销、并行度 |
| 120 | 一个 CUDA kernel 优化到什么程度可以停止？ | 性能目标、瓶颈顶线、工程成本 |

## Level 4: GPU 软件栈、库、编译器与框架

| # | 问题 | 学习重点 |
|---|---|---|
| 121 | CUDA Runtime API 和 Driver API 有什么区别？ | 抽象层次、控制粒度 |
| 122 | CUDA context 是什么？多线程程序中 context 管理要注意什么？ | 上下文、设备状态 |
| 123 | PTX 是什么？它和 SASS/cubin 有什么关系？ | 中间表示、机器码、JIT |
| 124 | `nvcc` 编译 CUDA 程序的大致流程是什么？ | host 编译、device 编译、fatbin |
| 125 | `-arch`、`-code`、`sm_xx`、`compute_xx` 分别表示什么？ | 架构目标、兼容性 |
| 126 | 为什么程序中可能同时包含 PTX 和 cubin？ | 前向兼容、JIT fallback |
| 127 | 如何查看 CUDA kernel 的寄存器使用和 shared memory 使用？ | ptxas info、Nsight Compute |
| 128 | 如何反汇编查看 SASS？ | cuobjdump、nvdisasm |
| 129 | PTX 优化和 SASS 优化有什么不同风险？ | 虚拟 ISA、架构相关性 |
| 130 | 什么是 JIT compilation？CUDA 中哪些场景会触发 JIT？ | PTX 装载、NVRTC、框架编译 |
| 131 | NVRTC 适合什么场景？ | 运行时编译、动态 kernel |
| 132 | Thrust、CUB、cuBLAS、cuDNN、NCCL 分别解决什么问题？ | CUDA 库生态 |
| 133 | 什么时候应该调用 cuBLAS 而不是自己写 GEMM？ | 性能、可靠性、维护成本 |
| 134 | cuBLAS 中 leading dimension 的含义是什么？ | 矩阵布局、列主序兼容 |
| 135 | cuDNN 如何加速深度学习中的卷积和归一化？ | 算法选择、Tensor Core |
| 136 | NCCL 为什么适合多 GPU 通信？ | collective、拓扑感知、带宽 |
| 137 | AllReduce、Broadcast、ReduceScatter、AllGather 分别是什么？ | 分布式训练通信原语 |
| 138 | CUDA-aware MPI 是什么？ | GPU buffer 直接通信 |
| 139 | GPUDirect RDMA 解决了什么问题？ | NIC-GPU 直连、绕过 CPU 拷贝 |
| 140 | GPUDirect Storage 适合哪些场景？ | 存储到 GPU 数据路径 |
| 141 | PyTorch Tensor 放到 CUDA 上之后，底层大致如何执行？ | 张量、kernel、库调用 |
| 142 | PyTorch 中 `.cuda()`、`.to(device)`、`non_blocking=True` 分别要注意什么？ | 设备迁移、异步拷贝 |
| 143 | PyTorch 自定义 CUDA extension 的基本组成是什么？ | C++ binding、CUDA kernel、setup |
| 144 | Triton 编程模型和 CUDA C++ 有什么区别？ | program instance、block-level DSL |
| 145 | Triton 中 block size、num warps、mask 的作用是什么？ | tile 映射、边界处理 |
| 146 | CUTLASS 是什么？为什么 GEMM 实现会非常复杂？ | 分层 tiling、Tensor Core、pipeline |
| 147 | TVM、XLA、TorchInductor 这类编译器试图自动解决什么问题？ | 图优化、kernel 生成、融合 |
| 148 | MLIR/LLVM/NVVM 在 GPU 编译链中可能扮演什么角色？ | 中间表示、优化 passes |
| 149 | Kernel fusion 的主要收益是什么？ | 减少访存和 launch |
| 150 | Kernel fusion 过度会带来什么问题？ | 寄存器压力、代码复杂度、并行度 |
| 151 | Lazy execution 和 eager execution 在 GPU 框架中有什么区别？ | 调度时机、优化空间 |
| 152 | 自动混合精度 AMP 的基本思想是什么？ | FP16/BF16、loss scaling |
| 153 | 为什么训练中常需要 loss scaling？ | FP16 下溢、梯度稳定性 |
| 154 | ONNX Runtime、TensorRT 和普通框架推理有什么差异？ | 推理优化、engine、部署 |
| 155 | TensorRT 优化模型时会做哪些典型转换？ | layer fusion、precision calibration、tactic |
| 156 | 什么是 CUDA Graphs 在 PyTorch 中的应用价值？ | 减少调度开销、静态工作负载 |
| 157 | 容器中运行 GPU 程序需要哪些组件配合？ | NVIDIA driver、container toolkit、CUDA libs |
| 158 | CUDA driver version 和 runtime version 不匹配时会发生什么？ | 兼容性、部署问题 |
| 159 | 如何设计一个可复现的 GPU 软件环境？ | driver、CUDA、cuDNN、容器、锁版本 |
| 160 | GPU 程序的单元测试和性能回归测试应该怎么组织？ | 正确性基准、容差、性能阈值 |

## Level 5: 高级架构、系统设计与大规模应用

| # | 问题 | 学习重点 |
|---|---|---|
| 161 | 一个 SM 内部通常有哪些主要执行资源？ | FP32/INT/Tensor/LDST/SFU 管线 |
| 162 | Instruction latency 和 throughput 的区别是什么？ | 延迟、发射率、调度 |
| 163 | Warp scheduler 如何在 ready warp 之间选择执行？ | scoreboard、依赖、隐藏延迟 |
| 164 | 什么是 scoreboard？它如何处理数据依赖？ | 指令依赖跟踪 |
| 165 | Memory consistency model 在 GPU 编程中为什么重要？ | 可见性、排序、同步 |
| 166 | `__threadfence()`、`__threadfence_block()`、`__threadfence_system()` 有什么区别？ | 内存栅栏作用域 |
| 167 | CUDA atomic 的 memory order 和 scope 概念解决什么问题？ | 现代同步语义 |
| 168 | Persistent kernel 是什么？适合哪些任务？ | 常驻线程、任务队列、低延迟 |
| 169 | Work stealing 或 task queue 在 GPU 上实现有哪些难点？ | 原子、负载均衡、同步 |
| 170 | Dynamic parallelism 和 persistent kernel 在任务生成上有什么取舍？ | 启动开销、灵活性 |
| 171 | Multi-Instance GPU (MIG) 解决什么问题？ | 资源隔离、多租户 |
| 172 | MPS 和 MIG 有什么区别？ | 进程共享、硬件分区 |
| 173 | 多 GPU 程序如何选择数据并行、模型并行、流水线并行？ | 扩展策略 |
| 174 | 大模型训练中的 tensor parallelism 和 pipeline parallelism 分别切分什么？ | 权重/层切分、通信模式 |
| 175 | ZeRO/FSDP 的核心思想是什么？ | 参数、梯度、优化器状态分片 |
| 176 | GPU 集群中的通信瓶颈如何分析？ | 计算通信比、拓扑、collective |
| 177 | Ring AllReduce 和 Tree AllReduce 各有什么特点？ | 带宽最优、延迟、规模 |
| 178 | NVLink、NVSwitch、InfiniBand 在集群中分别承担什么角色？ | 节点内和节点间互联 |
| 179 | GPU 任务调度系统需要考虑哪些资源？ | 显存、算力、互联、拓扑、MIG |
| 180 | GPU 虚拟化和容器化会带来哪些性能和隔离问题？ | device plugin、驱动共享、安全 |
| 181 | LLM 推理中 prefill 和 decode 阶段的瓶颈有什么不同？ | 计算密集、访存/带宽密集 |
| 182 | KV cache 为什么会成为 LLM 推理的重要显存开销？ | 序列长度、层数、batch |
| 183 | Continuous batching 解决了推理服务中的什么问题？ | 吞吐、延迟、动态请求 |
| 184 | Paged attention 的核心思想是什么？ | KV cache 管理、碎片减少 |
| 185 | Speculative decoding 如何提高推理吞吐？ | 草稿模型、验证、接受率 |
| 186 | FlashAttention 为什么比普通 attention 更省显存、更快？ | IO-aware、tiling、online softmax |
| 187 | Softmax 在 GPU 上优化时要注意哪些数值和性能问题？ | 最大值归约、指数、归一化 |
| 188 | LayerNorm/RMSNorm 为什么常被单独优化？ | 小规约、访存、融合 |
| 189 | Embedding lookup 在 GPU 上通常是什么瓶颈？ | 随机访存、缓存、带宽 |
| 190 | 稀疏计算为什么不一定比稠密计算快？ | 不规则访存、负载均衡、索引开销 |
| 191 | 图计算在 GPU 上的主要挑战是什么？ | 不规则性、分支、负载不均 |
| 192 | 数据库和数据分析系统如何利用 GPU 加速？ | 列式处理、过滤、聚合、join |
| 193 | GPU Direct Storage 或异步 IO 如何影响数据加载 pipeline？ | 数据通路、重叠、吞吐 |
| 194 | 如何为 GPU 服务设计端到端性能指标？ | P50/P99、吞吐、利用率、显存 |
| 195 | GPU 程序发生 OOM 时应如何系统排查？ | batch、碎片、缓存、生命周期 |
| 196 | 如何估算一个模型训练需要多少显存？ | 参数、梯度、优化器、激活、batch |
| 197 | 如何估算一个推理服务的最大 batch size？ | 显存、KV cache、延迟 SLA |
| 198 | GPU 安全和多租户环境中有哪些风险？ | 侧信道、数据残留、权限 |
| 199 | 设计一个 GPU 加速系统时，什么时候不应该使用 GPU？ | 数据规模、传输开销、开发成本 |
| 200 | 如果面试要求你从零优化一个 GPU kernel，你会按什么步骤展开？ | 正确性、测量、瓶颈、逐步优化、验证 |

## Recommended Practice Path

1. 先回答 Level 1-2：能画出 GPU 执行模型和 CUDA 内存层次，并手写 vector add、matrix transpose、reduction。
2. 再练 Level 3：每个经典 kernel 至少写 naive 和 optimized 两版，用 Nsight 或 `cudaEvent` 记录数据。
3. 然后进入 Level 4：理解 CUDA 到 PTX/cubin 的编译链，并能解释 PyTorch/Triton/CUTLASS/cuBLAS 的位置。
4. 最后学习 Level 5：关注多 GPU、LLM 推理、分布式通信和系统设计，把单 kernel 优化扩展到端到端性能。

