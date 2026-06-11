# CUDA/GPU 编程栈地图

> 目的：把 PTX、CUDA、CuTe DSL、Triton、JAX、Mojo 以及常见 GPU 框架放到同一张地图里，说明它们之间的层级关系，以及代码最终怎样在 GPU 上真正执行。

---

## 1. 先给结论

这些名字不是同一类东西。它们分布在从“应用框架”到“硬件指令”的不同层级：

```text
应用 / 模型 / 数组接口
PyTorch / TensorFlow / JAX / CuPy / RAPIDS / CUDA.jl
        |
图编译器 / 运行时 / 推理引擎
TorchDynamo + Inductor / XLA(OpenXLA) / TensorRT / ONNX Runtime / TVM
        |
Kernel DSL / Kernel 库 / 系统语言
CUDA C++ / Triton / CuTe DSL / cuTile / Mojo / Numba-CUDA / CUTLASS
        |
CUDA 库和工具链
cuBLAS / cuDNN / NCCL / CUB / Thrust / nvcc / NVRTC / ptxas
        |
中间表示和设备代码
HLO / StableHLO / MLIR / LLVM IR / NVVM IR / PTX / Tile IR / cubin
        |
NVIDIA Driver + Runtime
CUDA Runtime API / CUDA Driver API / JIT compiler / stream / context
        |
GPU 硬件
SM / Warp / Register / Shared Memory / HBM / Tensor Core / TMA
```

最关键的几句话：

- **CUDA** 是 NVIDIA GPU 的基础平台：包含编程模型、Runtime/Driver API、编译器、库、调试和性能分析工具。
- **PTX** 不是框架，也不是通常意义上的“最终机器码”；它是 NVIDIA GPU 的低层虚拟指令集/中间表示，通常还要被 `ptxas` 或驱动 JIT 编译成具体 GPU 架构的 native code。
- **Triton、CuTe DSL、cuTile、Mojo、Numba-CUDA** 更接近“写 kernel 的方式”，只是抽象层次和目标生态不同。
- **JAX、PyTorch、TensorFlow** 是更高层的数组/深度学习框架；它们一般不让你直接写 thread/block，而是通过图编译、库调用或自动生成 kernel 来用 GPU。
- **cuBLAS、cuDNN、NCCL、CUTLASS** 不是替代 CUDA 的东西，而是 CUDA 生态里的高性能库和 kernel 构建工具。

---

## 2. GPU 上“生效”的基本路径

无论你用 CUDA C++、Triton、JAX 还是 PyTorch，最后能让 GPU 工作，大体都绕不开这几个步骤：

1. **Host 端准备工作**
   - CPU 程序分配 GPU 显存，准备输入输出数据。
   - 可以显式 `cudaMalloc` / `cudaMemcpy`，也可以由框架管理，比如 PyTorch Tensor、JAX Array、CuPy ndarray。

2. **生成或选择 GPU kernel**
   - 直接写 CUDA C++ kernel：由 `nvcc` 编译。
   - 写 Triton/CuTe DSL/Numba/Mojo kernel：由各自 JIT/AOT 编译器生成设备代码。
   - 用 PyTorch/JAX/TensorFlow：框架可能调用 cuBLAS/cuDNN，也可能通过 XLA、Inductor、Triton 等生成 kernel。

3. **生成设备代码**
   - 常见中间层包括 MLIR、LLVM IR、NVVM IR、PTX。
   - NVIDIA GPU 最终执行的是面向具体 SM 架构的 native code，通常封装在 `cubin` 或 fatbin 中。

4. **Driver/Runtime 装载和启动**
   - CUDA Runtime API 或 Driver API 创建 context、管理 stream、装载 module、启动 kernel。
   - 如果程序里带的是 PTX，驱动可能在运行时 JIT 成当前 GPU 能执行的代码。
   - 如果程序里已经带有匹配架构的 cubin，驱动可以直接装载。

5. **GPU 硬件执行**
   - kernel 被切成 grid/block/thread。
   - block 被调度到 SM。
   - warp 通常以 32 个线程为基本调度单位执行。
   - 指令操作 register、shared memory、global memory；矩阵类指令可能使用 Tensor Core；新架构上的数据搬运可能用 TMA 等专用硬件。

---

## 3. 你提到的工具分别是什么

| 名称 | 它是什么 | 位于哪一层 | 怎么落到 GPU 上 | 适合解决什么问题 |
|------|----------|------------|------------------|------------------|
| **CUDA** | NVIDIA 的 GPU 计算平台、编程模型、API、工具链和库生态 | 基础平台 | CUDA C++/Python/库调用经过 CUDA Runtime/Driver 启动 kernel | 学 GPU 编程、写底层 kernel、调用 NVIDIA 高性能库 |
| **PTX** | Parallel Thread Execution，NVIDIA 的虚拟 ISA/低层 IR | 设备代码中间层 | `nvcc`、Triton、XLA 等可生成 PTX；再由 `ptxas` 或驱动 JIT 编译成具体 GPU 的 native code | 理解编译结果、做底层调优、检查指令和寄存器使用 |
| **CuTe DSL** | NVIDIA CUTLASS 生态里的 Python DSL，用于动态编译高性能 GPU kernel | Kernel DSL | Python 装饰器/DSL 表达 kernel，编译后通过 CUDA/NVIDIA 后端执行 | 想用 Python 写接近 CUTLASS/CuTe 风格的高性能 kernel |
| **Triton** | Python 风格的 GPU kernel DSL 和编译器 | Kernel DSL / 编译器 | Triton JIT 把 Python kernel 编译到 GPU 设备代码，NVIDIA 后端通常会经过 PTX/cubin 路径 | 深度学习自定义算子、融合算子、比 CUDA C++ 更快迭代 |
| **JAX** | Python 数组、自动微分、函数变换和 JIT 编译框架 | 高层数组/ML 框架 | Python 函数被 tracing 成 JAXPR/HLO/StableHLO，再交给 XLA/OpenXLA 编译或调用库，在 GPU 上执行 | 科研、自动微分、函数式数值计算、TPU/GPU/CPU 跨后端 |
| **Mojo** | Modular 推出的系统编程语言，面向 CPU/GPU/加速器 | 系统语言 / Kernel 编程 / MAX 生态 | Mojo GPU kernel 通过 `DeviceContext` 等接口编译并入队执行；在 NVIDIA 后端会使用 NVIDIA GPU 编译组件和驱动路径 | 探索 Python 风格但更接近系统语言的可移植 GPU 编程 |

### CUDA

CUDA 既可以指编程模型，也可以指工具包和运行时。写 CUDA C++ 时，通常是：

```text
.cu 源码
  -> nvcc
  -> host object + device fatbin
  -> fatbin 内含 PTX 和/或 cubin
  -> 程序运行时由 CUDA Runtime/Driver 选择或 JIT
  -> GPU 执行
```

CUDA 的核心抽象是 host/device、grid/block/thread、stream、显存、shared memory、同步、原子操作等。这个仓库前几阶段学习的内容就是 CUDA 编程模型本身。

### PTX

PTX 的位置容易被误解。它不是 Triton/JAX/CUDA 之上的框架，而是更底层的“虚拟 GPU 汇编/中间表示”。

```text
CUDA C++ / Triton / XLA / 其他编译器
        |
      PTX
        |
ptxas 或 CUDA Driver JIT
        |
SASS / cubin，即具体 GPU 架构上的 native code
```

为什么需要 PTX：

- 它比 CUDA C++ 更低层，能表达线程、寄存器、地址空间、内存访问、特殊指令等。
- 它比最终 native code 更抽象，给不同 GPU 架构留出重新编译空间。
- 发布程序时常见做法是 fatbin 里同时放某些架构的 cubin 和某个 compute capability 的 PTX，兼顾性能和前向兼容。

要注意：性能分析时，PTX 不是最终答案。真正跑在 SM 上的是对应架构的机器指令，通常要看 SASS、寄存器数、occupancy、访存行为和 Nsight Compute 指标。

### CuTe DSL、CUTLASS、CuTe C++

这几个名字也容易混在一起：

- **CUTLASS**：NVIDIA 的 CUDA C++ 模板库，主要用于构建高性能 GEMM、卷积和相关张量算子。
- **CuTe C++**：CUTLASS 3.x 以来的核心抽象之一，用 layout、tensor、tiling 来表达线程和数据布局。
- **CuTe DSL**：Python 版 DSL，继承 CuTe/CUTLASS 的思想，用 Python 装饰器和类型系统动态编译 GPU kernel。

CuTe 系列更偏“专家型 kernel 构建”。它适合认真控制数据布局、tiling、memory movement、Tensor Core 使用的场景；学习成本通常高于 Triton，但能更贴近 NVIDIA 官方库的底层组织方式。

### Triton

Triton 的定位是：用 Python 写 GPU kernel，但抽象粒度通常不是“一个 CUDA thread 做什么”，而是“一个 program/block 处理一块 tile”。

典型路径：

```text
@triton.jit Python kernel
  -> Triton 编译器 IR
  -> MLIR/LLVM/NVPTX 等后端路径
  -> PTX/cubin
  -> CUDA Driver 装载并启动
```

Triton 常见于：

- PyTorch 2.x `torch.compile`/Inductor 生成 kernel。
- 自写 fused activation、layer norm、attention、matmul 变体。
- 快速试验内存访问模式和 block/tile 大小。

Triton 的优势是迭代快、Python 生态友好、融合算子方便。它的限制是：不是所有硬件细节都能像 CUDA C++/CUTLASS 那样精细控制；复杂同步、特殊指令、非常规控制流时可能要回到 CUDA/CUTLASS。

### JAX

JAX 更高层。你通常写的是数组计算：

```python
@jax.jit
def f(x, y):
    return jnp.sin(x) + y * y
```

它不是把每一行 Python 都当成 GPU 线程执行。JAX 会追踪函数，形成中间表示，然后交给 XLA/OpenXLA 优化：

```text
Python + jax.numpy
  -> tracing
  -> JAXPR / StableHLO / HLO
  -> XLA/OpenXLA 优化：fusion、layout、memory planning
  -> 调用 cuBLAS/cuDNN 或生成 GPU kernel
  -> CUDA Driver/Runtime 执行
```

所以 JAX 适合“用数学表达式描述计算，让编译器做融合和后端生成”。如果要手写某个 warp 级算法，JAX 本身不是最直接的工具。

### Mojo

Mojo 不是 CUDA 的一个库，而是一门语言和 Modular 平台的一部分。它的目标是用接近 Python 的语法写高性能 CPU/GPU 代码，并支持多种硬件后端。

在 GPU 上，Mojo 暴露了 kernel、thread block、grid、device memory、异步执行等概念。公开文档中，Mojo 的 `DeviceContext` 负责编译并 enqueue GPU kernel。根据 Modular 文档，NVIDIA 后端涉及 PTX 编译组件和驱动兼容性；因此可以把它理解为：Mojo 在更高层提供语言和编译器，最终仍需要落到目标厂商的 GPU 编译与驱动执行路径。

Mojo 目前更适合关注 Modular/MAX 生态、可移植 GPU kernel、或者希望用系统语言方式写高性能代码的场景。它和 CUDA C++、Triton、CuTe DSL 有重叠，但生态成熟度和主流部署程度仍需要结合具体项目评估。

---

## 4. 还应该知道的常用 GPU 框架和工具

### 4.1 CUDA 官方核心库

这些通常是生产环境优先选择，而不是自己从头写 kernel：

| 工具/库 | 用途 | 和 CUDA 的关系 |
|---------|------|----------------|
| **cuBLAS / cuBLASLt** | BLAS、GEMM、batched GEMM、低精度矩阵乘 | 构建在 CUDA Runtime 之上，通常是矩阵乘首选 |
| **cuDNN** | 深度学习基础算子：卷积、attention、matmul、pooling、normalization 等 | PyTorch/TensorFlow/JAX 等经常调用 |
| **NCCL** | 多 GPU、多节点通信：all-reduce、broadcast、reduce-scatter 等 | 分布式训练/推理核心库 |
| **cuFFT** | FFT | 科学计算、信号处理常用 |
| **cuSPARSE** | 稀疏矩阵运算 | 稀疏线性代数 |
| **cuRAND** | GPU 随机数 | Monte Carlo、采样 |
| **Thrust** | 类 STL 的并行算法 | sort/reduce/scan 快速上手 |
| **CUB** | block/warp/device 级并行原语 | Thrust 底层常用组件，更灵活 |
| **CUTLASS** | GEMM/卷积等高性能 kernel 模板库 | 自定义高性能矩阵类 kernel 的重要基础 |

实践建议：如果你的需求能被 cuBLAS/cuDNN/NCCL 覆盖，通常先用库。只有库覆盖不了、融合机会明显、或需要特殊数据布局时，再写自定义 kernel。

### 4.2 新的 NVIDIA tile 路线：cuTile / CUDA Tile

除了 CUDA 的传统 SIMT 模型，NVIDIA 在 CUDA 13.x 文档中加入了 **CUDA Tile** 和 **cuTile Python**。它的思想是让程序员以 tile 为单位表达计算，把更底层的线程映射、Tensor Core、TMA 等交给编译器和 tile IR。

它和 CuTe DSL/Triton 有相似目标：降低高性能 kernel 的编写成本。但区别是：

- **cuTile** 是 NVIDIA CUDA 平台内的新 tile 编程模型，绑定 NVIDIA 生态更深。
- **Triton** 是独立 DSL/编译器，深度进入 PyTorch 生态。
- **CuTe DSL** 来自 CUTLASS/CuTe 思路，更偏 NVIDIA 官方高性能 kernel 构建体系。

### 4.3 深度学习框架

| 框架 | GPU 上怎么执行 | 典型场景 |
|------|----------------|----------|
| **PyTorch** | eager 模式调用 CUDA 库和自定义 kernel；`torch.compile` 可通过 Dynamo/Inductor/Triton 生成或融合 kernel | 最主流训练/推理研发框架 |
| **TensorFlow** | Graph/eager 执行，常调用 cuDNN/cuBLAS，也可用 XLA 编译 | 生产训练、Serving、移动端生态 |
| **JAX** | 通过 XLA/OpenXLA JIT 编译数组函数 | 科研、自动微分、函数式编程、TPU/GPU |
| **ONNX Runtime** | 通过 CUDA Execution Provider、TensorRT EP 等执行 ONNX 模型 | 跨框架推理部署 |
| **TensorRT / TensorRT-LLM** | 构建优化后的 inference engine，做 fusion、量化、kernel tuning | NVIDIA GPU 上高性能推理 |

### 4.4 Python 数组和数据科学

| 工具 | 作用 |
|------|------|
| **CuPy** | NumPy/SciPy 风格的 GPU 数组库，底层调用 CUDA 库，也支持自定义 kernel |
| **Numba-CUDA** | 用受限 Python 子集写 CUDA kernel，JIT 编译到 NVIDIA GPU |
| **RAPIDS** | GPU 数据科学套件，常见组件包括 cuDF、cuML、cuGraph，API 接近 pandas/scikit-learn/networkx |

这些工具的价值是让数据科学代码更少接触 CUDA C++，但底层仍通过 CUDA kernel 和 CUDA 库生效。

### 4.5 编译器和跨平台 kernel 框架

| 工具 | 定位 |
|------|------|
| **XLA / OpenXLA** | ML 编译器，服务 JAX、TensorFlow，也可接其他前端；优化 HLO/StableHLO 后生成目标设备代码 |
| **MLIR / LLVM** | 编译器基础设施；很多 GPU 编译器会把高层 IR 逐步 lower 到 LLVM/NVVM/PTX 或其他后端 |
| **TVM** | 机器学习编译框架，可从模型/算子描述生成不同硬件后端代码 |
| **Halide** | 图像和数组计算 DSL，把 algorithm 和 schedule 分离，可 target CUDA/OpenCL/Metal 等 |
| **Kokkos / RAJA** | HPC C++ 性能可移植框架，用同一套抽象 target CUDA/HIP/SYCL/OpenMP 等 |
| **OpenMP target / OpenACC** | 指令式 offload，适合把已有 C/C++/Fortran 程序逐步迁移到 GPU |
| **HIP / ROCm** | AMD GPU 生态，HIP API 很像 CUDA，也可以用于 CUDA/HIP 双后端代码 |
| **SYCL / oneAPI** | C++ 异构编程标准/生态，目标是跨 Intel/AMD/NVIDIA/CPU 等后端 |
| **OpenCL** | 更老牌的跨厂商异构计算标准，抽象更底层，现代 ML 生态里热度低于 CUDA/Triton/XLA |
| **CUDA.jl** | Julia 的 NVIDIA GPU 编程入口，既有数组抽象，也能写底层 CUDA kernel |

---

## 5. 几条典型执行链路

### 5.1 直接写 CUDA C++

```text
my_kernel.cu
  -> nvcc
  -> host code + fatbin(PTX/cubin)
  -> ./a.out
  -> cudaMalloc/cudaMemcpy/kernel<<<grid, block, shared, stream>>>
  -> CUDA Runtime
  -> CUDA Driver
  -> GPU
```

这是学习 GPU 执行模型最清楚的路径。

### 5.2 PyTorch 调用库

```text
torch.matmul(a, b)
  -> PyTorch dispatcher
  -> cuBLAS/cuBLASLt
  -> CUDA stream
  -> GPU GEMM kernel
```

大部分常规深度学习算子会尽量走成熟库。

### 5.3 PyTorch 生成 Triton kernel

```text
Python model
  -> torch.compile
  -> TorchDynamo 捕获图
  -> AOTAutograd/Inductor 优化
  -> Triton/C++ kernel codegen
  -> PTX/cubin
  -> CUDA Driver
  -> GPU
```

适合 elementwise fusion、reduction、部分 matmul/attention 变体。

### 5.4 JAX JIT

```text
@jax.jit function
  -> tracing
  -> JAXPR
  -> StableHLO/HLO
  -> XLA/OpenXLA
  -> library call 或 generated GPU kernel
  -> CUDA Driver
  -> GPU
```

JAX 的核心收益来自函数级优化、融合、自动微分和跨设备后端。

### 5.5 Triton 自定义算子

```text
@triton.jit kernel
  -> Triton JIT compile
  -> Triton IR / MLIR / LLVM path
  -> PTX/cubin
  -> launch on CUDA stream
  -> GPU
```

适合快速写自定义 GPU kernel，尤其是深度学习中的融合算子。

### 5.6 TensorRT 推理

```text
PyTorch/TensorFlow/ONNX model
  -> TensorRT build engine
  -> graph optimization / precision selection / tactic selection
  -> serialized engine
  -> enqueue inference on CUDA stream
  -> cuDNN/cuBLAS/custom TensorRT kernels
  -> GPU
```

适合部署阶段追求吞吐和低延迟。

---

## 6. 怎么选择

| 目标 | 优先选择 |
|------|----------|
| 学懂 GPU 执行模型 | CUDA C++、Nsight Compute、PTX/SASS 辅助理解 |
| 做常规矩阵乘、卷积、归约、通信 | cuBLAS/cuDNN/NCCL/CUB/Thrust |
| PyTorch 模型训练和研发 | PyTorch + CUDA 库；需要优化时看 `torch.compile`、Triton、自定义 CUDA extension |
| 科研型自动微分和函数式数值计算 | JAX + XLA/OpenXLA |
| 写深度学习自定义融合算子 | Triton 优先；极限优化再考虑 CUDA C++/CUTLASS/CuTe |
| 写 GEMM/attention/conv 等专家级 kernel | CUTLASS、CuTe C++、CuTe DSL、CUDA C++ |
| 部署 NVIDIA GPU 推理 | TensorRT、TensorRT-LLM、ONNX Runtime CUDA/TensorRT EP |
| Python 风格数组计算替代 NumPy | CuPy |
| Python 里直接写 CUDA kernel | Numba-CUDA 或 Triton |
| 数据科学 GPU 加速 | RAPIDS |
| 多硬件厂商可移植 HPC | Kokkos、RAJA、OpenMP target、SYCL、HIP |
| 探索新语言和可移植 GPU kernel | Mojo/MAX |

---

## 7. 学习顺序建议

如果目标是深入 CUDA/GPU 性能优化，可以按这个顺序：

1. **CUDA C++ 基础**
   - grid/block/thread、warp、memory hierarchy、stream、event。

2. **性能分析**
   - Nsight Systems 看时间线，Nsight Compute 看单 kernel 指标。
   - 理解 occupancy、memory throughput、bank conflict、coalescing、warp divergence。

3. **CUDA 库**
   - cuBLAS、cuDNN、NCCL、Thrust、CUB。
   - 明白什么时候不该自己写 kernel。

4. **自定义 kernel DSL**
   - Triton：快速写融合算子。
   - CUTLASS/CuTe：理解高性能 GEMM/attention 背后的 tiling 和数据搬运。

5. **编译器和框架层**
   - PyTorch compile、JAX/XLA、TensorRT、TVM、MLIR。
   - 重点理解“高层表达式如何变成 kernel”。

6. **低层代码**
   - 读 PTX/SASS、看寄存器数、反汇编 cubin。
   - 这一步用于解释性能瓶颈，不建议一开始就陷进去。

---

## 8. 常见误区

- **误区 1：PTX 就是 GPU 最终执行的汇编。**  
  更准确地说，PTX 是虚拟 ISA/中间表示；最终执行通常是具体 SM 架构的 native code。

- **误区 2：Triton/JAX/Mojo 替代了 CUDA。**  
  在 NVIDIA GPU 上，它们多数时候仍然依赖 CUDA driver、CUDA runtime、PTX/cubin 或 NVIDIA 库，只是隐藏了 CUDA C++ 细节。

- **误区 3：自己写 kernel 一定比库快。**  
  cuBLAS/cuDNN/NCCL/CUTLASS 里有大量架构特化和工程调优。自己写 kernel 的价值通常在特殊融合、特殊布局、特殊算子。

- **误区 4：高层框架“自动优化”就不用懂 CUDA。**  
  框架可以生成 kernel，但性能瓶颈仍然来自访存、同步、shape、layout、数据搬运和 kernel launch overhead。

- **误区 5：跨平台抽象没有代价。**  
  HIP、SYCL、Kokkos、Mojo 等能提升可移植性，但极限性能往往仍需要理解目标硬件后端。

---

## 9. 重点概念详解

这一节把下面这些概念单独展开：

```text
图/模型编译器：XLA(OpenXLA)、TVM
推理运行时：TensorRT、ONNX Runtime
Kernel DSL/模板库：Triton、CuTe DSL、cuTile、CUTLASS
高性能算子库：cuBLAS、cuDNN
编译器基础设施：MLIR
编译时机/机制：JIT compiler
```

先记一个判断方法：

- **输入是一个模型/计算图**：通常是 XLA、TVM、TensorRT、ONNX Runtime 这一类。
- **输入是一个 kernel 源码/DSL**：通常是 Triton、CuTe DSL、cuTile、CUTLASS 这一类。
- **输入是一次标准算子调用**：通常是 cuBLAS、cuDNN 这一类。
- **输入是 IR/dialect/pass**：通常在谈 MLIR、LLVM、编译器内部。
- **JIT** 不是某个固定软件，而是“运行时才编译”的机制，XLA、Triton、CuTe DSL、CUDA Driver 都可能有自己的 JIT。

### 9.1 总览表

| 概念 | 最准确定位 | 输入 | 输出/执行结果 | 主要用户 |
|------|------------|------|---------------|----------|
| **XLA / OpenXLA** | ML 图编译器生态 | TensorFlow/JAX/PyTorch 等框架产生的计算图、HLO/StableHLO | 面向 CPU/GPU/加速器的可执行程序或库调用 | JAX/TF/PyTorch-XLA 用户、ML 编译器开发者 |
| **TensorRT** | NVIDIA GPU 推理优化器 + 推理 runtime | 训练好的模型，常见入口是 ONNX/PyTorch/TensorFlow | TensorRT engine，运行时 enqueue 推理 | NVIDIA GPU 推理部署 |
| **ONNX Runtime** | 跨平台模型 runtime | ONNX 模型 | 按 Execution Provider 分配后的执行计划 | 跨框架推理部署 |
| **TVM** | 开源 ML 编译框架 | 模型 IR、算子描述、TensorIR/Relax/Relay 等 | 可部署 runtime module 和目标代码 | 编译器研究、自定义硬件、跨后端部署 |
| **Triton** | Python GPU kernel DSL + 编译器 | `@triton.jit` kernel | GPU kernel，NVIDIA 后端通常落到 PTX/cubin | 深度学习自定义/融合 kernel |
| **CuTe DSL** | CUTLASS/CuTe 思路的 Python DSL | `@kernel` / `@jit` DSL 代码 | 高性能 GPU kernel | 想用 Python 写接近 CUTLASS 风格 kernel 的开发者 |
| **cuTile** | NVIDIA tile 级 GPU 编程模型和 Python DSL | `@ct.kernel` tile kernel | 由 CUDA tile 编译路径生成的 GPU kernel | 想以 tile 为单位表达 GPU 计算的开发者 |
| **CUTLASS** | CUDA C++ 模板库和 Python DSL 集合 | C++ 模板参数、CuTe layout、kernel 配置 | GEMM/conv/attention 类 kernel | 高性能算子开发者 |
| **cuBLAS** | NVIDIA BLAS/GEMM 二进制库 | BLAS/GEMM API 调用 | 已优化的矩阵/向量运算 kernel | 需要矩阵乘和线代算子的所有人 |
| **cuDNN** | NVIDIA 深度学习算子库 | 卷积、matmul、attention、normalization 等图/算子描述 | 已优化的 DL kernel 或 fused kernel | 深度学习框架和推理/训练系统 |
| **MLIR** | 多层 IR 和编译器基础设施 | dialect、operation、pass、lowering pipeline | 更低层 IR、LLVM/NVVM、目标代码路径 | 编译器开发者 |
| **JIT compiler** | 运行时编译机制 | 源码/IR/PTX/shape-specialized graph | 当前机器/当前 shape 的可执行代码 | 框架、DSL、driver、runtime |

---

### 9.2 XLA / OpenXLA

**一句话**：XLA 是面向机器学习计算图的编译器；OpenXLA 是围绕 XLA、StableHLO、PJRT 等组件形成的开放编译生态。

它通常不直接面对 CUDA 线程，而是面对高层张量计算：

```text
JAX / TensorFlow / PyTorch-XLA
  -> tracing / graph capture
  -> HLO 或 StableHLO
  -> XLA 优化：fusion、layout、buffer assignment、collective、partitioning
  -> GPU backend
  -> 调用 cuBLAS/cuDNN 或生成 GPU kernel
  -> CUDA Driver
  -> GPU
```

几个关键点：

- **HLO**：High Level Optimizer/Operations，XLA 内部的高层张量 IR。
- **StableHLO**：更稳定、可序列化的 HLO 操作集，目标是作为 ML 框架和 ML 编译器之间的可移植边界。
- **PJRT**：OpenXLA 生态里的统一设备 API，用来让 JAX/TF/PyTorch-XLA 这类框架接入不同设备后端。
- **GPU 后端**：XLA 可以为 GPU 生成代码，也可以选择调用高度优化的库，比如 GEMM 走 cuBLAS/cuBLASLt，卷积/部分深度学习模式走 cuDNN。

适合场景：

- JAX 的 `jax.jit`、`pmap`、`pjit` 等。
- TensorFlow 的 XLA 编译。
- 需要把高层张量图编译到多种硬件。
- 希望由编译器自动做图级融合和内存规划。

不适合场景：

- 手写某个 warp 内算法。
- 直接控制 shared memory bank、warp shuffle、TMA 指令等底层细节。
- 只想部署一个 ONNX 模型并快速跑推理，这时 TensorRT/ONNX Runtime 更直接。

**和 CUDA 的关系**：XLA 在 NVIDIA GPU 上最终仍需要 CUDA 驱动、GPU 后端代码生成和/或 CUDA 库。它隐藏的是 CUDA kernel 编写细节，不是绕开 GPU 执行模型。

---

### 9.3 TensorRT

**一句话**：TensorRT 是 NVIDIA 面向推理部署的优化器和运行时，核心目标是让训练好的模型在 NVIDIA GPU 上低延迟、高吞吐地执行。

典型路径：

```text
PyTorch / TensorFlow / ONNX model
  -> TensorRT builder
  -> network definition
  -> graph optimization
  -> precision selection: FP32 / TF32 / FP16 / BF16 / FP8 / INT8
  -> tactic selection
  -> TensorRT engine
  -> TensorRT runtime enqueue
  -> CUDA stream 上执行 kernel
```

TensorRT 重点做的是：

- **图优化**：删除无用节点、常量折叠、层融合。
- **精度选择**：FP16、BF16、FP8、INT8 等，取决于 GPU、模型和校准/量化配置。
- **tactic selection**：同一个算子可能有多种实现，TensorRT 会根据 shape、dtype、workspace、硬件选择实现。
- **动态 shape 支持**：通过 optimization profile 描述 shape 范围。
- **plugin 机制**：模型里有 TensorRT 不支持的自定义算子时，可以写 plugin。
- **engine 序列化**：build 好的 engine 可保存，部署时加载，减少启动成本。

TensorRT 的边界：

- 它主要服务**推理**，不是通用训练框架。
- 它不是 CUDA C++ 替代品；底层仍是 CUDA kernel、CUDA stream、显存和 NVIDIA 库。
- 它通常对 NVIDIA GPU 最有价值，跨厂商部署不是它的目标。

**和 ONNX Runtime 的关系**：ONNX Runtime 可以通过 TensorRT Execution Provider 调用 TensorRT。也就是说，ONNX Runtime 是模型 runtime 的外壳，TensorRT 可以作为其中一个加速后端。

---

### 9.4 ONNX Runtime

**一句话**：ONNX Runtime 是运行 ONNX 模型的跨平台 runtime，核心机制是把 ONNX 图分配给不同 Execution Provider 执行。

典型路径：

```text
ONNX model
  -> ONNX Runtime InferenceSession
  -> graph optimization
  -> Execution Provider capability query
  -> 子图分配给 CUDA EP / TensorRT EP / CPU EP / 其他 EP
  -> 每个 EP 调用自己的 kernel 或加速库
  -> 汇总输出
```

核心概念：

- **ONNX**：模型交换格式，不是 runtime。
- **ONNX Runtime**：加载、优化、调度 ONNX 模型的 runtime。
- **Execution Provider, EP**：硬件/库后端适配层，例如 CPU EP、CUDA EP、TensorRT EP、OpenVINO EP、CoreML EP、DirectML EP 等。
- **CUDA EP**：用 CUDA/cuDNN/cuBLAS 等在 NVIDIA GPU 上执行支持的节点。
- **TensorRT EP**：把支持的子图交给 TensorRT 构建 engine，通常比通用 CUDA EP 更激进地优化推理。

使用 ONNX Runtime 时要特别注意：

- 如果某些节点当前 EP 不支持，可能会 fallback 到 CPU 或另一个 EP；这会导致额外数据搬运和性能下降。
- TensorRT EP 常常需要显式 shape range、engine cache、timing cache 等配置。
- ONNX Runtime 负责“图怎么分给后端”，但每个后端的真实性能由对应 EP、CUDA 库和硬件决定。

适合场景：

- 你有 ONNX 模型，想在不同硬件上部署。
- 你希望同一个应用可以切换 CPU/CUDA/TensorRT/OpenVINO/CoreML 等后端。
- 你不想直接写 TensorRT API，但想用 TensorRT 加速一部分模型。

---

### 9.5 TVM

**一句话**：TVM 是开源机器学习编译框架，目标是把模型和算子编译成能在不同硬件上部署的高性能代码。

典型路径可以理解为：

```text
模型 / 算子描述
  -> TVM 前端导入
  -> 高层 IR：Relax / Relay 等
  -> 算子级 IR：TensorIR / TIR
  -> schedule / transform / autotuning
  -> 目标后端代码：CUDA / LLVM / Metal / OpenCL / Vulkan / custom accelerator
  -> TVM runtime module
```

TVM 的关键词：

- **Python-first compiler API**：用 Python 组合和定制编译 pipeline。
- **Universal deployment**：生成小型可部署模块，面向数据中心、边缘设备和自定义硬件。
- **TensorIR/TIR**：描述底层循环、内存、线程绑定、向量化等算子实现。
- **Schedule**：把“计算定义”和“如何在硬件上执行”分开，例如 tiling、thread binding、cache read/write、vectorize、unroll。
- **Autotuning / MetaSchedule**：自动搜索不同 schedule，在具体硬件上找更快实现。

TVM 和 XLA 的差异：

- XLA 更常作为 JAX/TF/PyTorch-XLA 背后的生产编译器。
- TVM 更像一个可编程 ML 编译器工具箱，适合研究、自定义硬件、新后端、深度改造编译 pipeline。

TVM 和 TensorRT 的差异：

- TensorRT 是 NVIDIA GPU 推理部署优化器，用户更多是部署工程。
- TVM 是更通用的编译框架，能 target 多种硬件，也更适合做编译器实验。

---

### 9.6 Triton

**一句话**：Triton 是 Python 风格的 GPU kernel DSL 和编译器，常用于写深度学习里的自定义融合算子。

Triton 的编程模型不是“一个 Python 函数对应一个 CUDA thread”，而是更接近：

```text
一个 Triton program instance 处理一块 tile
许多 program instance 组成 grid
每个 program 内用向量化张量表达 tile 内计算
```

典型路径：

```text
@triton.jit kernel
  -> Triton AST/IR
  -> Triton 编译 pipeline
  -> LLVM / GPU backend
  -> PTX/cubin 或其他目标代码
  -> CUDA stream launch
  -> GPU
```

Triton 擅长：

- elementwise + reduction 融合。
- layer norm、softmax、embedding、indexing、masking。
- attention、matmul 变体的快速实验。
- PyTorch 2.x `torch.compile` / Inductor 生成 kernel 的后端之一。

Triton 不一定适合：

- 需要特别细的 warp 级同步和复杂共享内存协议。
- 需要直接使用某些最新硬件特性但 Triton 还没暴露。
- 已经能被 cuBLAS/cuDNN 完美覆盖的常规 GEMM/conv。

Triton 和 CUDA C++ 的关系：

- CUDA C++ 以 thread/block 为主，控制力更底层。
- Triton 以 tile/program 为主，开发效率更高。
- 最终在 NVIDIA GPU 上，Triton 仍要生成设备代码并通过 CUDA 驱动执行。

---

### 9.7 CuTe DSL

**一句话**：CuTe DSL 是 NVIDIA CUTLASS 生态里的 Python DSL，用 Python 写接近 CuTe/CUTLASS 风格的高性能 GPU kernel。

它的来源关系：

```text
CUTLASS 3.x
  -> 大量采用 CuTe C++ 作为 layout/tensor/tiling 抽象
  -> CuTe DSL 把这些思想带到 Python decorator DSL
```

典型路径：

```text
Python CuTe DSL
  -> @jit host function 或 @kernel GPU function
  -> dynamic/JIT compilation
  -> CUDA/NVIDIA GPU 后端
  -> GPU kernel 执行
```

CuTe DSL 的重点不是“隐藏硬件”，而是用更结构化的方式表达硬件相关设计：

- layout：多维数据布局和索引映射。
- tensor：把数据和 layout 组合起来。
- tiling/partitioning：描述线程块、warp、MMA tile 如何覆盖数据。
- pipeline：描述 global memory、shared memory、寄存器之间的数据流。
- Tensor Core：面向矩阵乘加的高性能路径。

它和 Triton 的区别：

| 对比点 | Triton | CuTe DSL |
|--------|--------|----------|
| 风格 | Python tile/program DSL | Python 版 CuTe/CUTLASS 思想 |
| 抽象目标 | 快速写高效 kernel | 更贴近 NVIDIA 官方 GEMM/kernel 构建体系 |
| 学习曲线 | 相对低 | 通常更高 |
| 控制粒度 | 比高层框架低，比 CUDA C++高 | 接近 CUTLASS/CuTe 的专家式控制 |
| 常见用户 | PyTorch 自定义算子、融合算子开发者 | GEMM/attention/conv 等高性能 kernel 工程师 |

如果你目标是先写出可用的深度学习融合 kernel，Triton 往往更快；如果目标是研究 CUTLASS 级别的 layout、MMA 和 pipeline，CuTe DSL 更贴近 NVIDIA 的专家路径。

---

### 9.8 cuTile / CUDA Tile

**一句话**：cuTile 是 NVIDIA 的 tile 级 GPU 并行编程模型和 Python DSL，用 tile 表达计算，让编译器自动利用 Tensor Core、TMA 等硬件能力。

一个直观对比：

```text
CUDA C++：我告诉 GPU 每个 thread 做什么
Triton：我告诉 GPU 每个 program/tile 做什么
cuTile：我用 tile 值和 tile 操作表达计算，更多映射细节交给 CUDA tile 编译路径
```

cuTile 文档中的几个核心对象：

- **kernel**：通过 `@ct.kernel` 标记的 GPU 程序入口。
- **launch**：host 端用 `ct.launch()` 把 kernel 入队到 GPU。
- **array**：位于 global memory 的真实数组，有物理布局，可读写。
- **tile**：kernel 内的不可变 tile 值，没有固定存储位置；tile 维度通常需要是编译期常量。
- **tile operation**：tile 上的 elementwise、matmul、reduction、shape manipulation 等。

典型路径：

```text
cuTile Python kernel
  -> CUDA tile 编程模型
  -> Tile IR / CUDA 编译路径
  -> 目标 NVIDIA GPU 的设备代码
  -> CUDA Driver 执行
```

它和 CuTe DSL/Triton 的关系：

- 三者都想让你以更高层的 tile 思维写 GPU kernel。
- Triton 是独立且已经深度进入 PyTorch 编译生态的 DSL。
- CuTe DSL 更像 CUTLASS/CuTe 在 Python 中的延伸。
- cuTile 是 NVIDIA CUDA 平台里新的 tile 编程路线，目标是让新硬件能力在不同 NVIDIA 架构上更自动地暴露。

目前学习时可以把 cuTile 放在“值得关注的新路线”。如果要做当前生产项目，仍需要结合 API 成熟度、文档、示例、框架集成情况评估。

---

### 9.9 CUTLASS

**一句话**：CUTLASS 是 NVIDIA 的 CUDA C++ 模板库，用来构建高性能 GEMM、卷积和相关张量运算 kernel。

CUTLASS 的位置介于“手写 CUDA kernel”和“调用 cuBLAS/cuDNN”之间：

```text
手写 CUDA C++ kernel
  -> 控制力强，但工程量大

CUTLASS
  -> 提供 GEMM/conv/epilogue/pipeline/MMA 的模板组件
  -> 你组合和特化模板
  -> 生成自己的 kernel

cuBLAS/cuDNN
  -> 直接调用封装好的二进制库
  -> 最省事，但自定义空间较小
```

CUTLASS 的核心价值：

- 把 GEMM 拆成 threadblock、warp、instruction tile 多层结构。
- 封装 Tensor Core MMA、shared memory layout、pipeline、epilogue 等复杂细节。
- 允许开发者为特殊 dtype、layout、epilogue fusion、problem shape 构建定制 kernel。
- CUTLASS 3.x 以后大量采用 CuTe 抽象，使 layout 和 tiling 表达更组合化。

什么时候用 CUTLASS：

- cuBLAS/cuDNN 无法表达你的 epilogue 或特殊 layout。
- 你需要把 GEMM 和 bias、activation、quant/dequant、scale、mask 等融合。
- 你要学习高性能 GEMM 的分层组织。
- 你愿意承担较高模板复杂度。

什么时候不用：

- 标准矩阵乘直接用 cuBLAS/cuBLASLt。
- 标准卷积/attention 优先试 cuDNN/TensorRT。
- 快速做原型时，Triton 可能更快。

---

### 9.10 cuBLAS / cuDNN

**一句话**：cuBLAS 和 cuDNN 是 NVIDIA GPU 上最常用的高性能二进制算子库；它们不是编程语言，也不是模型框架，而是大量调优后的 kernel 和调度逻辑。

#### cuBLAS

cuBLAS 是 CUDA 上的 BLAS 实现，主要面向：

- GEMM：矩阵乘。
- GEMV：矩阵向量乘。
- AXPY、DOT、NORM 等基础线性代数。
- batched GEMM、strided batched GEMM。
- cuBLASLt：更灵活的 matmul API，支持更多 layout、低精度、epilogue 和算法选择。
- cuBLASXt：面向多 GPU/CPU+GPU 的扩展接口。

典型调用链：

```text
torch.matmul / jnp.dot / 自己的 C++ 程序
  -> cuBLAS 或 cuBLASLt
  -> 内部选择高度优化 kernel
  -> CUDA stream 上执行
```

实践原则：

- 标准 GEMM 几乎总是先试 cuBLAS/cuBLASLt。
- 自己写 GEMM 的主要价值是学习、融合特殊逻辑、处理特殊 layout，或研究极端场景。

#### cuDNN

cuDNN 是深度学习算子库，常见覆盖：

- convolution forward/backward。
- pooling、normalization、activation。
- matmul、attention、RNN 相关能力。
- fused op 和 graph API。
- runtime fusion、heuristic selection、specialized patterns，例如 fused attention。

现代 cuDNN 越来越多地通过 **Graph API / Frontend API** 表达计算：

```text
framework graph pattern
  -> cuDNN frontend graph
  -> heuristics 选择 engine/config
  -> 执行 conv/matmul/attention/fusion kernel
```

cuDNN 和 TensorRT 的关系：

- cuDNN 是算子库，训练和推理都会用。
- TensorRT 是推理优化器/runtime，可能在 engine 中使用库、专用 kernel 或 plugin。
- PyTorch/JAX/TensorFlow 等框架经常直接调用 cuDNN。

---

### 9.11 MLIR

**一句话**：MLIR 是“多层中间表示”编译器基础设施，用来构建可扩展的编译器，不是一个直接拿来跑模型的 runtime。

传统编译器里常见路径是：

```text
源码
  -> AST
  -> LLVM IR
  -> 机器码
```

但机器学习和异构硬件需要表达更多层级：

```text
张量图
  -> 高层 tensor ops
  -> linalg / affine / scf
  -> gpu / nvgpu
  -> nvvm / llvm
  -> PTX / object code
```

MLIR 的核心机制：

- **Dialect**：一组 operation/type/attribute，例如 `stablehlo`、`linalg`、`gpu`、`nvgpu`、`nvvm`。
- **Operation**：IR 里的基本节点，可以很高层，也可以很低层。
- **Pass**：对 IR 做变换或优化。
- **Lowering**：把高层 dialect 逐步转换成低层 dialect。
- **Extensibility**：可以为新硬件、新语言、新领域定义自己的 dialect。

它和这些工具的关系：

- StableHLO 基于 MLIR 生态，用作 ML 框架和编译器之间的稳定操作集。
- Triton 编译链路中有 Triton/MLIR 相关 IR 和 lowering。
- XLA/OpenXLA、IREE、Torch-MLIR、部分 GPU 编译器都会使用或对接 MLIR 思想/组件。
- MLIR 本身不负责“跑 GPU”，它提供的是表达和变换计算的基础设施。

一句实用判断：当你在应用层写模型时，通常不会直接用 MLIR；当你在做编译器、DSL、新后端或调试 codegen 时，MLIR 才变成核心概念。

---

### 9.12 JIT Compiler

**一句话**：JIT compiler 指 Just-In-Time compiler，即运行时编译。它不是某一个具体软件，而是一种编译时机。

GPU 生态里至少有四类 JIT：

| JIT 类型 | 例子 | 编译输入 | 编译输出 | 触发时机 |
|----------|------|----------|----------|----------|
| **Framework graph JIT** | JAX `@jit`、TensorFlow XLA、部分 `torch.compile` 路径 | Python 函数捕获出的图/IR | 可执行图、库调用、GPU kernel | 第一次遇到某组 shape/dtype 时 |
| **Kernel DSL JIT** | Triton、CuTe DSL、Numba-CUDA | DSL/Python kernel | PTX/cubin/设备代码 | 第一次调用 kernel 或 shape/meta 参数变化时 |
| **CUDA Driver PTX JIT** | CUDA Driver | PTX | 当前 GPU 架构 native code | 程序加载 PTX 或第一次使用 kernel 时 |
| **Runtime specialization/build** | TensorRT engine build、cuDNN runtime fusion | 模型子图/算子图/shape/profile | engine、execution plan、fused kernel | build 阶段或首次运行前 |

JIT 的优点：

- 可以针对当前 GPU 架构生成代码。
- 可以针对 shape、dtype、stride、layout 做特化。
- 可以把多个算子融合，减少 kernel launch 和 HBM 读写。
- 可以延迟编译不一定会用到的路径。

JIT 的代价：

- 第一次运行慢，需要 warmup。
- 可能按 shape/dtype/meta 参数产生多份编译缓存。
- 线上服务要考虑 compile latency、cache 目录、driver/toolkit 版本、并发编译。
- 动态 shape 太多时，可能频繁重新编译或退化到泛化代码。

CUDA 层面的 JIT 要单独理解：

```text
fatbin 里有 sm_90 cubin，当前 GPU 是 H100(sm_90)
  -> 直接加载匹配 cubin

fatbin 里没有当前 GPU 的 cubin，但有 compute_xx PTX
  -> CUDA Driver JIT：PTX -> 当前 GPU native code
  -> 加载执行
```

这也是为什么发布 CUDA 程序时常见做法是同时带上多个 `sm_xx` cubin 和一个较高/合适版本的 PTX：前者保证已知架构启动快、性能可控，后者提供一定前向兼容。

---

### 9.13 它们之间最容易混淆的关系

#### XLA vs TVM

- 两者都是 ML 编译器。
- XLA 更常作为 JAX/TF/PyTorch-XLA 背后的 production compiler。
- TVM 更像可编程、可改造、可扩展到新后端的 ML compiler framework。

#### TensorRT vs ONNX Runtime

- TensorRT 是 NVIDIA 推理优化器和 runtime。
- ONNX Runtime 是 ONNX 模型 runtime。
- ONNX Runtime 可以用 TensorRT EP 调 TensorRT，也可以用 CUDA EP、CPU EP 等。

#### Triton vs NVIDIA Triton Inference Server

- **Triton language**：OpenAI 起源的 GPU kernel DSL/compiler。
- **NVIDIA Triton Inference Server**：NVIDIA 的模型 serving 系统。
- 两者名字相同，但不是同一个东西。这里讨论的是 Triton language。

#### Triton vs CuTe DSL vs cuTile

- 三者都偏 tile/kernel 编程。
- Triton 更偏 PyTorch 生态和快速写自定义 kernel。
- CuTe DSL 更贴近 CUTLASS/CuTe 的专家级 kernel 构建。
- cuTile 是 NVIDIA CUDA 平台的新 tile 编程模型，强调自动利用新硬件能力和跨 NVIDIA 架构可移植。

#### CUTLASS vs cuBLAS/cuDNN

- CUTLASS 是源码级模板库，你用它生成自己的 kernel。
- cuBLAS/cuDNN 是二进制库，你调用 API，库内部选择实现。
- 能用 cuBLAS/cuDNN 解决时，先用库；需要定制融合或特殊布局时，再考虑 CUTLASS/Triton/CuTe。

#### MLIR vs LLVM IR

- LLVM IR 更接近通用低层编译器 IR。
- MLIR 支持多个抽象层级和自定义 dialect，更适合连接张量图、循环变换、GPU lowering、硬件专用后端。

---

### 9.14 一条算子在不同工具里的可能路径

以 `C = A @ B + bias + gelu` 为例：

```text
PyTorch eager:
  torch.matmul -> cuBLAS/cuBLASLt
  bias/gelu -> 额外 CUDA kernel

PyTorch torch.compile:
  TorchDynamo 捕获图
  Inductor 可能用 Triton 生成融合 kernel

JAX:
  Python function -> JAXPR -> StableHLO/HLO -> XLA
  XLA 可能调用 cuBLASLt，也可能融合部分 elementwise

TensorRT:
  ONNX/PyTorch model -> TensorRT engine
  builder 选择 matmul tactic，并把 bias/gelu 尽量融合

ONNX Runtime:
  ONNX graph -> TensorRT EP 或 CUDA EP
  TensorRT EP 支持的子图进 TensorRT，不支持的节点交给 CUDA/CPU 等 EP

Triton:
  手写一个 fused matmul/activation kernel

CUTLASS/CuTe DSL:
  用 GEMM 模板或 CuTe layout/pipeline 写一个融合 epilogue 的 kernel

cuBLASLt:
  如果 epilogue 支持该模式，直接通过 matmul descriptor 调库
```

工程上最实际的选择顺序通常是：

1. 先看 cuBLAS/cuDNN/cuBLASLt/cuDNN Graph 能不能覆盖。
2. 推理部署优先试 TensorRT 或 ONNX Runtime + TensorRT EP。
3. PyTorch 自定义融合优先试 Triton。
4. 极限 GEMM/attention/conv 再看 CUTLASS、CuTe DSL、CUDA C++。
5. 做编译器或新硬件后端时看 XLA/OpenXLA、TVM、MLIR。

---

## 10. 参考资料

- NVIDIA CUDA Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/index.html>
- NVIDIA NVCC Compiler Driver: <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/>
- NVIDIA PTX ISA: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- NVIDIA CUDA Binary Utilities: <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>
- NVIDIA CUTLASS / CuTe DSL: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>
- NVIDIA CuTe DSL Introduction: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html>
- NVIDIA CUTLASS Overview: <https://docs.nvidia.com/cutlass/latest/overview.html>
- NVIDIA cuTile Python: <https://docs.nvidia.com/cuda/cutile-python/>
- NVIDIA CUDA Tile Programming Guide: <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html>
- Triton Documentation: <https://triton-lang.org/>
- JAX JIT Documentation: <https://docs.jax.dev/en/latest/jit-compilation.html>
- OpenXLA XLA Overview: <https://openxla.org/xla>
- OpenXLA StableHLO: <https://openxla.org/stablehlo>
- OpenXLA PJRT: <https://openxla.org/xla/pjrt>
- Mojo GPU Programming Fundamentals: <https://docs.modular.com/mojo/manual/gpu/fundamentals/>
- Mojo GPU Tutorial: <https://docs.modular.com/mojo/manual/gpu/intro-tutorial/>
- PyTorch `torch.compile`: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- NVIDIA cuBLAS: <https://docs.nvidia.com/cuda/cublas/>
- NVIDIA cuDNN: <https://developer.nvidia.com/cudnn>
- NVIDIA NCCL: <https://developer.nvidia.com/nccl>
- NVIDIA TensorRT Documentation: <https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html>
- ONNX Runtime CUDA Execution Provider: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>
- ONNX Runtime Execution Providers: <https://onnxruntime.ai/docs/execution-providers/>
- ONNX Runtime TensorRT Execution Provider: <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>
- CuPy Documentation: <https://docs.cupy.dev/>
- Numba-CUDA Documentation: <https://nvidia.github.io/numba-cuda/>
- RAPIDS: <https://rapids.ai/>
- Apache TVM: <https://tvm.apache.org/>
- Halide: <https://halide-lang.org/>
- MLIR GPU Dialect: <https://mlir.llvm.org/docs/Dialects/GPU/>
- MLIR Overview: <https://mlir.llvm.org/>
- NVIDIA PTX Compiler API: <https://docs.nvidia.com/cuda/ptx-compiler-api/index.html>
- AMD HIP Documentation: <https://rocm.docs.amd.com/projects/HIP/>
- Khronos OpenCL: <https://www.khronos.org/opencl/>
- CUDA.jl: <https://cuda.juliagpu.org/stable/>
- Kokkos Documentation: <https://kokkos.org/kokkos-core-wiki/>
