# GPU 与数据中心硬件术语速查

> 学习 CUDA、读 H100 架构图、看 DGX 规格表时，常会碰到 PCIe、OAM、NVMe 等**整机与互联**术语。本文按「和 GPU 算力关系远近」整理常用名词，并补充与编程/性能相关的要点。芯片内部结构（SM、HBM、NVLink 等）详见 [H100-Streaming-Multiprocessor-SM.md](H100-Streaming-Multiprocessor-SM.md)。

---

## 1. 总线、插槽与互联：数据怎么进出 GPU？

### 1.1 PCIe（Peripheral Component Interconnect Express）

**是什么**：主机（CPU/主板）与 GPU、网卡、NVMe 盘等扩展设备之间的**标准高速串行总线**。你在台式机或服务器里插的那张「显卡」，通常就是通过 **PCIe 插槽**连到 CPU 的。

| 要点 | 说明 |
|------|------|
| **代数** | PCIe 3.0 / 4.0 / 5.0 / 6.0，代数越高单 lane 带宽越高 |
| **lane 数** | 常见 `x16`（训练卡）、`x8`、`x4`；带宽 ≈ 单 lane 速率 × lane 数 |
| **与 CUDA 的关系** | 首次 `cudaMemcpy` H→D、kernel 启动命令、小数据频繁往返，都受 PCIe 带宽与延迟约束 |
| **典型瓶颈** | 单卡 H100 算力远高于 PCIe 5.0 x16 能喂饱的数据量时，容易出现 **PCIe-bound**（尤其小 batch、频繁同步） |

**和 NVLink 对比**：PCIe 走 **CPU 域**（经 Root Complex）；多卡 **GPU↔GPU** 大流量训练更常用 **NVLink / NVSwitch**（见下文），带宽远高于 PCIe。

```
CPU ←—— PCIe ——→ GPU（显存/HBM）
         ↑
    常见：x16 插槽、M.2 上的 NVMe 也走 PCIe
```

### 1.2 NVLink

**是什么**：NVIDIA 的 **GPU 间（或 GPU 与特定 CPU）高速直连**协议，带宽远高于同代 PCIe。H100 整卡底部可有十几路 NVLink，多卡服务器里常与 **NVSwitch** 配合。

| 场景 | 说明 |
|------|------|
| **多卡训练** | AllReduce、梯度同步走 NVLink/NVSwitch，避免所有流量挤 PCIe |
| **P2P** | `cudaDeviceEnablePeerAccess` 后，两卡显存可直连拷贝（拓扑允许时） |
| **与编程** | `nvidia-smi topo -m` 可看 GPU 之间是 NV#、PIX、PHB 等拓扑类型 |

### 1.3 NVSwitch

**是什么**：机内 **交换芯片**，把多张 GPU 的 NVLink 连成全互联或高连通拓扑（如 DGX 8 卡「每张卡都能高速访问其它卡」）。对大规模分布式训练的数据并行 / 张量并行很重要。

### 1.4 InfiniBand（IB）与 RoCE

**是什么**：**机间**高速网络，用于多节点训练（跨服务器 AllReduce）。常见搭配 **RDMA**（Remote Direct Memory Access），网卡可直接读写远端内存，降低 CPU 拷贝开销。

| 术语 | 含义 |
|------|------|
| **HCA** | Host Channel Adapter，InfiniBand 网卡 |
| **RDMA** | 远程直接内存访问，NCCL/MPI 多节点通信底层常用 |
| **RoCE** | 在以太网上跑 RDMA 的一类方案 |

单机多卡看 NVLink；**多机多卡**看 IB/RoCE + NCCL。

### 1.5 CXL（Compute Express Link）

**是什么**：基于 PCIe 物理层的**内存扩展与池化**协议（CXL.mem / CXL.cache 等）。和 CUDA 日常编程关系尚不如 PCIe/NVLink 直接，但在「CPU 旁路扩展内存、异构内存池」的整机方案里会出现。知道它是 **PCIe 生态下的内存/缓存一致性扩展**即可。

---

## 2. 加速卡形态：PCIe 卡、SXM、OAM

同一颗 GPU 芯片可以做成不同「封装/散热/供电」形态，影响功耗、带宽和是否适合密集集群。

### 2.1 PCIe 加速卡（Add-in Card）

**是什么**：插在主板 PCIe 槽里的 **独立显卡形态**（含消费级 GeForce、部分 Tesla/A 系列、H100 PCIe 版）。散热靠机箱风道或涡轮风扇，功耗常低于同芯片的 SXM/OAM 版本，**卡间互联**通常只有 PCIe + 少量 NVLink（视型号而定）。

### 2.2 SXM（Server PCI Express Module）

**是什么**：NVIDIA 服务器 GPU 的 **板载插座形态**（如 SXM2、SXM4、SXM5），芯片焊接在载板上，插在主板专用连接器上。特点：

- 更高 **TDP** 与供电能力（如 H100 SXM5 700W 级）
- 更多 **NVLink** 引脚，适合 4/8 卡 NVSwitch 机箱
- 常见于 **DGX**、HGX 等整机

「SXM」指的是**模块标准与插座**，不是另一种 GPU 架构名称。

### 2.3 OAM（OCP Accelerator Module）

**是什么**：**OCP（Open Compute Project）** 定义的开放加速器模块标准，全称 **OCP Accelerator Module**。把 GPU（或其它 AI 加速器）做成可热插拔、统一尺寸的模块，便于超大规模数据中心换型、运维和液冷。

| 对比 | SXM | OAM |
|------|-----|-----|
| **标准归属** | NVIDIA 生态为主 | OCP 开放标准，多厂商 |
| **典型产品** | H100 SXM5、HGX 基板 | H100 OAM（如 HGX H100 OAM）、部分 MI300 形态 |
| **散热** | 整机定制风冷/液冷 | 常设计为 **整柜液冷**、前后风道标准化 |
| **场景** | DGX、HGX 8-GPU | 云厂商大规模 AI 集群、开放硬件机柜 |

**记忆**：OAM 是「**开放标准的 AI 加速模块形态**」，不是某种显存或总线协议；和 PCIe、NVLink 正交——模块内部仍用 PCIe/NVLink 连 CPU 或其它 GPU。

### 2.4 HGX / DGX / Baseboard

| 术语 | 含义 |
|------|------|
| **HGX** | NVIDIA 的多 GPU **基板**参考设计（如 4 或 8 颗 SXM/OAM GPU + NVSwitch） |
| **DGX** | NVIDIA 整机 AI 服务器（如 DGX H100：8×GPU + NVSwitch + CPU + NVMe） |
| **Baseboard** | 承载多 GPU 与 NVSwitch 的载板，再装入机箱 |

---

## 3. 存储：NVMe、SSD 与训练 I/O

### 3.1 NVMe（Non-Volatile Memory Express）

**是什么**：面向 **SSD 等闪存** 的主机协议，跑在 **PCIe** 之上（物理上是 M.2 或 U.2 等接口）。名字里的「Non-Volatile」= 掉电不丢数据。

| 要点 | 说明 |
|------|------|
| **和 SATA SSD** | NVMe 延迟更低、队列更深、带宽更高（直连 PCIe） |
| **和 GPU** | 数据集从盘读到 **CPU 内存** 再 `cudaMemcpy` 到显存；大模型 checkpoint 加载常是 **存储 → DRAM → HBM** 链路 |
| **本地盘 vs 网络盘** | 训练节点常用本地 NVMe 做缓存；集群数据集可能在 **NFS / 对象存储 / 并行文件系统** |

### 3.2 相关术语

| 术语 | 含义 |
|------|------|
| **U.2 / M.2** | NVMe SSD 的常见物理接口形态 |
| **NFS / Lustre / GPFS** | 网络/并行文件系统，多机共享数据集 |
| **Checkpoint** | 训练中断点保存；写盘带宽与 NVMe 性能相关 |

**注意**：NVMe 是 **存储协议**，不是 GPU 显存；显存物理上多为 **HBM** 或 **GDDR**（见下节）。

---

## 4. 显存与内存层次（和 kernel 直接相关）

| 术语 | 英文 | 简要说明 |
|------|------|----------|
| **HBM** | High Bandwidth Memory | 3D 堆叠高带宽显存，H100/A100 等数据中心卡主力；CUDA **Global Memory** 的物理载体 |
| **GDDR** | Graphics DDR | 消费级显卡常用，带宽通常低于同代 HBM |
| **DRAM** | Dynamic RAM | 主机 **系统内存**（`malloc` / `new` 所在） |
| **SRAM** | Static RAM | 片上高速存储：SM 内 **Shared Memory、L1、寄存器** |
| **L2 Cache** | — | 全 GPU 共享二级缓存，在 SM 与 HBM 之间 |
| **ECC** | Error-Correcting Code | 显存/内存纠错，数据中心卡常开启，略占容量、保可靠性 |
| **UVM** | Unified Virtual Memory | `cudaMallocManaged` 统一虚拟地址，由驱动按需迁移页，易用但需关注访问模式 |
| **Pinned Memory** | Page-locked Host Memory | `cudaMallocHost` 锁页内存，**异步 H↔D 拷贝**（DMA）更快 |

访存优化见 [CoalesedMemoryAccess.md](CoalesedMemoryAccess.md)、[Bank_Conflict.md](Bank_Conflict.md)。

---

## 5. 芯片与算力相关术语

| 术语 | 含义 |
|------|------|
| **GPU / Die** | 单颗硅芯片；「满血 die」与「阉割 SKU」SM 数量可能不同 |
| **SM** | Streaming Multiprocessor，执行 CUDA 线程块的基本单元 |
| **GPC / TPC** | 芯片内物理分组（Graphics / Texture Processing Cluster） |
| **Tensor Core** | 矩阵乘加专用单元，FP16/BF16/FP8/INT8 等低精度高吞吐 |
| **CUDA Core** | 传统 FP32/INT32 标量/向量执行单元 |
| **Warp** | 32 线程为一组，调度与访存合并的基本单位 |
| **MIG** | Multi-Instance GPU，一张物理卡切成多个独立 GPU 实例（隔离租户） |
| **SKU** | 具体销售型号（如 H100 SXM5 80GB vs PCIe 80GB），SM 数、功耗、互联可能不同 |
| **TDP / TGP** | 热设计功耗 / 显卡总功耗上限，影响散热与供电 |
| **Architecture** | 架构代际：Volta、Ampere、Hopper（H100）、Blackwell（B200）等 |

---

## 6. CPU、主板与整机

| 术语 | 含义 |
|------|------|
| **CPU Socket** | CPU 与主板插座（如 Intel LGA、AMD SP5） |
| **NUMA** | Non-Uniform Memory Access；多路 CPU 时，访问「本地节点内存」比远端快。GPU 与哪个 NUMA 节点直连会影响 **PCIe 拷贝带宽** |
| **IOMMU** | 内存管理单元，虚拟化与 DMA 安全；GPU 直通（passthrough）场景会提到 |
| **BMC** | Baseboard Management Controller，带外管理（IPMI、远程开机、传感器） |
| **PSU** | Power Supply Unit，电源；8×700W GPU 整机对 PSU 要求极高 |
| **Liquid Cooling / CDU** | 液冷与冷却分配单元，高功耗 OAM/SXM 集群常见 |

**实践**：多路服务器上，尽量让 GPU 和发起 `cudaMemcpy` 的 CPU 内存在 **同一 NUMA 节点**（`numactl`、驱动拓扑工具可查）。

---

## 7. 软件栈里常见的「硬件味」缩写

| 术语 | 含义 |
|------|------|
| **CUDA** | NVIDIA 并行计算平台与 API |
| **Driver / Runtime** | 内核态驱动 + 用户态 CUDA Runtime（`libcudart`） |
| **NVML** | NVIDIA Management Library，`nvidia-smi` 等监控接口 |
| **NCCL** | 多 GPU / 多节点集合通信库（AllReduce 等） |
| **TCC vs WDDM** | Windows 上 Tesla/数据中心驱动模式（TCC）vs 显示驱动（WDDM）；Linux 服务器无此二分 |
| **MPS** | Multi-Process Service，多进程共享一卡 SM 资源 |
| **DCGM** | 数据中心 GPU 健康监控与诊断 |

---

## 8. 一张图串起来：训练节点里谁连谁

```
                    ┌──────── 其它服务器 ────────┐
                    │      InfiniBand / RoCE      │
                    └─────────────┬───────────────┘
                                  │
┌─────────────────────────────────┴─────────────────────────────────┐
│                         单机（如 DGX / HGX 节点）                    │
│  CPU ←—— PCIe ——→ GPU0 ←—— NVLink / NVSwitch ——→ GPU1 … GPU7      │
│   │                    ↑                                            │
│   │                    HBM（显存）                                     │
│   └── PCIe ——→ NVMe SSD（数据集 / checkpoint）                       │
│   └── DRAM（系统内存，DataLoader、Pinned Memory）                    │
└───────────────────────────────────────────────────────────────────┘
```

**数据路径小结**：

1. **训练算力**：主要在 **GPU SM + HBM** 上跑 kernel。  
2. **主机↔单卡**：**PCIe**（命令、中小规模 H↔D 拷贝）。  
3. **卡↔卡**：**NVLink + NVSwitch**（梯度同步）。  
4. **机↔机**：**InfiniBand/RoCE + NCCL**。  
5. **磁盘↔内存**：**NVMe**（本地 SSD），再进 GPU。

---

## 9. 速查对照表

| 缩写 | 全称 / 中文 | 一句话 |
|------|-------------|--------|
| **PCIe** | Peripheral Component Interconnect Express | CPU 与 GPU、NVMe 等扩展设备的高速总线 |
| **NVMe** | Non-Volatile Memory Express | 跑在 PCIe 上的 SSD 协议，不是显存 |
| **OAM** | OCP Accelerator Module | OCP 开放的 AI 加速模块形态，利于标准化液冷集群 |
| **SXM** | Server PCI Express Module | NVIDIA 服务器 GPU 插座模块，高功耗多 NVLink |
| **HBM** | High Bandwidth Memory | GPU 高带宽显存 |
| **NVLink** | — | GPU 间高速直连 |
| **NVSwitch** | — | 多 GPU NVLink 交换芯片 |
| **MIG** | Multi-Instance GPU | 一卡多实例切分 |
| **DGX / HGX** | — | NVIDIA 整机 / 多 GPU 基板设计 |
| **IB** | InfiniBand | 机间高速网络，常配 RDMA |
| **RDMA** | Remote Direct Memory Access | 网卡直连远端内存，少 CPU 拷贝 |
| **NUMA** | Non-Uniform Memory Access | 多路 CPU 下内存访问远近不一 |
| **ECC** | Error-Correcting Code | 内存/显存纠错 |
| **TDP** | Thermal Design Power | 热设计功耗 |

---

## 10. 延伸阅读（本仓库）

- 芯片内部：[H100-Streaming-Multiprocessor-SM.md](H100-Streaming-Multiprocessor-SM.md)（GPC、SM、HBM、L2、PCIe、NVLink）
- 访存：[CoalesedMemoryAccess.md](CoalesedMemoryAccess.md)、[Bank_Conflict.md](Bank_Conflict.md)
- 课程路线：[README.md](README.md)

**官方参考**（可选）：[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)、[NVIDIA Data Center GPU 文档](https://www.nvidia.com/en-us/data-center/)。
