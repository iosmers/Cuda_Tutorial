# GPU 硬件拆解与供应链地图

> 最后更新：2026-06-11  
> 目标：把一张 GPU 加速卡从“芯片内部”拆到“封装、显存、板卡、供电、散热、整机”，并记录每一层常见的供应商/代表厂商。  
> 注意：供应链会随具体型号、地区、客户和批次变化。下面是学习用的“常见供应商地图”，不是任何单一 GPU 的完整 BOM。

---

## 1. 一张 GPU 加速卡到底由什么组成？

很多人说“GPU 有很多 core”，但真实产品不是只有计算核心。以数据中心 GPU 为例，可以拆成五层：

```text
第 5 层：服务器 / 机柜
CPU、系统 DRAM、NVMe、NIC/DPU、NVSwitch、PSU、风冷/液冷、BMC
        |
第 4 层：GPU 模块 / 板卡
PCIe 卡、SXM、OAM、PCB、VRM、连接器、传感器、散热器、冷板
        |
第 3 层：先进封装
compute die/chiplet + HBM stack + silicon interposer/RDL bridge + ABF package substrate
        |
第 2 层：显存与片外 I/O
HBM/GDDR、memory controller、PHY、PCIe/NVLink/Infinity Fabric Link
        |
第 1 层：GPU 逻辑芯片内部
GPC/Shader Engine、SM/CU、Tensor Core/Matrix Core、L1/Shared、L2、scheduler、copy engine
```

学习 CUDA 时最常接触的是第 1、2 层：SM、warp、register、shared memory、L2、HBM。  
看供应链和硬件成本时最关键的是第 3、4、5 层：HBM、CoWoS/先进封装、ABF substrate、供电、散热、整机集成。

---

## 2. 芯片内部单元：这些通常没有“单独供应商”

CUDA Core、Tensor Core、SM、L2、memory controller 这些不是外面买来的小零件，而是 GPU 设计公司画进芯片版图里的电路。供应链上它们对应的是：

```text
GPU 架构设计公司
  -> NVIDIA / AMD / Intel / 自研 AI 芯片公司
EDA/IP 工具和部分接口 IP
  -> Synopsys / Cadence / Siemens EDA / Rambus / Alphawave 等
晶圆制造
  -> TSMC / Samsung Foundry / Intel Foundry 等
```

### 2.1 计算与调度单元

| 硬件单元 | NVIDIA 常见叫法 | AMD 常见叫法 | Intel 常见叫法 | 作用 | 供应商视角 |
|----------|----------------|--------------|----------------|------|------------|
| 顶层计算分组 | GPC | Shader Engine / XCD 内分组 | Slice / Render Slice | 把整颗 GPU 分成多个大区域，方便调度、布线和缓存组织 | 架构公司自研，制造在逻辑 die 中 |
| 中层计算分组 | TPC | Workgroup Processor / Compute Unit group | Xe Slice 内部子分组 | 再细分计算资源 | 同上 |
| 基本执行集群 | SM, Streaming Multiprocessor | CU, Compute Unit | Xe-core | CUDA block / wavefront / workgroup 最终落在这里执行 | 同上 |
| 线程束调度 | Warp Scheduler | Wavefront Scheduler | Thread/warp scheduler | 在多个 warp/wave 之间切换，隐藏内存延迟 | 同上 |
| 标量/向量算术 | CUDA Core | Stream Processor / SIMD lane | Vector engine | FP32、INT32、FP64 等传统算术 | 同上 |
| 矩阵乘单元 | Tensor Core | Matrix Core | XMX | FP16/BF16/TF32/FP8/INT8 等矩阵乘加，是 AI 训练/推理核心 | 同上 |
| 特殊函数单元 | SFU | Special Function Unit | Math/special function | sin/cos/rsqrt、插值等特殊函数 | 同上 |
| Load/Store 单元 | LD/ST | Load/Store | Load/store pipeline | 发起 global/shared/local memory 读写 | 同上 |
| 纹理单元 | Texture Unit | Texture Unit | Sampler | 图形纹理采样，也可服务只读缓存路径 | 同上 |
| 光追单元 | RT Core | Ray Accelerator | Ray tracing unit | BVH 遍历、ray/triangle 测试，游戏/渲染相关 | 同上 |
| 专用算法指令 | DPX 等 | Matrix/AI/HPC 特化指令 | XMX/DPAS 等 | 动态规划、AI 数据格式、矩阵指令等 | 同上 |

### 2.2 片上存储单元

| 硬件单元 | CUDA 编程中的对应 | 作用 | 性能特点 | 供应商视角 |
|----------|------------------|------|----------|------------|
| Register File | 局部变量优先放这里 | 每个线程私有寄存器 | 最低延迟、容量有限 | 芯片内部 SRAM |
| Shared Memory / LDS | `__shared__` | 一个 block/workgroup 内共享 | 低延迟、高带宽，需关注 bank conflict | 芯片内部 SRAM |
| L1 Cache | L1 / texture cache | SM/CU 附近缓存 | 命中时快，受访问模式影响 | 芯片内部 SRAM |
| L2 Cache | 全 GPU 共享 L2 | SM 与 HBM/GDDR 之间的大缓存 | 所有 SM 共享，影响复用和原子操作 | 芯片内部 SRAM |
| Constant / Read-only Cache | 常量/只读路径 | 加速广播式只读访问 | 适合所有线程读同一数据 | 芯片内部 SRAM |

这些 SRAM 都集成在 GPU die 上，不是像 HBM/GDDR 那样外接的独立内存颗粒。

### 2.3 片上互联与 I/O 单元

| 硬件单元 | 作用 | 你会在哪里看到它 |
|----------|------|------------------|
| NoC / crossbar / fabric | 连接 SM、L2、memory controller、I/O block | 架构白皮书、芯片 floorplan |
| Memory Controller | 管理 HBM/GDDR 通道、调度读写、ECC/RAS | 显存带宽、通道数、HBM stack 数 |
| HBM/GDDR PHY | 把片上数字信号变成高速外部电信号 | memory interface、signal integrity |
| PCIe Controller / PHY | CPU 与 GPU 之间的标准总线 | PCIe Gen4/5/6 x16 |
| NVLink / Infinity Fabric Link / Xe Link | GPU-GPU 或 GPU-CPU 高速互联 | 多卡训练、P2P、NCCL |
| Copy Engine / DMA Engine | 异步拷贝、P2P、H2D/D2H 数据搬运 | `cudaMemcpyAsync`、Nsight Systems 时间线 |
| NVDEC / NVENC / JPEG / Display Engine | 视频编解码、图像解码、显示输出 | 消费卡、推理视频 pipeline；部分数据中心卡没有显示 |
| RAS / ECC / Security / Firmware | 纠错、隔离、安全启动、遥测 | 数据中心 GPU 可靠性、MIG、DCGM |
| Clock / PLL / Power Gate / Sensor | 时钟、电压、功耗、温度管理 | TDP/TGP、boost clock、热降频 |

---

## 3. 显存与内存层次

### 3.1 从线程到显存的数据路径

```text
线程局部变量
  -> Register
  -> Shared Memory / L1
  -> L2 Cache
  -> HBM 或 GDDR
  -> PCIe / NVLink 到其它设备或主机内存
```

CUDA 优化里常说的“合并访存”“shared memory tiling”“bank conflict”“L2 复用”，本质都是围绕这条路径减少等待。

### 3.2 HBM 与 GDDR

| 类型 | 放在哪里 | 典型 GPU | 优点 | 代价 | 主要供应商 |
|------|----------|----------|------|------|------------|
| HBM / HBM2E / HBM3 / HBM3E / HBM4 | 和 GPU compute die 放在同一个先进封装内，通常通过 interposer/RDL 连接 | H100、H200、B200、MI300X、MI350 等数据中心加速器 | 极高带宽、能效好、封装内短连线 | 成本高、封装复杂、供应紧张 | SK hynix、Samsung、Micron |
| GDDR6 / GDDR6X / GDDR7 | 焊在 GPU PCB 上，围绕 GPU 封装摆放 | GeForce、Radeon、工作站卡、部分推理卡 | 成本低、生态成熟、板级设计灵活 | 带宽/能效低于高端 HBM，多颗粒占板面积 | Samsung、Micron、SK hynix |
| 片上 SRAM | GPU die 内部 | 所有 GPU | 低延迟、高带宽 | 容量昂贵 | 无独立供应商，属于逻辑 die |
| Host DRAM | CPU 主板 DIMM / SOCAMM 等 | 服务器系统内存 | 容量大，喂数据给 GPU | 远离 GPU，带宽/延迟受 PCIe/NUMA 影响 | Samsung、SK hynix、Micron，以及 DIMM 模组厂 |

HBM 的关键点：

- HBM 是多层 DRAM die 通过 TSV 垂直堆叠，再通过极宽接口连接到 GPU。
- HBM 通常不是插在 PCB 上，而是和 GPU compute die 一起进入 2.5D/3D 先进封装。
- 高端 AI GPU 的“内存墙”很大程度取决于 HBM 容量、HBM 带宽、L2 复用和算子访存模式。

GDDR 的关键点：

- GDDR 更常见于消费级/工作站 GPU。
- GDDR 走 PCB 级布线，接口没有 HBM 那么宽，但成本和供应链成熟度更好。
- GDDR7 使用 PAM3 信号，面向更高带宽和更好能效。

---

## 4. 先进封装：GPU 和 HBM 怎么放到一起？

高端 AI GPU 的核心不只是“晶圆先进制程”，还有先进封装。没有先进封装，compute die 和 HBM stack 无法以足够高密度连接。

### 4.1 典型 HBM GPU 封装结构

```text
散热器 / 冷板 / lid
        |
TIM 导热材料
        |
compute die / chiplet      HBM stack      HBM stack
        \                   |             /
         \                  |            /
          silicon interposer / RDL bridge / local interconnect
                    |
             ABF package substrate
                    |
              GPU module PCB / baseboard
```

### 4.2 封装相关部件与供应商

| 部件 | 作用 | 常见供应商/代表厂商 | 备注 |
|------|------|---------------------|------|
| Silicon interposer | 在 GPU die 和 HBM 之间提供高密度布线 | TSMC CoWoS-S；Samsung I-Cube；Intel EMIB/Foveros 相关技术 | HBM GPU 的关键瓶颈之一 |
| RDL / local silicon interconnect | 用重布线层或局部硅桥连接 chiplet | TSMC CoWoS-L/CoWoS-R、InFO；Intel EMIB；Samsung X-Cube/I-Cube | 不同厂商命名不同 |
| Advanced packaging | 把 die、HBM、interposer、substrate 组合成 package | TSMC、Samsung Foundry、Intel Foundry、ASE、Amkor、JCET 等 | 顶级 AI GPU 常见 TSMC CoWoS 路线 |
| ABF package substrate | 连接封装内部细线距和 PCB 大尺度焊球 | Ibiden、Unimicron、Shinko Electric、Nan Ya PCB、AT&S、Kinsus、Samsung Electro-Mechanics | AI GPU 封装面积大，对高端 ABF substrate 要求高 |
| ABF build-up film 材料 | ABF substrate 的绝缘材料 | Ajinomoto | ABF 即 Ajinomoto Build-up Film |
| Micro-bump / bump / solder ball | die 到 interposer、package 到 PCB 的电连接 | 封装厂、材料厂、设备厂共同完成 | 通常不是单一品牌可见部件 |
| Underfill / molding / lid / stiffener | 机械保护、控制翘曲、改善可靠性 | 封装材料厂、OSAT、foundry 生态 | 大封装翘曲和热机械应力是难点 |

### 4.3 CoWoS 为什么重要？

CoWoS 是 TSMC 的 Chip-on-Wafer-on-Substrate 先进封装路线。TSMC 官方说明中，CoWoS-S 使用 silicon interposer，面向 AI 和超算等高性能计算场景，可在大 interposer 上集成 logic chiplet 和 HBM stack。

对 AI GPU 来说，CoWoS 的作用可以简化成：

```text
把 GPU compute die 和多颗 HBM stack 放得足够近，
用足够宽、足够短、足够低功耗的连接喂饱计算单元。
```

所以，AI GPU 的供应瓶颈往往不只在晶圆，也可能在 HBM、CoWoS 产能、ABF substrate、先进封装设备、测试和液冷整机集成。

---

## 5. 板卡 / 模块：封装之外还有什么？

同一颗 GPU package 可以做成 PCIe 卡、SXM 模块、OAM 模块，取决于功耗、互联、散热和服务器设计。

| 形态 | 常见场景 | 特点 |
|------|----------|------|
| PCIe Add-in Card | 台式机、工作站、部分服务器 | 插 PCIe x16，板上有 GDDR/HBM GPU、VRM、散热器；通用性强 |
| SXM | NVIDIA 数据中心 GPU 模块 | 高功耗、高 NVLink 带宽，常用于 HGX/DGX |
| OAM | OCP Accelerator Module | 开放加速器模块标准，常见于云厂商和高密度 AI 服务器 |
| UBB / Baseboard | 多 GPU 统一基板 | 连接多颗 GPU、NVSwitch/Infinity Fabric、供电和管理 |

### 5.1 板卡上的主要硬件

| 部件 | 作用 | 常见供应商/代表厂商 |
|------|------|---------------------|
| GPU PCB / module substrate | 承载 GPU package、VRM、连接器、传感器 | Unimicron、Tripod、Compeq、Nan Ya PCB、TTM、深南电路等；具体取决于板卡/ODM |
| VRM controller / power stage | 把 12V/48V 转成 GPU 核心和显存需要的低压大电流 | Monolithic Power Systems、Infineon、Renesas、Texas Instruments、Analog Devices、onsemi、Vishay、Vicor |
| Inductor / capacitor / MLCC | 稳压、滤波、瞬态供电 | TDK、Murata、Taiyo Yuden、Samsung Electro-Mechanics、Yageo、Panasonic、Coilcraft、Delta |
| PCIe / NVLink / OAM / SXM 连接器 | 模块与主板/基板连接 | Amphenol、TE Connectivity、Molex、Samtec 等 |
| Retimer / redriver / clock buffer | 高速信号重整和时钟分发 | Broadcom、Astera Labs、Parade、Montage、Texas Instruments、Renesas |
| EEPROM / flash / MCU / sensor | VBIOS、板卡身份、温度/电流/电压监测 | Winbond、Macronix、ST、Nuvoton、Microchip、TI、ADI 等 |
| Air cooler / blower / fan | 风冷散热 | Delta、Nidec、Sunon、AVC、Auras、Cooler Master 等 |
| Cold plate / liquid loop | 液冷散热 | CoolIT、Asetek、Boyd、Delta、Vertiv、富士康/纬创等整机液冷生态 |

这里的供应商更容易按客户定制变化。比如同一款 GPU，不同 OEM 服务器或 AIB 显卡会使用不同 PCB、VRM、风扇和散热器供应商。

---

## 6. 服务器 / 机柜层：多 GPU 系统还需要什么？

一张 GPU 卡只是训练系统的一部分。大模型训练/推理还需要 CPU、主机内存、NVMe、网络、交换芯片、液冷、电源和整机集成。

| 系统部件 | 作用 | 常见供应商/代表厂商 |
|----------|------|---------------------|
| CPU | 启动程序、喂数据、管理 GPU、运行部分前后处理 | AMD EPYC、Intel Xeon、NVIDIA Grace、Arm server CPU 等 |
| System DRAM | DataLoader、CPU 侧缓存、OS、通信 buffer | Samsung、SK hynix、Micron；模组厂 Kingston、SMART Modular、Apacer 等 |
| NVMe SSD | 数据集缓存、checkpoint、日志 | Samsung、Kioxia、Solidigm、Micron、SK hynix、Western Digital |
| GPU-GPU switch | 机内 GPU 互联 | NVIDIA NVSwitch；AMD Infinity Fabric 平台互联 |
| NIC / DPU / HCA | 多机通信、RDMA、存储网络 | NVIDIA/Mellanox、Broadcom、Marvell、Intel、AMD Pensando |
| PSU / power shelf | 服务器或机柜供电 | Delta、Lite-On、Chicony、Flex、AcBel、Artesyn/Advanced Energy 等 |
| CDU / rear-door heat exchanger | 液冷分配、机柜级散热 | Vertiv、CoolIT、Boyd、Asetek、Delta、Schneider Electric 等 |
| BMC / management | 带外管理、传感器、远程控制 | ASPEED、Nuvoton、服务器 ODM 自研管理板 |
| Server ODM/OEM | 整机设计、生产、验证、交付 | Foxconn、Quanta/QCT、Wistron/Wiwynn、Inventec、Pegatron、Supermicro、Dell、HPE、Lenovo、Gigabyte、ASUS 等 |

NVIDIA 早期 HGX 伙伴计划就包括 Foxconn、Inventec、Quanta、Wistron 等 ODM。现在 AI 服务器供应链更复杂，常见组合是：

```text
GPU/平台设计：NVIDIA / AMD
晶圆制造：TSMC 等
HBM：SK hynix / Samsung / Micron
先进封装：TSMC CoWoS 等
基板/PCB/供电/散热：多个材料与部件供应商
整机 ODM/OEM：Foxconn / Quanta / Wistron / Inventec / Supermicro / Dell / HPE / Lenovo ...
云厂商部署：AWS / Azure / Google Cloud / Meta / Oracle / CoreWeave 等
```

---

## 7. 供应链总表

| 层级 | 拆出来的硬件/环节 | 它是什么 | 常见供应商/代表厂商 | 学习重点 |
|------|-------------------|----------|---------------------|----------|
| 架构设计 | GPU / AI accelerator 设计 | 决定 SM/CU、Tensor Core、cache、互联、软件栈 | NVIDIA、AMD、Intel；云厂自研 ASIC 如 Google TPU、AWS Trainium、Microsoft Maia | 架构白皮书、CUDA/ROCm/oneAPI 生态 |
| 晶圆制造 | Logic die / chiplet | 把设计变成硅片 | TSMC、Samsung Foundry、Intel Foundry | 制程节点、良率、reticle limit、chiplet |
| 计算核心 | SM/CU/Xe-core、CUDA Core、Tensor Core | 逻辑 die 内部电路 | 无独立供应商，属于 GPU 设计和 foundry 制造结果 | CUDA block/warp 如何落到 SM |
| 片上存储 | Register、Shared/L1、L2 | GPU die 内 SRAM | 无独立供应商 | latency、bandwidth、occupancy、bank conflict |
| 显存 | HBM | 封装内 3D 堆叠 DRAM | SK hynix、Samsung、Micron | 带宽、容量、TSV、interposer |
| 显存 | GDDR | PCB 上的图形 DRAM | Samsung、Micron、SK hynix | 消费 GPU 带宽、显存位宽、PAM3 |
| 接口 IP | HBM/GDDR/PCIe/CXL/SerDes PHY | 高速接口设计或 IP | Synopsys、Cadence、Rambus、Alphawave、内部自研 | PHY、signal integrity、协议 |
| 先进封装 | CoWoS / silicon interposer / RDL | 把 GPU 和 HBM 接在一起 | TSMC、Samsung、Intel、ASE、Amkor、JCET | 2.5D/3D、interposer、chiplet |
| 封装基板材料 | ABF build-up film | 高端 package substrate 的关键绝缘材料 | Ajinomoto | ABF 为什么是瓶颈材料 |
| 封装基板 | ABF package substrate | 封装与 PCB 之间的高密度基板 | Ibiden、Unimicron、Shinko Electric、Nan Ya PCB、AT&S、Kinsus、Samsung Electro-Mechanics | 大封装、翘曲、层数、良率 |
| 板卡 | PCIe/SXM/OAM module PCB | 承载 GPU package 和供电/连接器 | Unimicron、Tripod、Compeq、Nan Ya PCB、TTM、深南电路等 | 高速走线、供电、散热 |
| 电源 | VRM、power stage、inductor、capacitor | 给 GPU 供低压大电流 | MPS、Infineon、Renesas、TI、ADI、onsemi、Vishay、Vicor、TDK、Murata、Yageo 等 | TDP、瞬态电流、效率 |
| 高速连接 | PCIe/NVLink/OAM/SXM connector、retimer | 板级高速信号 | Amphenol、TE、Molex、Samtec、Broadcom、Astera Labs、Parade、Montage | PCIe/NVLink 拓扑和信号完整性 |
| 散热 | heatsink、fan、cold plate、CDU | 把 300W 到 1000W 级热量带走 | Delta、Nidec、Sunon、AVC、CoolIT、Asetek、Boyd、Vertiv 等 | 热阻、液冷、热降频 |
| 整机 | GPU server / rack | 多 GPU、CPU、网络、电源、液冷集成 | Foxconn、Quanta/QCT、Wistron/Wiwynn、Inventec、Supermicro、Dell、HPE、Lenovo、Gigabyte | 拓扑、NUMA、网络、运维 |

---

## 8. 典型产品拆解视角

### 8.1 NVIDIA H100 / GH100

NVIDIA Hopper 架构资料中，完整 GH100 GPU 包含 GPC、TPC、SM、L2 cache 和 HBM3 memory controller。NVIDIA 官方技术博客列出的完整 GH100 单元包括：

- 8 个 GPC、72 个 TPC、144 个 SM。
- 每个 SM 有 128 个 FP32 CUDA Core。
- 每个 SM 有 4 个第四代 Tensor Core。
- 6 个 HBM3 或 HBM2e stack，12 个 512-bit memory controller。
- 60 MB L2 cache。
- 第四代 NVLink 和 PCIe Gen5。

H100 SXM5 实际产品启用 132 个 SM、80GB HBM3、5 个 HBM3 stack 和 50MB L2。这个差异说明：架构图里的“完整 die”不一定等于最终销售 SKU，量产产品会因为良率、功耗、定位禁用一部分单元。

供应链视角：

```text
GPU 设计：NVIDIA
逻辑 die 制造：TSMC 4N
显存：HBM3，供应商随批次/产品而变
先进封装：HBM + GPU die 的 2.5D 封装路线
模块：SXM5 或 PCIe
整机：HGX/DGX 以及 OEM/ODM 服务器
```

### 8.2 NVIDIA Blackwell / B200/GB200

NVIDIA Blackwell 官方页面说明：Blackwell 架构 GPU 使用定制 TSMC 4NP 工艺，包含 2080 亿晶体管，并由两个 reticle-limited die 通过 10TB/s chip-to-chip interconnect 连接成统一 GPU。

从硬件学习角度，Blackwell 值得关注的是：

- 两颗大型 compute die 组成一个统一 GPU。
- 封装复杂度高于传统单 die GPU。
- HBM、chip-to-chip interconnect、CoWoS/先进封装、液冷整机都成为系统级瓶颈。

供应链视角：

```text
GPU 设计：NVIDIA
逻辑 die 制造：TSMC 4NP
显存：HBM3E/HBM4 代际随具体产品变化，主流供应商为 SK hynix / Samsung / Micron
先进封装：TSMC CoWoS 系列是关键路线之一
整机：GB200/GB300/NVL rack 涉及 ODM、液冷、电源、网络和 NVLink 系统
```

### 8.3 AMD Instinct MI300X

AMD MI300X 是 chiplet 和先进封装学习的好例子。AMD 数据表写明 MI300X 使用 CDNA 3 架构、304 个 compute unit、192GB HBM3，并使用 die stacking 和 chiplet 技术。AMD Hot Chips 资料列出 MI300X 为多 chiplet 加速器，1530 亿晶体管，使用 TSMC 5nm/6nm FinFET，并通过 Infinity Fabric 连接。

供应链视角：

```text
GPU 设计：AMD
逻辑 chiplet 制造：TSMC 5nm / 6nm
显存：HBM3
封装：多 chiplet + HBM 的先进封装
模块：OAM / 8-GPU platform
互联：Infinity Fabric Link
软件生态：ROCm
```

---

## 9. 从 CUDA 性能问题反推硬件单元

| 你看到的性能问题 | 可能关联的硬件单元 | 排查方向 |
|------------------|--------------------|----------|
| kernel 很慢但显存带宽不高 | SM、warp scheduler、occupancy、指令依赖 | 看活跃 warp、寄存器用量、指令吞吐 |
| 显存带宽打不满 | HBM/GDDR、L2、memory controller、LD/ST | 看 coalescing、stride、cache hit、内存事务 |
| shared memory 优化后没变快 | Shared/L1、bank、同步开销 | 看 bank conflict、`__syncthreads()`、tile 大小 |
| Tensor Core 没吃满 | Tensor Core、寄存器、shared memory、memory pipeline | 看 dtype、矩阵 shape、layout、mma 指令、数据搬运 |
| 多卡扩展差 | NVLink/NVSwitch/PCIe/NIC、CPU NUMA | 看 `nvidia-smi topo -m`、NCCL 日志、网络带宽 |
| 第一次运行很慢 | JIT、driver、kernel cache、TensorRT engine build | 做 warmup，固定 shape，开启 cache |
| 高负载后降频 | VRM、散热器、冷板、机箱风道、功耗限制 | 看功耗、温度、clock、DCGM/NVML 指标 |

---

## 10. 学习路线

### 阶段 1：先学芯片内部

- SM/CU/Xe-core 是什么。
- Warp/wavefront 为什么是执行和访存的基本单位。
- Register、Shared Memory、L1、L2、HBM 的层级关系。
- Tensor Core/Matrix Core 为什么适合矩阵乘。

对应本仓库：

- [Stage1_入门基础.md](Stage1_入门基础.md)
- [stage2_内存模型与调试.md](stage2_内存模型与调试.md)
- [H100-Streaming-Multiprocessor-SM.md](H100-Streaming-Multiprocessor-SM.md)

### 阶段 2：再学封装和显存

- HBM 为什么要放进封装里，而不是像 GDDR 那样放在 PCB 上。
- Silicon interposer、RDL bridge、CoWoS、ABF substrate 分别干什么。
- 为什么先进封装和 HBM 会成为 AI GPU 的供应瓶颈。

建议关键词：

- `HBM TSV`
- `2.5D packaging`
- `CoWoS`
- `silicon interposer`
- `ABF substrate`
- `chiplet`
- `reticle limit`

### 阶段 3：再学板卡和整机

- PCIe、SXM、OAM 的物理差异。
- VRM 怎么给 GPU 提供几百安培甚至上千安培瞬态电流。
- 风冷、液冷、冷板、CDU 的区别。
- NVLink/NVSwitch/InfiniBand/RoCE 在多卡训练中的位置。

对应本仓库：

- [Hardware.md](Hardware.md)
- [CUDA_GPU_编程栈地图.md](CUDA_GPU_编程栈地图.md)

### 阶段 4：最后看供应链

读 GPU 供应链时，按下面顺序拆：

```text
1. 谁设计芯片？NVIDIA / AMD / Intel / 自研 ASIC 公司
2. 谁代工晶圆？TSMC / Samsung Foundry / Intel Foundry
3. 用什么显存？HBM 还是 GDDR？供应商是谁？
4. 谁做先进封装？CoWoS / I-Cube / EMIB / Foveros / OSAT
5. 用什么 package substrate？ABF substrate 供应商是谁？
6. 谁做板卡/模块/整机？ODM/OEM 是谁？
7. 供电和散热是否支撑目标 TDP？
8. 多卡互联和网络是谁提供？
```

---

## 11. 读 GPU 规格表时看什么？

| 规格字段 | 你应该联想到的硬件 |
|----------|--------------------|
| SM / CU 数量 | 并行计算集群数量 |
| CUDA Core / Stream Processor | 标量/向量算术吞吐 |
| Tensor Core / Matrix Core | AI 矩阵乘吞吐 |
| FP32 / FP16 / BF16 / FP8 / INT8 TFLOPS/TOPS | 不同数据类型的峰值计算能力 |
| HBM/GDDR 容量 | 模型能不能放下，batch 能不能做大 |
| Memory bandwidth | 算子是否容易 memory-bound |
| L2 cache | 数据复用和跨 SM 共享缓存能力 |
| PCIe Gen / NVLink / Infinity Fabric | 主机和多卡通信能力 |
| TDP / TGP | 供电、散热、整机功耗 |
| Form factor | PCIe、SXM、OAM，决定服务器形态 |
| ECC / RAS / MIG | 数据中心可靠性和隔离能力 |

记住一个粗略判断：

```text
算子算术强度高：更看 Tensor Core / CUDA Core / clock
算子算术强度低：更看 HBM/GDDR 带宽、L2、访存合并
多卡训练：更看 NVLink/NVSwitch/IB/RoCE 和 NCCL 拓扑
部署落地：更看功耗、散热、供货、整机集成和软件生态
```

---

## 12. 术语速查

| 术语 | 含义 |
|------|------|
| Die | 一颗硅芯片。高端 GPU 可以是单 die，也可以是多 chiplet |
| Chiplet | 把大芯片拆成多个小 die，再通过先进封装连接 |
| Reticle limit | 光刻单次曝光面积限制，大 die 会接近这个上限 |
| SM | NVIDIA Streaming Multiprocessor，CUDA block 最终执行的核心集群 |
| CU | AMD Compute Unit，ROCm/HIP 里的核心执行集群 |
| Tensor Core | NVIDIA 矩阵乘加单元 |
| Matrix Core | AMD 矩阵乘加单元 |
| XMX | Intel 矩阵引擎 |
| HBM | High Bandwidth Memory，封装内高带宽堆叠显存 |
| GDDR | Graphics DDR，板级图形显存 |
| Interposer | 连接 GPU die 与 HBM 的高密度中介层 |
| CoWoS | TSMC 的 Chip-on-Wafer-on-Substrate 先进封装技术 |
| ABF substrate | 高端封装基板，使用 Ajinomoto Build-up Film 材料 |
| VRM | Voltage Regulator Module，给 GPU 提供低压大电流 |
| SXM | NVIDIA 高端服务器 GPU 模块形态 |
| OAM | OCP Accelerator Module，开放加速器模块标准 |
| NVLink | NVIDIA GPU 间高速互联 |
| NVSwitch | NVIDIA GPU 间交换芯片 |
| Infinity Fabric | AMD 芯片内/芯片间/卡间互联体系 |
| RAS | Reliability, Availability, Serviceability，可靠性/可用性/可维护性 |

---

## 13. 参考资料

- NVIDIA Hopper Architecture In-Depth: <https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/>
- NVIDIA Blackwell Architecture: <https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>
- NVIDIA H200 Tensor Core GPU: <https://www.nvidia.com/en-us/data-center/h200/>
- AMD Instinct MI300X Data Sheet: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf>
- AMD Instinct MI300X Hot Chips 2024 slides: <https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith%28MI300X%29.v1.Final.20240817.pdf>
- TSMC CoWoS: <https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm>
- Samsung HBM3E: <https://semiconductor.samsung.com/dram/hbm/hbm3e/>
- Micron HBM3E: <https://www.micron.com/products/memory/hbm/hbm3e>
- SK hynix HBM4 sample announcement: <https://news.skhynix.com/sk-hynix-ships-world-first-12-layer-hbm4-samples-to-customers/>
- Samsung GDDR7: <https://semiconductor.samsung.com/dram/gddr/gddr7/>
- Micron GDDR7: <https://www.micron.com/products/memory/graphics-memory/gddr7>
- Ajinomoto Build-up Film: <https://www.ajinomoto.com/innovation/our_innovation/buildupfilm>
- IBIDEN Flip Chip Package Substrate: <https://www.ibiden.com/product/electronics/merchandise/fliptippkg/>
- Shinko Semiconductor Package: <https://www.shinko.co.jp/english/product/package/>
- NVIDIA HGX ODM partner announcement: <https://nvidianews.nvidia.com/news/nvidia-partners-with-world-s-top-server-manufacturers-to-advance-ai-cloud-computing>
