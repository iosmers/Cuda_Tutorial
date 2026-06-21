# GPU 芯片规格汇总（LLM 训练 / 推理常用）

本文从 [README.md](./GPUs-Specs/README.md) 收录的各家官方 Datasheet / 架构白皮书中提取并整理关键规格，便于在大模型训练与推理选型时横向对比。

> 说明
> - 算力单位：`TFLOPS`（万亿次浮点/秒）、`TOPS`（万亿次整数/秒）、`PFLOPS = 1000 TFLOPS`。
> - NVIDIA 的 Tensor Core 算力，官方常以「**带 2:4 稀疏**」为标题数字，本文统一写成 `稠密 / 稀疏` 两个值（稀疏≈稠密×2）。
> - 内存带宽 `TB/s = 1000 GB/s`。
> - 同一型号存在 SXM / PCIe / 不同显存容量等多种变体，规格随之不同，下文已尽量标注。
> - 数据以厂商 Datasheet 公布的峰值理论值为准，实际可用算力会更低。

---

## 一、NVIDIA GPUs

### 架构演进速览

| 架构 | 代表产品 | 工艺 | 发布年 | 关键特性 |
|------|----------|------|--------|----------|
| Volta | V100 | TSMC 12nm | 2017 | 第一代 Tensor Core |
| Turing | T4 | TSMC 12nm | 2018 | INT8/INT4 推理、第二代 Tensor Core |
| Ampere | A100 / A30 / A40 / A10 | 7nm / 8nm | 2020 | TF32、第三代 Tensor Core、MIG、结构化稀疏 |
| Ada Lovelace | RTX 4090 / L40 / L40S / L20 | TSMC 4N | 2022 | FP8、第四代 Tensor Core |
| Hopper | H100 / H200 / H800 / H20 | TSMC 4N | 2022 | Transformer Engine、FP8、DPX、HBM3 |
| Blackwell | GB200 / B200 / RTX 5090 | TSMC 4NP | 2024 | 第二代 Transformer Engine、FP4、双 die |

---

### 1. Blackwell 架构

#### NVIDIA B200 / GB200

| 规格 | B200 (SXM) | GB200 (Superchip) |
|------|-----------|-------------------|
| 架构 | Blackwell（双 die） | Grace CPU + 2× B200 |
| 晶体管 | 208 B | — |
| 显存 | 192 GB HBM3e | 384 GB HBM3e（GPU 部分） |
| 显存带宽 | 8 TB/s | 16 TB/s（合计） |
| FP4 Tensor (稠密/稀疏) | 9 / 18 PFLOPS | 20 / 40 PFLOPS |
| FP8 / INT8 (稠密/稀疏) | 4.5 / 9 PFLOPS | 10 / 20 PFLOPS |
| FP16 / BF16 (稠密/稀疏) | 2.25 / 4.5 PFLOPS | 5 / 10 PFLOPS |
| TF32 (稠密/稀疏) | 1.1 / 2.2 PFLOPS | 2.5 / 5 PFLOPS |
| FP64 Tensor | 40 TFLOPS | 90 TFLOPS |
| NVLink | 1.8 TB/s（第 5 代） | 1.8 TB/s/GPU |
| 功耗 | 最高 1000 W | 2700 W（整卡） |

> GB200 NVL72：一个机架内 36 颗 Grace CPU + 72 颗 B200，组成单一 NVLink 域，FP4 推理算力可达 1.4 EFLOPS。

#### GeForce RTX 5090

| 规格 | 数值 |
|------|------|
| 核心 | GB202，TSMC 4NP |
| CUDA 核心 | 21,760 |
| 加速频率 | ~2.41 GHz |
| 显存 | 32 GB GDDR7 |
| 显存位宽 / 带宽 | 512-bit / 1792 GB/s |
| FP32 | ~104.8 TFLOPS |
| AI 算力 | 3352 TOPS (FP4) |
| 功耗 (TGP) | 575 W |
| 接口 | PCIe 5.0（无 NVLink） |

---

### 2. Hopper 架构

#### H100 / H200 / H800 / H20 对比

| 规格 | H100 SXM | H100 PCIe | H200 SXM | H800 SXM | H20 SXM |
|------|----------|-----------|----------|----------|---------|
| 核心 | GH100 | GH100 | GH100 | GH100 | GH100 |
| 工艺 | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N |
| 晶体管 | 80 B | 80 B | 80 B | 80 B | 80 B |
| 显存 | 80 GB HBM3 | 80 GB HBM2e | 141 GB HBM3e | 80 GB HBM3 | 96 GB HBM3 |
| 显存带宽 | 3.35 TB/s | 2.0 TB/s | 4.8 TB/s | 3.35 TB/s | 4.0 TB/s |
| FP64 / FP64 Tensor | 34 / 67 TFLOPS | 26 / 51 TFLOPS | 34 / 67 TFLOPS | 1 (阉割) | ~1 TFLOPS |
| FP32 | 67 TFLOPS | 51 TFLOPS | 67 TFLOPS | 67 TFLOPS | ~44 TFLOPS |
| TF32 (稠密/稀疏) | 494 / 989 | 378 / 756 | 494 / 989 | 494 / 989 | ~74 / 148 |
| FP16/BF16 (稠密/稀疏) | 989 / 1979 | 756 / 1513 | 989 / 1979 | 989 / 1979 | ~148 / 296 |
| FP8 (稠密/稀疏) | 1979 / 3958 | 1513 / 3026 | 1979 / 3958 | 1979 / 3958 | ~296 / 592 |
| INT8 (稠密/稀疏 TOPS) | 1979 / 3958 | 1513 / 3026 | 1979 / 3958 | 1979 / 3958 | ~296 / 592 |
| NVLink | 900 GB/s | 600 GB/s (桥) | 900 GB/s | **400 GB/s** | 900 GB/s |
| 功耗 | 最高 700 W | 350 W | 最高 700 W | 最高 700 W | 400 W |

> 对华特供卡说明
> - **H800**：算力与 H100 几乎一致，主要削减 NVLink 带宽（900→400 GB/s）以满足出口管制。
> - **H20**：内存与带宽很高（96 GB / 4.0 TB/s），但 Tensor 算力大幅削减（约为 H100 的 1/7），定位为「大显存、低算力」的推理卡，适合 KV-Cache 密集型场景。
> - **H100 NVL**：双卡形态，合计 188 GB HBM3、7.8 TB/s，专为大模型推理优化。

---

### 3. Ada Lovelace 架构

| 规格 | RTX 4090 | L40 | L40S | L20 |
|------|----------|-----|------|-----|
| 核心 | AD102 | AD102 | AD102 | AD102 |
| CUDA 核心 | 16,384 | 18,176 | 18,176 | — |
| 显存 | 24 GB GDDR6X | 48 GB GDDR6 ECC | 48 GB GDDR6 ECC | 48 GB GDDR6 |
| 显存带宽 | 1008 GB/s | 864 GB/s | 864 GB/s | 864 GB/s |
| FP32 | 82.6 TFLOPS | 90.5 TFLOPS | 91.6 TFLOPS | ~59.8 TFLOPS |
| FP16 Tensor (稀疏) | 660 TFLOPS | 362 TFLOPS | 733 TFLOPS | — |
| FP8 Tensor (稀疏) | 1321 TFLOPS | 1466 TFLOPS | 1466 TFLOPS | ~239 TFLOPS |
| 功耗 | 450 W | 300 W | 350 W | 275 W |
| 接口 | PCIe 4.0 | PCIe 4.0 | PCIe 4.0 | PCIe 4.0 |

> L40S 相对 L40 提升了 Tensor/AI 吞吐，定位推理 + 轻量训练 + 图形一体；L20 为对华特供版。

---

### 4. Ampere 架构

#### 数据中心旗舰：A100 / A800

| 规格 | A100 SXM 80GB | A100 PCIe 80GB | A100 40GB | A800 80GB |
|------|---------------|----------------|-----------|-----------|
| 核心 | GA100 | GA100 | GA100 | GA100 |
| 工艺 | TSMC 7nm | TSMC 7nm | TSMC 7nm | TSMC 7nm |
| 晶体管 | 54.2 B | 54.2 B | 54.2 B | 54.2 B |
| SM / CUDA 核心 | 108 / 6912 | 108 / 6912 | 108 / 6912 | 108 / 6912 |
| 显存 | 80 GB HBM2e | 80 GB HBM2e | 40 GB HBM2 | 80 GB HBM2e |
| 显存带宽 | 2039 GB/s | 1935 GB/s | 1555 GB/s | 2039 GB/s |
| FP64 / FP64 Tensor | 9.7 / 19.5 | 9.7 / 19.5 | 9.7 / 19.5 | 9.7 / 19.5 |
| FP32 | 19.5 TFLOPS | 19.5 TFLOPS | 19.5 TFLOPS | 19.5 TFLOPS |
| TF32 (稠密/稀疏) | 156 / 312 | 156 / 312 | 156 / 312 | 156 / 312 |
| FP16/BF16 (稠密/稀疏) | 312 / 624 | 312 / 624 | 312 / 624 | 312 / 624 |
| INT8 (稠密/稀疏 TOPS) | 624 / 1248 | 624 / 1248 | 624 / 1248 | 624 / 1248 |
| NVLink | 600 GB/s | 600 GB/s (桥) | 600 GB/s | **400 GB/s** |
| 功耗 | 400 W | 300 W | 400 W | 400 W |

> **A800** 为对华特供版，算力与 A100 相同，仅将 NVLink 由 600 GB/s 降至 400 GB/s。

#### 其他 Ampere 卡

| 规格 | A40 | A30 | A10 | A16 | A2 |
|------|-----|-----|-----|-----|-----|
| 核心 | GA102 | GA100 | GA102 | 4× GA107 | GA107 |
| 显存 | 48 GB GDDR6 ECC | 24 GB HBM2 | 24 GB GDDR6 | 64 GB (4×16) GDDR6 | 16 GB GDDR6 |
| 显存带宽 | 696 GB/s | 933 GB/s | 600 GB/s | 4×200 GB/s | 200 GB/s |
| FP32 | 37.4 TFLOPS | 10.3 TFLOPS | 31.2 TFLOPS | 4×4.5 TFLOPS | 4.5 TFLOPS |
| FP16 Tensor (稀疏) | 299 TFLOPS | 330 TFLOPS | 250 TFLOPS | — | 36 TFLOPS |
| INT8 (稀疏 TOPS) | 599 | 661 | 500 | — | 72 |
| 功耗 | 300 W | 165 W | 150 W | 250 W | 40–60 W |
| 定位 | 图形/训练 | 推理/训练 | 推理/视觉 | 高密度 VDI | 边缘/入门推理 |

---

### 5. Turing 架构

#### NVIDIA T4

| 规格 | 数值 |
|------|------|
| 核心 | TU104，TSMC 12nm |
| CUDA / Tensor 核心 | 2560 / 320 |
| 显存 | 16 GB GDDR6 |
| 显存带宽 | 320 GB/s |
| FP32 | 8.1 TFLOPS |
| FP16 Tensor | 65 TFLOPS |
| INT8 / INT4 | 130 / 260 TOPS |
| 功耗 | 70 W（单槽、被动散热） |

> T4 凭借 70W 超低功耗 + INT8/INT4 推理，长期是云端推理的主力低功耗卡。

---

### 6. Volta 架构

#### NVIDIA V100

| 规格 | V100 SXM2 | V100 PCIe |
|------|-----------|-----------|
| 核心 | GV100，TSMC 12nm | GV100 |
| 晶体管 | 21.1 B | 21.1 B |
| CUDA / Tensor 核心 | 5120 / 640 | 5120 / 640 |
| 显存 | 16 / 32 GB HBM2 | 16 / 32 GB HBM2 |
| 显存带宽 | 900 GB/s | 900 GB/s |
| FP64 | 7.8 TFLOPS | 7.0 TFLOPS |
| FP32 | 15.7 TFLOPS | 14.0 TFLOPS |
| Tensor (FP16) | 125 TFLOPS | 112 TFLOPS |
| NVLink | 300 GB/s | — (仅 PCIe) |
| 功耗 | 300 W | 250 W |

---

## 二、AMD Instinct GPUs

| 规格 | MI325X | MI300X | MI300A (APU) |
|------|--------|--------|--------------|
| 架构 | CDNA 3 | CDNA 3 | CDNA 3 + Zen 4 |
| 工艺 | TSMC 5nm/6nm | TSMC 5nm/6nm | TSMC 5nm/6nm |
| CPU | — | — | 24 核 Zen 4 |
| 显存 | 256 GB HBM3e | 192 GB HBM3 | 128 GB HBM3（统一内存） |
| 显存带宽 | 6.0 TB/s | 5.3 TB/s | 5.3 TB/s |
| FP64 (Vector/Matrix) | 81.7 / 163.4 TFLOPS | 81.7 / 163.4 TFLOPS | 61.3 / 122.6 TFLOPS |
| FP32 | 163.4 TFLOPS | 163.4 TFLOPS | 122.6 TFLOPS |
| TF32 (稀疏) | 653 TFLOPS | 653 TFLOPS | 490 TFLOPS |
| FP16/BF16 (稠密/稀疏) | 1307 / 2614 | 1307 / 2614 | 980 / 1961 |
| FP8 (稠密/稀疏) | 2614 / 5229 | 2614 / 5229 | 1961 / 3922 |
| INT8 (稠密/稀疏 TOPS) | 2614 / 5229 | 2614 / 5229 | 1961 / 3922 |
| 互联 | Infinity Fabric | Infinity Fabric | Infinity Fabric |
| 功耗 | 1000 W | 750 W | 550 W（峰值 760 W） |

> AMD Instinct 系列以**超大显存 + 超高带宽**著称（MI300X 192GB / MI325X 256GB），在单卡放下更大模型、减少跨卡通信方面有优势；MI300A 为 CPU+GPU 统一内存 APU，主打 HPC + AI 融合（用于 El Capitan 超算）。

---

## 三、选型速查

### 按训练 / 推理场景

| 场景 | 推荐 | 理由 |
|------|------|------|
| 超大模型训练（万卡集群） | GB200 / B200 / H100 / H200 | FP8/FP4 + 高速 NVLink + Transformer Engine |
| 大模型训练（受管制地区） | H800 / A800 | 算力接近旗舰，仅互联受限 |
| 大模型推理（长上下文/大 KV-Cache） | H200 / H20 / MI300X / MI325X | 大显存 + 高带宽 |
| 性价比推理 | L40S / L20 / RTX 4090 | FP8 支持、单卡成本低 |
| 低功耗 / 边缘推理 | T4 / A2 / A10 | 70–150 W，被动/单槽 |
| HPC + AI 融合 | MI300A / H100 | 强 FP64 |

### 关键维度记忆

- **显存容量榜**：MI325X (256) > MI300X (192) ≈ B200 (192) > H200 (141) > H20 (96) > H100/A100 (80)
- **显存带宽榜**：B200 (8) > MI325X (6) > MI300X/A (5.3) > H200 (4.8) > H20 (4.0) > H100 SXM (3.35) > A100 (2.04) TB/s
- **FP8 算力榜（稠密）**：B200 (4.5P) > MI300X/MI325X (2.6P) > H100 SXM (1.98P) > L40S (0.73P)
- **对华特供识别**：H20（砍算力）、H800/A800（砍 NVLink）、L20（Ada 特供）

---

*数据来源：各厂商官方 Datasheet 与架构白皮书（见 [README.md](./GPUs-Specs/README.md) 链接）。规格以官方峰值理论值为准，不同批次/固件/散热条件下实测会有差异。*
