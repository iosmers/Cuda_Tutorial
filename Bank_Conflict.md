# Bank Conflict（共享内存 Bank 冲突）

> **关联练习**：阶段二「矩阵转置」、阶段三「并行归约」——Bank Conflict 只发生在访问 **`__shared__`（Shared Memory）** 时，与 Global Memory 的**合并访存（Coalesced Access）**是两套独立规则。详见 [CoalesedMemoryAccess.md](CoalesedMemoryAccess.md)、[MatrixTranspose.md](MatrixTranspose.md)。

---

## 1. 概念：什么是 Bank Conflict？

Shared Memory 是 SM 上的一块**低延迟、高带宽**片上 SRAM，由同一线程块（Block）内的线程共享。为了支持一个 Warp（32 个线程）在同一周期内并行访问，硬件把 Shared Memory 划分为 **32 个 Bank**（与 Warp 宽度一致）。

典型规则（以 `float` 为例，每个元素 4 字节）：

- 每个 Bank 宽度为 **4 字节**（32 bit）
- 字节地址 `addr` 对应的 Bank 编号：`bank = (addr / 4) % 32`
- **同一 Warp** 内，若多个线程在同一时刻访问**不同地址但落在同一 Bank** → 硬件无法并行完成 → **Bank Conflict**，请求被**串行化**

一句话总结：

> **同一 Warp 内，32 个线程各访问不同 Bank → 1 拍完成；32 个线程挤在同一 Bank → 最多 32 拍串行，带宽暴跌。**

> **注意**：Bank Conflict **只**讨论 Shared Memory。Global Memory 看的是地址是否连续、能否合并，见 [CoalesedMemoryAccess.md](CoalesedMemoryAccess.md)。

---

## 2. 硬件视角：三种访问结果

| 访问模式 | 同一 Warp 内 32 线程的行为 | 结果 |
|----------|---------------------------|------|
| 各访问**不同 Bank** | 32 个不同 Bank 并行服务 | ✅ 无冲突，1 周期 |
| 各访问**同一地址** | 广播（Broadcast）给所有线程 | ✅ 无冲突 |
| 各访问**同一 Bank 的不同地址** | 硬件排队、串行服务 | ❌ **N-way Bank Conflict**（N = 同 Bank 请求数） |

**关键**：冲突的判断单位是 **Warp**，不是整个 Block。不同 Warp 之间的 Bank 访问互不影响。

**如何选 Warp 分析**：CUDA 中 `threadIdx.x` 变化最快，因此 `threadIdx.y` 相同、`threadIdx.x = 0..31` 的 32 个线程通常属于同一 Warp（一维 block 时就是 `threadIdx.x` 连续的 32 个线程）。

---

## 3. Bank 编号：先会算，再会避

Shared Memory 按 4 字节一格划入 32 个 Bank，循环使用：

```text
字节偏移:  0    4    8   ...  124   128   132  ...
Bank:      0    1    2   ...   31     0     1   ...
           └──── 32 格 ────┘   └── 又从 Bank 0 开始
```

对 `float` 数组 `sdata[]`，第 `i` 个元素的 Bank：

```text
bank(i) = i % 32
```

对二维数组 `tile[row][col]`（行主序，行宽为 `WIDTH`）：

```text
float 下标 = row * WIDTH + col
bank       = (row * WIDTH + col) % 32
```

后面所有例子，都归结为：**同一 Warp 里，32 个线程算出的 `bank` 是否相同**。

---

## 4. 举例说明

### 例 1：无冲突 —— 连续下标 `sdata[threadIdx.x]`

```cpp
__shared__ float sdata[256];

// 假设 blockDim.x = 256，考察前 32 个线程（一个 Warp）
int tid = threadIdx.x;          // 0, 1, 2, ..., 31
float v = sdata[tid];           // 访问 sdata[0], sdata[1], ..., sdata[31]
```

| 线程 `tid` | float 下标 | Bank |
|------------|-----------|------|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| … | … | … |
| 31 | 31 | 31 |

32 个线程命中 **32 个不同 Bank** → **无冲突**。这是 Shared Memory 最理想的访问模式之一。

---

### 例 2：2-way 冲突 —— 步长为 2 的访问 `sdata[threadIdx.x * 2]`

```cpp
__shared__ float sdata[256];

int tid = threadIdx.x;          // 0, 1, 2, ..., 31
float v = sdata[tid * 2];       // 访问 sdata[0], sdata[2], sdata[4], ...
```

| 线程 `tid` | float 下标 | Bank = 下标 % 32 |
|------------|-----------|------------------|
| 0 | 0 | 0 |
| 1 | 2 | 2 |
| 2 | 4 | 4 |
| … | … | … |
| 15 | 30 | 30 |
| 16 | 32 | **0** ← 与 tid=0 同 Bank |
| 17 | 34 | **2** ← 与 tid=1 同 Bank |
| … | … | … |
| 31 | 62 | **30** |

前 16 个线程占用 Bank 0, 2, 4, …, 30；后 16 个线程再次占用同一组 Bank。每个 Bank 上有 **2 个请求** → **2-way Bank Conflict**，硬件把并行度减半。

**规律**：步长为 `S` 时，若 `gcd(S, 32) = G` 且 `G > 1`，同一 Warp 内会有 **G-way** 冲突。这里 `S = 2`，`gcd(2, 32) = 2`。

---

### 例 3：32-way 冲突 —— 矩阵转置按列读 `tile[32][32]`

这是 CUDA 教程里**最经典**的 Bank Conflict 案例，详见 [MatrixTranspose.md](MatrixTranspose.md) 第 4 节。

```cpp
#define TILE 32
__shared__ float tile[TILE][TILE];   // 无 padding

// 阶段 B：从 shared 按列读出（转置方向）
int tx = threadIdx.x;   // 0..31
int ty = threadIdx.y;   // 固定，例如 0
float v = tile[tx][ty]; // 读 tile[0][0], tile[1][0], ..., tile[31][0]
```

一个 Warp：`threadIdx.y` 相同，`threadIdx.x = 0..31`。

| 线程 `tx` | 访问元素 | float 下标 = tx * 32 + ty | Bank（ty=0 时） |
|-----------|----------|---------------------------|-----------------|
| 0 | tile[0][0] | 0 | 0 |
| 1 | tile[1][0] | 32 | 0 |
| 2 | tile[2][0] | 64 | 0 |
| … | … | … | 0 |
| 31 | tile[31][0] | 992 | 0 |

相邻两行间隔 **32 个 float** → Bank 编号差 `32 % 32 = 0` → **每一列上的元素永远落在同一 Bank**。

```text
无 padding，tile[32][32] 固定 col=0 按列看：

  row 0  tile[0][0]  ─┐
  row 1  tile[1][0]   │  32 个元素
  row 2  tile[2][0]   │  全在 Bank 0
  ...                 │
  row31  tile[31][0] ─┘
```

**32 个线程全部落在 Bank 0** → **32-way Bank Conflict**，Shared Memory 有效带宽约降为原来的 **1/32**。

**根因**：行宽 32 恰好等于 Bank 数量 32，行与行之间地址差是 32 的整数倍，列方向访问时 Bank 编号不变。

---

### 例 4：Padding 消除转置冲突 —— `tile[32][33]`

在列方向多加 1 个元素，**不参与计算**，只用来「垫高」行宽，打破 Bank 对齐：

```cpp
__shared__ float tile[TILE][TILE + 1];   // 行宽 33，不是 32

float v = tile[tx][ty];   // 仍按列读
```

此时 float 下标 = `tx * 33 + ty`，Bank = `(tx * 33 + ty) % 32`。

因为 `33 % 32 = 1`，所以：

```text
Bank = (tx * 33 + ty) % 32 = (tx + ty) % 32
```

`ty` 固定、`tx` 从 0 到 31 时，`(tx + ty) % 32` 遍历 0～31 各一次：

```text
有 padding，tile[32][33] 固定 col=0 按列看：

  row 0  tile[0][0]  → Bank 0
  row 1  tile[1][0]  → Bank 1
  row 2  tile[2][0]  → Bank 2
  ...
  row31  tile[31][0] → Bank 31
```

| | `tile[32][32]` | `tile[32][33]`（+1 padding） |
|--|----------------|-------------------------------|
| 行宽（float） | 32 | 33 |
| 相邻两行同一列的 Bank 差 | 32 mod 32 = **0** | 33 mod 32 = **1** |
| Warp 按列读 | 32 线程 → 1 个 Bank | 32 线程 → 32 个 Bank |
| 结果 | 32-way 冲突 | ✅ 无冲突 |

阶段 A 按行写 `tile[ty][tx]` 时，`tx` 连续 → Bank 本来就连续，**padding 不会破坏行方向访问**；它专门修的是阶段 B 按列读时的冲突。

---

### 例 5：归约中的冲突 —— Interleaved vs Sequential

阶段三归约练习中，两种寻址方式对比鲜明（见 [stage3_性能优化.md](stage3_性能优化.md)）。

**有冲突的 Interleaved 寻址（v1 思想）**：

```cpp
__shared__ float sdata[256];
int tid = threadIdx.x;

// 第一轮：s = 1，活跃线程 tid = 0, 2, 4, ..., 30
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

当 `s = 16` 时，活跃线程 `tid = 0, 32, 64, …`（若 block 更大）。在一个 Warp 内看 `tid = 0` 与 `tid = 32`：

- 读 `sdata[0]` 与 `sdata[32]` → Bank 0 与 Bank 0 → **冲突**
- 读 `sdata[16]` 与 `sdata[48]` → 同理

线程索引间隔大、且为 2 的幂时，容易让同一 Warp 内多个线程撞上同一 Bank。

**无冲突的 Sequential 寻址（v3）**：

```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];   // tid 与 tid+s 在前半/后半，Bank 错开
    }
    __syncthreads();
}
```

当 `s = 16`、`tid = 0..15` 时，线程 `tid` 访问 `sdata[tid]` 和 `sdata[tid + 16]`：

| tid | 读 sdata[tid] Bank | 读 sdata[tid+16] Bank |
|-----|-------------------|----------------------|
| 0 | 0 | 16 |
| 1 | 1 | 17 |
| … | … | … |
| 15 | 15 | 31 |

同一 Warp 内 16 个活跃线程，每次访问的两个地址落在**不同 Bank**，且无两线程争抢同一 Bank → **无冲突**。这也是现代归约实现更推荐 sequential addressing 的原因之一。

---

### 例 6：广播 —— 多线程读同一地址（无冲突）

```cpp
__shared__ float sdata[256];
float v = sdata[0];   // 整个 Warp 32 个线程都读 sdata[0]
```

32 个线程访问**同一地址、同一 Bank** → 硬件做 **Broadcast**，一次读出、广播给所有 lane → **不算 Bank Conflict**。

这在归约写回 `sdata[0]`、或所有线程读同一配置常量时很常见。

---

## 5. 如何发现与验证 Bank Conflict？

### 5.1 手工分析（写 kernel 前）

1. 选定一个 Warp（通常 `threadIdx.x` 连续 32 个线程）
2. 写出每个线程访问的 Shared Memory 地址或 float 下标
3. 计算 `bank = (字节偏移 / 4) % 32`
4. 统计同一周期内、同一 Bank 上有几个不同地址 → 即为 **N-way conflict**

### 5.2 工具确认（优化后）

```bash
# Nsight Compute：查看 Shared Load/Store Bank Conflicts 相关指标
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./your_app
```

或在 Nsight Compute GUI 中查看 **Shared Memory Bank Conflicts**、**L1 Wavefronts Shared Excessive** 等。优化前后对比该指标，比单纯看总时间更能确认「是不是 Bank 在拖后腿」。

`compute-sanitizer --tool racecheck` 用于检测 **数据竞争**，不能替代 Bank Conflict 分析。

---

## 6. 消除 Bank Conflict 的常用手段

| 手段 | 做法 | 适用场景 |
|------|------|----------|
| **改访问模式** | 让相邻线程访问相邻地址（例 1、归约 v3） | 能改算法或循环结构时首选 |
| **Padding** | `tile[T][T+1]`，行宽与 32 互质 | 二维 tile 按列访问（例 4） |
| **重塑数组** | 一维 `sdata[T*T]` 手动算下标，控制 Bank 分布 | 灵活但易错 |
| **拆成多次访问** | 用 `float4` 或循环分 phase，每 phase 无冲突 | 部分 GEMM / 卷积实现 |
| **避开 Shared** | Warp Shuffle（`__shfl_down_sync`）在寄存器间交换 | 归约、扫描等 warp 内通信 |

**Padding 经验**：`TILE_DIM` 为 32 的倍数时，加 **1** 列通常即可（`[32][33]`）；更一般地，行宽 `W` 满足 `gcd(W, 32) = 1` 即可让「行号每 +1，Bank 编号 +1」，从而打散列访问。

---

## 7. Bank Conflict vs 合并访存（对照）

| | **Bank Conflict** | **合并访存（Coalesced）** |
|--|-------------------|---------------------------|
| 发生层次 | Shared Memory（`__shared__`） | Global Memory（HBM） |
| 分析单位 | 同一 **Warp** | 同一 **Warp** |
| 理想模式 | 32 线程 → 32 个不同 Bank | 32 线程 → 32 个连续地址 |
| 典型翻车 | 转置 `tile[32][32]` 按列读 | 行主序矩阵按列读 global |
| 经典修复 | `+1` padding | Tiling + Shared Memory 中转 |

矩阵转置同时要修两边：用 Shared Memory 分块修 **Global 合并**；用 `+1` padding 修 **Shared Bank Conflict**。两个问题独立，需分别分析。

---

## 8. 自检清单

- [ ] 能说出 Bank 编号公式：`(addr / 4) % 32`
- [ ] 能解释为何 `sdata[threadIdx.x]` 无冲突、`sdata[threadIdx.x * 2]` 有 2-way 冲突
- [ ] 能推导 `tile[32][32]` 按列读为何 32-way 冲突
- [ ] 能说明 `tile[32][33]` 的 padding 如何打散 Bank
- [ ] 能区分 Bank Conflict（Shared）与合并访存（Global）
- [ ] 知道用 Nsight Compute 查看 Shared Bank Conflict 指标

---

## 9. 延伸阅读

| 资料 | 内容 |
|------|------|
| [MatrixTranspose.md](MatrixTranspose.md) | 转置完整链路：Naive → Tiling → Padding |
| [CoalesedMemoryAccess.md](CoalesedMemoryAccess.md) | Global Memory 合并访存 |
| [stage2_内存模型与调试.md](stage2_内存模型与调试.md) | Shared Memory 与同步、转置练习 |
| [stage3_性能优化.md](stage3_性能优化.md) | 归约 7 版：Interleaved vs Sequential |
| [NVIDIA 官方转置博客](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) | Naive → Coalesced → No Bank Conflict |
