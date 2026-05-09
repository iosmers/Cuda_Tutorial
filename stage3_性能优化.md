# 阶段三：CUDA 性能优化

> **目标**：能独立识别瓶颈并优化 kernel。

---

## 1. 性能思维：瓶颈分类

一个 kernel 的性能通常被限制在其中之一：

- **Memory Bound（访存瓶颈）**：计算很少，等内存。常见于 SAXPY、reduction、transpose。
- **Compute Bound（计算瓶颈）**：ALU 跑满，内存够用。常见于稠密 matmul、卷积。
- **Latency Bound（延迟瓶颈）**：并行度不够（occupancy 低），SM 空转。

优化的第一步永远是：**用 profiler 确认瓶颈类型**，再对症下药。

---

## 2. 访存优化：Coalesced Access

GPU 以 **32 线程（一个 warp）** 为单位访问 global memory，如果 32 线程访问的地址**连续且对齐**，硬件合并成 1~2 次事务。

### 合并 vs 非合并

```cpp
// 合并：相邻线程访问相邻地址
int i = blockIdx.x * blockDim.x + threadIdx.x;
float v = a[i];

// 非合并：stride=2，warp 访问分散
float v = a[i * 2];
```

### AoS vs SoA

```cpp
// Array of Structures（对 GPU 不友好）
struct Particle { float x, y, z, vx, vy, vz; };
Particle particles[N];        // 读 x 时会顺带读到 y,z,vx... 浪费带宽

// Structure of Arrays（GPU 友好）
struct Particles {
    float *x, *y, *z, *vx, *vy, *vz;    // 每个属性独立数组
};
```
**规则**：GPU 代码优先 SoA。

---

## 3. Warp 级原语

同一 warp 的 32 线程可以用 **shuffle** 指令直接交换寄存器数据，不经过 shared memory。

```cpp
// 从 warp 内 lane=srcLane 的线程读取 val
int v = __shfl_sync(0xffffffff, val, srcLane);

// 向下偏移（lane i 拿到 lane i+delta 的 val）
int v = __shfl_down_sync(0xffffffff, val, delta);

// warp 内求和（32 线程）
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// 此时 lane 0 的 val 就是 32 个值之和
```

### Warp Divergence
```cpp
if (threadIdx.x % 2 == 0) { /* A */ }
else                      { /* B */ }
// 同一 warp 内线程走不同分支 → 串行执行 A、B，性能减半
```
**尽量让一个 warp 内线程走同一分支**。

---

## 4. 原子操作

```cpp
atomicAdd(&counter, 1);     // 正确但性能受冲突严重影响
```
- 冲突少 → 几乎无代价
- 所有线程打一个地址 → 完全串行，极慢
- **策略**：先在 shared memory 局部聚合，再 global atomic 合并

---

## 5. Stream 与并发

默认 stream 下所有操作串行。多 stream 可实现 **H2D 拷贝 / kernel / D2H 拷贝** 三者重叠。

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);

// 异步拷贝必须使用 pinned host memory
float *h_buf; cudaMallocHost(&h_buf, bytes);

cudaMemcpyAsync(d1, h_buf,    sz, cudaMemcpyHostToDevice, s1);
kernel<<<g, b, 0, s1>>>(d1);
cudaMemcpyAsync(h_out, d1,    sz, cudaMemcpyDeviceToHost, s1);

// s2 可同时处理另一块数据
cudaStreamSynchronize(s1);
```

**前提**：host 侧必须是 **pinned memory**（`cudaMallocHost`），否则退化为同步拷贝。

---

## 6. Profiling 工具

```bash
# 系统级：看 kernel / copy 时间线
nsys profile -o report ./my_app

# Kernel 级：看指令、内存、occupancy
ncu --set full -o report ./my_app
```

关键指标：
- **SM Occupancy**：活跃 warp / 最大 warp，越高越能隐藏延迟
- **Memory Throughput**：是否打满带宽
- **Warp Stall Reasons**：为什么 warp 在等

---

## 7. 代码练习

### 练习 1：并行归约（Reduction）7 版优化

求 `sum = sum(a[0..N-1])`。

**v1 - Interleaved addressing（有 bank conflict + divergence）**
```cpp
__global__ void reduce_v1(const float* in, float* out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0)              // 大量 warp divergence
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
```

**v2 - 解决 divergence**
```cpp
for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x)
        sdata[index] += sdata[index + s];
    __syncthreads();
}
```

**v3 - Sequential addressing（消除 bank conflict）**
```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

**v4 - 首次加载时做一次加法（线程减半）**
```cpp
int i = blockIdx.x * (blockDim.x * 2) + tid;
sdata[tid] = (i < N ? in[i] : 0) + (i + blockDim.x < N ? in[i + blockDim.x] : 0);
__syncthreads();
// 后续同 v3
```

**v5 - 展开最后一个 warp（无需同步）**
```cpp
for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
if (tid < 32) {
    volatile float* v = sdata;
    v[tid] += v[tid + 32]; v[tid] += v[tid + 16];
    v[tid] += v[tid + 8];  v[tid] += v[tid + 4];
    v[tid] += v[tid + 2];  v[tid] += v[tid + 1];
}
```

**v6 - 使用 Warp Shuffle（现代版首选）**
```cpp
__global__ void reduce_shuffle(const float* in, float* out, int N) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = (i < N) ? in[i] : 0.0f;

    // warp 内归约
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);

    // 每个 warp 的 lane 0 把结果写 shared
    __shared__ float warpSum[32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) warpSum[wid] = v;
    __syncthreads();

    // 第一个 warp 再归约 warpSum
    if (wid == 0) {
        v = (tid < blockDim.x / 32) ? warpSum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) out[blockIdx.x] = v;
    }
}
```

**v7 - 多元素 per thread + grid-stride loop**：每线程处理多个元素，提高算术密度。

**实验要求**：
- 用 `cudaEvent` 测每版时间
- 记录 Bandwidth（GB/s）：`N * sizeof(float) / time`
- 观察 v1→v7 带宽如何一步步逼近硬件峰值

---

### 练习 2：直方图（Histogram）

统计 `data[i]` 落在各 bin 的数量（如 256 个 bin）。

**v1 - 全局原子（慢）**
```cpp
__global__ void hist_v1(const uint8_t* data, int N, int* hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(&hist[data[i]], 1);     // 大量冲突
}
```

**v2 - Shared memory 局部聚合**
```cpp
__global__ void hist_v2(const uint8_t* data, int N, int* hist) {
    __shared__ int local[256];
    int tid = threadIdx.x;
    // 初始化 shared
    for (int j = tid; j < 256; j += blockDim.x) local[j] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i < N) atomicAdd(&local[data[i]], 1);   // Block 内原子，快得多
    __syncthreads();

    // 合并到全局
    for (int j = tid; j < 256; j += blockDim.x)
        atomicAdd(&hist[j], local[j]);
}
```

在实际数据上 v2 通常比 v1 快 **5~20 倍**。

---

### 练习 3：Stream 重叠

把大数组切成 N 块，用 2 个 stream 轮流处理：

```cpp
const int CHUNKS = 8;
int chunk = N / CHUNKS;
for (int i = 0; i < CHUNKS; ++i) {
    cudaStream_t s = streams[i & 1];    // 交替使用 2 个 stream
    cudaMemcpyAsync(d + i*chunk, h + i*chunk, chunk*sizeof(float),
                    cudaMemcpyHostToDevice, s);
    kernel<<<g, b, 0, s>>>(d + i*chunk, chunk);
    cudaMemcpyAsync(h_out + i*chunk, d + i*chunk, chunk*sizeof(float),
                    cudaMemcpyDeviceToHost, s);
}
cudaDeviceSynchronize();
```
用 `nsys` 查看 timeline，确认拷贝与计算在时间上是**重叠**的。

---

## 8. 阶段产出：Profiling 报告

对 reduction v1~v7 产出一份表格：

| 版本 | 时间 (ms) | 带宽 (GB/s) | 峰值占比 | 主要瓶颈 |
|------|----------|------------|---------|---------|
| v1 | ... | ... | ...% | Divergence |
| v2 | ... | ... | ...% | Bank conflict |
| ... | | | | |
| v7 | ... | ... | >80% | 接近带宽上限 |

---

## 9. 自检清单

- [ ] 能判断一个 kernel 是 Memory / Compute / Latency Bound
- [ ] 理解合并访存的规则，能识别非合并模式
- [ ] 会用 `__shfl_down_sync` 写 warp 级归约
- [ ] 能解释 warp divergence 的代价
- [ ] 知道 `atomicAdd` 冲突热点的优化思路
- [ ] 能配置 pinned memory + 多 stream 实现重叠
- [ ] Reduction v1→v7 各版本都跑通并记录性能曲线
- [ ] 会用 Nsight Compute 看 Occupancy 与 Stall Reasons
