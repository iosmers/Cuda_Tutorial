# 阶段二：CUDA 内存模型与调试

> **目标**：掌握 CUDA 内存层次，会用 shared memory 优化，会调试。

---

## 1. CUDA 内存层次全景

| 内存类型 | 作用域 | 生命周期 | 延迟 | 大小 | 典型用途 |
|---------|-------|---------|-----|------|---------|
| **Register** | 线程私有 | 线程 | ~1 cycle | 每 SM 几万个 | 局部变量 |
| **Local** | 线程私有 | 线程 | 高（实际在 global） | - | 寄存器溢出 |
| **Shared** | Block 内共享 | Block | ~20 cycles | 每 SM 48~164 KB | Block 内协作、缓存 |
| **Global** | 所有线程 | Application | 400~800 cycles | 几 GB | 主要数据存储 |
| **Constant** | 所有线程只读 | Application | 缓存命中极快 | 64 KB | 只读参数 |
| **Texture** | 所有线程只读 | Application | 有空间局部性缓存 | - | 图像采样 |

**优化关键**：**减少 global 访问**，**多用 shared memory 复用数据**。

---

## 2. Shared Memory 与同步

### 声明方式

```cpp
__global__ void kernel() {
    __shared__ float tile[16][16];         // 静态分配
    // 或动态分配：
    extern __shared__ float sdata[];       // kernel<<<g,b,bytes>>>
}
```

### 同步原语
```cpp
__syncthreads();   // Block 内所有线程等齐（必须所有线程都执行到）
```

**陷阱**：`__syncthreads()` 绝不能放在**分支内**（只有部分线程执行会死锁）。

---

## 3. Bank Conflict（先理解概念）

- Shared memory 被分成 **32 个 bank**（对应 warp 的 32 线程）
- 同一 warp 内不同线程同时访问**同一 bank 的不同地址** → 串行化（冲突）
- 访问**同一地址**（广播）或**不同 bank** → 并行

例：`sdata[threadIdx.x]` 一般无冲突；`sdata[threadIdx.x * 2]` 可能冲突。

---

## 4. 错误处理：`CUDA_CHECK` 宏

CUDA API 失败默认不抛异常。必须手动检查。

```cpp
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// kernel 启动后检查
CUDA_CHECK(cudaGetLastError());       // 启动错误
CUDA_CHECK(cudaDeviceSynchronize());  // 执行期错误
```

---

## 5. 调试工具

```bash
# 越界 / 未初始化访存检查（替代 cuda-memcheck）
compute-sanitizer ./my_app

# 检查特定类型
compute-sanitizer --tool memcheck  ./my_app   # 越界
compute-sanitizer --tool racecheck ./my_app   # shared mem 数据竞争
```

---

## 6. 性能计时：`cudaEvent_t`

```cpp
cudaEvent_t s, e;
cudaEventCreate(&s); cudaEventCreate(&e);
cudaEventRecord(s);
kernel<<<g, b>>>(...);
cudaEventRecord(e);
cudaEventSynchronize(e);
float ms;
cudaEventElapsedTime(&ms, s, e);
cudaEventDestroy(s); cudaEventDestroy(e);
```

比 `clock()` 准确，因为它测量的是 GPU 时间线。

---

## 7. 代码练习

### 练习 1：矩阵乘法 - Naive 版

计算 `C = A * B`，其中 A 是 MxK，B 是 KxN，C 是 MxN。

```cpp
__global__ void matMulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 启动
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

**问题**：每个 C 元素都要从 global memory 读 2K 个 float，**数据复用为 0**。

---

### 练习 2：矩阵乘法 - Tiled 版（Shared Memory）

思想：把 A、B 按 `TILE x TILE` 分块加载到 shared memory，Block 内 `TILE*TILE` 个线程复用。

```cpp
#define TILE 16

__global__ void matMulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // 协作加载一个 tile
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();                        // 等 tile 加载完

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();                        // 等大家算完再加载下一块
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**性能对比**：在 1024x1024x1024 矩阵上，Tiled 版通常比 Naive 快 **3~8 倍**。

---

### 练习 3：矩阵转置（体会合并访存）

Naive 转置会写非合并，先实现它，再用 shared memory 解决。

```cpp
// Naive：读合并，写不合并
__global__ void transposeNaive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // in 的列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // in 的行
    if (x < N && y < M)
        out[x * M + y] = in[y * N + x];
}

// Shared memory 版：读写都合并
#define BDIM 32
__global__ void transposeShared(const float* in, float* out, int M, int N) {
    __shared__ float tile[BDIM][BDIM + 1];   // +1 消除 bank conflict

    int x = blockIdx.x * BDIM + threadIdx.x;
    int y = blockIdx.y * BDIM + threadIdx.y;
    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // 重新计算写出坐标（交换 block）
    x = blockIdx.y * BDIM + threadIdx.x;
    y = blockIdx.x * BDIM + threadIdx.y;
    if (x < M && y < N)
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
}
```

**关键点**：
- `+1` padding 技巧：消除 shared memory bank conflict
- 写出时以"目标块"视角重算索引，保证写也是合并的

---

## 8. 性能测量：计算 GFLOPS

矩阵乘法浮点次数 = `2 * M * N * K`（每个输出元素 K 次乘加 = 2K flops）

```cpp
double flops = 2.0 * M * N * K;
double gflops = flops / (ms * 1e6);        // ms→s: /1000；flops→Gflops: /1e9
printf("%.2f GFLOPS\n", gflops);
```

---

## 9. 阶段产出：性能对比表

建议在 README 里填一张表（示例，实际以你机器为准）：

| 实验 | 规模 | Naive (ms) | Optimized (ms) | 加速比 | GFLOPS |
|------|------|-----------|----------------|--------|--------|
| MatMul | 1024³ | 25.3 | 4.8 | 5.3× | 447 |
| Transpose | 4096² | 3.2 | 0.9 | 3.6× | - |

---

## 10. 自检清单

- [ ] 能说出 6 种内存的作用域与相对延迟
- [ ] 理解 Bank Conflict 的产生条件与消除技巧
- [ ] 会写 `CUDA_CHECK` 宏并在所有 API 调用处使用
- [ ] 能用 `compute-sanitizer` 发现越界 bug
- [ ] Tiled matmul 结果与 Naive 一致且更快
- [ ] Transpose 的 shared memory 版本无 bank conflict（`+1` padding）
- [ ] 能用 `cudaEvent` 准确计时并算出 GFLOPS
