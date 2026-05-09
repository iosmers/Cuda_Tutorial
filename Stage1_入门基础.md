# 阶段一：CUDA 入门基础

> **目标**：理解 GPU 并行模型，能写并运行第一个 kernel。

---

## 1. 前置知识：为什么需要 GPU？

### CPU vs GPU 架构差异

| 维度 | CPU | GPU |
|------|-----|-----|
| 核心数 | 少（4~64），复杂核心 | 多（上千），简单核心 |
| 优化目标 | 低延迟（单线程快） | 高吞吐（大量线程并行） |
| 适合场景 | 串行、分支多、复杂逻辑 | 数据并行、相同操作作用于大量数据 |
| 缓存 | 大 L1/L2/L3 | 小，但显存带宽极高 |

**关键理念**：GPU 通过"**大量线程隐藏内存延迟**"达到高吞吐，而不是靠单线程快。

---

## 2. CUDA 编程模型

### 核心层级
```
Grid（网格）
├── Block（线程块）  ← 同一 Block 内线程可共享 shared memory、可同步
│   ├── Thread（线程）
│   ├── Thread
│   └── ...
└── Block
    └── ...
```

### 硬件对应
- **SM (Streaming Multiprocessor)**：GPU 上的物理核心集群，一个 Block 被调度到一个 SM 上执行
- **Warp**：32 个线程组成的执行单元，SM 以 warp 为粒度调度，warp 内 32 线程**同指令同步执行**（SIMT）

### 关键限制（记住数量级）
- 每个 Block 最多 1024 线程
- Warp 大小固定为 32
- Grid 和 Block 都可以是 1D/2D/3D

---

## 3. 关键语法

### 函数修饰符
| 修饰符 | 调用位置 | 执行位置 |
|--------|---------|---------|
| `__global__` | Host | Device（即 kernel） |
| `__device__` | Device | Device |
| `__host__` | Host | Host（默认） |

### Kernel 启动语法
```cpp
kernel_name<<<gridDim, blockDim>>>(args...);
```
- `gridDim`：Block 的数量（可为 dim3）
- `blockDim`：每个 Block 的线程数（可为 dim3）

### 内置变量（kernel 内可用）
- `threadIdx.x/y/z`：线程在 Block 内的索引
- `blockIdx.x/y/z`：Block 在 Grid 内的索引
- `blockDim.x/y/z`：每个 Block 的维度
- `gridDim.x/y/z`：Grid 的维度

### 线程全局索引计算（最常用）
```cpp
// 1D
int i = blockIdx.x * blockDim.x + threadIdx.x;

// 2D（如图像处理）
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## 4. 内存管理三板斧

```cpp
float *d_A;
cudaMalloc(&d_A, N * sizeof(float));              // 在 GPU 分配
cudaMemcpy(d_A, h_A, N*sizeof(float),
           cudaMemcpyHostToDevice);                // CPU → GPU
// ... kernel 调用 ...
cudaMemcpy(h_A, d_A, N*sizeof(float),
           cudaMemcpyDeviceToHost);                // GPU → CPU
cudaFree(d_A);                                     // 释放
```

**记忆要点**：GPU 有独立显存，任何数据都要显式拷贝过去。

---

## 5. 环境准备

```bash
# 查看驱动 + CUDA 版本
nvidia-smi
nvcc --version

# 编译运行
nvcc hello.cu -o hello
./hello
```

---

## 6. 代码练习

### 练习 1：Hello World from GPU

```cpp
// hello.cu
#include <cstdio>

__global__ void hello() {
    printf("Hello from thread %d in block %d\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    hello<<<2, 4>>>();           // 2 个 block，每 block 4 个线程 = 8 个线程
    cudaDeviceSynchronize();     // 等待 GPU 完成，否则 printf 可能不输出
    return 0;
}
```
**要点**：`cudaDeviceSynchronize()` 必须加，否则 host 程序提前退出 printf 丢失。

---

### 练习 2：向量加法 `C = A + B`

```cpp
// vec_add.cu
#include <cstdio>
#include <cstdlib>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];   // 边界判断必不可少
}

int main() {
    const int N = 1 << 20;           // 1M 元素
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (N + block - 1) / block;   // 向上取整
    vecAdd<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 验证
    bool ok = true;
    for (int i = 0; i < N; ++i) if (h_C[i] != 3.0f) { ok = false; break; }
    printf("Result: %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

**思考题**：
- 如果 `N` 不是 256 的整数倍，会发生什么？（答：`if (i < N)` 防止越界）
- 改用 `block = 1024`，grid 怎么变？

---

### 练习 3：SAXPY `y = a*x + y`

```cpp
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

**练习**：在 main 里加上 `cudaEvent_t` 计时，输出 GB/s（带宽）。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
cudaEventRecord(start);
saxpy<<<grid, block>>>(N, 2.0f, d_x, d_y);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms; cudaEventElapsedTime(&ms, start, stop);
// SAXPY 读 2N + 写 N = 3N 个 float
float gbps = (3.0f * N * sizeof(float)) / (ms * 1e6f);
printf("%.2f GB/s\n", gbps);
```

---

### 练习 4：矩阵加法（2D Grid/Block）

```cpp
__global__ void matAdd(const float* A, const float* B, float* C, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 行
    if (x < N && y < M) {
        int idx = y * N + x;                        // 行主序
        C[idx] = A[idx] + B[idx];
    }
}

// 启动
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matAdd<<<grid, block>>>(d_A, d_B, d_C, M, N);
```

**要点**：2D 线程组织更贴合图像/矩阵的结构，索引计算仍是 `row * width + col`。

---

## 7. 阶段产出

建立项目目录：
```
cuda-learning/
├── 01_hello/hello.cu
├── 02_vec_add/vec_add.cu
├── 03_saxpy/saxpy.cu
├── 04_mat_add/mat_add.cu
├── Makefile
└── README.md
```

`Makefile` 示例：
```makefile
NVCC := nvcc
FLAGS := -O2 -arch=sm_70
all:
	$(NVCC) $(FLAGS) 01_hello/hello.cu -o 01_hello/hello
	$(NVCC) $(FLAGS) 02_vec_add/vec_add.cu -o 02_vec_add/vec_add
	# ...
```

---

## 8. 自检清单

- [ ] 能解释 Grid/Block/Thread/Warp 关系
- [ ] 能独立写线程索引公式（1D & 2D）
- [ ] 理解 `__global__` 的调用/执行位置
- [ ] 会用 `cudaMalloc/Memcpy/Free` 三件套
- [ ] 知道为什么要 `cudaDeviceSynchronize()`
- [ ] 4 个练习全部通过 CPU 结果验证
