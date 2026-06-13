# 从 LeetGPU Prefix Sum 入手学习 CUDA：并行扫描（Scan）完整讲解

> **题目**：LeetGPU — Prefix Sum  
> **目标**：给定 GPU 上的 `float* input` 和长度 `N`，计算 inclusive prefix sum：
>
> ```text
> output[i] = input[0] + input[1] + ... + input[i]
> ```
>
> CUDA 提交接口：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N);
> ```
>
> 这篇不只是给一份能提交的代码，而是通过这道题学习 CUDA 里的几个核心概念：
>
> - thread / block / grid 的任务划分
> - global memory 与 shared memory
> - `__syncthreads()` 的作用和限制
> - 为什么普通 kernel 不能跨 block 同步
> - block 内并行 scan
> - 多 block 分层 scan
> - kernel launch 如何作为全局同步边界

---

## 1. 题目到底要求什么？

Prefix Sum 又叫 Scan。给一个数组：

```text
input = [1, 2, 3, 4]
```

输出是每个位置之前所有元素加上自己：

```text
output[0] = 1
output[1] = 1 + 2 = 3
output[2] = 1 + 2 + 3 = 6
output[3] = 1 + 2 + 3 + 4 = 10

output = [1, 3, 6, 10]
```

LeetGPU 这题是 **inclusive scan**，也就是输出包含当前元素。

再看一个带负数的例子：

```text
input  = [5, -2, 3, 1, -4]
output = [5,  3, 6, 7,  3]
```

因为：

```text
5
5 + (-2) = 3
5 + (-2) + 3 = 6
5 + (-2) + 3 + 1 = 7
5 + (-2) + 3 + 1 + (-4) = 3
```

---

## 2. CPU 串行写法很简单，但 GPU 不能直接照抄

CPU 上最自然的写法是：

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    sum += input[i];
    output[i] = sum;
}
```

这段代码的特点是：

```text
output[i] 依赖 output[i - 1]
```

也就是说它有很强的前后依赖。

如果直接在 GPU 上只开一个线程来跑这个循环：

```cpp
__global__ void badPrefixSum(const float* input, float* output, int N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += input[i];
            output[i] = sum;
        }
    }
}
```

这当然是正确的，但它几乎没有利用 GPU 的并行能力。

GPU 擅长的是：

```text
很多线程同时干活
```

所以 Prefix Sum 的核心问题是：

> 看起来有前后依赖的累加，如何拆成多个线程协作完成？

答案是使用并行 scan 算法。

---

## 3. Inclusive Scan 与 Exclusive Scan

学习 GPU scan 前，要先区分两个概念。

假设：

```text
input = [a, b, c, d]
```

### 3.1 Inclusive Scan

Inclusive scan 包含当前位置元素：

```text
inclusive = [a, a+b, a+b+c, a+b+c+d]
```

例子：

```text
input     = [1, 2, 3, 4]
inclusive = [1, 3, 6, 10]
```

LeetGPU 这题要的就是 inclusive scan。

### 3.2 Exclusive Scan

Exclusive scan 不包含当前位置元素，只累加它之前的元素：

```text
exclusive = [0, a, a+b, a+b+c]
```

例子：

```text
input     = [1, 2, 3, 4]
exclusive = [0, 1, 3, 6]
```

两者可以互相转换：

```text
inclusive[i] = exclusive[i] + input[i]
```

后面的代码会先用 Blelloch 算法得到 exclusive scan，再加回原始值，得到 inclusive scan。

---

## 4. CUDA 编程模型：这题会用到哪些 CUDA 知识？

### 4.1 `solve` 是 Host 函数

LeetGPU 给的接口是：

```cpp
extern "C" void solve(const float* input, float* output, int N);
```

这里的 `solve` 是在 CPU 上执行的 host 函数。

但是：

```text
input 和 output 是 device pointer
```

也就是说，`input` 和 `output` 指向 GPU 显存，不应该在 CPU 代码里直接写：

```cpp
// 不要这样做
float x = input[0];
```

正确做法是：

```cpp
// 在 solve 里启动 CUDA kernel
someKernel<<<grid, block>>>(input, output, N);
```

### 4.2 Kernel 是在 GPU 上执行的函数

CUDA kernel 用 `__global__` 声明：

```cpp
__global__ void myKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx];
    }
}
```

启动方式是：

```cpp
myKernel<<<gridDim, blockDim>>>(input, output, N);
```

其中：

| 名称 | 含义 |
|---|---|
| `threadIdx.x` | 当前线程在本 block 内的编号 |
| `blockIdx.x` | 当前 block 在 grid 内的编号 |
| `blockDim.x` | 每个 block 有多少个线程 |
| `gridDim.x` | grid 中有多少个 block |

常见的全局线程编号是：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

### 4.3 Global Memory 与 Shared Memory

CUDA 中常见两类内存：

| 内存 | 特点 | 本题用途 |
|---|---|---|
| Global Memory | 显存，大，但是慢；所有 block 都能访问 | `input`、`output`、`blockSums` |
| Shared Memory | 片上内存，小，但是快；一个 block 内线程共享 | block 内做 scan 的临时数组 |

本题的核心优化就是：

```text
先把一段 input 搬到 shared memory，
在 shared memory 中做并行 scan，
再写回 output。
```

### 4.4 `__syncthreads()` 只能同步同一个 block 内的线程

`__syncthreads()` 的意思是：

```text
当前 block 内所有线程都走到这里以后，大家再一起继续。
```

它不能同步不同 block。

这是 Prefix Sum 需要分层做的根本原因。

一个普通 CUDA kernel 中，不同 block 之间没有安全的全局同步。也就是说：

```text
block 0 不知道 block 1 执行到哪里了
block 1 也不知道 block 0 执行到哪里了
```

所以全数组 scan 通常不能只靠一个 kernel 完成。

我们会用多个 kernel：

```text
kernel 1：每个 block 扫自己的局部段
kernel 2：扫描每个 block 的总和
kernel 3：把 block 偏移量加回去
```

一个 kernel launch 结束后，再启动下一个 kernel，这就是天然的全局同步边界。

---

## 5. 单个 Block 内如何做 Prefix Sum？

我们先只考虑一个 block 扫一小段数据。

提交代码中使用：

```cpp
BLOCK_THREADS = 1024
ITEMS_PER_BLOCK = 2048
```

也就是：

```text
一个 block 有 1024 个线程
每个线程负责 2 个元素
一个 block 一共处理 2048 个元素
```

为什么每个线程处理 2 个元素？

1. 1024 是 NVIDIA GPU 常见的每 block 最大线程数。
2. 处理 2048 个元素可以减少 block 数量。
3. 2048 是 2 的幂，适合树形 scan。
4. shared memory 只需要 `2048 * sizeof(float) = 8192` 字节，开销不大。

---

## 6. Blelloch Scan：先变成树，再展开树

本题的 block 内 scan 使用经典的 **Blelloch Scan**。

它分两步：

```text
1. upsweep：从叶子往根归约，算出整段总和
2. downsweep：从根往叶子传播，得到 exclusive scan
```

最后：

```text
inclusive[i] = exclusive[i] + input[i]
```

为了方便手算，下面用一个玩具例子：

```text
ITEMS_PER_BLOCK = 8
THREADS = 4
input = [3, 1, 7, 0, 4, 1, 6, 3]
```

注意：真实提交代码用的是 2048 个元素，这里只是为了讲清楚。

---

### 6.1 初始状态

shared memory 中的数组：

```text
temp = [3, 1, 7, 0, 4, 1, 6, 3]
```

4 个线程各读两个元素：

| 线程 | 负责的元素 |
|---|---|
| thread 0 | `temp[0]`, `temp[4]` |
| thread 1 | `temp[1]`, `temp[5]` |
| thread 2 | `temp[2]`, `temp[6]` |
| thread 3 | `temp[3]`, `temp[7]` |

真实代码中对应：

```cpp
int i0 = base + threadIdx.x;
int i1 = base + BLOCK_THREADS + threadIdx.x;
```

---

### 6.2 Upsweep：构造归约树

第一轮，把相邻两个元素相加，结果放在右边：

```text
[3, 1, 7, 0, 4, 1, 6, 3]
    ↓     ↓     ↓     ↓
[3, 4, 7, 7, 4, 5, 6, 9]
```

解释：

```text
temp[1] = temp[1] + temp[0] = 1 + 3 = 4
temp[3] = temp[3] + temp[2] = 0 + 7 = 7
temp[5] = temp[5] + temp[4] = 1 + 4 = 5
temp[7] = temp[7] + temp[6] = 3 + 6 = 9
```

第二轮，每 4 个元素合并一次：

```text
[3, 4, 7, 7, 4, 5, 6, 9]
          ↓           ↓
[3, 4, 7, 11, 4, 5, 6, 14]
```

解释：

```text
temp[3] = temp[3] + temp[1] = 7 + 4 = 11
temp[7] = temp[7] + temp[5] = 9 + 5 = 14
```

第三轮，整个数组合并：

```text
[3, 4, 7, 11, 4, 5, 6, 14]
                           ↓
[3, 4, 7, 11, 4, 5, 6, 25]
```

现在最后一个元素 `25` 就是整个 block 的总和：

```text
3 + 1 + 7 + 0 + 4 + 1 + 6 + 3 = 25
```

在代码中，这个总和会写入：

```cpp
blockSums[blockIdx.x]
```

---

### 6.3 把最后一个元素置 0

Blelloch scan 的 downsweep 会生成 exclusive scan。

所以先把最后一个元素置为 0：

```text
[3, 4, 7, 11, 4, 5, 6, 25]
变成
[3, 4, 7, 11, 4, 5, 6, 0]
```

---

### 6.4 Downsweep：从树根往下传播

第一轮：

```text
[3, 4, 7, 11, 4, 5, 6, 0]
变成
[3, 4, 7, 0, 4, 5, 6, 11]
```

第二轮：

```text
[3, 4, 7, 0, 4, 5, 6, 11]
变成
[3, 0, 7, 4, 4, 11, 6, 16]
```

第三轮：

```text
[3, 0, 7, 4, 4, 11, 6, 16]
变成
[0, 3, 4, 11, 11, 15, 16, 22]
```

这就是 exclusive scan：

```text
exclusive = [0, 3, 4, 11, 11, 15, 16, 22]
```

检查一下：

```text
input[0] 前面没有元素                         = 0
input[1] 前面是 3                              = 3
input[2] 前面是 3 + 1                          = 4
input[3] 前面是 3 + 1 + 7                      = 11
input[4] 前面是 3 + 1 + 7 + 0                  = 11
input[5] 前面是 3 + 1 + 7 + 0 + 4              = 15
input[6] 前面是 3 + 1 + 7 + 0 + 4 + 1          = 16
input[7] 前面是 3 + 1 + 7 + 0 + 4 + 1 + 6      = 22
```

最后加回原始值：

```text
input     = [3, 1, 7, 0, 4, 1, 6, 3]
exclusive = [0, 3, 4, 11, 11, 15, 16, 22]
------------------------------------------------
inclusive = [3, 4, 11, 11, 15, 16, 22, 25]
```

---

## 7. 为什么还需要多 Block 分层扫描？

单个 block 最多只能处理一段数据。

本题 `N` 可能很大，LeetGPU 的约束中 `N` 可到很大，性能测试也不是只有几个元素。

所以我们要把数组切段：

```text
input = [ block0 ][ block1 ][ block2 ] ...
```

每个 block 先独立扫描自己的段。

但问题是：

```text
block1 的所有结果，都要加上 block0 的总和
block2 的所有结果，都要加上 block0 + block1 的总和
block3 的所有结果，都要加上 block0 + block1 + block2 的总和
```

因此需要保存每个 block 的总和：

```text
blockSums = [sum(block0), sum(block1), sum(block2), ...]
```

然后对 `blockSums` 再做一次 prefix sum。

---

## 8. 多 Block 示例 1：刚好分成多个段

为了方便手算，假设每个 block 只能处理 4 个元素。

真实代码每个 block 处理 2048 个元素。

输入：

```text
input = [1, 2, 3, 4,   10, 20, 30, 40,   5, 5]
```

切成三段：

```text
block0: [1, 2, 3, 4]
block1: [10, 20, 30, 40]
block2: [5, 5]
```

### 8.1 每个 block 先做局部 inclusive scan

```text
block0 local scan = [1, 3, 6, 10]
block1 local scan = [10, 30, 60, 100]
block2 local scan = [5, 10]
```

此时如果直接拼起来，会得到：

```text
[1, 3, 6, 10, 10, 30, 60, 100, 5, 10]
```

这还不是最终答案。

### 8.2 记录每个 block 的总和

```text
blockSums = [10, 100, 10]
```

### 8.3 对 blockSums 做 prefix sum

```text
scannedBlockSums = [10, 110, 120]
```

含义是：

```text
block0 结束后累计 10
block1 结束后累计 110
block2 结束后累计 120
```

### 8.4 给每个 block 加偏移量

每个 block 需要加的是它之前所有 block 的总和：

| block | 应该加的 offset |
|---|---:|
| block0 | 0 |
| block1 | `scannedBlockSums[0] = 10` |
| block2 | `scannedBlockSums[1] = 110` |

所以：

```text
block0: [1, 3, 6, 10] + 0
      = [1, 3, 6, 10]

block1: [10, 30, 60, 100] + 10
      = [20, 40, 70, 110]

block2: [5, 10] + 110
      = [115, 120]
```

最终结果：

```text
output = [1, 3, 6, 10, 20, 40, 70, 110, 115, 120]
```

你可以用 CPU 串行累加验证：

```text
1
1+2=3
1+2+3=6
1+2+3+4=10
10+10=20
20+20=40
40+30=70
70+40=110
110+5=115
115+5=120
```

---

## 9. 多 Block 示例 2：长度不是 block 容量的整数倍

仍然假设每个 block 只能处理 4 个元素。

输入：

```text
input = [1, 2, 3, 4, 5]
```

切段：

```text
block0: [1, 2, 3, 4]
block1: [5]
```

局部扫描：

```text
block0 local scan = [1, 3, 6, 10]
block1 local scan = [5]
```

block 总和：

```text
blockSums = [10, 5]
```

扫描 blockSums：

```text
scannedBlockSums = [10, 15]
```

加偏移量：

```text
block0 offset = 0
block1 offset = scannedBlockSums[0] = 10
```

最终：

```text
block0 = [1, 3, 6, 10]
block1 = [5] + 10 = [15]

output = [1, 3, 6, 10, 15]
```

代码中通过边界判断处理这种情况：

```cpp
if (i0 < N) { ... }
if (i1 < N) { ... }
```

超出 `N` 的位置在 shared memory 中补 0，不影响总和。

---

## 10. 多 Block 示例 3：带负数

输入：

```text
input = [5, -2, 3, 1,   -4, 10]
```

假设每 block 处理 4 个元素：

```text
block0: [5, -2, 3, 1]
block1: [-4, 10]
```

局部扫描：

```text
block0 local scan = [5, 3, 6, 7]
block1 local scan = [-4, 6]
```

block 总和：

```text
blockSums = [7, 6]
```

扫描 blockSums：

```text
scannedBlockSums = [7, 13]
```

偏移量：

```text
block0 offset = 0
block1 offset = scannedBlockSums[0] = 7
```

最终：

```text
block0 = [5, 3, 6, 7]
block1 = [-4, 6] + 7 = [3, 13]

output = [5, 3, 6, 7, 3, 13]
```

---

## 11. 对 LeetGPU 测试规模的理解

提交代码中：

```cpp
ITEMS_PER_BLOCK = 2048
```

如果 LeetGPU 测试：

```text
N = 250000
```

需要的 block 数大约是：

```text
ceil(250000 / 2048) = 123
```

流程就是：

```text
1. 用 123 个 block 扫 input，得到 output 的局部 scan 和 blockSums[123]
2. blockSums 只有 123 个元素，一个 block 就能扫完
3. 把 scannedBlockSums 作为 offset 加回原 output
```

所以对 `N = 250000`，这份代码通常只需要几次 kernel launch。

如果 `N = 100000000`：

```text
ceil(100000000 / 2048) = 48829
```

则递归结构大概是：

```text
input:      100000000 elements -> 48829 block sums
blockSums:      48829 elements ->    24 block sums
blockSums:         24 elements ->     1 block sum
```

依然是分层解决。

---

## 12. LeetGPU 可直接提交代码

下面这份代码是完整提交版。

```cpp
#include <cuda_runtime.h>

// 一个 block 用 1024 个线程处理 2048 个元素。
// 2048 是 2 的幂，方便使用 Blelloch scan。
static constexpr int BLOCK_THREADS = 1024;
static constexpr int ITEMS_PER_BLOCK = 2 * BLOCK_THREADS;

// -----------------------------------------------------------------------------
// scanBlockKernel
//
// 作用：
//   对每个 block 负责的一段 input 做 inclusive prefix sum。
//
// 输入：
//   input      : 原始数组，device pointer
//   output     : 输出数组，device pointer
//   blockSums  : 每个 block 的总和，可为 nullptr
//   N          : 数组长度
//
// 每个 block 处理 ITEMS_PER_BLOCK = 2048 个元素。
// 每个线程处理两个元素：
//   i0 = base + threadIdx.x
//   i1 = base + BLOCK_THREADS + threadIdx.x
// -----------------------------------------------------------------------------
__global__ void scanBlockKernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                float* __restrict__ blockSums,
                                int N) {
    // shared memory 是一个 block 内所有线程共享的快速片上内存。
    // 这里只存当前 block 负责的 2048 个元素。
    __shared__ float temp[ITEMS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int base = blockIdx.x * ITEMS_PER_BLOCK;

    const int i0 = base + tid;
    const int i1 = base + BLOCK_THREADS + tid;

    // 每个线程读取两个元素。
    // 如果越界，就补 0，这样不会影响 prefix sum。
    const float v0 = (i0 < N) ? input[i0] : 0.0f;
    const float v1 = (i1 < N) ? input[i1] : 0.0f;

    temp[tid] = v0;
    temp[BLOCK_THREADS + tid] = v1;

    // -------------------------------------------------------------------------
    // 第一阶段：upsweep / reduce
    //
    // 目标：构造一棵求和树，让 temp[ITEMS_PER_BLOCK - 1] 变成整段总和。
    // -------------------------------------------------------------------------
    int offset = 1;
    for (int d = ITEMS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        // 必须同步，确保上一层写入 temp 的结果已经完成。
        __syncthreads();

        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset <<= 1;
    }

    // 现在 temp[ITEMS_PER_BLOCK - 1] 是当前 block 负责区间的总和。
    // 把它保存到 blockSums，供后续跨 block 加偏移量使用。
    //
    // 然后把最后一个位置设为 0，准备做 downsweep。
    // 这一步是 Blelloch exclusive scan 的标准操作。
    if (tid == 0) {
        if (blockSums != nullptr) {
            blockSums[blockIdx.x] = temp[ITEMS_PER_BLOCK - 1];
        }
        temp[ITEMS_PER_BLOCK - 1] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // 第二阶段：downsweep
    //
    // 目标：把求和树展开，得到 exclusive prefix sum。
    // -------------------------------------------------------------------------
    for (int d = 1; d < ITEMS_PER_BLOCK; d <<= 1) {
        offset >>= 1;

        // 必须同步，确保上一层交换和累加完成。
        __syncthreads();

        if (tid < d) {
            const int ai = offset * (2 * tid + 1) - 1;
            const int bi = offset * (2 * tid + 2) - 1;

            const float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // 确保 downsweep 最后一层写入完成。
    __syncthreads();

    // temp 现在是 exclusive scan。
    // LeetGPU 要 inclusive scan，所以加回当前位置原始值。
    if (i0 < N) {
        output[i0] = temp[tid] + v0;
    }
    if (i1 < N) {
        output[i1] = temp[BLOCK_THREADS + tid] + v1;
    }
}

// -----------------------------------------------------------------------------
// addBlockOffsetsKernel
//
// 作用：
//   上一个 kernel 只完成了每个 block 内部的局部 scan。
//   这个 kernel 把前面 block 的累计和加到当前 block 的每个元素上。
//
// 对于第 b 个 block：
//   offset = scannedBlockSums[b - 1]
//
// 第 0 个 block 前面没有元素，所以 offset = 0，不需要加。
// -----------------------------------------------------------------------------
__global__ void addBlockOffsetsKernel(float* __restrict__ output,
                                      const float* __restrict__ scannedBlockSums,
                                      int N) {
    const int tid = threadIdx.x;
    const int block = blockIdx.x;

    if (block == 0) {
        return;
    }

    const float offset = scannedBlockSums[block - 1];
    const int base = block * ITEMS_PER_BLOCK;

    const int i0 = base + tid;
    const int i1 = base + BLOCK_THREADS + tid;

    if (i0 < N) {
        output[i0] += offset;
    }
    if (i1 < N) {
        output[i1] += offset;
    }
}

// -----------------------------------------------------------------------------
// scanRecursive
//
// 作用：
//   对任意长度 N 的 device 数组做 inclusive scan。
//
// 为什么需要递归？
//   如果 N > ITEMS_PER_BLOCK，一个 block 扫不完。
//   先每个 block 扫一段，得到 blockSums。
//   然后对 blockSums 再做 scan。
//   blockSums 本身也可能很长，所以递归处理。
// -----------------------------------------------------------------------------
static void scanRecursive(const float* input, float* output, int N) {
    if (N <= 0) {
        return;
    }

    const int numBlocks = (N + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;

    float* blockSums = nullptr;
    float* scannedBlockSums = nullptr;

    // 如果只有一个 block，不需要跨 block 偏移量。
    // 如果有多个 block，需要一个数组保存每个 block 的总和。
    if (numBlocks > 1) {
        cudaMalloc(reinterpret_cast<void**>(&blockSums),
                   static_cast<size_t>(numBlocks) * sizeof(float));
    }

    // 第一步：每个 block 扫自己的局部区间。
    // 输出：
    //   output     中暂时是每个 block 内部的 prefix sum
    //   blockSums  中是每个 block 的总和
    scanBlockKernel<<<numBlocks, BLOCK_THREADS>>>(input, output, blockSums, N);

    if (numBlocks > 1) {
        cudaMalloc(reinterpret_cast<void**>(&scannedBlockSums),
                   static_cast<size_t>(numBlocks) * sizeof(float));

        // 第二步：递归扫描 blockSums。
        // scannedBlockSums[b] 表示从 block0 到 block b 的累计和。
        scanRecursive(blockSums, scannedBlockSums, numBlocks);

        // 第三步：给每个 block 的局部结果加上前面所有 block 的累计和。
        addBlockOffsetsKernel<<<numBlocks, BLOCK_THREADS>>>(output,
                                                            scannedBlockSums,
                                                            N);

        cudaFree(scannedBlockSums);
        cudaFree(blockSums);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    scanRecursive(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 13. 逐段理解提交代码

### 13.1 为什么是两个下标 `i0` 和 `i1`？

代码：

```cpp
const int i0 = base + tid;
const int i1 = base + BLOCK_THREADS + tid;
```

假设：

```text
BLOCK_THREADS = 1024
blockIdx.x = 0
```

那么：

| threadIdx.x | i0 | i1 |
|---:|---:|---:|
| 0 | 0 | 1024 |
| 1 | 1 | 1025 |
| 2 | 2 | 1026 |
| ... | ... | ... |
| 1023 | 1023 | 2047 |

一个 block 正好覆盖：

```text
[0, 2047]
```

如果：

```text
blockIdx.x = 1
```

则：

```text
base = 1 * 2048 = 2048
```

这个 block 覆盖：

```text
[2048, 4095]
```

所以整个数组被切成连续的 2048 元素一段。

---

### 13.2 为什么需要 `v0` 和 `v1`？

代码：

```cpp
const float v0 = (i0 < N) ? input[i0] : 0.0f;
const float v1 = (i1 < N) ? input[i1] : 0.0f;
```

后面会把 shared memory 中的 `temp` 变成 exclusive scan。

但题目要 inclusive scan。

所以最后要：

```cpp
output[i0] = temp[tid] + v0;
output[i1] = temp[BLOCK_THREADS + tid] + v1;
```

如果不保存原始值，到了最后 `temp` 已经被算法改写，就没法再知道当前位置原始输入是多少。

---

### 13.3 为什么 `__syncthreads()` 不能放在 `if` 里面？

在 block 内，所有线程必须都执行到同一个 `__syncthreads()`。

错误例子：

```cpp
if (tid < d) {
    // 危险：不是所有线程都会进入这个 if
    __syncthreads();
}
```

这样可能死锁。

正确写法是：

```cpp
__syncthreads();
if (tid < d) {
    // 部分线程做计算
}
```

本题中 upsweep 和 downsweep 每一层都要同步，因为下一层会读取上一层写入 shared memory 的结果。

---

### 13.4 为什么 `block == 0` 不加 offset？

第 0 段前面没有元素。

例如：

```text
block0: input[0..2047]
```

它的局部 prefix sum 就是全局 prefix sum，不需要加任何东西。

第 1 段要加的是第 0 段总和：

```text
block1 offset = scannedBlockSums[0]
```

第 2 段要加的是第 0 段和第 1 段总和：

```text
block2 offset = scannedBlockSums[1]
```

所以通用公式是：

```cpp
offset = scannedBlockSums[blockIdx.x - 1];
```

---

### 13.5 为什么 `scanRecursive` 里面可以连续启动多个 kernel？

CUDA kernel launch 默认在同一个 stream 中按顺序执行。

也就是说：

```cpp
kernelA<<<...>>>();
kernelB<<<...>>>();
```

在默认 stream 里，`kernelB` 会在 `kernelA` 之后执行。

这就给了我们阶段之间的全局同步：

```text
scanBlockKernel 完成所有 block 的局部扫描
然后 scanRecursive(blockSums, ...)
然后 addBlockOffsetsKernel 加 offset
```

注意：

```text
不能在同一个普通 kernel 内等待所有 block 完成。
```

所以这里用多次 kernel launch 来组织算法。

---

## 14. 这道题里学到的 CUDA 编程知识

### 14.1 数据并行不是所有问题都能一行解决

像 vector add：

```text
output[i] = a[i] + b[i]
```

每个元素相互独立，很容易并行。

Prefix Sum 不同：

```text
output[i] 依赖 input[0..i]
```

所以需要并行算法设计，而不是简单地给每个元素开一个线程。

### 14.2 Shared Memory 适合 block 内协作

本题中每个 block 把 2048 个元素加载进 shared memory：

```cpp
__shared__ float temp[ITEMS_PER_BLOCK];
```

然后所有线程一起在 `temp` 上做树形 scan。

这比反复读写 global memory 更高效。

### 14.3 `__syncthreads()` 是 block 内屏障

它解决的是：

```text
同一个 block 内，线程 A 写 shared memory，线程 B 之后要读这个结果
```

但是它不能同步不同 block。

这也是为什么跨 block 的 prefix sum 要拆成多个 kernel。

### 14.4 Kernel launch 可以作为阶段边界

本题的三个阶段：

```text
局部 scan -> 扫 blockSums -> 加 offset
```

是通过多个 kernel launch 串起来的。

这是很多 CUDA 算法常见的组织方式。

### 14.5 边界判断非常重要

因为 `N` 不一定是 2048 的整数倍。

所以所有 global memory 读写都要判断：

```cpp
if (i < N) {
    ...
}
```

否则就可能越界访问。

---

## 15. 常见错误

### 错误 1：把 inclusive 和 exclusive 搞反

如果只写：

```cpp
output[i] = temp[i];
```

那输出是 exclusive scan。

LeetGPU 要的是 inclusive scan，所以必须：

```cpp
output[i] = temp[i] + input[i];
```

在代码中我们用 `v0` / `v1` 保存了 `input[i]`。

---

### 错误 2：第 b 个 block 加了错误的 offset

错误写法：

```cpp
offset = scannedBlockSums[block];
```

这会把当前 block 自己的总和也加进去。

正确写法：

```cpp
offset = scannedBlockSums[block - 1];
```

因为第 `block` 段只需要它之前所有段的和。

---

### 错误 3：以为 `__syncthreads()` 能同步所有 block

不能。

`__syncthreads()` 只能同步当前 block 内的线程。

跨 block 同步通常需要：

```text
结束当前 kernel，启动下一个 kernel
```

---

### 错误 4：没有处理最后一个不完整 block

如果 `N = 2050`，那么：

```text
block0 处理 0..2047
block1 只处理 2048..2049
```

block1 的很多线程对应的下标都越界。

所以必须写：

```cpp
if (i < N) {
    output[i] = ...;
}
```

读取 input 时越界位置补 0：

```cpp
float v = (i < N) ? input[i] : 0.0f;
```

---

### 错误 5：用 `atomicAdd` 试图做 prefix sum

`atomicAdd` 可以安全地累加到一个全局变量，但它不能自然地产生每个位置的有序前缀和。

即使强行写，也会严重串行化，而且线程执行顺序不等于数组顺序。

Prefix Sum 应该用 scan 算法，而不是 atomic。

---

## 16. 复杂度分析

对于每个层级：

```text
每个元素被读写常数次
```

所以总工作量接近：

```text
O(N)
```

每个 block 内部的 Blelloch scan 是：

```text
O(log ITEMS_PER_BLOCK) 层同步
```

因为 `ITEMS_PER_BLOCK = 2048`，所以：

```text
log2(2048) = 11
```

upsweep 11 层，downsweep 11 层。

额外空间：

```text
blockSums + scannedBlockSums
```

大约是：

```text
O(N / ITEMS_PER_BLOCK)
```

比原数组小很多。

---

## 17. 如何手动检查代码是否正确？

可以用下面几个测试思路。

### 测试 1：长度为 1

```text
input  = [7]
output = [7]
```

### 测试 2：刚好一个 block 内

真实代码中一个 block 是 2048 个元素。

如果 `N <= 2048`：

```text
numBlocks = 1
```

这时不需要 `blockSums` 偏移量，局部 scan 就是全局结果。

### 测试 3：刚好超过一个 block

如果：

```text
N = 2049
```

那么：

```text
block0: 2048 个元素
block1: 1 个元素
```

`output[2048]` 应该等于：

```text
前 2048 个元素的和 + input[2048]
```

这正好检验 `addBlockOffsetsKernel` 是否正确。

### 测试 4：最后一个输出

无论输入是什么，最后一个输出一定等于整个数组总和：

```text
output[N - 1] = sum(input[0..N-1])
```

这是检查 prefix sum 的一个简单方法。

---

## 18. 总结

这道 Prefix Sum 是很适合学习 CUDA 的题，因为它不像 vector add 那样每个元素完全独立，需要真正理解并行算法。

核心思路可以压缩成三句话：

```text
1. 每个 block 用 shared memory 做局部 scan。
2. 保存每个 block 的总和，对 blockSums 再做 scan。
3. 把前面 block 的累计和作为 offset 加回每个 block 的局部结果。
```

对应到 CUDA 知识：

```text
shared memory 负责 block 内协作
__syncthreads() 负责 block 内同步
多次 kernel launch 负责跨 block 阶段同步
```

只要理解了这个模式，很多 CUDA 并行算法都会变得更容易理解，例如：

- stream compaction
- radix sort
- histogram 后处理
- parallel filtering
- sparse matrix 前处理
- GPU 上的并行分配与重排
