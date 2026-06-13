# 从 LeetGPU Histogramming 入手学习 CUDA：直方图统计、原子操作与 Shared Memory 优化

> **题目**：LeetGPU — Histogramming  
> **目标**：给定 GPU 上的 `int* input`，长度 `N`，以及桶数量 `num_bins`，统计每个整数出现了多少次。
>
> ```text
> histogram[b] = input 中值等于 b 的元素个数
> b 的范围是 [0, num_bins)
> ```
>
> CUDA 提交接口：
>
> ```cpp
> extern "C" void solve(const int* input, int* histogram, int N, int num_bins);
> ```
>
> 这题的重点不是“怎么写一个 for 循环”，而是学习 GPU 上非常常见的几个概念：
>
> - 多线程同时更新同一个位置会产生 data race
> - `atomicAdd` 如何保证计数正确
> - 为什么直接对 global memory 做 atomic 会慢
> - 如何用 shared memory 做 block 内局部聚合
> - 如何减少 global atomic 次数
> - 如何用 grid-stride loop 处理超大数组

---

## 1. 题目到底要求什么？

输入：

```text
input = [0, 1, 2, 1, 0]
N = 5
num_bins = 3
```

需要输出：

```text
histogram = [2, 2, 1]
```

原因是：

```text
0 出现 2 次
1 出现 2 次
2 出现 1 次
```

再看一个例子：

```text
input = [3, 3, 3, 3]
N = 4
num_bins = 5
```

输出：

```text
histogram = [0, 0, 0, 4, 0]
```

因为只有数字 `3` 出现了 4 次。

题目保证：

```text
0 <= input[i] < num_bins
1 <= num_bins <= 1024
```

所以 kernel 里不用额外判断 `input[i]` 是否越界。

---

## 2. CPU 串行写法

CPU 上最直接的写法是：

```cpp
for (int b = 0; b < num_bins; ++b) {
    histogram[b] = 0;
}

for (int i = 0; i < N; ++i) {
    int bin = input[i];
    histogram[bin] += 1;
}
```

这个逻辑很简单：

```text
读一个 input[i]
找到对应桶 bin
把 histogram[bin] 加 1
```

但是 GPU 上不能直接让很多线程同时执行：

```cpp
histogram[input[i]] += 1;
```

因为多个线程可能同时更新同一个桶。

---

## 3. 为什么普通 `histogram[bin]++` 是错的？

假设两个线程同时看到：

```text
histogram[3] = 10
```

它们都想执行：

```cpp
histogram[3] = histogram[3] + 1;
```

这条语句在硬件上不是一个不可分割的操作，大概会分成：

```text
1. 从内存读 histogram[3]
2. 在寄存器里加 1
3. 写回 histogram[3]
```

如果两个线程交错执行：

```text
线程 A 读到 10
线程 B 读到 10
线程 A 写回 11
线程 B 写回 11
```

最终结果是 `11`，但正确结果应该是 `12`。

这就是 **data race**。

所以 histogram 这题必须使用原子操作。

---

## 4. 最朴素的 CUDA 写法：global atomic

一个最容易想到、也一定正确的 GPU 版本是：

```cpp
__global__ void histGlobalAtomicKernel(const int* input,
                                       int* histogram,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = input[i];
        atomicAdd(&histogram[bin], 1);
    }
}
```

`atomicAdd` 的含义是：

```text
对某个地址做加法，并保证这个读-改-写过程不会被其他线程打断。
```

这样就不会丢计数。

但是这个版本有一个严重问题：

```text
所有线程都在直接竞争 global memory 里的 num_bins 个桶。
```

如果 `N = 50,000,000`，`num_bins = 256`，那么会有 5000 万次 global atomic。

如果输入分布很集中，比如很多元素都等于 `0`，那大量线程会同时争抢：

```cpp
histogram[0]
```

这会非常慢。

---

## 5. 优化思路：先在每个 block 内统计局部直方图

更好的思路是两阶段：

```text
阶段 1：每个 block 在 shared memory 里统计自己的 local histogram
阶段 2：每个 block 把 local histogram 合并到 global histogram
```

也就是：

```text
input 数据很多
        |
        v
block0 统计 local_hist0
block1 统计 local_hist1
block2 统计 local_hist2
...
        |
        v
把所有 local_hist 加到全局 histogram
```

这样做的好处是：

```text
大量 atomicAdd 发生在 shared memory 中，速度比 global memory 快很多。
global atomic 的次数从 N 次下降到 blocks * num_bins 次。
```

以性能测试常见参数为例：

```text
N = 50,000,000
num_bins = 256
blocks = 4096
```

global atomic 版本大约需要：

```text
50,000,000 次 global atomic
```

shared memory 局部聚合版本最后合并阶段只需要大约：

```text
4096 * 256 = 1,048,576 次 global atomic
```

global atomic 数量下降了几十倍。

---

## 6. 再进一步：每个 warp 一份局部 histogram

一个 block 内有多个 warp。

如果整个 block 只有一份 shared histogram：

```text
block local histogram: local_hist[num_bins]
```

那么同一个 block 内的 256 个线程仍然可能竞争同一个 shared memory 地址。

更稳一点的做法是：

```text
每个 warp 一份 local histogram
```

如果一个 block 有 256 个线程：

```text
256 threads / 32 threads per warp = 8 warps
```

那么 shared memory 中开：

```text
shared_hist[8][num_bins]
```

每个线程只更新自己所在 warp 的那份 histogram：

```cpp
int warp_id = threadIdx.x / 32;
atomicAdd(&shared_hist[warp_id][bin], 1);
```

这样可以把 block 内的竞争拆散到 8 份局部 histogram 上。

最后每个 block 再把 8 份 warp histogram 求和，合并到 global histogram。

---

## 7. LeetGPU 可直接提交代码

下面这份是推荐提交版：

- 使用 `cudaMemset` 初始化输出 histogram
- 使用 shared memory 做 block 内局部聚合
- 每个 warp 一份局部 histogram，降低 shared atomic 竞争
- 使用 grid-stride loop 处理大数组
- block 数量做上限限制，避免合并阶段 global atomic 太多

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
static constexpr int MAX_BLOCKS = 4096;

// -----------------------------------------------------------------------------
// histogramKernel
//
// input     : device pointer，长度为 N
// histogram : device pointer，长度为 num_bins
// N         : input 元素数量
// num_bins  : 桶数量，题目保证 1 <= num_bins <= 1024
//
// shared memory 布局：
//   s_hist[warp_id * num_bins + bin]
//
// 也就是每个 warp 一份局部 histogram。
// -----------------------------------------------------------------------------
__global__ void histogramKernel(const int* __restrict__ input,
                                int* __restrict__ histogram,
                                int N,
                                int num_bins) {
    extern __shared__ int s_hist[];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;     // tid / 32
    const int num_warps = blockDim.x >> 5;

    // -------------------------------------------------------------------------
    // 1. 初始化 shared memory 中的 warp-private histograms
    // -------------------------------------------------------------------------
    const int shared_hist_size = num_warps * num_bins;
    for (int i = tid; i < shared_hist_size; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. 每个线程用 grid-stride loop 读取多个 input 元素
    //    然后累加到自己 warp 对应的局部 histogram 中。
    // -------------------------------------------------------------------------
    int* my_hist = s_hist + warp_id * num_bins;

    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    for (int i = global_tid; i < N; i += stride) {
        const int bin = input[i];
        atomicAdd(&my_hist[bin], 1);
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 3. 把当前 block 中所有 warp 的局部 histogram 合并，
    //    然后加到 global histogram 中。
    //
    //    每个 bin 只由一个线程负责求和，避免重复写。
    // -------------------------------------------------------------------------
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        int sum = 0;
        for (int w = 0; w < num_warps; ++w) {
            sum += s_hist[w * num_bins + bin];
        }

        if (sum != 0) {
            atomicAdd(&histogram[bin], sum);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (num_bins <= 0) {
        return;
    }

    // 不能假设 LeetGPU 已经帮我们把 histogram 清零。
    cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));

    if (N <= 0) {
        cudaDeviceSynchronize();
        return;
    }

    int blocks = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    // 每个 warp 一份 histogram。
    // num_bins 最大 1024，因此 shared memory 最大：
    //   8 * 1024 * sizeof(int) = 32 KB
    // 普通 CUDA GPU 都可以接受。
    const size_t shared_bytes =
        static_cast<size_t>(WARPS_PER_BLOCK) * num_bins * sizeof(int);

    histogramKernel<<<blocks, BLOCK_THREADS, shared_bytes>>>(input,
                                                             histogram,
                                                             N,
                                                             num_bins);
    cudaDeviceSynchronize();
}
```

---

## 8. 逐段理解提交代码

### 8.1 为什么要先 `cudaMemset`？

`histogram` 是输出数组，但题目没有保证它初始值一定是 0。

如果不清零：

```cpp
atomicAdd(&histogram[bin], 1);
```

就是在未知旧值上累加，结果会错。

所以必须先写：

```cpp
cudaMemset(histogram, 0, num_bins * sizeof(int));
```

`histogram` 是 device pointer，`cudaMemset` 会在 GPU 显存里把它清零。

---

### 8.2 为什么使用动态 shared memory？

题目里的 `num_bins` 是运行时参数：

```text
1 <= num_bins <= 1024
```

所以不能简单写死：

```cpp
__shared__ int local[256];
```

因为 `num_bins` 可能不是 256。

提交代码使用：

```cpp
extern __shared__ int s_hist[];
```

然后在 kernel launch 时指定 shared memory 大小：

```cpp
histogramKernel<<<blocks, BLOCK_THREADS, shared_bytes>>>(...);
```

其中：

```cpp
shared_bytes = WARPS_PER_BLOCK * num_bins * sizeof(int);
```

这就是 CUDA 的动态 shared memory。

---

### 8.3 shared memory 的布局是什么？

代码中：

```cpp
int* my_hist = s_hist + warp_id * num_bins;
```

表示第 `warp_id` 个 warp 使用：

```text
s_hist[warp_id * num_bins ... warp_id * num_bins + num_bins - 1]
```

如果：

```text
num_bins = 256
WARPS_PER_BLOCK = 8
```

那么布局是：

```text
warp 0: s_hist[0      ... 255]
warp 1: s_hist[256    ... 511]
warp 2: s_hist[512    ... 767]
...
warp 7: s_hist[1792   ... 2047]
```

每个 warp 有自己独立的一份 histogram。

---

### 8.4 为什么初始化 shared memory 要用循环？

shared memory 的元素数量是：

```text
num_warps * num_bins
```

最大可能是：

```text
8 * 1024 = 8192 个 int
```

而一个 block 只有 256 个线程。

所以一个线程可能需要初始化多个元素：

```cpp
for (int i = tid; i < shared_hist_size; i += blockDim.x) {
    s_hist[i] = 0;
}
```

这是一种常见写法：

```text
第 tid 个线程处理 tid, tid + blockDim.x, tid + 2 * blockDim.x, ...
```

---

### 8.5 为什么初始化后要 `__syncthreads()`？

初始化 shared memory 是所有线程协作完成的。

如果没有：

```cpp
__syncthreads();
```

有些线程可能还没把 `s_hist` 清零，另一些线程就开始 `atomicAdd` 了。

这样会把计数加到未初始化的旧值上，结果错误。

所以初始化后必须同步：

```cpp
__syncthreads();
```

---

### 8.6 为什么用 grid-stride loop？

普通写法是一个线程处理一个元素：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    ...
}
```

但是这题的 `N` 可能很大：

```text
N <= 100,000,000
```

如果完全按照 `N / 256` 开 block，block 数会非常多。

合并阶段每个 block 都要把 local histogram 加到 global histogram：

```text
global atomic 次数 = blocks * num_bins
```

block 太多会让合并阶段变慢。

所以提交代码把 block 数限制到：

```cpp
MAX_BLOCKS = 4096
```

然后每个线程通过 grid-stride loop 处理多个元素：

```cpp
for (int i = global_tid; i < N; i += stride) {
    int bin = input[i];
    atomicAdd(&my_hist[bin], 1);
}
```

其中：

```cpp
stride = blockDim.x * gridDim.x
```

这样即使只开 4096 个 block，也能覆盖完整数组。

---

### 8.7 为什么最后还需要 global atomic？

每个 block 都有自己的局部 histogram。

例如：

```text
block0 统计到 bin 3 有 100 个
block1 统计到 bin 3 有 120 个
block2 统计到 bin 3 有  90 个
```

最终全局的：

```text
histogram[3] = 100 + 120 + 90 + ...
```

多个 block 会同时合并到同一个 global histogram 桶，所以这里依然需要：

```cpp
atomicAdd(&histogram[bin], sum);
```

区别是，global atomic 的次数已经从“每个 input 元素一次”变成了“每个 block 每个 bin 最多一次”。

---

## 9. 更短但较慢的版本

如果只想先写一个最容易理解的正确版本，可以这样：

```cpp
#include <cuda_runtime.h>

__global__ void histGlobalAtomicKernel(const int* input,
                                       int* histogram,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&histogram[input[i]], 1);
    }
}

extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    histGlobalAtomicKernel<<<blocks, threads>>>(input, histogram, N);

    cudaDeviceSynchronize();
}
```

这个版本逻辑简单，也容易验证正确性。

但是在大数据量、桶数量少、数据分布集中的情况下，性能会明显差于 shared memory 局部聚合版本。

---

## 10. 常见错误

### 错误 1：忘记初始化 `histogram`

错误写法：

```cpp
// 没有 cudaMemset
histogramKernel<<<blocks, threads>>>(input, histogram, N, num_bins);
```

如果 `histogram` 里原来有垃圾值，最后结果一定错。

正确写法：

```cpp
cudaMemset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));
```

---

### 错误 2：不用 atomic

错误写法：

```cpp
histogram[input[i]] += 1;
```

多个线程同时更新同一个桶时会丢计数。

正确写法：

```cpp
atomicAdd(&histogram[input[i]], 1);
```

或者先在 shared memory 中用 atomic 聚合，再合并到 global memory。

---

### 错误 3：shared memory 初始化后忘记同步

错误写法：

```cpp
for (int i = tid; i < shared_hist_size; i += blockDim.x) {
    s_hist[i] = 0;
}

// 少了 __syncthreads()
atomicAdd(&my_hist[bin], 1);
```

正确写法：

```cpp
for (int i = tid; i < shared_hist_size; i += blockDim.x) {
    s_hist[i] = 0;
}
__syncthreads();
```

---

### 错误 4：统计完 shared histogram 后忘记同步

错误写法：

```cpp
for (int i = global_tid; i < N; i += stride) {
    atomicAdd(&my_hist[input[i]], 1);
}

// 少了 __syncthreads()
// 有些线程可能还没统计完，另一些线程就开始合并了
for (int bin = tid; bin < num_bins; bin += blockDim.x) {
    ...
}
```

正确写法：

```cpp
for (int i = global_tid; i < N; i += stride) {
    atomicAdd(&my_hist[input[i]], 1);
}
__syncthreads();
```

---

### 错误 5：把 `num_bins` 写死成 256

LeetGPU 的性能测试常用：

```text
num_bins = 256
```

但题目约束是：

```text
1 <= num_bins <= 1024
```

所以不要写：

```cpp
__shared__ int local[256];
```

推荐使用动态 shared memory：

```cpp
extern __shared__ int s_hist[];
```

---

### 错误 6：block 开太多，合并阶段变慢

如果直接：

```cpp
int blocks = (N + 255) / 256;
```

当：

```text
N = 50,000,000
```

会有大约：

```text
195,313 个 block
```

如果每个 block 最后合并 256 个 bin，那么合并阶段 global atomic 次数约为：

```text
195,313 * 256 ≈ 50,000,000
```

这又退化得很严重。

所以推荐限制 block 数，然后用 grid-stride loop：

```cpp
if (blocks > MAX_BLOCKS) {
    blocks = MAX_BLOCKS;
}
```

---

## 11. 复杂度分析

每个输入元素会被读取一次，并累加到某个局部 histogram：

```text
O(N)
```

每个 block 最后会合并 `num_bins` 个桶：

```text
O(blocks * num_bins)
```

所以整体工作量大约是：

```text
O(N + blocks * num_bins)
```

因为 `num_bins <= 1024`，并且我们限制了 `blocks <= 4096`，所以合并阶段是可控的。

额外 shared memory：

```text
WARPS_PER_BLOCK * num_bins * sizeof(int)
```

在提交代码中：

```text
8 * num_bins * 4 bytes
```

当 `num_bins = 1024` 时：

```text
8 * 1024 * 4 = 32768 bytes = 32 KB
```

---

## 12. 如何手动检查代码是否正确？

### 测试 1：只有一个元素

```text
input = [0]
N = 1
num_bins = 1
output = [1]
```

### 测试 2：多个桶均匀分布

```text
input = [0, 1, 2, 1, 0]
N = 5
num_bins = 3
output = [2, 2, 1]
```

### 测试 3：全部落到同一个桶

```text
input = [3, 3, 3, 3]
N = 4
num_bins = 5
output = [0, 0, 0, 4, 0]
```

这个测试可以检查 atomic 是否正确。

### 测试 4：有些桶没有出现

```text
input = [0, 4, 4, 0]
N = 4
num_bins = 5
output = [2, 0, 0, 0, 2]
```

这个测试可以检查 `cudaMemset` 是否正确清零。

### 测试 5：`num_bins` 不是 256

```text
input = [0, 1, 1, 2, 2, 2]
N = 6
num_bins = 3
output = [1, 2, 3]
```

这个测试可以检查有没有把桶数量写死。

---

## 13. 总结

Histogramming 这题的核心是：

```text
很多线程会同时更新少量桶，所以必须处理写冲突。
```

最重要的三点：

```text
1. 正确性靠 atomicAdd。
2. 性能靠 shared memory 局部聚合。
3. 大输入靠 grid-stride loop + 控制 block 数量。
```

最终推荐实现可以概括为：

```text
先 cudaMemset 清零输出
每个 block 在 shared memory 里统计局部 histogram
每个 warp 使用独立的局部 histogram 减少竞争
最后把局部 histogram atomicAdd 到 global histogram
```

这类模式在 GPU 编程中很常见，不只适用于 histogram，也适用于：

- 词频统计
- bucket counting
- radix sort 中的 digit histogram
- 图算法中的度数统计
- 稀疏矩阵格式转换前的 row count
- 数据分桶和特征统计
