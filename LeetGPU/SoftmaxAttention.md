# 从 LeetGPU Softmax Attention 入手学习 CUDA：稳定 Softmax、按行 Attention 与 Shared Memory 实现

> **题目**：LeetGPU — Softmax Attention  
> **目标**：给定三个 GPU 上的矩阵：
>
> - `Q`: `M × d`
> - `K`: `N × d`
> - `V`: `N × d`
>
> 计算：
>
> ```text
> output = softmax(Q K^T / sqrt(d)) V
> ```
>
> 其中 softmax 是 **按行 row-wise** 做的。CUDA 提交接口：
>
> ```cpp
> extern "C" void solve(const float* Q,
>                       const float* K,
>                       const float* V,
>                       float* output,
>                       int M,
>                       int N,
>                       int d);
> ```
>
> 这篇文档不只是给一份可提交代码，而是通过这题理解：
>
> - Attention 公式到底在算什么
> - 为什么 softmax 要做数值稳定处理
> - 为什么不要显式存完整 `M × N` attention 矩阵
> - 一个 block 负责一个 query row 的实现思路
> - 如何用 shared memory 缓存一行 scores
> - 大 `N` 情况下如何用 streaming fallback 保证通用性

---

## 1. 题目到底要求什么？

给定：

```text
Q: M × d
K: N × d
V: N × d
```

要输出：

```text
output: M × d
```

数学公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

拆开看，对于第 `i` 个 query row：

```text
Q_i = Q[i, :]
```

它会和每一个 key row 做点积：

```text
score[i, j] = dot(Q_i, K_j) / sqrt(d)
```

其中：

```text
j = 0, 1, ..., N - 1
```

然后对这一整行 `score[i, :]` 做 softmax：

```text
weight[i, j] = exp(score[i, j]) / sum_t exp(score[i, t])
```

最后用这个权重对 `V` 的每一行做加权求和：

```text
output[i, col] = sum_j weight[i, j] * V[j, col]
```

其中：

```text
col = 0, 1, ..., d - 1
```

---

## 2. 一个很小的例子

假设：

```text
M = 1
N = 2
d = 2
```

```text
Q = [[1, 2]]
K = [[1, 0],
     [0, 1]]
V = [[3, 4],
     [5, 6]]
```

先算 `QK^T`：

```text
dot(Q0, K0) = 1 * 1 + 2 * 0 = 1
dot(Q0, K1) = 1 * 0 + 2 * 1 = 2
```

除以：

```text
sqrt(d) = sqrt(2)
```

得到：

```text
score = [1 / sqrt(2), 2 / sqrt(2)]
      ≈ [0.707, 1.414]
```

做 softmax：

```text
weight ≈ [0.330, 0.670]
```

最后加权 `V`：

```text
output[0, 0] = 0.330 * 3 + 0.670 * 5 ≈ 4.34
output[0, 1] = 0.330 * 4 + 0.670 * 6 ≈ 5.34
```

所以输出约为：

```text
[[4.34, 5.34]]
```

---

## 3. CPU 串行写法

CPU 上最直观的写法是三步：

1. 计算 scores
2. 对 scores 做 softmax
3. 用 softmax 权重乘 V

伪代码如下：

```cpp
for (int i = 0; i < M; ++i) {
    // 1. 计算 score[i, j]
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        score[j] = dot / sqrtf((float)d);
    }

    // 2. softmax
    float denom = 0.0f;
    for (int j = 0; j < N; ++j) {
        prob[j] = expf(score[j]);
        denom += prob[j];
    }
    for (int j = 0; j < N; ++j) {
        prob[j] /= denom;
    }

    // 3. prob * V
    for (int col = 0; col < d; ++col) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            acc += prob[j] * V[j * d + col];
        }
        output[i * d + col] = acc;
    }
}
```

这个逻辑正确，但 GPU 上不能简单照搬。

---

## 4. 为什么不能直接存完整 attention 矩阵？

最直接的 GPU 思路可能是：

```text
S = QK^T              // M × N
P = softmax(S)        // M × N
output = P V          // M × d
```

但是这会显式创建 `M × N` 中间矩阵。

题目约束里：

```text
M, N 最大都可能到 100,000
```

如果真的存一个 `M × N` float 矩阵：

```text
100000 × 100000 × 4 bytes = 40 GB
```

这还只是一个矩阵。如果再存 softmax 后的 `P`，还要再来一份。

所以更好的做法是：

```text
一次只处理一行 Q_i。
只在 shared memory 中保存当前 query row 对所有 key 的 scores。
算完这一行 output 后就丢掉 scores。
```

这样不用分配完整 `M × N` 中间矩阵。

---

## 5. Softmax 必须做数值稳定处理

普通 softmax 是：

```text
softmax(x_j) = exp(x_j) / sum_t exp(x_t)
```

如果 `x_j` 很大，例如：

```text
x_j = 100
```

那么：

```text
exp(100)
```

可能溢出成 `inf`。

标准稳定写法是先减去最大值：

```text
m = max_j x_j
softmax(x_j) = exp(x_j - m) / sum_t exp(x_t - m)
```

为什么这样不改变结果？

```text
exp(x_j - m) / sum_t exp(x_t - m)
= exp(x_j) / exp(m) / sum_t (exp(x_t) / exp(m))
= exp(x_j) / sum_t exp(x_t)
```

减去最大值后，所有指数的输入都满足：

```text
x_j - m <= 0
```

所以不会因为正数太大而爆掉。

---

## 6. 每一行 Attention 的稳定计算公式

对于第 `i` 行 query，先定义：

```text
score_j = dot(Q_i, K_j) / sqrt(d)
```

稳定 softmax 需要：

```text
row_max = max_j score_j
```

然后：

```text
denominator = sum_j exp(score_j - row_max)
```

最后输出：

```text
output[i, col]
= sum_j exp(score_j - row_max) * V[j, col] / denominator
```

注意这里不一定需要真的保存 softmax 权重 `weight[j]`。

可以直接使用：

```text
unnormalized_weight[j] = exp(score_j - row_max)
```

最后统一除以 `denominator`。

---

## 7. CUDA 实现思路

LeetGPU 这题的性能测试说明里给了：

```text
M = 512
N = 256
d <= 128
```

这里 `N = 256` 很小，非常适合使用：

```text
一个 CUDA block 负责一个 query row
```

也就是：

```text
blockIdx.x = query row i
```

每个 block 内部做：

```text
1. 并行计算这一行的 N 个 score
2. block 内 reduce 得到 row_max
3. 并行计算 exp(score - row_max)，并 reduce 得到 denominator
4. 用这 N 个权重乘 V，得到 output[i, 0..d-1]
```

因为 `N <= 1024` 的情况可以把当前行的 scores 放进 shared memory：

```cpp
__shared__ float scores[1024];
```

这样后面 softmax 和乘 V 时不用重新计算 score。

---

## 8. 为什么还需要大 N fallback？

题目约束写的是：

```text
1 <= N <= 100,000
```

如果 `N` 很大，就不能再用：

```cpp
__shared__ float scores[100000];
```

shared memory 放不下。

所以提交代码做了两个路径：

```text
N <= 1024:
    使用 fast kernel：一个 block 处理一行，scores 放 shared memory。

N > 1024:
    使用 streaming fallback：不存 scores，按 output 元素流式重算。
```

fallback 会慢一些，因为会为每个 output column 重新计算 dot product；但它不需要巨大 shared memory，通用性更好。

---

## 9. LeetGPU 可直接提交代码

下面这份是完整提交版。

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;
static constexpr int MAX_D = 128;
static constexpr int FAST_MAX_N = 1024;
static constexpr float NEG_INF = -3.4028234663852886e38f;

// -----------------------------------------------------------------------------
// reduceMaxInBlock / reduceSumInBlock
//
// 使用 shared memory 做一个 block 内的简单规约。
// blockDim.x 固定使用 256。
// -----------------------------------------------------------------------------
__device__ float reduceMaxInBlock(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    return shared[0];
}

__device__ float reduceSumInBlock(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    return shared[0];
}

// -----------------------------------------------------------------------------
// attentionSmallNKernel
//
// 适用于 N <= 1024 的主路径。
// 一个 block 负责 Q 的一行。
//
// Q      : M × d
// K      : N × d
// V      : N × d
// output : M × d
//
// 共享内存：
//   q_shared[MAX_D]       当前 query row
//   scores[FAST_MAX_N]    当前 query row 对所有 key 的 score / exp(score-max)
//   reduce[BLOCK_THREADS] block 内规约用
// -----------------------------------------------------------------------------
__global__ void attentionSmallNKernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ output,
                                      int M,
                                      int N,
                                      int d) {
    __shared__ float q_shared[MAX_D];
    __shared__ float scores[FAST_MAX_N];
    __shared__ float reduce[BLOCK_THREADS];
    __shared__ float row_max_shared;
    __shared__ float denom_shared;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    // -------------------------------------------------------------------------
    // 1. 把当前 Q row 缓存到 shared memory。
    //    d <= 128，所以最多只需要 128 个 float。
    // -------------------------------------------------------------------------
    if (tid < d) {
        q_shared[tid] = Q[row * d + tid];
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(d));

    // -------------------------------------------------------------------------
    // 2. 并行计算 score[j] = dot(Q[row], K[j]) / sqrt(d)
    //    同时每个线程维护自己的 local max。
    // -------------------------------------------------------------------------
    float local_max = NEG_INF;

    for (int j = tid; j < N; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += q_shared[k] * K[j * d + k];
        }

        const float score = dot * scale;
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }

    const float row_max = reduceMaxInBlock(local_max, reduce);
    if (tid == 0) {
        row_max_shared = row_max;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 3. 计算 exp(score - row_max)，并求 denominator。
    //    这里直接把 scores[j] 覆盖成 exp(score - row_max)，避免再开数组。
    // -------------------------------------------------------------------------
    float local_sum = 0.0f;
    const float m = row_max_shared;

    for (int j = tid; j < N; j += blockDim.x) {
        const float e = expf(scores[j] - m);
        scores[j] = e;
        local_sum += e;
    }

    const float denom = reduceSumInBlock(local_sum, reduce);
    if (tid == 0) {
        denom_shared = denom;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 4. 计算 output[row, col]
    //
    //    output[row, col]
    //      = sum_j exp(score_j - max) * V[j, col] / denominator
    //
    //    d <= 128，所以这里通常前 d 个线程各负责一个 col。
    // -------------------------------------------------------------------------
    const float inv_denom = 1.0f / denom_shared;

    for (int col = tid; col < d; col += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            acc += scores[j] * V[j * d + col];
        }
        output[row * d + col] = acc * inv_denom;
    }
}

// -----------------------------------------------------------------------------
// attentionStreamingKernel
//
// 大 N fallback。
// 一个 block 负责一个 output 元素：output[row, col]。
//
// 不保存 scores，而是：
//   第 1 遍：扫描所有 key，求 row_max
//   第 2 遍：重新计算 score，累加 denominator 和 numerator
//
// 这个版本比 small-N 版本慢，但不需要 O(N) shared memory。
// -----------------------------------------------------------------------------
__global__ void attentionStreamingKernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ output,
                                         int M,
                                         int N,
                                         int d) {
    __shared__ float q_shared[MAX_D];
    __shared__ float reduce1[BLOCK_THREADS];
    __shared__ float reduce2[BLOCK_THREADS];
    __shared__ float row_max_shared;

    const int row = blockIdx.x;
    const int col = blockIdx.y;
    const int tid = threadIdx.x;

    if (row >= M || col >= d) {
        return;
    }

    // 当前 query row 缓存到 shared memory。
    if (tid < d) {
        q_shared[tid] = Q[row * d + tid];
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(d));

    // -------------------------------------------------------------------------
    // 1. 第一遍：求这一行 score 的最大值。
    // -------------------------------------------------------------------------
    float local_max = NEG_INF;

    for (int j = tid; j < N; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += q_shared[k] * K[j * d + k];
        }
        local_max = fmaxf(local_max, dot * scale);
    }

    const float row_max = reduceMaxInBlock(local_max, reduce1);
    if (tid == 0) {
        row_max_shared = row_max;
    }
    __syncthreads();

    // -------------------------------------------------------------------------
    // 2. 第二遍：累加 denominator 和当前 col 的 numerator。
    // -------------------------------------------------------------------------
    float local_denom = 0.0f;
    float local_num = 0.0f;
    const float m = row_max_shared;

    for (int j = tid; j < N; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < d; ++k) {
            dot += q_shared[k] * K[j * d + k];
        }

        const float e = expf(dot * scale - m);
        local_denom += e;
        local_num += e * V[j * d + col];
    }

    const float denom = reduceSumInBlock(local_denom, reduce1);
    const float numer = reduceSumInBlock(local_num, reduce2);

    if (tid == 0) {
        output[row * d + col] = numer / denom;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q,
                      const float* K,
                      const float* V,
                      float* output,
                      int M,
                      int N,
                      int d) {
    if (M <= 0 || N <= 0 || d <= 0) {
        return;
    }

    if (N <= FAST_MAX_N) {
        // 主路径：LeetGPU 性能测试 N = 256，非常适合这个 kernel。
        attentionSmallNKernel<<<M, BLOCK_THREADS>>>(Q, K, V, output, M, N, d);
    } else {
        // fallback：避免为超大 N 分配过大的 shared memory。
        dim3 grid(M, d);
        attentionStreamingKernel<<<grid, BLOCK_THREADS>>>(Q, K, V, output, M, N, d);
    }

    cudaDeviceSynchronize();
}
```

---

## 10. 逐段理解提交代码

### 10.1 矩阵在内存中怎么索引？

LeetGPU 传进来的 `Q`, `K`, `V`, `output` 都是一维 device pointer。

但是逻辑上它们是二维矩阵。

默认按 row-major 存储，也就是：

```text
matrix[row][col] -> matrix[row * d + col]
```

所以：

```cpp
Q[row * d + k]
K[j * d + k]
V[j * d + col]
output[row * d + col]
```

分别表示：

```text
Q[row, k]
K[j, k]
V[j, col]
output[row, col]
```

---

### 10.2 为什么 `scale = rsqrtf((float)d)`？

Attention 公式中有：

```text
QK^T / sqrt(d)
```

也就是每个 dot product 都要除以 `sqrt(d)`。

代码中写：

```cpp
const float scale = rsqrtf(static_cast<float>(d));
```

`rsqrtf(x)` 是：

```text
1 / sqrt(x)
```

所以：

```cpp
score = dot * scale;
```

等价于：

```cpp
score = dot / sqrtf((float)d);
```

---

### 10.3 为什么要把 Q 的当前行放进 shared memory？

对于一个 query row：

```text
Q[row, :]
```

它会被反复用于计算：

```text
dot(Q[row], K[0])
dot(Q[row], K[1])
...
dot(Q[row], K[N-1])
```

也就是说同一行 Q 会被重复读取很多次。

把它先放到 shared memory：

```cpp
if (tid < d) {
    q_shared[tid] = Q[row * d + tid];
}
__syncthreads();
```

后面所有线程都可以快速访问：

```cpp
dot += q_shared[k] * K[j * d + k];
```

因为 `d <= 128`，这只需要很少的 shared memory。

---

### 10.4 为什么 `scores[j]` 可以放 shared memory？

性能测试中：

```text
N = 256
```

代码主路径支持：

```text
N <= 1024
```

所以当前 query row 的所有 scores 最多是：

```text
1024 个 float = 4096 bytes
```

这很小，非常适合放 shared memory。

代码中：

```cpp
__shared__ float scores[FAST_MAX_N];
```

先保存原始 score：

```cpp
scores[j] = score;
```

后面再覆盖成：

```cpp
scores[j] = expf(score - row_max);
```

这样不用额外再开一个概率数组。

---

### 10.5 为什么计算 softmax 需要两次 reduce？

稳定 softmax 需要：

```text
1. row_max = max(score)
2. denom = sum(exp(score - row_max))
```

所以需要两个 block 内规约：

```cpp
const float row_max = reduceMaxInBlock(local_max, reduce);
```

以及：

```cpp
const float denom = reduceSumInBlock(local_sum, reduce);
```

它们分别对应：

```text
max reduction
sum reduction
```

---

### 10.6 为什么每个线程计算多个 key？

代码中：

```cpp
for (int j = tid; j < N; j += blockDim.x) {
    ...
}
```

这是 block 内的 stride loop。

假设：

```text
blockDim.x = 256
N = 1024
```

那么：

```text
thread 0 处理 j = 0, 256, 512, 768
thread 1 处理 j = 1, 257, 513, 769
...
thread 255 处理 j = 255, 511, 767, 1023
```

如果 `N = 256`，每个线程刚好处理一个 key。

如果 `N < 256`，只有前 `N` 个线程工作。

---

### 10.7 为什么 output 阶段通常一个线程负责一个 col？

代码中：

```cpp
for (int col = tid; col < d; col += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += scores[j] * V[j * d + col];
    }
    output[row * d + col] = acc * inv_denom;
}
```

因为：

```text
d <= 128
blockDim.x = 256
```

所以通常：

```text
thread 0 负责 output[row, 0]
thread 1 负责 output[row, 1]
...
thread d-1 负责 output[row, d-1]
```

每个线程沿着 `N` 个 value row 做加权求和。

---

## 11. 为什么这个实现适合 LeetGPU 这题？

题目的性能测试主要是：

```text
M = 512
N = 256
d <= 128
```

主 kernel 的工作方式是：

```text
一个 block 处理一个 query row
总 block 数 = M = 512
```

每个 block 内：

```text
256 个线程并行计算 256 个 key 的 score
```

这正好匹配 `N = 256`。

并且只在 shared memory 中保存一行 score：

```text
256 floats
```

不会创建完整的 `M × N` attention matrix。

---

## 12. 更容易理解但更慢的版本

如果只想先写一个最朴素的正确版本，可以让一个 block 负责一个 output 元素：

```text
blockIdx.x = row
blockIdx.y = col
```

每个 block 做：

```text
1. 扫描所有 key，求 max score
2. 再扫描所有 key，求 denominator 和 numerator
3. output[row, col] = numerator / denominator
```

这就是提交代码里的 `attentionStreamingKernel`。

它的优点是：

```text
不需要保存 scores
可以处理较大的 N
```

缺点是：

```text
同一个 row 的 score 会被 d 个 output column 重复计算
```

当 `d = 128` 时，会浪费很多计算。

所以 LeetGPU 性能测试的小 N 场景，推荐使用 `attentionSmallNKernel`。

---

## 13. 常见错误

### 错误 1：忘记除以 `sqrt(d)`

错误写法：

```cpp
score = dot;
```

正确写法：

```cpp
score = dot * rsqrtf((float)d);
```

少了这个缩放，softmax 分布会不一样，结果会错。

---

### 错误 2：softmax 没有减最大值

不稳定写法：

```cpp
float e = expf(score);
```

推荐写法：

```cpp
float e = expf(score - row_max);
```

这样可以避免指数溢出。

---

### 错误 3：softmax 对整个矩阵做，而不是按行做

Attention 中的 softmax 是：

```text
对每个 query row 的 N 个 score 单独做 softmax
```

也就是：

```text
softmax(score[i, :])
```

不是对整个 `M × N` 矩阵一起做 softmax。

---

### 错误 4：矩阵索引写错

容易错的地方是 `K^T`。

数学上是：

```text
QK^T
```

但内存里 `K` 仍然是 `N × d` row-major。

所以 dot product 应该是：

```cpp
dot += Q[row * d + k] * K[j * d + k];
```

不要写成：

```cpp
K[k * N + j]
```

除非你真的把 K 转置存储了。

---

### 错误 5：显式分配 `M × N` 中间矩阵

如果写：

```cpp
cudaMalloc(&scores, M * N * sizeof(float));
```

在大输入时会非常占显存。

这题更适合：

```text
按 query row 流式计算
只保存当前 row 的 scores
```

---

### 错误 6：shared memory 大小不考虑 N

如果写死：

```cpp
__shared__ float scores[256];
```

当 `N > 256` 就会越界。

提交代码里写的是：

```cpp
static constexpr int FAST_MAX_N = 1024;
__shared__ float scores[FAST_MAX_N];
```

并且 host 端只在：

```cpp
if (N <= FAST_MAX_N)
```

时才使用这个 kernel。

---

## 14. 复杂度分析

对每个 query row：

### 14.1 计算 scores

每个 score 是一个长度为 `d` 的 dot product。

总共有 `N` 个 key，所以：

```text
O(N * d)
```

### 14.2 softmax

对 `N` 个 score 做 max 和 sum：

```text
O(N)
```

### 14.3 乘 V

输出有 `d` 个 column，每个 column 都要累加 `N` 个 value：

```text
O(N * d)
```

所以每个 query row 总体是：

```text
O(N * d)
```

所有 `M` 行合起来：

```text
O(M * N * d)
```

这和 attention 本身的计算量一致。

额外空间方面，主 kernel 每个 block 使用：

```text
q_shared: 128 floats
scores:   1024 floats
reduce:   256 floats
```

大约：

```text
(128 + 1024 + 256) * 4 bytes ≈ 5.5 KB
```

非常小。

---

## 15. 如何手动检查代码是否正确？

### 测试 1：`N = 1`

如果只有一个 key/value：

```text
N = 1
```

softmax 结果一定是：

```text
[1]
```

所以：

```text
output[i, :] = V[0, :]
```

无论 Q 和 K 是什么。

---

### 测试 2：Q 和 K 全是 0

如果所有 score 都是 0：

```text
score[j] = 0
```

那么 softmax 是均匀分布：

```text
weight[j] = 1 / N
```

所以输出应该是 V 所有行的平均值：

```text
output[i, col] = average_j V[j, col]
```

---

### 测试 3：某个 key 明显最大

如果 `Q[row]` 和 `K[best]` 点积远大于其他 key，那么 softmax 会接近 one-hot：

```text
weight[best] ≈ 1
```

输出应该接近：

```text
V[best, :]
```

---

### 测试 4：LeetGPU 示例 2

输入：

```text
Q = [[1, 2]]
K = [[1, 0],
     [0, 1]]
V = [[3, 4],
     [5, 6]]
d = 2
```

输出约为：

```text
[[4.34, 5.34]]
```

这可以检查：

```text
1. dot product 是否正确
2. 是否除以 sqrt(d)
3. softmax 是否按行做
4. V 的加权求和是否正确
```

---

## 16. 和 FlashAttention 的关系

这题是标准 attention：

```text
softmax(QK^T / sqrt(d))V
```

真正的 FlashAttention 会进一步做分块 streaming softmax：

```text
不保存完整 scores
按 K/V block 一块一块扫描
维护 running max 和 running denominator
```

它的核心也是稳定 softmax，只是把：

```text
row_max 和 denominator
```

做成在线更新。

本题中 `N = 256`，所以直接把一行 scores 放进 shared memory 已经很合适；等后面做到 Causal Attention、Sliding Window Attention 或 FlashAttention 时，再学习在线 softmax 会更自然。

---

## 17. 总结

Softmax Attention 这题可以压缩成三句话：

```text
1. 对每个 query row，计算它和所有 key 的 scaled dot product。
2. 对这一行 scores 做稳定 softmax。
3. 用 softmax 权重对 V 做加权求和。
```

CUDA 实现的关键点是：

```text
一个 block 处理一个 query row
scores 放 shared memory
block 内 reduction 求 max 和 sum
不显式存完整 M × N attention 矩阵
```

提交代码中的主路径适合 LeetGPU 的性能测试：

```text
M = 512, N = 256, d <= 128
```

同时保留了大 `N` fallback，避免 shared memory 越界。
