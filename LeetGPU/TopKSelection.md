# LeetGPU Top K Selection 解题思路：分块 Top-K 与两阶段合并

> **题目**：Top K Selection  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* input, float* output, int N, int k);
> ```
>
> 从长度为 `N` 的数组中选出最大的 `k` 个元素，并按降序写入：
>
> ```text
> output[0] >= output[1] >= ... >= output[k-1]
> ```
>
> 性能测试常见规模：
>
> ```text
> N = 50,000,000
> k = 100
> ```

---

## 1. CPU 串行思路

最简单的方法是排序：

```cpp
sort(input, input + N, greater<float>());
copy first k elements to output;
```

但是完整排序复杂度是：

```text
O(N log N)
```

而我们只需要前 `k` 个，不需要把所有元素都排好。

当 `k = 100` 且 `N = 50,000,000` 时，更适合做 selection。

---

## 2. GPU 两阶段思路

把输入切成很多 chunk：

```text
chunk0, chunk1, chunk2, ...
```

第一阶段：每个 block 负责一个 chunk，找出这个 chunk 内部的 top k。

```text
chunk0 -> local top k
chunk1 -> local top k
chunk2 -> local top k
...
```

第二阶段：把所有 block 产生的 local top k 合并，再选出全局 top k。

为什么这样正确？

> 如果一个元素连自己 chunk 内的 top k 都进不去，那么同一个 chunk 里至少有 k 个元素不小于它，所以它不可能成为全局 top k。

因此：

```text
全局 top k 一定包含在所有 local top k 的并集中
```

---

## 3. 如何处理重复值？

如果数组里有重复值，比如：

```text
[5, 5, 5, 4]
k = 2
```

输出应该是：

```text
[5, 5]
```

为了不把重复值误删，选择时不能只用：

```text
value < previous_value
```

而应该把元素看成一个 pair：

```text
(value, index)
```

比较规则：

```text
value 越大越靠前
value 相等时，index 越小越靠前
```

这样每个元素都有唯一顺序，重复值也可以被选多次。

---

## 4. LeetGPU 参考提交代码

下面代码针对 LeetGPU 常见测试 `k = 100` 设计，使用两阶段：

```text
localTopKKernel：每个 chunk 选 k 个候选
finalTopKKernel：从候选数组中选全局 k 个
```

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;
static constexpr int CHUNK_SIZE = 4096;
static constexpr float NEG_INF = -3.4028234663852886e38f;

__device__ bool betterPair(float a_val, int a_idx, float b_val, int b_idx) {
    if (a_idx < 0) return false;
    if (b_idx < 0) return true;
    if (a_val > b_val) return true;
    if (a_val < b_val) return false;
    return a_idx < b_idx;
}

__device__ bool belowPrevious(float val,
                              int idx,
                              float prev_val,
                              int prev_idx,
                              bool has_prev) {
    if (!has_prev) return true;
    if (val < prev_val) return true;
    if (val > prev_val) return false;
    return idx > prev_idx;
}

// 每个 block 处理一个连续 chunk，输出该 chunk 的 top k。
__global__ void localTopKKernel(const float* __restrict__ input,
                                float* __restrict__ candidates,
                                int N,
                                int k) {
    __shared__ float s_val[BLOCK_THREADS];
    __shared__ int s_idx[BLOCK_THREADS];

    const int tid = threadIdx.x;
    const int start = blockIdx.x * CHUNK_SIZE;
    const int end = (start + CHUNK_SIZE < N) ? (start + CHUNK_SIZE) : N;

    float prev_val = 0.0f;
    int prev_idx = -1;
    bool has_prev = false;

    for (int rank = 0; rank < k; ++rank) {
        float best_val = NEG_INF;
        int best_idx = -1;

        for (int idx = start + tid; idx < end; idx += blockDim.x) {
            const float v = input[idx];
            if (belowPrevious(v, idx, prev_val, prev_idx, has_prev) &&
                betterPair(v, idx, best_val, best_idx)) {
                best_val = v;
                best_idx = idx;
            }
        }

        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (betterPair(s_val[tid + offset], s_idx[tid + offset],
                               s_val[tid], s_idx[tid])) {
                    s_val[tid] = s_val[tid + offset];
                    s_idx[tid] = s_idx[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            candidates[blockIdx.x * k + rank] = (s_idx[0] >= 0) ? s_val[0] : NEG_INF;
        }

        __syncthreads();

        prev_val = s_val[0];
        prev_idx = s_idx[0];
        has_prev = (prev_idx >= 0);

        __syncthreads();
    }
}

// 从所有候选里再选全局 top k。
__global__ void finalTopKKernel(const float* __restrict__ candidates,
                                float* __restrict__ output,
                                int total_candidates,
                                int k) {
    __shared__ float s_val[BLOCK_THREADS];
    __shared__ int s_idx[BLOCK_THREADS];

    const int tid = threadIdx.x;

    float prev_val = 0.0f;
    int prev_idx = -1;
    bool has_prev = false;

    for (int rank = 0; rank < k; ++rank) {
        float best_val = NEG_INF;
        int best_idx = -1;

        for (int idx = tid; idx < total_candidates; idx += blockDim.x) {
            const float v = candidates[idx];
            if (belowPrevious(v, idx, prev_val, prev_idx, has_prev) &&
                betterPair(v, idx, best_val, best_idx)) {
                best_val = v;
                best_idx = idx;
            }
        }

        s_val[tid] = best_val;
        s_idx[tid] = best_idx;
        __syncthreads();

        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                if (betterPair(s_val[tid + offset], s_idx[tid + offset],
                               s_val[tid], s_idx[tid])) {
                    s_val[tid] = s_val[tid + offset];
                    s_idx[tid] = s_idx[tid + offset];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[rank] = s_val[0];
        }

        __syncthreads();

        prev_val = s_val[0];
        prev_idx = s_idx[0];
        has_prev = (prev_idx >= 0);

        __syncthreads();
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0) {
        return;
    }

    const int num_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    float* candidates = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&candidates),
               static_cast<size_t>(num_chunks) * k * sizeof(float));

    localTopKKernel<<<num_chunks, BLOCK_THREADS>>>(input, candidates, N, k);
    finalTopKKernel<<<1, BLOCK_THREADS>>>(candidates, output, num_chunks * k, k);

    cudaFree(candidates);
    cudaDeviceSynchronize();
}
```

---

## 5. 复杂度

第一阶段每个 chunk 选 `k` 次，每次扫描 chunk：

```text
O(N × k)
```

第二阶段候选数量是：

```text
num_chunks × k
```

再选 `k` 次：

```text
O(num_chunks × k²)
```

对 LeetGPU 常见参数：

```text
N = 50,000,000
k = 100
CHUNK_SIZE = 4096
num_chunks ≈ 12208
候选数量 ≈ 1,220,800
```

候选数组大约：

```text
1,220,800 × 4 bytes ≈ 4.9 MB
```

可接受。

---

## 6. 更高性能的方向

这份代码重在清楚和通用，进一步优化可以考虑：

- 每个线程维护小 top-k，再 block 内合并
- 使用 bitonic sort 对每个 block 的候选排序
- 用 radix/select 找阈值，再 compact 大于阈值的元素
- 对 `k <= 128` 做专门模板展开
- 减少 final 单 block 合并的串行瓶颈

---

## 7. 常见错误

### 错误 1：只找不同的值，丢掉重复值

Top K 选的是元素，不是去重后的值。

### 错误 2：完整排序整个 N

完整排序能做，但 `N = 50,000,000` 时成本太高。

### 错误 3：输出没有降序

题目要求：

```text
output 按 descending order 排列
```

两阶段选择每次选“当前剩余最大”，天然得到降序。
