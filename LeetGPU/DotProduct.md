# LeetGPU Dot Product 解题思路：向量点积与并行归约

> **题目**：Dot Product  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* B, float* result, int N);
> ```
>
> 计算：
>
> ```text
> result = sum_i A[i] * B[i]
> ```

---

## 1. CPU 串行思路

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    sum += A[i] * B[i];
}
*result = sum;
```

这本质上是一个 reduction：很多元素最终归约成一个标量。

---

## 2. CUDA 并行思路

直接让所有线程都：

```cpp
atomicAdd(result, A[i] * B[i]);
```

虽然正确，但 global atomic 次数是 `N`，会很慢。

更好的方式：

```text
每个线程处理多个元素，得到 local_sum
每个 block 内用 shared memory reduction 得到 block_sum
每个 block 只 atomicAdd 一次到 result
```

这样 global atomic 次数从：

```text
N 次
```

变成：

```text
blocks 次
```

---

## 3. LeetGPU 可提交参考代码

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;
static constexpr int MAX_BLOCKS = 4096;

__global__ void dotProductKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ result,
                                 int N) {
    __shared__ float sdata[BLOCK_THREADS];

    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = global_tid; i < N; i += stride) {
        sum += A[i] * B[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemset(result, 0, sizeof(float));

    if (N <= 0) {
        cudaDeviceSynchronize();
        return;
    }

    int blocks = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    dotProductKernel<<<blocks, BLOCK_THREADS>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
```

---

## 4. 为什么要 `cudaMemset(result, 0)`？

因为 kernel 里使用：

```cpp
atomicAdd(result, block_sum);
```

如果 `result` 初始不是 0，就会在旧值上累加，答案错误。

---

## 5. 复杂度

每个元素读一次：

```text
时间复杂度 O(N)
```

额外空间只有每个 block 的 shared memory：

```text
256 floats
```

---

## 6. 常见错误

### 错误 1：一个线程串行算全部

这样正确但没有利用 GPU 并行能力。

### 错误 2：每个元素都 global atomic

会产生大量 atomic contention。

### 错误 3：忘记清零 result

使用 atomic 累加前必须清零输出。
