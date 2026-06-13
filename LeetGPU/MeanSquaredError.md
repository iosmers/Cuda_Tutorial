# LeetGPU Mean Squared Error 解题思路：平方误差均值与并行归约

> **题目**：Mean Squared Error  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* predictions,
>                       const float* targets,
>                       float* mse,
>                       int N);
> ```
>
> 计算：
>
> ```text
> mse = (1 / N) * sum_i (predictions[i] - targets[i])^2
> ```

---

## 1. CPU 串行思路

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) {
    float diff = predictions[i] - targets[i];
    sum += diff * diff;
}
*mse = sum / N;
```

这和 Dot Product 很像，本质也是 reduction。

---

## 2. CUDA 并行思路

每个线程处理多个元素：

```cpp
diff = predictions[i] - targets[i]
local_sum += diff * diff
```

然后：

```text
block 内 shared memory reduction
每个 block atomicAdd 一次到 mse
```

为了少一个除法 kernel，可以让每个 block 写入：

```text
block_sum / N
```

所有 block 加起来就是：

```text
sum(block_sum) / N
```

也就是最终 MSE。

---

## 3. LeetGPU 可提交参考代码

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;
static constexpr int MAX_BLOCKS = 4096;

__global__ void mseKernel(const float* __restrict__ predictions,
                          const float* __restrict__ targets,
                          float* __restrict__ mse,
                          int N) {
    __shared__ float sdata[BLOCK_THREADS];

    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = global_tid; i < N; i += stride) {
        const float diff = predictions[i] - targets[i];
        sum += diff * diff;
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
        atomicAdd(mse, sdata[0] / static_cast<float>(N));
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions,
                      const float* targets,
                      float* mse,
                      int N) {
    cudaMemset(mse, 0, sizeof(float));

    if (N <= 0) {
        cudaDeviceSynchronize();
        return;
    }

    int blocks = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
    if (blocks > MAX_BLOCKS) {
        blocks = MAX_BLOCKS;
    }

    mseKernel<<<blocks, BLOCK_THREADS>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
```

---

## 4. 和 Dot Product 的关系

MSE 可以看成对向量：

```text
diff = predictions - targets
```

做一次 dot product：

```text
mse = dot(diff, diff) / N
```

所以优化套路完全一样：

```text
grid-stride loop + block reduction + 少量 global atomic
```

---

## 5. 常见错误

### 错误 1：忘记除以 N

题目要的是均值，不是平方误差和。

### 错误 2：每个元素都 atomicAdd

这样 global atomic 次数是 `N`，性能差。

### 错误 3：没有清零 mse

使用 atomic 累加前必须：

```cpp
cudaMemset(mse, 0, sizeof(float));
```
