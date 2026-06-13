# LeetGPU Sparse Matrix-Vector Multiplication 解题思路：稀疏矩阵向量乘 SpMV

> **题目**：Sparse Matrix-Vector Multiplication  
> **接口**：
>
> ```cpp
> extern "C" void solve(const float* A, const float* x, float* y,
>                       int M, int N, int nnz);
> ```
>
> `A` 是 `M × N` 的矩阵，`x` 是长度为 `N` 的向量，输出 `y` 长度为 `M`：
>
> ```text
> y = A × x
> y[i] = sum_j A[i, j] * x[j]
> ```
>
> 注意：这道题虽然叫 Sparse Matrix-Vector Multiplication，但接口里给的是 **dense row-major 矩阵 A**，不是 CSR/COO 格式；`nnz` 只是告诉你非零元素数量，不能直接拿到非零元素位置。

---

## 1. CPU 串行思路

如果在 CPU 上写，就是每一行做一次 dot product：

```cpp
for (int row = 0; row < M; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
        sum += A[row * N + col] * x[col];
    }
    y[row] = sum;
}
```

因为题目说矩阵约 60%~70% 是 0，所以可以稍微加一个判断：

```cpp
float a = A[row * N + col];
if (a != 0.0f) {
    sum += a * x[col];
}
```

但是由于输入不是 CSR 格式，我们仍然必须扫描整行 `N` 个元素才能知道哪些位置非零。

---

## 2. CUDA 并行划分

性能测试规模大致是：

```text
M = 1000
N = 10000
```

比较自然的划分是：

```text
一个 block 负责 A 的一行
blockIdx.x = row
block 内 256 个线程并行扫这一行的不同列
```

也就是：

```text
thread 0 处理 col = 0, 256, 512, ...
thread 1 处理 col = 1, 257, 513, ...
...
```

每个线程算出自己的局部和，再在 block 内用 shared memory 做 reduction，最后由 `thread 0` 写出：

```text
y[row]
```

---

## 3. 为什么不用 atomicAdd？

可以让很多线程都执行：

```cpp
atomicAdd(&y[row], partial);
```

但这会让同一行的所有线程竞争同一个 global memory 地址。

更好的方式是：

```text
block 内先归约到一个 sum
只写一次 y[row]
```

这避免了大量 global atomic。

---

## 4. LeetGPU 可提交参考代码

```cpp
#include <cuda_runtime.h>

static constexpr int BLOCK_THREADS = 256;

__global__ void spmvKernel(const float* __restrict__ A,
                           const float* __restrict__ x,
                           float* __restrict__ y,
                           int M,
                           int N) {
    __shared__ float sdata[BLOCK_THREADS];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= M) {
        return;
    }

    float sum = 0.0f;
    const int row_base = row * N;

    // 一个 block 处理一整行，block 内线程分摊列方向工作。
    for (int col = tid; col < N; col += blockDim.x) {
        const float a = A[row_base + col];
        if (a != 0.0f) {
            sum += a * x[col];
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    // block 内 reduction。
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        y[row] = sdata[0];
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y,
                      int M, int N, int nnz) {
    // 这个接口没有 CSR/COO 的索引数组，所以 nnz 无法直接用于跳过非零位置。
    // 为了避免 unused parameter warning，也可以显式忽略它。
    (void)nnz;

    if (M <= 0 || N <= 0) {
        return;
    }

    spmvKernel<<<M, BLOCK_THREADS>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
```

---

## 5. 复杂度

因为输入是 dense row-major，所以每个元素都要至少读一次：

```text
时间复杂度：O(M × N)
```

如果题目给的是 CSR 格式，复杂度可以变成：

```text
O(nnz)
```

但本题接口没有 `row_ptr / col_idx / values`，因此不能做到真正 CSR SpMV。

---

## 6. 常见错误

### 错误 1：误以为 A 是 CSR

本题签名只有：

```cpp
const float* A
```

没有：

```cpp
row_ptr, col_idx, values
```

所以 `A[row * N + col]` 才是正确索引方式。

### 错误 2：忘记 block 内 reduction

每个线程只算一部分列，必须把 partial sum 合起来。

### 错误 3：把 `nnz` 当成矩阵长度

矩阵真实存储长度仍是：

```text
M × N
```

`nnz` 只是非零元素数量，不代表 `A` 的数组长度。
