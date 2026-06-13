# LeetGPU General Matrix Multiplication (GEMM) 解题思路：FP16 输入、FP32 累加、Shared Memory 分块

> **题目**：General Matrix Multiplication (GEMM)  
> **接口**：
>
> ```cpp
> extern "C" void solve(const half* A, const half* B, half* C,
>                       int M, int N, int K,
>                       float alpha, float beta);
> ```
>
> 计算：
>
> ```text
> C = alpha * (A × B) + beta * C_initial
> ```
>
> 其中：
>
> - `A`: `M × K`, FP16
> - `B`: `K × N`, FP16
> - `C`: `M × N`, FP16，既是输入的 `C_initial`，也是输出
> - 要求乘法累加用 FP32，再写回 FP16

---

## 1. CPU 串行思路

每个输出元素：

```text
C[row, col] = alpha * sum_t A[row, t] * B[t, col] + beta * C_initial[row, col]
```

CPU 伪代码：

```cpp
for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
        float acc = 0.0f;
        for (int t = 0; t < K; ++t) {
            acc += float(A[row * K + t]) * float(B[t * N + col]);
        }
        C[row * N + col] = half(alpha * acc + beta * float(C[row * N + col]));
    }
}
```

---

## 2. CUDA 并行划分

最自然的 CUDA 映射：

```text
一个线程负责 C 的一个元素
```

也就是：

```text
row = blockIdx.y * TILE + threadIdx.y
col = blockIdx.x * TILE + threadIdx.x
```

但是如果每个线程都直接从 global memory 读取 A 和 B，会重复读取很多数据。

所以使用 shared memory tiled GEMM：

```text
每个 block 计算 C 的一个 TILE × TILE 小块
每轮加载 A 的一块和 B 的一块到 shared memory
在 shared memory 中做 TILE 次乘加
循环扫完整个 K 维度
```

---

## 3. 为什么要 FP32 累加？

题目要求：

```text
输入输出是 FP16，但 accumulation 用 FP32
```

原因是 FP16 精度低，如果 1024 个乘积都用 FP16 累加，误差会明显变大。

代码中需要：

```cpp
float acc = 0.0f;
acc += __half2float(a) * __half2float(b);
```

最后再：

```cpp
C[...] = __float2half(result);
```

---

## 4. LeetGPU 可提交参考代码：Shared Memory 版

这份代码不使用 cuBLAS，只用 CUDA native 特性。它不是 Tensor Core 极限性能版本，但结构清晰、正确性好，适合理解 GEMM 分块。

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr int TILE = 16;

__global__ void gemmTiledKernel(const half* __restrict__ A,
                                const half* __restrict__ B,
                                half* __restrict__ C,
                                int M,
                                int N,
                                int K,
                                float alpha,
                                float beta) {
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int tile = 0; tile < K; tile += TILE) {
        const int a_col = tile + tx;
        const int b_row = tile + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }

        if (b_row < K && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const int idx = row * N + col;
        const float old_c = __half2float(C[idx]);
        const float out = alpha * acc + beta * old_c;
        C[idx] = __float2half(out);
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K,
                      float alpha, float beta) {
    if (M <= 0 || N <= 0 || K <= 0) {
        return;
    }

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    gemmTiledKernel<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
```

---

## 5. 进一步优化方向

LeetGPU 这题允许 WMMA。真正追性能时，可以用 Tensor Core：

```text
nvcuda::wmma::fragment<matrix_a, 16,16,16, half, row_major>
nvcuda::wmma::fragment<matrix_b, 16,16,16, half, row_major>
nvcuda::wmma::fragment<accumulator, 16,16,16, float>
```

WMMA 的核心优势是：

```text
一个 warp 直接计算 16×16×16 MMA tile
使用 Tensor Cores，远快于普通 CUDA core 的逐元素乘加
```

但 WMMA 边界处理和 fragment 布局更复杂；先掌握 shared memory tiled GEMM 更重要。

---

## 6. 常见错误

### 错误 1：把矩阵维度写反

本题：

```text
A: M × K
B: K × N
C: M × N
```

索引是：

```cpp
A[row * K + t]
B[t * N + col]
C[row * N + col]
```

### 错误 2：忘记 beta * C_initial

题目不是简单 `C = A × B`，而是：

```text
C = alpha * A × B + beta * C_initial
```

所以写回前要先读旧的 `C[idx]`。

### 错误 3：用 FP16 累加

应该：

```cpp
float acc = 0.0f;
```

不要用：

```cpp
half acc;
```
