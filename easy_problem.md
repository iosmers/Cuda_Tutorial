# LeetGPU Easy 题速刷笔记：CUDA 入门题其实就这几种套路

最近把仓库里 LeetGPU 的 Easy 题重新过了一遍。刷完之后感觉很明显：这些题并不是在考什么很花的算法，更多是在训练 CUDA 的基本手感——线程怎么编号、二维矩阵怎么展平、图像通道怎么索引、边界判断别忘、什么时候一个线程算一个输出。

这篇就按我自己的理解做个整理。每道题都放了三块：**题目在做什么、怎么想、可提交代码**。代码默认符合 LeetGPU 的风格：输入输出已经在 GPU 上，`solve` 里只负责启动 kernel，最后 `cudaDeviceSynchronize()` 等待执行完成。

先记住一个最常用的模板：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    output[i] = f(input[i]);
}
```

LeetGPU Easy 里有一大半题，本质上都是这个模板换个外壳。

---

## 1. Vector Addition：CUDA 第一题

### 题目介绍

给两个长度为 `N` 的 float 数组 `A`、`B`，输出：

```text
C[i] = A[i] + B[i]
```

这是最标准的 CUDA 入门题。CPU 上是一个 for 循环，GPU 上就是把每个元素分给一个线程。

### 思路

一个线程负责一个下标 `i`。因为 block 数通常是向上取整算出来的，所以最后一个 block 里可能有多余线程，必须加 `if (i < N)`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

---

## 2. Matrix Addition：二维矩阵先当一维数组看

### 题目介绍

给两个 `N × N` 矩阵逐元素相加：

```text
C[row, col] = A[row, col] + B[row, col]
```

矩阵是 row-major 存储，但这题完全可以先不管 `row` 和 `col`，直接把矩阵看成长度为 `N * N` 的数组。

### 思路

总元素数是 `N * N`，每个线程处理一个线性下标 `idx`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

---

## 3. Matrix Copy：最朴素的 global memory 读写

### 题目介绍

复制一个 `N × N` 矩阵：

```text
B[i] = A[i]
```

这题当然可以想到 `cudaMemcpy`，但在线上题里一般希望你写一个 kernel，所以还是按一线程一元素来。

### 思路

依然把矩阵展平成 `N * N` 个元素，每个线程负责拷贝一个元素。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_copy_kernel(const float* A, float* B, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        B[idx] = A[idx];
    }
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    matrix_copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, total);
    cudaDeviceSynchronize();
}
```

---

## 4. Color Inversion：注意 RGBA 的 alpha 不要动

### 题目介绍

输入是一张 RGBA 图像，每个像素 4 个 `unsigned char`：

```text
R, G, B, A
```

要求颜色反转：

```text
R = 255 - R
G = 255 - G
B = 255 - B
A 不变
```

### 思路

一个线程处理一个像素，而不是一个通道。像素编号是 `pixel`，对应数组起点是：

```cpp
int base = pixel * 4;
```

然后只改 `base + 0/1/2`，不要改 alpha。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void color_inversion_kernel(unsigned char* image, int width, int height) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel < total_pixels) {
        int base = pixel * 4;
        image[base + 0] = 255 - image[base + 0];
        image[base + 1] = 255 - image[base + 1];
        image[base + 2] = 255 - image[base + 2];
        // image[base + 3] 是 alpha，保持不变
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    color_inversion_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```

---

## 5. RGB to Grayscale：三个通道变一个灰度值

### 题目介绍

输入是 RGB 格式的一维数组：

```text
R, G, B, R, G, B, ...
```

每个像素输出一个灰度值：

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### 思路

一个线程处理一个像素。和上一题不同，这里每个像素是 3 个 float，输出只有 1 个 float。

常见错误就是把输入当成 RGBA，或者把输出长度写成 `width * height * 3`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output,
                                        int width, int height) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (pixel < total_pixels) {
        int base = pixel * 3;
        float r = input[base + 0];
        float g = input[base + 1];
        float b = input[base + 2];
        output[pixel] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
```

---

## 6. ReLU：最简单的激活函数

### 题目介绍

ReLU 的公式是：

```text
ReLU(x) = max(0, x)
```

输入输出长度都是 `N`。

### 思路

典型 elementwise 题，一个线程读一个 `x`，判断一下正负，然后写回输出。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.0f;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 7. Leaky ReLU：负数区域留一点斜率

### 题目介绍

Leaky ReLU 和 ReLU 很像，只是负数不直接变成 0，而是乘一个小系数。题目里 `alpha = 0.01`：

```text
f(x) = x,         if x > 0
f(x) = 0.01 * x, if x <= 0
```

### 思路

和 ReLU 只有一行表达式的区别，依然是一线程一元素。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = x > 0.0f ? x : 0.01f * x;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 8. Sigmoid Activation：记得用 expf

### 题目介绍

Sigmoid 公式：

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

### 思路

每个线程算一个元素。CUDA C++ 里 float 版本指数函数用 `expf`，不要随手写成 double 版本的 `exp`。

### 实现代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = X[i];
        Y[i] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
```

---

## 9. SiLU：Sigmoid 再乘回 x

### 题目介绍

SiLU，也常叫 Swish，公式是：

```text
SiLU(x) = x * sigmoid(x)
```

### 思路

还是 elementwise。先算 sigmoid，再乘上原来的 `x`。

### 实现代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        output[i] = x * sig;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
```

---

## 10. Value Clipping：把数夹在区间里

### 题目介绍

给定上下界 `lo` 和 `hi`，把每个元素裁剪到这个范围内：

```text
output[i] = lo,       if input[i] < lo
output[i] = hi,       if input[i] > hi
output[i] = input[i], otherwise
```

也就是常说的 clamp。

### 思路

每个线程处理一个元素，判断两次即可。想写短一点也可以用 `fminf(fmaxf(x, lo), hi)`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output,
                            float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        if (x < lo) x = lo;
        if (x > hi) x = hi;
        output[i] = x;
    }
}

extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```

---

## 11. SwiGLU：输入拆成两半

### 题目介绍

SwiGLU 是门控激活。题目里输入长度 `N` 为偶数，先拆成两半：

```text
x1 = input[0 .. N/2 - 1]
x2 = input[N/2 .. N - 1]
```

然后：

```text
SwiGLU(x1, x2) = SiLU(x1) * x2
```

输出长度是 `N / 2`。

### 思路

线程数开 `N/2` 个就够了。第 `i` 个线程读：

```cpp
x1 = input[i]
x2 = input[halfN + i]
```

然后算 `x1 * sigmoid(x1) * x2`。

### 实现代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[halfN + i];
        float sig = 1.0f / (1.0f + expf(-x1));
        float silu = x1 * sig;
        output[i] = silu * x2;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

---

## 12. GEGLU：GELU 放在第二半输入上

### 题目介绍

GEGLU 也是门控激活。输入同样拆成两半：

```text
x1 = input[0 .. N/2 - 1]
x2 = input[N/2 .. N - 1]
```

GELU 公式：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

题目里的 GEGLU 是：

```text
output[i] = x1[i] * GELU(x2[i])
```

### 思路

这题容易错的点是：GELU 是作用在第二半 `x2` 上，不是作用在 `x1` 上。输出长度也是 `N/2`。

### 实现代码

```cpp
#include <cuda_runtime.h>
#include <math.h>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[halfN + i];
        float gelu = 0.5f * x2 * (1.0f + erff(x2 * 0.7071067811865476f));
        output[i] = x1 * gelu;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
```

---

## 13. Reverse Array：原地反转只处理前一半

### 题目介绍

把数组原地反转：

```text
input[i] <-> input[N - 1 - i]
```

### 思路

只需要处理前 `N/2` 个元素。假如你让所有 `N` 个线程都去交换，会出现一个很经典的问题：前半部分刚换完，后半部分又换回来，最后等于没换。

奇数长度时，中间那个元素不用动。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void reverse_array_kernel(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        int j = N - 1 - i;
        float tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
    }
}

extern "C" void solve(float* input, int N) {
    int half = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (half + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```

---

## 14. Interleave Arrays：一个线程写两个位置

### 题目介绍

给两个长度为 `N` 的数组 `A` 和 `B`，交错写到输出中：

```text
output = [A0, B0, A1, B1, A2, B2, ...]
```

公式就是：

```text
output[2*i]     = A[i]
output[2*i + 1] = B[i]
```

### 思路

最直接的写法是一个线程处理一对元素，也就是读 `A[i]`、`B[i]`，写 `output[2*i]` 和 `output[2*i+1]`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[2 * i] = A[i];
        output[2 * i + 1] = B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
```

---

## 15. 1D Convolution：一个线程算一个输出窗口

### 题目介绍

一维 valid convolution。输入长度是 `input_size`，卷积核长度是 `kernel_size`，输出长度为：

```text
output_size = input_size - kernel_size + 1
```

题目实际采用的是直接相关形式：

```text
output[i] = sum_j input[i + j] * kernel[j]
```

也就是说，不需要把 kernel 翻转。

### 思路

一个线程负责一个输出位置 `i`。线程内部循环 `kernel_size` 次，把对应窗口里的元素乘起来再累加。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel,
                                      float* output, int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (i < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            sum += input[i + j] * kernel[j];
        }
        output[i] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output,
                                                              input_size, kernel_size);
    cudaDeviceSynchronize();
}
```

---

## 16. Matrix Multiplication：Naive GEMM baseline

### 题目介绍

给矩阵：

```text
A: M × N
B: N × K
C: M × K
```

计算：

```text
C[row, col] = sum_t A[row, t] * B[t, col]
```

这题是 GEMM 的最朴素版本。先写对 naive 版，再去看 shared memory tiled GEMM 会顺很多。

### 思路

一个线程负责 `C` 的一个元素。这里建议用二维 block：

```cpp
col = blockIdx.x * blockDim.x + threadIdx.x;
row = blockIdx.y * blockDim.y + threadIdx.y;
```

row-major 下标要记清楚：

```text
A[row * N + t]
B[t * K + col]
C[row * K + col]
```

最容易写错的是 `B` 的下标，因为 `B` 的列数是 `K`。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int t = 0; t < N; ++t) {
            sum += A[row * N + t] * B[t * K + col];
        }
        C[row * K + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

---

## 17. Matrix Transpose：索引题，也是访存优化的第一课

### 题目介绍

输入矩阵形状为 `rows × cols`，输出矩阵形状为 `cols × rows`：

```text
output[col, row] = input[row, col]
```

row-major 下标是：

```text
input[row * cols + col]
output[col * rows + row]
```

### 思路

最简单版本还是一个线程负责一个元素。二维线程映射中，`x` 方向对应列，`y` 方向对应行。

这题很适合后面学优化：naive 版本读 input 是连续的，但写 output 是跨 stride 的；想进一步提速，需要用 shared memory 做 tile 转置，再合并写回。不过 Easy 阶段先把索引写对更重要。

### 实现代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output,
                                        int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

---

## 18. Rainbow Table：每个线程里面多跑几轮 hash

### 题目介绍

对每个输入整数独立执行 `R` 轮 FNV-1a hash：

```text
value = input[i]
repeat R times:
    value = fnv1a_hash(value)
output[i] = value
```

虽然名字听起来像密码学题，但 Easy 版本仍然是 map：每个元素之间没有依赖。

### 思路

一个线程处理一个 `input[i]`，线程内部跑 `R` 轮小循环。hash 函数在 device 端调用，所以要写成 `__device__` 函数。

### 实现代码

```cpp
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;

    unsigned int hash = OFFSET_BASIS;
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int value = static_cast<unsigned int>(input[i]);
        for (int r = 0; r < R; ++r) {
            value = fnv1a_hash(value);
        }
        output[i] = value;
    }
}

extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
```

---

## 19. Simple Inference：这题是 PyTorch GPU tensor，不是 CUDA C++ kernel

### 题目介绍

给定 GPU 上的输入 tensor 和一个训练好的 `torch.nn.Linear` 模型，计算：

```text
output = input @ weight.T + bias
```

题目允许直接使用 PyTorch。

### 思路

既然 `input`、`model`、`output` 都已经在 GPU 上，最简单的写法就是调用 `model(input)`，再用 `output.copy_(...)` 写入题目给定的输出 tensor。

注意加上 `torch.no_grad()`，推理时不需要构建计算图。

### 实现代码

```python
import torch
import torch.nn as nn

# input, model, output 都已经在 GPU 上
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    with torch.no_grad():
        output.copy_(model(input))
```

如果不用 `model(input)`，手写公式也就是：

```python
output.copy_(input @ model.weight.T + model.bias)
```

---

## 最后总结：Easy 题主要练这几件事

刷完这些 Easy 题，我觉得最值得带走的不是某一道题的答案，而是下面这些固定动作：

1. **一维索引一定要熟**  
   `int i = blockIdx.x * blockDim.x + threadIdx.x;`

2. **二维矩阵先搞清楚 row-major**  
   `idx = row * num_cols + col`，矩阵乘法、转置基本都卡在这里。

3. **边界判断不要省**  
   block 数向上取整后，多出来的线程一定会存在，`if (idx < total)` 是基本功。

4. **输出长度要重新算**  
   比如 1D Convolution 的输出是 `input_size - kernel_size + 1`，SwiGLU/GEGLU 的输出是 `N/2`。

5. **图像题先看通道布局**  
   RGBA 是 4 个通道，RGB 是 3 个通道；Color Inversion 还要注意 alpha 不变。

6. **Easy 不等于没价值**  
   Vector Add、Matrix Add、Activation 这些题看着简单，但它们把 CUDA kernel 的基本形式练顺了。后面做 reduction、scan、GEMM、attention，本质上也是在这些基础上继续叠复杂度。

如果刚开始学 CUDA，我建议不要急着上来就看各种极限优化。先把这些 Easy 题用最朴素的版本写对，尤其是索引和边界。等这些不再需要脑子想了，再回头看 shared memory、coalesced access、bank conflict，理解会轻松很多。
