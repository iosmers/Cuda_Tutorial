# LeetGPU Matrix Transpose 解题思路

> **难度**：Easy  
> **题型**：索引变换 / 访存优化  
> **详细长文**：[MatrixTranspose.md](MatrixTranspose.md)

---

## 1. 题目要求

输入矩阵 `input` 形状为：

```text
rows × cols
```

输出矩阵 `output` 形状为：

```text
cols × rows
```

转置公式：

```text
output[col, row] = input[row, col]
```

row-major 索引：

```cpp
input[row * cols + col]
output[col * rows + row]
```

---

## 2. Naive 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
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

## 3. 优化思路

Naive transpose 的问题是：

```text
读 input 是连续的，但写 output 是跨 stride 的，不利于合并访存。
```

优化版使用 shared memory tile：

```text
1. 以 TILE×TILE 读入 input，读是 coalesced。
2. 在 shared memory 中交换 row/col。
3. 再以 coalesced 方式写 output。
```

更详细解释见：[MatrixTranspose.md](MatrixTranspose.md)。
