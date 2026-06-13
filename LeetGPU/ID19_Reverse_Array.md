# LeetGPU Reverse Array 解题思路

> **难度**：Easy  
> **题型**：索引变换 / 原地交换  
> **接口**：
>
> ```cpp
> extern "C" void solve(float* input, int N);
> ```

---

## 1. 题目要求

原地反转数组：

```text
input[i] <-> input[N - 1 - i]
```

只需要处理前一半元素，后一半会被交换过来。

---

## 2. CUDA 并行思路

一个线程负责一对交换：

```cpp
i = thread id
j = N - 1 - i
```

只允许：

```cpp
i < N / 2
```

否则会重复交换，把数组又换回去。

---

## 3. 可提交代码

```cpp
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        int j = N - 1 - i;
        float tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int half = N / 2;
    int blocksPerGrid = (half + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
```

---

## 4. 常见错误

- 让所有 `N` 个线程都交换，会重复交换。
- 奇数长度时中间元素不用动。
- 原地操作不需要额外 output。
