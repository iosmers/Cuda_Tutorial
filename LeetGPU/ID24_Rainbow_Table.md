# LeetGPU Rainbow Table 解题思路

> **难度**：Easy  
> **题型**：每线程内部小循环 / Parallel Hashing  
> **接口**：
>
> ```cpp
> extern "C" void solve(const int* input, unsigned int* output, int N, int R);
> ```

---

## 1. 题目要求

对每个输入整数独立执行 `R` 轮 FNV-1a hash：

```text
value = input[i]
repeat R times:
    value = fnv1a_hash(value)
output[i] = value
```

每个元素之间没有依赖，非常适合一线程一元素。

---

## 2. 可提交代码

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

## 3. 讲课重点

- 这仍然是 map：每个元素独立。
- 不同的是每个线程内部有一个长度为 `R` 的小循环。
- hash 函数应声明为 `__device__`，在 kernel 内调用。
