# 阶段四：CUDA 进阶与实战项目

> **目标**：会用 CUDA 生态库，完成一个综合项目。

---

## 1. CUDA 生态库概览

自己写 kernel 适合学习与极致优化；生产中优先用成熟库。

| 库 | 用途 |
|----|------|
| **cuBLAS** | 稠密线性代数（GEMM、GEMV、AXPY 等） |
| **cuBLASLt / cuDNN** | 深度学习基础算子（cuDNN 面向 conv/pool/rnn） |
| **cuRAND** | GPU 随机数生成 |
| **cuFFT** | 快速傅里叶变换 |
| **cuSPARSE** | 稀疏矩阵运算 |
| **Thrust** | 类 STL 的高层并行算法（sort/reduce/scan） |
| **CUB** | Thrust 的底层构建块，灵活性更高 |
| **NCCL** | 多 GPU 通信集合 |

---

## 2. cuBLAS GEMM 实战

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// C = alpha * A * B + beta * C
// 注意：cuBLAS 使用 column-major，常见做法：把 A*B 等价算成 B^T * A^T
const float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,              // 注意顺序
            &alpha,
            d_B, N,               // B is KxN in row-major => NxK col-major
            d_A, K,               // A is MxK in row-major => KxM col-major
            &beta,
            d_C, N);              // C MxN row-major => NxM col-major

cublasDestroy(handle);
```

**对比实验**：
- 自写 Tiled GEMM（阶段二） vs cuBLAS
- 通常 cuBLAS 能达到理论峰值的 80~95%

编译：`nvcc xxx.cu -lcublas`

---

## 3. Thrust 快速上手

类 STL 风格，一行代码搞定并行排序/归约/扫描。

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

thrust::device_vector<int> d(h_begin, h_end);   // 自动 H2D

thrust::sort(d.begin(), d.end());               // 排序
int sum = thrust::reduce(d.begin(), d.end());   // 求和
thrust::inclusive_scan(d.begin(), d.end(), d.begin());  // 前缀和
```

---

## 4. 进阶主题（选一深入）

### A. Unified Memory

```cpp
float *data;
cudaMallocManaged(&data, N * sizeof(float));
// CPU 直接写
for (int i = 0; i < N; ++i) data[i] = i;
// GPU 直接读
kernel<<<g, b>>>(data);
cudaDeviceSynchronize();
```
**优点**：代码简洁，适合原型；
**代价**：隐式迁移可能有性能损失，生产代码仍推荐显式拷贝。

### B. Cooperative Groups

更安全的同步抽象：
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k() {
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);
    // warp 级归约
    int v = warp.shfl_down(val, 16);
    block.sync();
}
```

### C. Tensor Cores（WMMA）

FP16/BF16/TF32 矩阵乘加，算力比 FP32 CUDA Core 高 8~16 倍。

```cpp
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, a_ptr, K);
wmma::load_matrix_sync(b_frag, b_ptr, K);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
```
要求 GPU 架构 ≥ sm_70（Volta+）。

---

## 5. 延伸阅读（了解趋势）

- **Triton**（OpenAI）：Python DSL，写比 CUDA 更简洁的 GPU kernel，PyTorch 广泛采用
- **CUTLASS**（NVIDIA）：CUDA 模板库，用来构建 GEMM/卷积等高性能算子
- **FlashAttention**：通过 tiling + recomputation 减少 HBM 访问，attention 加速数倍的经典案例——非常值得阅读论文理解"**算子融合 + 内存层次感知**"的优化思想

---

## 6. 综合项目

### 选项 A：小型 CNN 推理引擎（MNIST）

**架构**：`Input(28x28) → Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → FC → Softmax`

**模块拆分**：
```
cnn_infer/
├── tensor.h / .cu          // 简单 tensor 容器，封装 cudaMalloc
├── conv2d.cu               // 卷积 kernel（im2col + cuBLAS GEMM，或直接 kernel）
├── relu.cu                 // element-wise
├── maxpool.cu              // 2x2 pooling
├── fc.cu                   // 全连接 = GEMM
├── softmax.cu              // 含 numerical stability
├── load_weights.cpp        // 读取训练好的权重（PyTorch 导出）
└── main.cu                 // 装配流水线，跑测试集
```

**实现建议**：
1. 先用 PyTorch 训练一个小 LeNet，导出权重为二进制文件
2. 实现各算子的 naive 版本，对单张图片验证正确性
3. 逐个算子优化（conv 用 im2col+GEMM 最简单有效）
4. 测试集准确率应与 PyTorch 相差 < 0.1%

**ReLU 示例**：
```cpp
__global__ void relu(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = fmaxf(0.0f, x[i]);
}
```

**Softmax（含数值稳定）**：
```cpp
__global__ void softmax(float* x, int N) {
    // 1) max
    // 2) exp(x - max)
    // 3) sum
    // 4) divide
    // 单 block 实现，配合 shared memory reduction
}
```

---

### 选项 B：K-means 聚类 GPU 版

**算法流程**：
```
repeat:
  1. Assign: 每个点找最近的中心   ← 高度并行（每点独立）
  2. Update: 用各簇均值更新中心    ← reduce + divide
until converged
```

**Kernel 设计**：
```cpp
// 每个线程处理一个点
__global__ void assign(const float* points, const float* centers,
                       int* labels, int N, int K, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float bestDist = FLT_MAX;
    int   bestK    = 0;
    for (int k = 0; k < K; ++k) {
        float d = 0;
        for (int dim = 0; dim < D; ++dim) {
            float diff = points[i*D + dim] - centers[k*D + dim];
            d += diff * diff;
        }
        if (d < bestDist) { bestDist = d; bestK = k; }
    }
    labels[i] = bestK;
}
```
- `update` 阶段：每个簇求均值，用 `atomicAdd` 累加坐标和计数
- 扩展：把 centers 放 constant memory（只读且小）、对 D 做循环展开

---

## 7. 性能报告模板

综合项目必须产出报告：

```
# 项目：MNIST CNN Inference on CUDA

## 硬件
- GPU: RTX 3060 (sm_86, 12GB)
- CUDA 12.3

## 性能对比（10000 张测试集）
| 实现            | 总耗时 (ms) | 单张 (ms) | 准确率 |
|-----------------|-------------|-----------|--------|
| CPU (numpy)     | 8200        | 0.82      | 98.9%  |
| CUDA naive      | 420         | 0.042     | 98.9%  |
| CUDA + cuBLAS   | 135         | 0.0135    | 98.9%  |

## 优化历程
1. 初版 conv 是 naive 7 层循环展开
2. 改 im2col + cuBLAS SGEMM → 3× 加速
3. FC 层合并到一个 GEMM → 再 1.5×
...
```

---

## 8. 学习路径延伸

学完四个阶段后，推荐方向（按兴趣选）：

- **深度学习系统**：学 Triton，阅读 PyTorch 自定义 CUDA 算子源码
- **高性能计算**：OpenMP Target Offload / SYCL / HIP（跨平台）
- **图形渲染**：CUDA + OptiX、光线追踪
- **大模型推理**：FasterTransformer、vLLM、TensorRT-LLM 源码

---

## 9. 自检清单

- [ ] 会用 cuBLAS 做 GEMM 并与自写版本对比
- [ ] 会用 Thrust 一行完成 sort / reduce / scan
- [ ] 至少深入学过一项进阶主题（UM / CG / Tensor Core）
- [ ] 综合项目跑通，端到端正确性验证
- [ ] 项目至少有 2 次迭代优化，有性能对比数据
- [ ] 写了一篇总结笔记/博客，梳理 30 天所学

---

## 结语

CUDA 的核心能力 = **并行思维** + **硬件感知**。

工具会变（Triton、SYCL 不断演进），但对内存层次、warp 行为、访存模式的直觉一旦建立，迁移到任何异构编程平台都游刃有余。持续练、持续 profile。
