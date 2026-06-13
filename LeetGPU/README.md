# LeetGPU 题解速通与网课化大纲

> 目标：快速熟悉本目录内 LeetGPU 题目，把它们整理成可讲授的 CUDA 网课路线。  
> 阅读方式：先看「分类总览」建立题型地图，再按「课程模块」逐章学习，最后用「题目速查表」回顾每道题的核心解法。

---

## 0. 一句话总览：LeetGPU 其实在考哪些 CUDA 模板？

这些题虽然名字很多，但底层反复出现的是几类并行模式：

| CUDA 模板 | 典型特征 | 常用技术 | 代表题 |
|---|---|---|---|
| **Map / Elementwise** | 每个输出元素独立 | 一线程一元素、grid-stride loop | RoPE、Weight Dequant、Jacobi、Pooling |
| **Reduction** | 多个元素归约成一个值 | shared memory reduction、warp reduction、atomicAdd | Reduction、Dot Product、MSE、Monte Carlo |
| **Scan / Prefix Sum** | 每个位置依赖前缀 | Blelloch scan、多 kernel 全局同步 | Prefix Sum、Segmented Scan、Stream Compaction |
| **Histogram / Count** | 多线程更新少量桶 | atomicAdd、shared local histogram | Histogramming、Count 1D/2D/3D |
| **Stencil / Convolution** | 每个输出看邻域 | 2D/3D grid、halo、shared tile | 2D Conv、3D Conv、Gaussian Blur、Jacobi |
| **GEMM / Linear Algebra** | 三重循环矩阵乘 | tiled shared memory、FP32 accumulate、WMMA 思想 | GEMM、Batched GEMM、INT8/INT4 MatMul |
| **Sort / Select / Sampling** | 排序、TopK、采样 | bitonic/radix、selection、prefix/histogram | Sorting、Radix Sort、Top K、Top-p |
| **Attention / Transformer** | softmax(QKᵀ)V 及变体 | stable softmax、按 head/query 分块、KV cache | Softmax Attention、MHA、GQA、Llama Block |
| **Graph / Iterative** | 需要多轮同步或状态推进 | frontier、atomicCAS、多 kernel loop | BFS、APSP、K-Means、SSM |
| **Signal / FFT** | 蝶形计算或二维可分离 | butterfly、bit reversal、行列两阶段 | FFT、2D FFT |

> 讲课建议：不要按题号讲。应该按「并行模式」讲。学生掌握模板后，很多题只是换了外壳。

---

## 1. 推荐网课路线

### 预备章：Easy 模式速通——一线程一元素的 CUDA 入门题

**目标**：先用 Easy 题建立 CUDA 最基本的代码肌肉记忆。Easy 题大多不是考复杂算法，而是考：

```text
1. 会不会把 CPU for 循环改成 GPU 并行线程。
2. 会不会写正确的线程索引和边界判断。
3. 会不会处理 row-major / image / vector 的内存布局。
4. 会不会把简单数学函数映射到每个元素。
```

建议题目：

- [Vector Addition](ID01_Vector_Addition.md)
- [Matrix Addition](ID08_Matrix_Addition.md) / [Matrix Copy](ID31_Matrix_Copy.md)
- [Color Inversion](ID07_Color_Inversion.md) / [RGB to Grayscale](ID66_RGB_to_Grayscale.md)
- [ReLU](ID21_ReLU.md) / [Leaky ReLU](ID23_Leaky_ReLU.md) / [Sigmoid](ID68_Sigmoid_Activation.md) / [SiLU](ID52_Sigmoid_Linear_Unit.md) / [GEGLU](ID65_Gaussian_Error_Gated_Linear_Unit.md) / [SwiGLU](ID54_Swish_Gated_Linear_Unit.md)
- [Reverse Array](ID19_Reverse_Array.md) / [Interleave Arrays](ID63_Interleave_Arrays.md)
- [1D Convolution](ID09_1D_Convolution.md)
- [Matrix Multiplication](ID02_Matrix_Multiplication.md)
- [Matrix Transpose](MatrixTranspose.md)

通用模板：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    output[i] = f(input[i]);
}
```

讲课重点：

- Easy 题的重点不是“难算法”，而是建立 CUDA kernel 的基本格式。
- 一维数组、二维矩阵、图像 RGB 都可以先展平成一维线性 index。
- 所有 kernel 都必须写边界判断：`if (idx < total)`。
- Activation 类题目是最好的 elementwise 训练题。
- Matrix Transpose 虽然是 Easy，但非常适合作为 coalesced access / shared memory 的第一道性能优化题。

---

### 第一章：CUDA 并行思维入门

**目标**：建立 thread/block/grid、global/shared memory、访存合并的基本直觉。

建议题目：

1. [Matrix Transpose](MatrixTranspose.md)
2. [Reduction](ID04_Reduction.md)
3. [Dot Product](DotProduct.md)

核心讲法：

- Matrix Transpose 用来讲 coalesced memory access、shared memory tile、bank conflict。
- Reduction 用来讲「很多值变一个值」的基本套路。
- Dot Product 是 Reduction 的最小业务化版本。

---

### 第二章：Reduction 家族：求和、均值、最大值、损失函数

**目标**：学会 grid-stride loop + block reduction + 少量 atomic。

建议题目：

- [Reduction](ID04_Reduction.md)
- [Dot Product](DotProduct.md)
- [FP16 Dot Product](ID58_FP16_Dot_Product.md)
- [Mean Squared Error](MeanSquaredError.md)
- [Monte Carlo Integration](ID35_Monte_Carlo_Integration.md)
- [Subarray Sum](ID47_Subarray_Sum.md)
- [2D Subarray Sum](ID48_2D_Subarray_Sum.md)
- [3D Subarray Sum](ID49_3D_Subarray_Sum.md)
- [RMS Normalization](ID50_RMS_Normalization.md)
- [Batch Normalization](ID40_Batch_Normalization.md)
- [Categorical Cross Entropy Loss](ID25_Categorical_Cross_Entropy_Loss.md)

通用公式：

```text
local_sum = 每个线程处理多个元素
block_sum = block 内 shared memory reduction
global_sum = block_sum 再 atomicAdd 或二次 reduction
```

讲课重点：

- 为什么不能让每个元素都 `atomicAdd` 到全局结果。
- 为什么输出标量要先 `cudaMemset`。
- FP16 输入时为什么最好 FP32 累加。
- Softmax / Cross Entropy 为什么要先减最大值。

---

### 第三章：Scan / Prefix Sum：并行前缀依赖问题

**目标**：理解看似串行依赖的前缀和如何并行化。

建议题目：

- [Prefix Sum](PrefixSum.md)
- [Segmented Exclusive Prefix Sum](ID70_Segmented_Exclusive_Prefix_Sum.md)
- [Stream Compaction](ID72_Stream_Compaction.md)
- [Parallel Merge](ID71_Parallel_Merge.md)
- [Max Subarray Sum](ID51_Max_Subarray_Sum.md)

通用思路：

```text
block 内 scan
保存 block sums
对 block sums 再 scan
把 block offset 加回每个 block
```

讲课重点：

- `__syncthreads()` 只能同步一个 block。
- 跨 block 同步靠多次 kernel launch。
- Stream Compaction = flag + prefix sum + scatter。
- Segmented Scan = 普通 scan 加上 segment flag 的组合规则。

---

### 第四章：Histogram / Count / Atomic：多线程写同一个桶

**目标**：理解 data race、atomicAdd 和 shared memory 局部聚合。

建议题目：

- [Histogramming](Histogramming.md)
- [Count Array Element](ID43_Count_Array_Element.md)
- [Count 2D Array Element](ID44_Count_2D_Array_Element.md)
- [Count 3D Array Element](ID45_Count_3D_Array_Element.md)

通用思路：

```text
错误：global_hist[value]++
正确：atomicAdd(&global_hist[value], 1)
优化：每个 block 在 shared memory 里统计 local histogram，再合并到 global
```

讲课重点：

- `hist[x]++` 为什么不是原子操作。
- global atomic 和 shared atomic 的性能差异。
- 为什么要先清零 histogram。
- 每个 warp 一份 local histogram 可以降低冲突。

---

### 第五章：Stencil / Convolution / Pooling：邻域计算

**目标**：掌握 2D/3D 输出坐标映射、边界处理、halo/tile 思想。

建议题目：

- [2D Convolution](ID10_2D_Convolution.md)
- [3D Convolution](ID11_3D_Convolution.md)
- [Gaussian Blur](ID28_Gaussian_Blur.md)
- [2D Jacobi Stencil](ID69_2D_Jacobi_Stencil.md)
- [2D Max Pooling](ID42_2D_Max_Pooling.md)
- [Causal Depthwise Conv1d](ID90_Causal_Depthwise_Conv1d.md)

通用思路：

```text
一个线程负责一个输出点
根据输出坐标找到输入邻域
遍历 kernel/window
处理边界
写 output
```

讲课重点：

- row-major / NCHW / 3D flatten 索引。
- padding、stride、kernel center 的计算。
- input/output 分离，不能原地 Jacobi。
- shared memory tile + halo 是进一步优化方向。

---

### 第六章：GEMM 与线性代数：CUDA 里的主菜

**目标**：理解矩阵乘法是深度学习 kernel 的核心，掌握 tiled GEMM。

建议题目：

- [General Matrix Multiplication (GEMM)](GEMM.md)
- [Batched Matrix Multiplication](ID30_Batched_Matrix_Multiplication.md)
- [FP16 Batched Matrix Multiplication](ID57_FP16_Batched_Matrix_Multiplication.md)
- [Sparse Matrix-Vector Multiplication](SparseMatrixVectorMultiplication.md)
- [Sparse Matrix-Dense Matrix Multiplication](ID75_Sparse_Matrix_Dense_Matrix_Multiplication.md)
- [Matrix Power](ID37_Matrix_Power.md)
- [Ordinary Least Squares](ID33_Ordinary_Least_Squares.md)
- [Logistic Regression](ID34_Logistic_Regression.md)

通用 GEMM 公式：

```text
C[row, col] = sum_k A[row, k] * B[k, col]
```

通用 CUDA 结构：

```text
一个 block 负责 C 的一个 TILE × TILE
每轮把 A/B 的 tile 加载到 shared memory
同步
在 shared memory 中做 TILE 次乘加
循环扫完整个 K
```

讲课重点：

- A/B/C 的 row-major 索引。
- FP16 输入、FP32 accumulate、FP16 输出。
- alpha/beta GEMM：`C = alpha * AB + beta * C_initial`。
- Batched GEMM 只是多一个 batch offset。
- Sparse 题如果接口没给 CSR/COO，就只能 dense scan + 跳过 0。

---

### 第七章：量化与低精度计算

**目标**：理解 INT8/INT4 反量化、zero point、scale、group size。

建议题目：

- [INT8 Quantized MatMul](ID32_INT8_Quantized_MatMul.md)
- [INT4 Weight-Only Quantized MatMul](ID81_INT4_Weight_Only_Quantized_MatMul.md)
- [Weight Dequantization](ID64_Weight_Dequantization.md)
- [INT8 KV-Cache Attention](ID96_INT8_KV_Cache_Attention.md)

通用思路：

```text
int8/int4 存储只是压缩形式
真正计算前要反量化：real = scale * (q - zero_point)
MatMul 中 int32/float accumulate
最后再量化或写 half/float
```

讲课重点：

- INT8 MatMul 的 int32 accumulate。
- INT4 一个 byte 打包两个 4-bit，nibble 顺序容易错。
- group-wise scale：按 group_size 找 scale。
- KV cache 量化：attention score 和 value 加权前要反量化。

---

### 第八章：Sort / Select / Sampling：从排序到生成式模型采样

**目标**：掌握 bitonic/radix/select/top-k/top-p 的区别。

建议题目：

- [Sorting](ID15_Sorting.md)
- [Radix Sort](ID36_Radix_Sort.md)
- [Top K Selection](TopKSelection.md)
- [MoE Top-K Gating](ID67_MoE_Top_K_Gating.md)
- [Top-p Sampling](ID60_Top_p_Sampling.md)

分类讲法：

```text
Sorting：把所有元素排好
TopK：只要最大的 k 个，不需要完整排序
Radix Sort：整数按 bit/digit 稳定排序
Top-p：按概率排序后取 cumulative probability >= p 的集合再采样
MoE Top-K：每个 token 从专家 logits 中选 k 个 expert，并对 top-k 做 softmax
```

讲课重点：

- 完整排序 vs selection 的复杂度差异。
- 重复值处理：把元素看成 `(value, index)`。
- Radix Sort 要稳定，否则低位排序会被破坏。
- Top-p 是 nucleus sampling，不是 top-k。

---

### 第九章：Attention 与 Transformer Kernel

**目标**：掌握现代 LLM 计算的核心：softmax attention、mask、head、RoPE、GQA、KV cache。

建议题目：

- [Softmax Attention](SoftmaxAttention.md)
- [Multi-Head Attention](ID12_Multi_Head_Attention.md)
- [Causal Self-Attention](ID53_Causal_Self_Attention.md)
- [Attention with Linear Biases](ID55_Attention_with_Linear_Biases.md)
- [Sliding Window Self-Attention](ID59_Sliding_Window_Self_Attention.md)
- [Grouped Query Attention](ID80_Grouped_Query_Attention.md)
- [Decaying Causal Attention](ID92_Decaying_Causal_Attention.md)
- [Linear Self-Attention](ID56_Linear_Self_Attention.md)
- [Rotary Positional Embedding](ID61_Rotary_Positional_Embedding.md)
- [INT8 KV-Cache Attention](ID96_INT8_KV_Cache_Attention.md)
- [Speculative Decoding Verification](ID87_Speculative_Decoding_Verification.md)

标准 attention 模板：

```text
score[j] = dot(Q_i, K_j) / sqrt(d)
row_max = max_j score[j]
denominator = sum_j exp(score[j] - row_max)
output[i, col] = sum_j exp(score[j] - row_max) * V[j, col] / denominator
```

变体区别：

| 变体 | 本质变化 |
|---|---|
| Multi-Head Attention | 把 d_model 切成多个 head，各自 attention 后拼接 |
| Causal Attention | 只能看 j <= i，mask 掉未来 token |
| ALiBi | score 上加线性位置 bias |
| Sliding Window | 只看局部窗口，降低复杂度 |
| GQA | 多个 query heads 共享较少的 KV heads |
| Decaying Attention | score 加随距离衰减项 |
| Linear Attention | 用核函数/特征映射替代 softmax attention |
| RoPE | 对 Q/K 做二维旋转位置编码 |
| INT8 KV Cache | K/V cache 以 int8 存储，使用前反量化 |

讲课重点：

- stable softmax 是 attention 的生命线。
- 一个 block 负责一个 query/head 是最容易理解的 baseline。
- Causal/window/bias/GQA 都是在 score 计算或 K/V 选择上改动。
- FlashAttention 的核心是 streaming softmax，不显式存完整 `QKᵀ`。

---

### 第十章：Transformer Block 与 LLM 组合题

**目标**：把多个基础 kernel 串成完整模型 block。

建议题目：

- [GPT-2 Transformer Block](ID74_GPT_2_Transformer_Block.md)
- [Llama Transformer Block](ID93_Llama_Transformer_Block.md)
- [SwiGLU MLP Block](ID84_SwiGLU_MLP_Block.md)
- [LoRA Linear](ID85_LoRA_Linear.md)
- [Adder Transformer Inference](ID76_Adder_Transformer_Inference.md)
- [SSM Selective Scan](ID94_SSM_Selective_Scan.md)
- [Linear Recurrence](ID82_Linear_Recurrence.md)

模块拆解：

```text
GPT-2 Block = LayerNorm + QKV GEMM + Causal MHA + Projection + Residual + MLP(GELU)
Llama Block = RMSNorm + QKV/GQA + RoPE + Causal Attention + SwiGLU MLP + Residual
LoRA Linear = xW + scale * (xA)B
SwiGLU = silu(xW_gate) * (xW_up) -> W_down
SSM/Linear Recurrence = 时间维递推，不能简单逐元素并行
```

讲课重点：

- 大模型题不要一上来写一个巨型 kernel，先模块化。
- weights packed buffer 的 offset 是工程题核心。
- Transformer block 由 GEMM、Norm、Attention、Activation、Residual 组合而成。
- SSM/Recurrence 的难点是时间依赖，可以从串行 baseline 到 associative scan 逐步优化。

---

### 第十一章：图算法、模拟与迭代算法

**目标**：理解需要多轮 kernel launch 的算法。

建议题目：

- [BFS Shortest Path](ID46_BFS_Shortest_Path.md)
- [All-Pairs Shortest Paths](ID73_All_Pairs_Shortest_Paths.md)
- [K-Means Clustering](ID20_K_Means_Clustering.md)
- [Multi-Agent Simulation](ID14_Multi_Agent_Simulation.md)
- [Nearest Neighbor](ID38_Nearest_Neighbor.md)

通用结构：

```text
初始化状态
for iteration / level / k:
    launch kernel 更新一轮状态
    必要时交换 buffer
    检查是否结束
```

讲课重点：

- BFS frontier 每一层之间必须同步。
- Floyd-Warshall 的 k 循环天然需要全局同步。
- K-Means 是 assign + reduce/update 的迭代模式。
- Multi-Agent/Nearest Neighbor 常见 baseline 是 O(N²)，优化方向是空间分桶。

---

### 第十二章：FFT 与信号处理

**目标**：理解 butterfly 结构和可分离二维变换。

建议题目：

- [Fast Fourier Transform](ID39_Fast_Fourier_Transform.md)
- [2D FFT](ID78_2D_FFT.md)

通用思路：

```text
1D FFT：bit reversal + logN 层 butterfly
2D FFT：先对每一行做 1D FFT，再对每一列做 1D FFT
```

讲课重点：

- 复数内存布局。
- 每一 stage 的 butterfly 可以并行。
- stage 之间需要同步。
- 2D FFT 的行列分离思想。

---

## 2. 题目速查表：每题一句话解法

### Easy 入门题速查表

| 题目 | 类型 | 一句话解法 | 课程作用 |
|---|---|---|---|
| [Vector Addition](ID01_Vector_Addition.md) | Elementwise | `C[i] = A[i] + B[i]`，一线程一元素 | 第一个 CUDA kernel，讲 thread index 和边界判断 |
| [Matrix Addition](ID08_Matrix_Addition.md) | Elementwise / 2D Flatten | 把 `N×N` 矩阵展平成 `N*N` 个元素做加法 | 讲二维数据一维化索引 |
| [Matrix Copy](ID31_Matrix_Copy.md) | Memory Copy | `output[i] = input[i]` | 讲 device pointer、global memory 读写 |
| [Matrix Transpose](ID03_Matrix_Transpose.md) | Memory Pattern | `output[col*rows + row] = input[row*cols + col]` | 讲 coalesced access、shared tile、bank conflict |
| [Color Inversion](ID07_Color_Inversion.md) | Image Elementwise | 每个像素/通道执行颜色反转，alpha 不变 | 讲 RGBA 图像数组和 elementwise map |
| [RGB to Grayscale](ID66_RGB_to_Grayscale.md) | Image Elementwise | 每个像素读 RGB，写 `0.299R+0.587G+0.114B` | 讲 RGB AoS 图像布局和多输入元素生成一个输出 |
| [Reverse Array](ID19_Reverse_Array.md) | Index Transform | 原地交换 `i` 与 `N-1-i`，只处理前半段 | 讲索引变换和避免重复交换 |
| [Interleave Arrays](ID63_Interleave_Arrays.md) | Index Transform | `out[2i]=A[i]`, `out[2i+1]=B[i]` | 讲一个线程写多个连续位置 |
| [Value Clipping](ID62_Value_Clipping.md) | Elementwise | `output[i] = min(max(input[i], lo), hi)` | 讲分支、clamp、数值范围 |
| [ReLU](ID21_ReLU.md) | Activation | `max(x, 0)` | 神经网络 activation 入门 |
| [Leaky ReLU](ID23_Leaky_ReLU.md) | Activation | `x > 0 ? x : 0.01*x` | 讲简单条件分支 |
| [Sigmoid Activation](ID68_Sigmoid_Activation.md) | Activation | `1 / (1 + exp(-x))` | 讲 `expf` 和数学函数 |
| [Sigmoid Linear Unit (SiLU)](ID52_Sigmoid_Linear_Unit.md) | Activation | `x * sigmoid(x)` | 讲复合 elementwise 函数 |
| [Gaussian Error Gated Linear Unit (GEGLU)](ID65_Gaussian_Error_Gated_Linear_Unit.md) | Gated Activation | 输入拆两半：`x1 * GELU(x2)` | 讲 gated activation 和半长输出 |
| [Swish-Gated Linear Unit (SwiGLU)](ID54_Swish_Gated_Linear_Unit.md) | Gated Activation | 输入拆两半：`SiLU(x1) * x2` | 连接到 LLM 的 SwiGLU MLP |
| [1D Convolution](ID09_1D_Convolution.md) | Stencil | 一个线程算一个输出位置，遍历 kernel window | 从 elementwise 过渡到邻域计算 |
| [Matrix Multiplication](ID02_Matrix_Multiplication.md) | GEMM Baseline | 一个线程算一个 `C[row,col]`，循环 K 维 | GEMM 的 naive 版本，为 shared tiled GEMM 铺垫 |
| [Rainbow Table](ID24_Rainbow_Table.md) | Iterative Hash | 每个元素独立做 R 轮 FNV-1a hash | 讲“每个线程内部有小循环”的 map 模式 |
| [Simple Inference](ID41_Simple_Inference.md) | Mini NN / PyTorch | 执行 `model(input)` 并 `copy_` 到 output | 把 elementwise + GEMM 串成小模型 |

Easy 题可以分成 5 个小专题：

```text
1. 纯 elementwise：Vector Add / ReLU / Sigmoid / Clip
2. 索引变换：Reverse / Interleave / Transpose
3. 图像处理：Color Inversion / RGB to Grayscale
4. 邻域计算：1D Convolution
5. 小模型组件：Gated Activation / Simple Inference / Matrix Multiplication
```

> 网课建议：Easy 题适合做“课上现场写代码”。每道题 5~15 分钟，重点让学生形成固定 kernel 模板，而不是追求性能极限。


### A. 基础 Reduction / Softmax / Loss

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Reduction | grid-stride loop 累加，block 内 reduction，block sum 合并 | [ID04](ID04_Reduction.md) |
| Dot Product | 先乘再归约，本质 Reduction | [DotProduct](DotProduct.md) |
| FP16 Dot Product | half 输入转 float 累加，最后写 half | [ID58](ID58_FP16_Dot_Product.md) |
| Mean Squared Error | 对 `(pred-target)^2` 做 reduction，再除以 N | [MeanSquaredError](MeanSquaredError.md) |
| Monte Carlo Integration | 对 y_samples 求平均再乘 `(b-a)` | [ID35](ID35_Monte_Carlo_Integration.md) |
| Softmax | 先求 max，再求 `sum exp(x-max)`，最后归一化 | [ID05](ID05_Softmax.md) |
| Categorical Cross Entropy | `-(logit_label - max - log(sum_exp))`，避免显式 softmax | [ID25](ID25_Categorical_Cross_Entropy_Loss.md) |
| RMS Normalization | `x / sqrt(mean(x^2)+eps) * gamma + beta` | [ID50](ID50_RMS_Normalization.md) |
| Batch Normalization | 每个 channel 求 mean/var，再 normalize | [ID40](ID40_Batch_Normalization.md) |

### B. Prefix / Scan / Compact / Merge

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Prefix Sum | block scan + block sums scan + offset add back | [PrefixSum](PrefixSum.md) |
| Segmented Exclusive Prefix Sum | scan 的组合规则加入 segment flag | [ID70](ID70_Segmented_Exclusive_Prefix_Sum.md) |
| Stream Compaction | flag + exclusive scan + scatter | [ID72](ID72_Stream_Compaction.md) |
| Parallel Merge | 每个输出 rank 二分找到 A/B 划分点 | [ID71](ID71_Parallel_Merge.md) |
| Max Subarray Sum | prefix/window sum + max reduction | [ID51](ID51_Max_Subarray_Sum.md) |

### C. Histogram / Count

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Histogramming | shared local histogram + global atomic merge | [Histogramming](Histogramming.md) |
| Count Array Element | 一维 histogram | [ID43](ID43_Count_Array_Element.md) |
| Count 2D Array Element | 二维数组展平后 histogram | [ID44](ID44_Count_2D_Array_Element.md) |
| Count 3D Array Element | 三维数组展平后 histogram | [ID45](ID45_Count_3D_Array_Element.md) |

### D. Convolution / Stencil / Pooling

| 题目 | 核心解法 | 文档 |
|---|---|---|
| 2D Convolution | 一个线程算一个输出像素，遍历二维 kernel | [ID10](ID10_2D_Convolution.md) |
| 3D Convolution | 一个线程算一个输出体素，遍历三维 kernel | [ID11](ID11_3D_Convolution.md) |
| Gaussian Blur | 2D convolution 的高斯核特例 | [ID28](ID28_Gaussian_Blur.md) |
| 2D Jacobi Stencil | 输入输出分离，读取上下左右邻居更新 | [ID69](ID69_2D_Jacobi_Stencil.md) |
| 2D Max Pooling | 一个线程负责一个 NCHW 输出窗口最大值 | [ID42](ID42_2D_Max_Pooling.md) |
| Causal Depthwise Conv1d | 每通道独立，只看当前和过去 token | [ID90](ID90_Causal_Depthwise_Conv1d.md) |

### E. GEMM / 线性代数 / 稀疏

| 题目 | 核心解法 | 文档 |
|---|---|---|
| GEMM | shared memory tiled matrix multiplication | [GEMM](GEMM.md) |
| Batched Matrix Multiplication | GEMM 加 batch offset | [ID30](ID30_Batched_Matrix_Multiplication.md) |
| FP16 Batched Matrix Multiplication | half 输入，float accumulate，可进一步 WMMA | [ID57](ID57_FP16_Batched_Matrix_Multiplication.md) |
| Sparse Matrix-Vector Multiplication | 接口是 dense A，只能按行扫描并跳过 0 | [SparseMatrixVectorMultiplication](SparseMatrixVectorMultiplication.md) |
| Sparse Matrix-Dense Matrix Multiplication | dense 接口下扫描 K 维，跳过 A 中 0 | [ID75](ID75_Sparse_Matrix_Dense_Matrix_Multiplication.md) |
| Matrix Power | 反复 GEMM 或矩阵快速幂 | [ID37](ID37_Matrix_Power.md) |
| Ordinary Least Squares | 计算 `XᵀX` 和 `Xᵀy`，解线性方程 | [ID33](ID33_Ordinary_Least_Squares.md) |
| Logistic Regression | sigmoid + gradient reduction + beta update | [ID34](ID34_Logistic_Regression.md) |

### F. 量化 / 低精度

| 题目 | 核心解法 | 文档 |
|---|---|---|
| INT8 Quantized MatMul | int8 减 zero point，int32 accumulate，再按 scale 输出 | [ID32](ID32_INT8_Quantized_MatMul.md) |
| INT4 Weight-Only Quantized MatMul | uint8 解包两个 int4，按 group scale 反量化 | [ID81](ID81_INT4_Weight_Only_Quantized_MatMul.md) |
| Weight Dequantization | `Y = X * scale[tile/group]` | [ID64](ID64_Weight_Dequantization.md) |
| INT8 KV-Cache Attention | K/V cache 先反量化，再做 decode attention | [ID96](ID96_INT8_KV_Cache_Attention.md) |

### G. Sort / Select / Sampling

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Sorting | bitonic sort baseline，compare-swap 网络 | [ID15](ID15_Sorting.md) |
| Radix Sort | 按 bit/digit 做 stable partition/scatter | [ID36](ID36_Radix_Sort.md) |
| Top K Selection | 每块 local top-k，再合并候选 top-k | [TopKSelection](TopKSelection.md) |
| MoE Top-K Gating | 每个 token 选 top-k expert，再对 top-k softmax | [ID67](ID67_MoE_Top_K_Gating.md) |
| Top-p Sampling | softmax 后按概率降序累积到 p，再采样 | [ID60](ID60_Top_p_Sampling.md) |

### H. Attention / Transformer

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Softmax Attention | 一行 query 一个 block，stable softmax 后加权 V | [SoftmaxAttention](SoftmaxAttention.md) |
| Multi-Head Attention | d_model 切 head，每个 head 独立 attention | [ID12](ID12_Multi_Head_Attention.md) |
| Causal Self-Attention | attention 只看 `j <= i` | [ID53](ID53_Causal_Self_Attention.md) |
| Attention with Linear Biases | score 加 ALiBi 线性位置 bias | [ID55](ID55_Attention_with_Linear_Biases.md) |
| Sliding Window Self-Attention | attention 只看局部窗口 | [ID59](ID59_Sliding_Window_Self_Attention.md) |
| Grouped Query Attention | 多个 Q head 共享少量 KV head | [ID80](ID80_Grouped_Query_Attention.md) |
| Decaying Causal Attention | causal score 加距离衰减项 | [ID92](ID92_Decaying_Causal_Attention.md) |
| Linear Self-Attention | 用核特征映射把 softmax attention 线性化 | [ID56](ID56_Linear_Self_Attention.md) |
| Rotary Positional Embedding | 对偶数/奇数维做旋转位置编码 | [ID61](ID61_Rotary_Positional_Embedding.md) |
| Speculative Decoding Verification | 按 target/draft 概率接受或拒绝 draft token | [ID87](ID87_Speculative_Decoding_Verification.md) |
| GPT-2 Transformer Block | LN + causal MHA + MLP(GELU) + residual | [ID74](ID74_GPT_2_Transformer_Block.md) |
| Llama Transformer Block | RMSNorm + RoPE/GQA attention + SwiGLU + residual | [ID93](ID93_Llama_Transformer_Block.md) |
| SwiGLU MLP Block | `silu(xW_gate) * xW_up -> W_down` | [ID84](ID84_SwiGLU_MLP_Block.md) |
| LoRA Linear | `xW + scale*(xA)B` | [ID85](ID85_LoRA_Linear.md) |
| Adder Transformer Inference | 小 transformer 端到端推理组合题 | [ID76](ID76_Adder_Transformer_Inference.md) |
| SSM Selective Scan | 时间递推状态空间扫描 | [ID94](ID94_SSM_Selective_Scan.md) |
| Linear Recurrence | `h_t = a_t h_{t-1} + x_t`，时间维递推 | [ID82](ID82_Linear_Recurrence.md) |

### I. Graph / Iterative / Search

| 题目 | 核心解法 | 文档 |
|---|---|---|
| BFS Shortest Path | frontier BFS，每层 kernel，同步后扩展下一层 | [ID46](ID46_BFS_Shortest_Path.md) |
| All-Pairs Shortest Paths | Floyd-Warshall，每个 k 一轮全局更新 | [ID73](ID73_All_Pairs_Shortest_Paths.md) |
| K-Means Clustering | assign labels + 聚合 centroid + normalize，迭代 | [ID20](ID20_K_Means_Clustering.md) |
| Multi-Agent Simulation | 双缓冲更新 agent 状态，baseline O(N²) | [ID14](ID14_Multi_Agent_Simulation.md) |
| Nearest Neighbor | 每个点扫描所有其他点，block 内 min reduction | [ID38](ID38_Nearest_Neighbor.md) |

### J. FFT / Signal

| 题目 | 核心解法 | 文档 |
|---|---|---|
| Fast Fourier Transform | bit reversal + 多层 butterfly | [ID39](ID39_Fast_Fourier_Transform.md) |
| 2D FFT | 先行 FFT，再列 FFT | [ID78](ID78_2D_FFT.md) |

---

## 3. 讲课时建议反复强调的 CUDA 关键词

### 3.1 必讲概念

- **线程映射**：一个线程负责一个输出元素，还是一个 block 负责一行/一个 query。
- **Grid-stride loop**：大数组通用遍历模板。
- **Shared memory**：用于 block 内复用数据、局部 histogram、局部 reduction。
- **`__syncthreads()`**：只同步一个 block，不能跨 block。
- **Kernel launch 作为全局同步**：scan、Floyd、BFS 等题都依赖这个思想。
- **Atomic 操作**：正确但可能慢，优化方向是减少 global atomic 次数。
- **Numerical stability**：softmax / cross entropy / attention 必须减 max。
- **Row-major indexing**：大部分 bug 都是索引错。
- **Ping-pong buffer**：排序、矩阵幂、迭代算法、Jacobi、BFS 常见。

### 3.2 常用代码模板

#### 一维 grid-stride loop

```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = tid; i < N; i += stride) {
    // process i
}
```

#### block 内 reduction

```cpp
sdata[threadIdx.x] = local;
__syncthreads();
for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
        sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }
    __syncthreads();
}
```

#### stable softmax

```text
m = max(x)
s = sum(exp(x - m))
y[i] = exp(x[i] - m) / s
```

#### tiled GEMM

```text
for tile in K dimension:
    load A tile to shared
    load B tile to shared
    sync
    multiply accumulate
    sync
```

#### stream compaction

```text
flag[i] = predicate(input[i])
pos[i] = exclusive_scan(flag)[i]
if flag[i]: out[pos[i]] = input[i]
```

---

## 4. 适合网课作业的分层安排


### Level 0：Easy 入门暖身

- [Vector Addition](ID01_Vector_Addition.md)
- [Matrix Addition](ID08_Matrix_Addition.md) / [Matrix Copy](ID31_Matrix_Copy.md)
- [Color Inversion](ID07_Color_Inversion.md) / [RGB to Grayscale](ID66_RGB_to_Grayscale.md)
- [ReLU](ID21_ReLU.md) / [Leaky ReLU](ID23_Leaky_ReLU.md) / [Sigmoid](ID68_Sigmoid_Activation.md) / [SiLU](ID52_Sigmoid_Linear_Unit.md)
- [Reverse Array](ID19_Reverse_Array.md) / [Interleave Arrays](ID63_Interleave_Arrays.md)
- [1D Convolution](ID09_1D_Convolution.md)
- [Matrix Multiplication naive 版](ID02_Matrix_Multiplication.md)
- [Matrix Transpose naive 版](ID03_Matrix_Transpose.md)

目标：能熟练写出 `idx` 计算、边界判断、global memory 读写和简单数学函数。

### Level 1：必须独立写出来

- Reduction
- Dot Product
- MSE
- Softmax
- Histogramming
- Prefix Sum
- 2D Convolution
- GEMM

### Level 2：理解模板迁移

- FP16 Dot Product
- BatchNorm / RMSNorm
- Count 1D/2D/3D
- Stream Compaction
- Batched GEMM
- Max Pooling
- Top K Selection

### Level 3：性能优化专题

- Matrix Transpose
- GEMM shared memory tile
- Histogram shared local aggregation
- Radix Sort
- FFT
- Sparse Matrix-Dense Matrix Multiplication
- INT8/INT4 MatMul

### Level 4：大模型专题

- Softmax Attention
- Causal Attention
- Multi-Head Attention
- GQA
- RoPE
- KV Cache Attention
- SwiGLU
- LoRA
- GPT-2 / Llama Block
- SSM Selective Scan

---

## 5. 最后的学习建议

1. **先记模板，不要死记题目。** 看到「求和/均值/最大值」就想到 reduction；看到「过滤」就想到 flag + scan + scatter；看到「attention」就想到 row-wise stable softmax。
2. **每道题先写 baseline。** CUDA 学习里，正确性优先于极限性能。
3. **每次优化只改一个点。** 比如先从 global memory 版变 shared memory 版，再考虑 warp-level primitive。
4. **把所有复杂题拆成简单题组合。** Transformer block = GEMM + Norm + Attention + Activation + Residual。
5. **讲课时一定画数据流。** GPU 题目难点往往不是公式，而是数据在 thread/block/shared/global 之间怎么流动。
