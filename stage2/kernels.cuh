#pragma once

// 矩阵乘法：Naive（global only）与 Tiled（shared memory）
__global__ void matMulNaive(const float* A, const float* B, float* C, int M,
                            int N, int K);

__global__ void matMulTiled(const float* A, const float* B, float* C, int M,
                            int N, int K);

// 矩阵转置：Naive 与 Shared（合并读写 + padding 消 bank conflict）
__global__ void transposeNaive(const float* in, float* out, int M, int N);

__global__ void transposeShared(const float* in, float* out, int M, int N);
