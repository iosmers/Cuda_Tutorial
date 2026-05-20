#include "kernels.cuh"

// ---------------------------------------------------------------------------
// 矩阵乘法
// ---------------------------------------------------------------------------

__global__ void matMulNaive(const float* A, const float* B, float* C, int M,
                            int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define TILE 16

__global__ void matMulTiled(const float* A, const float* B, float* C, int M,
                            int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// 矩阵转置
// ---------------------------------------------------------------------------

__global__ void transposeNaive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M) {
        out[x * M + y] = in[y * N + x];
    }
}

#define BDIM 32

__global__ void transposeShared(const float* in, float* out, int M, int N) {
    __shared__ float tile[BDIM][BDIM + 1];

    int x = blockIdx.x * BDIM + threadIdx.x;
    int y = blockIdx.y * BDIM + threadIdx.y;
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    x = blockIdx.y * BDIM + threadIdx.x;
    y = blockIdx.x * BDIM + threadIdx.y;
    if (x < M && y < N) {
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}
