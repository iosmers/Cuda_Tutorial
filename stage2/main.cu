#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.h"
#include "kernels.cuh"

namespace {

void matMulCpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void transposeCpu(const float* in, float* out, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            out[col * M + row] = in[row * N + col];
        }
    }
}

bool allClose(const float* a, const float* b, int n, float rtol = 1e-4f,
              float atol = 1e-4f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(a[i] - b[i]) > atol + rtol * std::fabs(b[i])) {
            return false;
        }
    }
    return true;
}

float timeKernelMs(void (*record)(cudaStream_t), int warmup = 2,
                   int repeats = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        record(0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        record(0);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(repeats);
}

double matMulGflops(int M, int N, int K, float ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    return flops / (static_cast<double>(ms) * 1e6);
}

void runMatMulBench(int M, int N, int K) {
    const size_t aBytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bBytes = static_cast<size_t>(K) * N * sizeof(float);
    const size_t cBytes = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N), h_ref(M * N);
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>(i % 97) * 0.01f;
    }
    for (size_t i = 0; i < h_B.size(); ++i) {
        h_B[i] = static_cast<float>(i % 53) * 0.02f;
    }

    matMulCpu(h_A.data(), h_B.data(), h_ref.data(), M, N, K);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, aBytes));
    CUDA_CHECK(cudaMalloc(&d_B, bBytes));
    CUDA_CHECK(cudaMalloc(&d_C, cBytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), aBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    dim3 gridTiled((N + 15) / 16, (M + 15) / 16);
    dim3 blockTiled(16, 16);

    auto launchNaive = [&](cudaStream_t) {
        matMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchTiled = [&](cudaStream_t) {
        matMulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    };

    launchTiled(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, cBytes, cudaMemcpyDeviceToHost));
    if (!allClose(h_C.data(), h_ref.data(), M * N)) {
        fprintf(stderr, "matMulTiled 结果与 CPU 不一致\n");
        exit(EXIT_FAILURE);
    }
    printf("matMulTiled 正确性: OK\n");

    float msNaive = timeKernelMs(launchNaive);
    float msTiled = timeKernelMs(launchTiled);

    printf("\n=== MatMul %d x %d x %d ===\n", M, K, N);
    printf("| 版本   | 时间 (ms) | GFLOPS | 相对 Naive |\n");
    printf("|--------|-----------|--------|------------|\n");
    printf("| Naive  | %8.3f | %6.1f | 1.00x      |\n", msNaive,
           matMulGflops(M, N, K, msNaive));
    printf("| Tiled  | %8.3f | %6.1f | %.2fx      |\n", msTiled,
           matMulGflops(M, N, K, msTiled), msNaive / msTiled);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void runTransposeBench(int M, int N) {
    const size_t inBytes = static_cast<size_t>(M) * N * sizeof(float);
    const size_t outBytes = static_cast<size_t>(N) * M * sizeof(float);

    std::vector<float> h_in(M * N), h_out(N * M), h_ref(N * M);
    for (size_t i = 0; i < h_in.size(); ++i) {
        h_in[i] = static_cast<float>(i % 127) * 0.001f;
    }

    transposeCpu(h_in.data(), h_ref.data(), M, N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), inBytes, cudaMemcpyHostToDevice));

    dim3 blockNaive(16, 16);
    dim3 gridNaive((N + blockNaive.x - 1) / blockNaive.x,
                   (M + blockNaive.y - 1) / blockNaive.y);

    dim3 blockShared(32, 32);
    dim3 gridShared((N + 31) / 32, (M + 31) / 32);

    auto launchNaive = [&](cudaStream_t) {
        transposeNaive<<<gridNaive, blockNaive>>>(d_in, d_out, M, N);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchShared = [&](cudaStream_t) {
        transposeShared<<<gridShared, blockShared>>>(d_in, d_out, M, N);
        CUDA_CHECK(cudaGetLastError());
    };

    launchShared(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    if (!allClose(h_out.data(), h_ref.data(), N * M)) {
        fprintf(stderr, "transposeShared 结果与 CPU 不一致\n");
        exit(EXIT_FAILURE);
    }
    printf("transposeShared 正确性: OK\n");

    float msNaive = timeKernelMs(launchNaive);
    float msShared = timeKernelMs(launchShared);

    const double bytes = 2.0 * static_cast<double>(M) * N * sizeof(float);
    auto gbps = [&](float ms) {
        return bytes / (static_cast<double>(ms) * 1e6);
    };

    printf("\n=== Transpose %d x %d ===\n", M, N);
    printf("| 版本    | 时间 (ms) | 带宽 (GB/s) | 相对 Naive |\n");
    printf("|---------|-----------|-------------|------------|\n");
    printf("| Naive   | %8.3f | %9.2f | 1.00x      |\n", msNaive,
           gbps(msNaive));
    printf("| Shared  | %8.3f | %9.2f | %.2fx      |\n", msShared,
           gbps(msShared), msNaive / msShared);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

}  // namespace

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    int tM = 4096, tN = 4096;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (argc >= 6) {
        tM = std::atoi(argv[4]);
        tN = std::atoi(argv[5]);
    }

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    runMatMulBench(M, N, K);
    runTransposeBench(tM, tN);

    return 0;
}
