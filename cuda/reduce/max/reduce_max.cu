#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "utils.cuh"

// CPU 实现
void host_reduce(float* x, int n, float* max_val) {
    *max_val = x[0];
    for (int i = 1; i < n; ++i) {
        *max_val = fmaxf(*max_val, x[i]);
    }
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void reduce_max(const float* x, float* partial, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    float val = -FLT_MAX;

    for (int i = 0; i < 4; ++i) {
        int load_idx = idx + blockDim.x * i;
        if (load_idx < n) {
            val = fmaxf(val, x[load_idx]);
        }
    }

    smem[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
        if (tid < offset) {
            smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        val = smem[tid];
        if constexpr (BLOCK_SIZE >= 64) {
            val = fmaxf(smem[tid], smem[tid + 32]);
        }

        val = warp_reduce_max(val);

        if (tid == 0) {
            partial[blockIdx.x] = val;
        }
    }
}

float max_host_array(const float* x, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        max_val = fmaxf(max_val, x[i]);
    }
    return max_val;
}

template <typename KernelLauncher>
float benchmark_kernel_ms(KernelLauncher&& launcher, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
}

struct BenchmarkResult {
    float host_max;
    float gpu_max;
    float abs_err;
    float avg_ms;
    double bandwidth_gbs;
};

template <typename KernelLauncher>
BenchmarkResult run_benchmark(KernelLauncher&& launcher, float* d_partial, float* h_partial,
                              int partial_size, int input_elems, int warmup, int iters,
                              float host_max) {
    launcher();
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(
        cudaMemcpy(h_partial, d_partial, sizeof(float) * partial_size, cudaMemcpyDeviceToHost));

    float gpu_max = max_host_array(h_partial, partial_size);
    float abs_err = std::fabs(host_max - gpu_max);
    float avg_ms = benchmark_kernel_ms(launcher, warmup, iters);
    double input_bytes = static_cast<double>(input_elems) * sizeof(float);
    double bandwidth_gbs = input_bytes / (avg_ms * 1.0e6);

    return BenchmarkResult{
        host_max, gpu_max, abs_err, avg_ms, bandwidth_gbs,
    };
}

int main() {
    constexpr int N = (1 << 24) + 123;
    constexpr int BLOCK_SIZE = 256;
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    const int grid_size = CEIL_DIV(N, BLOCK_SIZE * 4);

    float* h_x = static_cast<float*>(std::malloc(sizeof(float) * N));
    float* h_partial = static_cast<float*>(std::malloc(sizeof(float) * grid_size));

    for (int i = 0; i < N; ++i) {
        h_x[i] = -100.0f + static_cast<float>(i % 97) * 0.01f;
    }

    const int max_index = 5 * BLOCK_SIZE * 4 + 37 + BLOCK_SIZE * 2;
    h_x[max_index] = 12345.0f;

    float host_max = 0.0f;
    host_reduce(h_x, N, &host_max);

    float* d_x = nullptr;
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * grid_size));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice));

    auto launch = [&]() { reduce_max<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(d_x, d_partial, N); };

    BenchmarkResult result =
        run_benchmark(launch, d_partial, h_partial, grid_size, N, kWarmup, kIters, host_max);

    const char* pass = result.abs_err < 1e-6f ? "pass" : "fail";

    std::printf("N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
    std::printf("host_max   = %.6f\n", result.host_max);
    std::printf("gpu_max    = %.6f\n", result.gpu_max);
    std::printf("%-8s %-12s %-18s %-12s %-12s\n", "version", "time(ms)", "bandwidth(GB/s)",
                "abs_err", "correctness");
    std::printf("%-8s %-12.4f %-18.2f %-12.6f %-12s\n", "v4", result.avg_ms, result.bandwidth_gbs,
                result.abs_err, pass);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial));
    std::free(h_x);
    std::free(h_partial);

    return result.abs_err < 1e-6f ? 0 : 1;
}
