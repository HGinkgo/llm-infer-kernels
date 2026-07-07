#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

// CPU 实现
void host_reduce(float* x, int N, double* sum) {
    *sum = 0.0;
    for (int i = 0; i < N; ++i) {
        *sum += x[i];
    }
}

// reduce_v1：每个线程加载 1 个元素，然后在 block 内用 shared memory 归约
template <int BLOCK_SIZE>
__global__ void reduce_sum_v1(float* x, float* partial, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    smem[tid] = idx < n ? x[idx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}

// reduce_v2：每个线程加载两个元素
template <int BLOCK_SIZE>
__global__ void reduce_sum_v2(float* x, float* partial, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    float val = 0.0f;
    if (idx < n) {
        val += x[idx];
    }

    if (idx + blockDim.x < n) {
        val += x[idx + blockDim.x];
    }

    smem[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = smem[0];
    }
}

// reduce_v3：把最后一个 warp 的归约从 shared memory + __syncthreads() 改成
// __shfl_down_sync
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void reduce_sum_v3(float* x, float* partial, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    float val = 0.0f;
    if (idx < n) {
        val += x[idx];
    }

    if (idx + blockDim.x < n) {
        val += x[idx + blockDim.x];
    }

    smem[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }

    if (tid < 32) {
        val = smem[tid];

        if constexpr (BLOCK_SIZE >= 64) {
            val += smem[tid + 32];
        }

        val = warp_reduce_sum(val);

        if (tid == 0) {
            partial[blockIdx.x] = val;
        }
    }
}

// 一个线程处理 4 个数据
template <int BLOCK_SIZE>
__global__ void reduce_sum_v4(float* x, float* partial, int n) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of 32");

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    float val = 0.0f;

    for (int i = 0; i < 4; ++i) {
        if (idx + blockDim.x * i < n) {
            val += x[idx + blockDim.x * i];
        }
    }

    smem[tid] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 32; offset >>= 1) {
        if (tid < offset) {
            smem[tid] += smem[tid + offset];
        }

        __syncthreads();
    }

    if (tid < 32) {
        val = smem[tid];

        if constexpr (BLOCK_SIZE >= 64) {
            val += smem[tid + 32];
        }

        val = warp_reduce_sum(val);

        if (tid == 0) {
            partial[blockIdx.x] = val;
        }
    }
}

double sum_host_array(const float* x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
    }
    return sum;
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
    const char* name;
    double rel_err;
    float avg_ms;
    double bandwidth_gbs;
    double speedup;
};

template <typename KernelLauncher>
BenchmarkResult run_benchmark(const char* name, KernelLauncher&& launcher, float* d_partial,
                              float* h_partial, int partial_size, int input_elems, int warmup,
                              int iters, double host_sum, float baseline_ms = 0.0f) {
    launcher();
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(
        cudaMemcpy(h_partial, d_partial, sizeof(float) * partial_size, cudaMemcpyDeviceToHost));

    double gpu_sum = sum_host_array(h_partial, partial_size);
    double rel_err = std::fabs(host_sum - gpu_sum) / std::fabs(host_sum);
    float avg_ms = benchmark_kernel_ms(launcher, warmup, iters);
    double input_bytes = static_cast<double>(input_elems) * sizeof(float);
    double bandwidth_gbs = input_bytes / (avg_ms * 1.0e6);
    double speedup = baseline_ms > 0.0f ? baseline_ms / avg_ms : 1.0;

    return BenchmarkResult{
        name, rel_err, avg_ms, bandwidth_gbs, speedup,
    };
}

int main() {
    constexpr int N = (1 << 24) + 123;
    constexpr int BLOCK_SIZE = 256;
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    const int grid_size_v1 = CEIL_DIV(N, BLOCK_SIZE);
    const int grid_size_v2 = CEIL_DIV(N, BLOCK_SIZE * 2);
    const int grid_size_v4 = CEIL_DIV(N, BLOCK_SIZE * 4);

    float* h_x = static_cast<float*>(std::malloc(sizeof(float) * N));
    float* h_partial_v1 = static_cast<float*>(std::malloc(sizeof(float) * grid_size_v1));
    float* h_partial_v2 = static_cast<float*>(std::malloc(sizeof(float) * grid_size_v2));
    float* h_partial_v3 = static_cast<float*>(std::malloc(sizeof(float) * grid_size_v2));
    float* h_partial_v4 = static_cast<float*>(std::malloc(sizeof(float) * grid_size_v4));
    double host_sum = 0.0;

    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.0f + static_cast<float>(i % 7) * 0.001f;
    }

    host_reduce(h_x, N, &host_sum);

    float* d_x = nullptr;
    float* d_partial_v1 = nullptr;
    float* d_partial_v2 = nullptr;
    float* d_partial_v3 = nullptr;
    float* d_partial_v4 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_partial_v1, sizeof(float) * grid_size_v1));
    CUDA_CHECK(cudaMalloc(&d_partial_v2, sizeof(float) * grid_size_v2));
    CUDA_CHECK(cudaMalloc(&d_partial_v3, sizeof(float) * grid_size_v2));
    CUDA_CHECK(cudaMalloc(&d_partial_v4, sizeof(float) * grid_size_v4));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice));

    auto launch_v1 = [&]() {
        reduce_sum_v1<BLOCK_SIZE><<<grid_size_v1, BLOCK_SIZE>>>(d_x, d_partial_v1, N);
    };
    auto launch_v2 = [&]() {
        reduce_sum_v2<BLOCK_SIZE><<<grid_size_v2, BLOCK_SIZE>>>(d_x, d_partial_v2, N);
    };
    auto launch_v3 = [&]() {
        reduce_sum_v3<BLOCK_SIZE><<<grid_size_v2, BLOCK_SIZE>>>(d_x, d_partial_v3, N);
    };
    auto launch_v4 = [&]() {
        reduce_sum_v4<BLOCK_SIZE><<<grid_size_v4, BLOCK_SIZE>>>(d_x, d_partial_v4, N);
    };

    BenchmarkResult result_v1 = run_benchmark("v1", launch_v1, d_partial_v1, h_partial_v1,
                                              grid_size_v1, N, kWarmup, kIters, host_sum);

    BenchmarkResult result_v2 =
        run_benchmark("v2", launch_v2, d_partial_v2, h_partial_v2, grid_size_v2, N, kWarmup, kIters,
                      host_sum, result_v1.avg_ms);

    BenchmarkResult result_v3 =
        run_benchmark("v3", launch_v3, d_partial_v3, h_partial_v3, grid_size_v2, N, kWarmup, kIters,
                      host_sum, result_v1.avg_ms);

    BenchmarkResult result_v4 =
        run_benchmark("v4", launch_v4, d_partial_v4, h_partial_v4, grid_size_v4, N, kWarmup, kIters,
                      host_sum, result_v1.avg_ms);

    std::printf("N = %d, BLOCK_SIZE = %d\n", N, BLOCK_SIZE);
    std::printf("host_sum   = %.8f\n", host_sum);
    std::printf("%-8s %-12s %-18s %-10s %-12s\n", "version", "time(ms)", "bandwidth(GB/s)",
                "speedup(x)", "correctness");
    const char* pass_v1 = result_v1.rel_err < 1e-4 ? "pass" : "fail";
    const char* pass_v2 = result_v2.rel_err < 1e-4 ? "pass" : "fail";
    const char* pass_v3 = result_v3.rel_err < 1e-4 ? "pass" : "fail";
    const char* pass_v4 = result_v4.rel_err < 1e-4 ? "pass" : "fail";
    std::printf("%-8s %-12.4f %-18.2f %.2fx %-12s\n", result_v1.name, result_v1.avg_ms,
                result_v1.bandwidth_gbs, result_v1.speedup, pass_v1);
    std::printf("%-8s %-12.4f %-18.2f %.2fx %-12s\n", result_v2.name, result_v2.avg_ms,
                result_v2.bandwidth_gbs, result_v2.speedup, pass_v2);
    std::printf("%-8s %-12.4f %-18.2f %.2fx %-12s\n", result_v3.name, result_v3.avg_ms,
                result_v3.bandwidth_gbs, result_v3.speedup, pass_v3);
    std::printf("%-8s %-12.4f %-18.2f %.2fx %-12s\n", result_v4.name, result_v4.avg_ms,
                result_v4.bandwidth_gbs, result_v4.speedup, pass_v4);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_partial_v1));
    CUDA_CHECK(cudaFree(d_partial_v2));
    CUDA_CHECK(cudaFree(d_partial_v3));
    CUDA_CHECK(cudaFree(d_partial_v4));
    std::free(h_x);
    std::free(h_partial_v1);
    std::free(h_partial_v2);
    std::free(h_partial_v3);
    std::free(h_partial_v4);

    return (result_v1.rel_err < 1e-4 && result_v2.rel_err < 1e-4 && result_v3.rel_err < 1e-4 &&
            result_v4.rel_err < 1e-4)
               ? 0
               : 1;
}
