
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

__global__ void elementwise_add_v1(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

template <int ITEMS_PER_THREAD>
__global__ void elementwise_add_v2(const float* a, const float* b, float* c, int n) {
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * ITEMS_PER_THREAD + tid;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = base + item * blockDim.x;

        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

__global__ void elementwise_add_v3_float4(const float* __restrict__ a, const float* __restrict__ b,
                                          float* __restrict__ c, int n) {
    int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_count = n / 4;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    if (vector_idx < vector_count) {
        float4 value_a = a4[vector_idx];
        float4 value_b = b4[vector_idx];
        float4 value_c;

        value_c.x = value_a.x + value_b.x;
        value_c.y = value_a.y + value_b.y;
        value_c.z = value_a.z + value_b.z;
        value_c.w = value_a.w + value_b.w;

        c4[vector_idx] = value_c;
    }

    int tail_idx = vector_count * 4 + vector_idx;
    if (tail_idx < n) {
        c[tail_idx] = a[tail_idx] + b[tail_idx];
    }
}

void elementwise_add_host(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

float max_abs_error(const float* actual, const float* expected, int n) {
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = fmaxf(max_error, std::fabs(actual[i] - expected[i]));
    }
    return max_error;
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

int main() {
    constexpr int N = (1 << 24) + 3;
    constexpr int BLOCK_SIZE = 256;
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    float* h_a = static_cast<float*>(std::malloc(bytes));
    float* h_b = static_cast<float*>(std::malloc(bytes));
    float* h_c = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));

    if (h_a == nullptr || h_b == nullptr || h_c == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_a);
        std::free(h_b);
        std::free(h_c);
        std::free(h_ref);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i) {
        h_a[i] = -4.0f + static_cast<float>(i % 31) * 0.25f;
        h_b[i] = 2.0f - static_cast<float>(i % 17) * 0.125f;
    }
    elementwise_add_host(h_a, h_b, h_ref, N);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int vector_work_items = ceil_div(N, 4);
    int grid_size = ceil_div(vector_work_items, BLOCK_SIZE);
    auto launch = [&]() { elementwise_add_v3_float4<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N); };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_c, h_ref, N);
    float avg_ms = benchmark_kernel_ms(launch, kWarmup, kIters);
    double effective_bytes = static_cast<double>(bytes) * 3.0;
    double bandwidth_gbs = effective_bytes / (avg_ms * 1.0e6);
    bool pass = max_error < 1e-6f;

    std::printf("N = %d, BLOCK_SIZE = %d, VECTOR_WORK_ITEMS = %d, GRID_SIZE = %d\n", N, BLOCK_SIZE,
                vector_work_items, grid_size);
    std::printf("time(ms)   = %.4f\n", avg_ms);
    std::printf("bandwidth  = %.2f GB/s\n", bandwidth_gbs);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
    std::free(h_ref);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
