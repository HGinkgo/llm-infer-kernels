#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

constexpr int kWarpSize = 32;

constexpr int kV1BlockSize = 256;
constexpr int kV1ItemsPerThread = 4;
constexpr int kV1MaxColumns = kV1BlockSize * kV1ItemsPerThread;

constexpr int kV2WarpsPerBlock = 8;
constexpr int kV2BlockSize = kV2WarpsPerBlock * kWarpSize;
constexpr int kV2ItemsPerThread = 4;
constexpr int kV2MaxColumns = kWarpSize * kV2ItemsPerThread;

template <int BLOCK_SIZE>
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int BLOCK_SIZE>
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val, float* warp_smem) {
    constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;

    int tid = threadIdx.x;
    int lane_id = tid % kWarpSize;
    int warp_id = tid / kWarpSize;

    val = warp_reduce_max<BLOCK_SIZE>(val);

    if (lane_id == 0) {
        warp_smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? warp_smem[lane_id] : -FLT_MAX;
        val = warp_reduce_max<BLOCK_SIZE>(val);
        if (lane_id == 0) {
            warp_smem[0] = val;
        }
    }
    __syncthreads();

    return warp_smem[0];
}

template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val, float* warp_smem) {
    constexpr int NUM_WARPS = BLOCK_SIZE / kWarpSize;

    int tid = threadIdx.x;
    int lane_id = tid % kWarpSize;
    int warp_id = tid / kWarpSize;

    val = warp_reduce_sum<BLOCK_SIZE>(val);

    if (lane_id == 0) {
        warp_smem[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? warp_smem[lane_id] : 0.0f;
        val = warp_reduce_sum<BLOCK_SIZE>(val);
        if (lane_id == 0) {
            warp_smem[0] = val;
        }
    }
    __syncthreads();

    return warp_smem[0];
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void softmax_martix_v1(const float* x, float* y, int m, int n) {
    static_assert(BLOCK_SIZE % kWarpSize == 0, "BLOCK_SIZE must be a multiple of 32");

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= m) {
        return;
    }

    const float* row_x = x + row * n;
    float* row_y = y + row * n;

    __shared__ float warp_smem[BLOCK_SIZE / kWarpSize];

    float values[ITEMS_PER_THREAD];
    float exp_values[ITEMS_PER_THREAD];
    float local_max = -FLT_MAX;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;

        values[item] = col < n ? row_x[col] : -FLT_MAX;
        local_max = fmaxf(local_max, values[item]);
    }

    float row_max = block_reduce_max<BLOCK_SIZE>(local_max, warp_smem);

    float local_sum = 0.0f;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;

        exp_values[item] = col < n ? expf(values[item] - row_max) : 0.0f;
        local_sum += exp_values[item];
    }

    float row_sum = block_reduce_sum<BLOCK_SIZE>(local_sum, warp_smem);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = tid + item * BLOCK_SIZE;

        if (col < n) {
            row_y[col] = exp_values[item] / row_sum;
        }
    }
}

__inline__ __device__ float warp_all_reduce_max(float val) {
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float warp_all_reduce_sum(float val) {
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

bool is_aligned_16(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % alignof(float4) == 0;
}

template <int WARPS_PER_BLOCK, int ITEMS_PER_THREAD>
__global__ void softmax_martix_v2(const float* x, float* y, int m, int n) {
    int tid = threadIdx.x;
    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (row >= m) {
        return;
    }

    const float* row_x = x + row * n;
    float* row_y = y + row * n;

    float values[ITEMS_PER_THREAD];
    float exp_values[ITEMS_PER_THREAD];
    float local_max = -FLT_MAX;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = lane_id + item * kWarpSize;

        values[item] = col < n ? row_x[col] : -FLT_MAX;
        local_max = fmaxf(local_max, values[item]);
    }

    float row_max = warp_all_reduce_max(local_max);

    float local_sum = 0.0f;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = lane_id + item * kWarpSize;

        exp_values[item] = col < n ? expf(values[item] - row_max) : 0.0f;
        local_sum += exp_values[item];
    }

    float row_sum = warp_all_reduce_sum(local_sum);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int col = lane_id + item * kWarpSize;

        if (col < n) {
            row_y[col] = exp_values[item] / row_sum;
        }
    }
}

void launch_softmax_matrix_v3(const float* x, float* y, int m, int n) {
    if (m <= 0 || n <= 0) {
        return;
    }

    // 先看列数是不是小于第二种方法的 Block_Size,是就是短行
    if (n <= kV2MaxColumns) {
        int grid_size = ceil_div(m, kV2WarpsPerBlock);

        softmax_martix_v2<kV2WarpsPerBlock, kV2ItemsPerThread>
            <<<grid_size, kV2BlockSize>>>(x, y, m, n);
        return;
    }

    if (n <= kV1MaxColumns) {
        softmax_martix_v1<kV1BlockSize, kV1ItemsPerThread><<<m, kV1BlockSize>>>(x, y, m, n);
        return;
    }

    std::fprintf(stderr, "Unsupported row width: n = %d, max supported = %d\n", n, kV1MaxColumns);
    std::exit(EXIT_FAILURE);
}

template <int BLOCK_SIZE>
__global__ void softmax_matrix_v4_float4(const float* __restrict__ x, float* __restrict__ y, int m,
                                         int n) {
    static_assert(BLOCK_SIZE % kWarpSize == 0, "BLOCK_SIZE must be a multiple of 32");

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= m) {
        return;
    }

    const float* row_x = x + row * n;
    float* row_y = y + row * n;

    const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
    float4* row_y4 = reinterpret_cast<float4*>(row_y);

    int vector_count = n / 4;
    __shared__ float warp_smem[BLOCK_SIZE / kWarpSize];

    float4 values = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    float local_max = -FLT_MAX;

    if (tid < vector_count) {
        values = row_x4[tid];
        local_max = fmaxf(fmaxf(values.x, values.y), fmaxf(values.z, values.w));
    } // 一个线程处理相邻的 4 个元素

    float row_max = block_reduce_max<BLOCK_SIZE>(local_max, warp_smem);

    float4 exp_values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float local_sum = 0.0f;

    if (tid < vector_count) {
        exp_values.x = expf(values.x - row_max);
        exp_values.y = expf(values.y - row_max);
        exp_values.z = expf(values.z - row_max);
        exp_values.w = expf(values.w - row_max);

        local_sum = exp_values.x + exp_values.y + exp_values.z + exp_values.w;
    }

    float row_sum = block_reduce_sum<BLOCK_SIZE>(local_sum, warp_smem);

    if (tid < vector_count) {
        float4 output;
        output.x = exp_values.x / row_sum;
        output.y = exp_values.y / row_sum;
        output.z = exp_values.z / row_sum;
        output.w = exp_values.w / row_sum;
        row_y4[tid] = output;
    }
}
void launch_softmax_matrix_v4(const float* x, float* y, int m, int n) {
    constexpr int kV1BlockSize = 256;
    constexpr int kV1MaxColumns = 1024;

    if (m <= 0 || n <= 0) {
        return;
    }

    if (n <= kV2MaxColumns) {
        int grid_size = ceil_div(m, kV2WarpsPerBlock);
        softmax_martix_v2<kV2WarpsPerBlock, kV2ItemsPerThread>
            <<<grid_size, kV2BlockSize>>>(x, y, m, n);
        return;
    }

    if (n <= kV1MaxColumns && n % 4 == 0 && is_aligned_16(x) && is_aligned_16(y)) {
        softmax_matrix_v4_float4<kV1BlockSize><<<m, kV1BlockSize>>>(x, y, m, n);
        return;
    }

    if (n <= kV1MaxColumns) {
        softmax_martix_v1<kV1BlockSize, kV1ItemsPerThread><<<m, kV1BlockSize>>>(x, y, m, n);
        return;
    }

    std::fprintf(stderr, "Unsupported row width: n = %d\n", n);
    std::exit(EXIT_FAILURE);
}

void softmax_matrix_host(const float* x, float* y, int m, int n) {
    for (int row = 0; row < m; ++row) {
        const float* row_x = x + row * n;
        float* row_y = y + row * n;

        float row_max = -FLT_MAX;
        for (int col = 0; col < n; ++col) {
            row_max = fmaxf(row_max, row_x[col]);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            row_y[col] = expf(row_x[col] - row_max);
            row_sum += row_y[col];
        }

        for (int col = 0; col < n; ++col) {
            row_y[col] /= row_sum;
        }
    }
}

float max_abs_error(const float* actual, const float* expected, int count) {
    float max_error = 0.0f;
    for (int i = 0; i < count; ++i) {
        max_error = fmaxf(max_error, std::fabs(actual[i] - expected[i]));
    }
    return max_error;
}

float max_row_sum_error(const float* x, int m, int n) {
    float max_error = 0.0f;
    for (int row = 0; row < m; ++row) {
        float row_sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            row_sum += x[row * n + col];
        }
        max_error = fmaxf(max_error, std::fabs(row_sum - 1.0f));
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

bool run_softmax_matrix_case(int m, int n) {
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    const int element_count = m * n;
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);

    float* h_x = static_cast<float*>(std::malloc(bytes));
    float* h_y = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    if (h_x == nullptr || h_y == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_x);
        std::free(h_y);
        std::free(h_ref);
        return false;
    }

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int index = row * n + col;
            h_x[index] = -4.0f + static_cast<float>((row * 17 + col * 13) % 29) * 0.25f;
        }
        h_x[row * n + (row * 37) % n] = 8.0f;
    }
    softmax_matrix_host(h_x, h_ref, m, n);

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    auto launch = [&]() { launch_softmax_matrix_v4(d_x, d_y, m, n); };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_y, h_ref, element_count);
    float max_sum_error = max_row_sum_error(h_y, m, n);
    float avg_ms = benchmark_kernel_ms(launch, kWarmup, kIters);
    double effective_bytes = static_cast<double>(bytes) * 2.0;
    double bandwidth_gbs = effective_bytes / (avg_ms * 1.0e6);
    bool pass = max_error < 1e-5f && max_sum_error < 1e-5f;

    std::printf("M = %d, N = %d\n", m, n);
    std::printf("max_row_sum_err = %.8f\n", max_sum_error);
    std::printf("time(ms)        = %.4f\n", avg_ms);
    std::printf("bandwidth       = %.2f GB/s\n", bandwidth_gbs);
    std::printf("max_abs_err     = %.8f\n", max_error);
    std::printf("correctness     = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);

    return pass;
}

int main() {
    bool v2_pass = run_softmax_matrix_case(127, 127);
    bool v2_boundary_pass = run_softmax_matrix_case(127, 128);
    bool v4_partial_pass = run_softmax_matrix_case(127, 256);
    bool v4_boundary_pass = run_softmax_matrix_case(127, 1024);
    bool v1_pass = run_softmax_matrix_case(127, 1003);
    return v2_pass && v2_boundary_pass && v4_partial_pass && v4_boundary_pass && v1_pass
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
