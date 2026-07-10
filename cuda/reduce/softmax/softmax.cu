#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

constexpr int kWarpSize = 32;

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
    int lane_id = tid % kWarpSize; // warp 内部的线程 id
    int warp_id = tid / kWarpSize; // warp 的 id

    val = warp_reduce_max<BLOCK_SIZE>(val);

    if (lane_id == 0) {
        warp_smem[warp_id] = val;
    } // block 里面的 warp 自己的 max 已经算出来了
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

// 1D softmax
template <int BLOCK_SIZE>
__global__ void softmax_1d(const float* x, float* y, int n) {
    int tid = threadIdx.x;

    __shared__ float smem[BLOCK_SIZE];

    // 1. load x，越界线程填 -FLT_MAX
    float val = -FLT_MAX;
    if (tid < n) {
        val = x[tid];
    }

    smem[tid] = val;
    __syncthreads();

    // 2. block 内 reduce max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }

    float max_val = smem[0];

    // 3. 计算 exp(x - max)
    float exp_val = 0.0f;
    if (tid < n) {
        exp_val = expf(x[tid] - max_val);
    }

    smem[tid] = exp_val;
    __syncthreads();

    // 4. block 内 reduce sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            smem[tid] = smem[tid] + smem[tid + offset];
        }

        __syncthreads();
    }

    float sum_val = smem[0];

    // 5. 写回 softmax
    if (tid < n) {
        y[tid] = exp_val / sum_val;
    }
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void softmax_1d_v2(const float* x, float* y, int n) {
    static_assert(BLOCK_SIZE % kWarpSize == 0, "BLOCK_SIZE must be a multiple of 32");

    int tid = threadIdx.x;

    __shared__ float warp_smem[BLOCK_SIZE / kWarpSize];

    float values[ITEMS_PER_THREAD];
    float exp_values[ITEMS_PER_THREAD];

    // 1. 每线程加载多个元素，并在寄存器中求 local max
    float local_max = -FLT_MAX;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = tid + item * BLOCK_SIZE;

        values[item] = idx < n ? x[idx] : -FLT_MAX;
        local_max = fmaxf(local_max, values[item]);
    }

    // 2. 归约得到整个 vector 的 max
    float max_val = block_reduce_max<BLOCK_SIZE>(local_max, warp_smem);

    // 3. 在寄存器中计算 exp，并求 local sum
    float local_sum = 0.0f;

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = tid + item * BLOCK_SIZE;

        exp_values[item] = idx < n ? expf(values[item] - max_val) : 0.0f;
        local_sum += exp_values[item];
    }

    float sum_val = block_reduce_sum<BLOCK_SIZE>(local_sum, warp_smem);

#pragma unroll
    for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
        int idx = tid + BLOCK_SIZE * item;

        if (idx < n) {
            y[idx] = exp_values[item] / sum_val;
        }
    }
}

void softmax_host(const float* x, float* y, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; ++i) {
        max_val = fmaxf(max_val, x[i]);
    }

    float sum_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        y[i] = expf(x[i] - max_val);
        sum_val += y[i];
    }

    for (int i = 0; i < n; ++i) {
        y[i] /= sum_val;
    }
}

float max_abs_error(const float* actual, const float* expected, int n) {
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = fmaxf(max_error, std::fabs(actual[i] - expected[i]));
    }
    return max_error;
}

float sum_host_array(const float* x, int n) {
    float sum = 0.0f;
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

int main() {
    constexpr int N = 1003;
    constexpr int BLOCK_SIZE = 256;
    constexpr int ITEMS_PER_THREAD = 4;
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    float* h_x = static_cast<float*>(std::malloc(sizeof(float) * N));
    float* h_y = static_cast<float*>(std::malloc(sizeof(float) * N));
    float* h_ref = static_cast<float*>(std::malloc(sizeof(float) * N));

    for (int i = 0; i < N; ++i) {
        h_x[i] = -4.0f + static_cast<float>(i % 23) * 0.25f;
    }
    h_x[73] = 8.0f;
    softmax_host(h_x, h_ref, N);

    float* d_x = nullptr;
    float* d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, sizeof(float) * N, cudaMemcpyHostToDevice));

    auto launch = [&]() {
        softmax_1d_v2<BLOCK_SIZE, ITEMS_PER_THREAD><<<1, BLOCK_SIZE>>>(d_x, d_y, N);
    };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_y, d_y, sizeof(float) * N, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_y, h_ref, N);
    float output_sum = sum_host_array(h_y, N);
    float avg_ms = benchmark_kernel_ms(launch, kWarmup, kIters);
    double effective_bytes = static_cast<double>(N) * sizeof(float) * 2.0;
    double bandwidth_gbs = effective_bytes / (avg_ms * 1.0e6);
    bool pass = max_error < 1e-5f && std::fabs(output_sum - 1.0f) < 1e-5f;

    std::printf("N = %d, BLOCK_SIZE = %d, ITEMS_PER_THREAD = %d\n", N, BLOCK_SIZE,
                ITEMS_PER_THREAD);
    std::printf("output_sum = %.8f\n", output_sum);
    std::printf("time(ms)   = %.4f\n", avg_ms);
    std::printf("bandwidth  = %.2f GB/s\n", bandwidth_gbs);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    std::free(h_x);
    std::free(h_y);
    std::free(h_ref);

    return pass ? 0 : 1;
}
