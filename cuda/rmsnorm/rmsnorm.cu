#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

constexpr int warpSize = 32;

template <int BLOCK_SIZE>
__global__ void rmsnorm_v1(const float* input, const float* weight, float* output, int rows,
                           int cols, float epsilon) {
    extern __shared__ float shared_sum[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float value = input_row[col];
        local_sum += value * value;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_square = shared_sum[0] / static_cast<float>(cols);
        shared_sum[0] = rsqrtf(mean_square + epsilon);
    }
    __syncthreads();

    float inv_rms = shared_sum[0];

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        output_row[col] = input_row[col] * inv_rms * weight[col];
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_v2(const float* input, const float* weight, float* output, int rows,
                           int cols, float epsilon) {
    constexpr int WARP_SIZE = 32;
    static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of warp size");

    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    if (row >= rows) {
        return;
    }

    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float value = input_row[col];
        local_sum += value * value;
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane_id == 0) {
            float mean_square = block_sum / static_cast<float>(cols);
            warp_sums[0] = rsqrtf(mean_square + epsilon);
        }
    }
    __syncthreads();

    float inv_rms = warp_sums[0];

    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        output_row[col] = input_row[col] * inv_rms * weight[col];
    }
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_v3(const float* input, const float* weight, float* output, int rows,
                           int cols, float epsilon) {
    static_assert(BLOCK_SIZE % warpSize == 0, "BLCOK_SIZE must be a multiple of warp size");

    constexpr int NUM_WARPS = BLOCK_SIZE / warpSize;
    __shared__ float warp_sums[NUM_WARPS];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    if (row >= rows) {
        return;
    }

    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    const float4* input_row4 = reinterpret_cast<const float4*>(input_row);
    float4* output_row4 = reinterpret_cast<float4*>(output_row);
    const float4* weight4 = reinterpret_cast<const float4*>(weight);

    int vector_count = cols / 4;
    float local_sum = 0.0f;

    for (int vector_idx = tid; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        float4 input_value = input_row4[vector_idx];

        local_sum += input_value.x * input_value.x;
        local_sum += input_value.y * input_value.y;
        local_sum += input_value.z * input_value.z;
        local_sum += input_value.w * input_value.w;
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane_id == 0) {
            float mean_square = block_sum / static_cast<float>(cols);
            warp_sums[0] = rsqrtf(mean_square + epsilon);
        }
    }
    __syncthreads();

    float inv_rms = warp_sums[0];

    for (int vector_idx = tid; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        float4 input_value = input_row4[vector_idx];
        float4 weight_value = weight4[vector_idx];

        float4 output_value;

        output_value.x = input_value.x * inv_rms * weight_value.x;
        output_value.y = input_value.y * inv_rms * weight_value.y;
        output_value.z = input_value.z * inv_rms * weight_value.z;
        output_value.w = input_value.w * inv_rms * weight_value.w;

        output_row4[vector_idx] = output_value;
    }
}

template <int BLOCK_SIZE>
__global__ void rmsnorm_v4(const float* __restrict__ input, const float* __restrict__ weight,
                           float* __restrict__ output, int rows, int cols, float epsilon) {
    static_assert(BLOCK_SIZE % warpSize == 0, "BLOCK_SIZE must be a multiple of warp size");

    constexpr int ROWS_PER_BLOCK = BLOCK_SIZE / warpSize;
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int row_id = blockIdx.x * ROWS_PER_BLOCK + tid / warpSize;

    if (row_id >= rows) {
        return;
    }

    const float* input_row = input + row_id * cols;
    float* output_row = output + row_id * cols;

    const float4* input_row4 = reinterpret_cast<const float4*>(input_row);
    float4* output_row4 = reinterpret_cast<float4*>(output_row);
    const float4* weight4 = reinterpret_cast<const float4*>(weight);

    int vector_count = cols / 4;
    float local_sum = 0.0f;

    for (int vector_idx = lane_id; vector_idx < vector_count; vector_idx += warpSize) {
        float4 x = input_row4[vector_idx];

        local_sum = fmaf(x.x, x.x, local_sum);
        local_sum = fmaf(x.y, x.y, local_sum);
        local_sum = fmaf(x.z, x.z, local_sum);
        local_sum = fmaf(x.w, x.w, local_sum);
    }

    float square_sum = warp_reduce_sum(local_sum);
    square_sum = __shfl_sync(0xffffffff, square_sum, 0);

    float inv_rms = rsqrtf(square_sum / static_cast<float>(cols) + epsilon);

    for (int vector_idx = lane_id; vector_idx < vector_count; vector_idx += warpSize) {
        float4 x = input_row4[vector_idx];
        float4 w = weight4[vector_idx];
        float4 y;

        y.x = x.x * inv_rms * w.x;
        y.y = x.y * inv_rms * w.y;
        y.z = x.z * inv_rms * w.z;
        y.w = x.w * inv_rms * w.w;

        output_row4[vector_idx] = y;
    }
}

template <int BLOCK_SIZE>
void launch_rmsnorm(const float* input, const float* weight, float* output, int rows, int cols,
                    float epsilon) {
    if (cols % 4 == 0) {
        constexpr int ROWS_PER_BLOCK = BLOCK_SIZE / warpSize;
        int grid_size = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

        rmsnorm_v4<BLOCK_SIZE>
            <<<grid_size, BLOCK_SIZE>>>(input, weight, output, rows, cols, epsilon);
    } else {
        rmsnorm_v2<BLOCK_SIZE><<<rows, BLOCK_SIZE>>>(input, weight, output, rows, cols, epsilon);
    }
}

void rmsnorm_host(const float* input, const float* weight, float* output, int rows, int cols,
                  float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const float* input_row = input + row * cols;
        float* output_row = output + row * cols;

        double square_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            double value = static_cast<double>(input_row[col]);
            square_sum += value * value;
        }

        float mean_square = static_cast<float>(square_sum / static_cast<double>(cols));
        float inv_rms = 1.0f / std::sqrt(mean_square + epsilon);

        for (int col = 0; col < cols; ++col) {
            output_row[col] = input_row[col] * inv_rms * weight[col];
        }
    }
}

float max_abs_error(const float* actual, const float* expected, int n) {
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = fmaxf(max_error, std::fabs(actual[i] - expected[i]));
    }
    return max_error;
}

int main() {
    constexpr int ROWS = 37;
    constexpr int COLS = 1024;
    constexpr int BLOCK_SIZE = 256;
    constexpr float EPSILON = 1e-5f;

    const int element_count = ROWS * COLS;
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(COLS) * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    float* h_weight = static_cast<float*>(std::malloc(weight_bytes));
    float* h_output = static_cast<float*>(std::malloc(bytes));
    float* h_reference = static_cast<float*>(std::malloc(bytes));
    if (h_input == nullptr || h_weight == nullptr || h_output == nullptr ||
        h_reference == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_input);
        std::free(h_weight);
        std::free(h_output);
        std::free(h_reference);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < element_count; ++i) {
        h_input[i] = static_cast<float>((i * 17) % 101 - 50) * 0.02f;
    }
    for (int col = 0; col < COLS; ++col) {
        h_weight[col] = 0.5f + static_cast<float>(col % 13) * 0.05f;
    }
    rmsnorm_host(h_input, h_weight, h_reference, ROWS, COLS, EPSILON);

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    launch_rmsnorm<BLOCK_SIZE>(d_input, d_weight, d_output, ROWS, COLS, EPSILON);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_output, h_reference, element_count);
    bool pass = max_error < 1e-5f;

    std::printf("ROWS = %d, COLS = %d, BLOCK_SIZE = %d\n", ROWS, COLS, BLOCK_SIZE);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_weight);
    std::free(h_output);
    std::free(h_reference);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
