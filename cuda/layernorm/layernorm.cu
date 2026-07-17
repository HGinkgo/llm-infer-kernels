
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

constexpr int warpSize = 32;
/*
    RMSNorm:
    x * rsqrt(mean(x²) + epsilon) * weight

    LayerNorm:
    (x - mean(x)) * rsqrt(variance + epsilon) * weight + bias
*/
__global__ void layernorm_v1(const float* input, const float* weight, const float* bias,
                             float* output, int rows, int cols, float epsilon) {
    // 一个线程处理一行
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    float sum = 0.0f;

    for (int col = 0; col < cols; ++col) {
        sum += input_row[col];
    }

    float mean = sum / static_cast<float>(cols);
    float variance_sum = 0.0f;

    for (int col = 0; col < cols; ++col) {
        float diff = input_row[col] - mean;
        variance_sum += diff * diff;
    }

    float variance = variance_sum / static_cast<float>(cols);
    float inv_std = rsqrtf(variance + epsilon);

    for (int col = 0; col < cols; ++col) {
        float normalized = (input_row[col] - mean) * inv_std;
        output_row[col] = normalized * weight[col] + bias[col];
    }
}

__device__ __inline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void layernorm_v2(const float* input, const float* weight, const float* bias,
                             float* output, int rows, int cols, float epsilon) {

    // 一个 block 处理一行
    static_assert(BLOCK_SIZE % warpSize == 0, "BLOCK_SIZE must be a multiple of warp size");

    constexpr int NUM_WARPS = BLOCK_SIZE / warpSize;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float shared_mean;
    __shared__ float shared_variance;

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
        local_sum += input_row[col];
    }

    local_sum = warp_reduce_sum(local_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane_id == 0) {
            shared_mean = block_sum / static_cast<float>(cols);
        }
    }
    __syncthreads();

    float mean = shared_mean;
    float diff_sum = 0.0f;
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        float diff = input_row[col] - mean;
        diff_sum += diff * diff;
    }
    diff_sum = warp_reduce_sum(diff_sum);

    if (lane_id == 0) {
        warp_sums[warp_id] = diff_sum;
    }
    __syncthreads();

    float variance_sum = 0.0f;
    if (warp_id == 0) {
        variance_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        variance_sum = warp_reduce_sum(variance_sum);

        if (lane_id == 0) {
            shared_variance = variance_sum / static_cast<float>(cols);
        }
    }
    __syncthreads();

    float variance = shared_variance;
    float inv_std = rsqrtf(variance + epsilon);
    for (int col = tid; col < cols; col += BLOCK_SIZE) {
        output_row[col] = (input_row[col] - mean) * inv_std * weight[col] + bias[col];
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_v3(const float* input, const float* weight, const float* bias,
                             float* output, int rows, int cols, float epsilon) {

    static_assert(BLOCK_SIZE % warpSize == 0, "BLOCK_SIZE must be a multiple of warp size");

    constexpr int NUM_WARPS = BLOCK_SIZE / warpSize;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float shared_mean;
    __shared__ float shared_variance;

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
    const float4* weight4 = reinterpret_cast<const float4*>(weight);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* output_row4 = reinterpret_cast<float4*>(output_row);

    int vector_count = cols / 4;
    float local_sum = 0.0f;

    for (int vector_idx = tid; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        float4 input_value = input_row4[vector_idx];
        local_sum += input_value.x + input_value.y + input_value.z + input_value.w;
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        float block_sum = warp_reduce_sum(local_sum);

        if (lane_id == 0) {
            shared_mean = block_sum / static_cast<float>(cols);
        }
    }
    __syncthreads();
    float mean = shared_mean;

    float variance_sum = 0.0f;
    for (int vector_idx = tid; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        float4 input_value = input_row4[vector_idx];
        variance_sum += (input_value.x - mean) * (input_value.x - mean);
        variance_sum += (input_value.y - mean) * (input_value.y - mean);
        variance_sum += (input_value.z - mean) * (input_value.z - mean);
        variance_sum += (input_value.w - mean) * (input_value.w - mean);
    }

    variance_sum = warp_reduce_sum(variance_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = variance_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        variance_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;
        variance_sum = warp_reduce_sum(variance_sum);

        if (lane_id == 0) {
            shared_variance = variance_sum / static_cast<float>(cols);
        }
    }
    __syncthreads();
    float variance = shared_variance;

    float inv_std = rsqrtf(variance + epsilon);
    for (int vector_idx = tid; vector_idx < vector_count; vector_idx += BLOCK_SIZE) {
        float4 input_value = input_row4[vector_idx];
        float4 weight_value = weight4[vector_idx];
        float4 bias_value = bias4[vector_idx];
        float4 output_value;

        output_value.x = (input_value.x - mean) * inv_std * weight_value.x + bias_value.x;
        output_value.y = (input_value.y - mean) * inv_std * weight_value.y + bias_value.y;
        output_value.z = (input_value.z - mean) * inv_std * weight_value.z + bias_value.z;
        output_value.w = (input_value.w - mean) * inv_std * weight_value.w + bias_value.w;

        output_row4[vector_idx] = output_value;
    }
}

template <int BLOCK_SIZE>
__global__ void layernorm_v4(const float* input, const float* weight, const float* bias,
                             float* output, int rows, int cols, float epsilon) {

    static_assert(BLOCK_SIZE % warpSize == 0, "BLOCK_SIZE must be a multiple of warp size");
    constexpr int ROWS_PER_BLOCK = BLOCK_SIZE / warpSize;

    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;
    int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;

    if (row >= rows) {
        return;
    }

    const float* input_row = input + row * cols;
    float* output_row = output + row * cols;

    const float4* input_row4 = reinterpret_cast<const float4*>(input_row);
    const float4* weight4 = reinterpret_cast<const float4*>(weight);
    const float4* bias4 = reinterpret_cast<const float4*>(bias);
    float4* output_row4 = reinterpret_cast<float4*>(output_row);

    int vector_count = cols / 4;

    float local_sum = 0.0f;
    for (int vector_idx = lane_id; vector_idx < vector_count; vector_idx += warpSize) {
        float4 input_value = input_row4[vector_idx];

        local_sum += input_value.x + input_value.y + input_value.z + input_value.w;
    }
    float warp_sum = warp_reduce_sum(local_sum);

    float mean = 0.0f;
    if (lane_id == 0) {
        mean = warp_sum / static_cast<float>(cols);
    }

    mean = __shfl_sync(0xffffffff, mean, 0);

    float variance_local_sum = 0.0f;
    for (int vector_idx = lane_id; vector_idx < vector_count; vector_idx += warpSize) {
        float4 input_value = input_row4[vector_idx];

        variance_local_sum += (input_value.x - mean) * (input_value.x - mean);
        variance_local_sum += (input_value.y - mean) * (input_value.y - mean);
        variance_local_sum += (input_value.z - mean) * (input_value.z - mean);
        variance_local_sum += (input_value.w - mean) * (input_value.w - mean);
    }
    float variance_sum = warp_reduce_sum(variance_local_sum);

    float variance = 0.0f;
    if (lane_id == 0) {
        variance = variance_sum / static_cast<float>(cols);
    }
    variance = __shfl_sync(0xffffffff, variance, 0);

    float inv_std = rsqrtf(variance + epsilon);
    for (int vector_idx = lane_id; vector_idx < vector_count; vector_idx += warpSize) {
        float4 input_value = input_row4[vector_idx];
        float4 weight_value = weight4[vector_idx];
        float4 bias_value = bias4[vector_idx];
        float4 output_value;

        output_value.x = (input_value.x - mean) * inv_std * weight_value.x + bias_value.x;
        output_value.y = (input_value.y - mean) * inv_std * weight_value.y + bias_value.y;
        output_value.z = (input_value.z - mean) * inv_std * weight_value.z + bias_value.z;
        output_value.w = (input_value.w - mean) * inv_std * weight_value.w + bias_value.w;

        output_row4[vector_idx] = output_value;
    }
}

void layernorm_host(const float* input, const float* weight, const float* bias, float* output,
                    int rows, int cols, float epsilon) {
    for (int row = 0; row < rows; ++row) {
        const float* input_row = input + row * cols;
        float* output_row = output + row * cols;

        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            sum += static_cast<double>(input_row[col]);
        }
        double mean = sum / static_cast<double>(cols);

        double variance_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            double difference = static_cast<double>(input_row[col]) - mean;
            variance_sum += difference * difference;
        }

        float variance = static_cast<float>(variance_sum / static_cast<double>(cols));
        float inv_std = 1.0f / std::sqrt(variance + epsilon);

        for (int col = 0; col < cols; ++col) {
            float normalized = (input_row[col] - static_cast<float>(mean)) * inv_std;
            output_row[col] = normalized * weight[col] + bias[col];
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

template <typename KernelLauncher>
float benchmark_kernel_ms(KernelLauncher&& launcher, int warmup, int iterations) {
    for (int i = 0; i < warmup; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launcher();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(iterations);
}

int main() {
    constexpr int ROWS = 37;
    constexpr int COLS = 1003;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;
    constexpr float EPSILON = 1e-5f;

    const int element_count = ROWS * COLS;
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);
    const size_t parameter_bytes = static_cast<size_t>(COLS) * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    float* h_weight = static_cast<float*>(std::malloc(parameter_bytes));
    float* h_bias = static_cast<float*>(std::malloc(parameter_bytes));
    float* h_output = static_cast<float*>(std::malloc(bytes));
    float* h_reference = static_cast<float*>(std::malloc(bytes));

    if (h_input == nullptr || h_weight == nullptr || h_bias == nullptr || h_output == nullptr ||
        h_reference == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_input);
        std::free(h_weight);
        std::free(h_bias);
        std::free(h_output);
        std::free(h_reference);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < element_count; ++i) {
        h_input[i] = static_cast<float>((i * 17) % 101 - 50) * 0.02f;
    }
    for (int col = 0; col < COLS; ++col) {
        h_weight[col] = 0.5f + static_cast<float>(col % 13) * 0.05f;
        h_bias[col] = static_cast<float>(col % 7 - 3) * 0.03f;
    }

    layernorm_host(h_input, h_weight, h_bias, h_reference, ROWS, COLS, EPSILON);

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_bias = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_weight, parameter_bytes));
    CUDA_CHECK(cudaMalloc(&d_bias, parameter_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight, parameter_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias, parameter_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    int grid_size = ROWS;
    auto launch = [&]() {
        layernorm_v2<BLOCK_SIZE>
            <<<grid_size, BLOCK_SIZE>>>(d_input, d_weight, d_bias, d_output, ROWS, COLS, EPSILON);
    };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_output, h_reference, element_count);
    float average_ms = benchmark_kernel_ms(launch, WARMUP, ITERATIONS);
    bool pass = max_error < 1e-5f;

    std::printf("ROWS = %d, COLS = %d, BLOCK_SIZE = %d\n", ROWS, COLS, BLOCK_SIZE);
    std::printf("time(ms) = %.4f\n", average_ms);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_weight);
    std::free(h_bias);
    std::free(h_output);
    std::free(h_reference);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
