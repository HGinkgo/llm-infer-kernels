#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

constexpr int warpSize = 32;

__global__ void gemv_v1(const float* matrix, const float* vector, float* output, int rows,
                        int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    const float* matrix_row = matrix + row * cols;
    float sum = 0.0f;

    for (int col = 0; col < cols; ++col) {
        sum = fmaf(matrix_row[col], vector[col], sum);
    }

    output[row] = sum;
}

__device__ __inline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void gemv_v2(const float* matrix, const float* vector, float* output, int rows,
                        int cols) {

    constexpr int NUMS_PER_BLOCK = BLOCK_SIZE / warpSize;

    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;
    int row = blockIdx.x * NUMS_PER_BLOCK + warp_id;

    if (row >= rows) {
        return;
    }

    const float* matrix_row = matrix + row * cols;
    float sum = 0.0f;

    // 前面写错写成了 tid
    for (int col = lane_id; col < cols; col += warpSize) {
        sum += matrix_row[col] * vector[col];
    }

    sum = warp_reduce_sum(sum);
    if (lane_id == 0) {
        output[row] = sum;
    }
}

template <int BLOCK_SIZE>
__global__ void gemv_v3(const float* matrix, const float* vector, float* output, int rows,
                        int cols) {
    static_assert(BLOCK_SIZE % warpSize == 0);

    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / warpSize;

    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;

    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= rows) {
        return;
    }

    const float* matrix_row = matrix + cols * row;
    float sum = 0.0f;

    if ((cols & 3) == 0) {
        const float4* matrix_row_vec = reinterpret_cast<const float4*>(matrix_row);
        const float4* vector_vec = reinterpret_cast<const float4*>(vector);

        int vec_cols = cols / 4;

        for (int index = lane_id; index < vec_cols; index += warpSize) {
            float4 matrix_value = matrix_row_vec[index];
            float4 vector_value = vector_vec[index];

            sum = fmaf(matrix_value.x, vector_value.x, sum);
            sum = fmaf(matrix_value.y, vector_value.y, sum);
            sum = fmaf(matrix_value.z, vector_value.z, sum);
            sum = fmaf(matrix_value.w, vector_value.w, sum);
        }
    } else {
        for (int col = lane_id; col < cols; col += warpSize) {
            sum = fmaf(matrix_row[col], vector[col], sum);
        }
    }

    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        output[row] = sum;
    }
}

void gemv_host(const float* matrix, const float* vector, float* output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        const float* matrix_row = matrix + row * cols;
        float sum = 0.0f;

        for (int col = 0; col < cols; ++col) {
            sum += matrix_row[col] * vector[col];
        }

        output[row] = sum;
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
    constexpr int ROWS = 1024;
    constexpr int COLS = 1024;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;

    const int matrix_elements = ROWS * COLS;
    const size_t matrix_bytes = static_cast<size_t>(matrix_elements) * sizeof(float);
    const size_t vector_bytes = static_cast<size_t>(COLS) * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(ROWS) * sizeof(float);

    float* h_matrix = static_cast<float*>(std::malloc(matrix_bytes));
    float* h_vector = static_cast<float*>(std::malloc(vector_bytes));
    float* h_output = static_cast<float*>(std::malloc(output_bytes));
    float* h_reference = static_cast<float*>(std::malloc(output_bytes));
    if (h_matrix == nullptr || h_vector == nullptr || h_output == nullptr ||
        h_reference == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_matrix);
        std::free(h_vector);
        std::free(h_output);
        std::free(h_reference);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < matrix_elements; ++i) {
        h_matrix[i] = static_cast<float>((i * 17) % 101 - 50) * 0.02f;
    }
    for (int col = 0; col < COLS; ++col) {
        h_vector[col] = static_cast<float>((col * 13) % 29 - 14) * 0.03f;
    }
    gemv_host(h_matrix, h_vector, h_reference, ROWS, COLS);

    float* d_matrix = nullptr;
    float* d_vector = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHECK(cudaMalloc(&d_vector, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vector, h_vector, vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / warpSize;
    int grid_size = ceil_div(ROWS, WARPS_PER_BLOCK);
    auto launch = [&]() {
        gemv_v3<BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(d_matrix, d_vector, d_output, ROWS, COLS);
    };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_output, h_reference, ROWS);
    float average_ms = benchmark_kernel_ms(launch, WARMUP, ITERATIONS);
    bool pass = max_error < 1e-5f;

    std::printf("ROWS = %d, COLS = %d, BLOCK_SIZE = %d\n", ROWS, COLS, BLOCK_SIZE);
    std::printf("time(ms) = %.4f\n", average_ms);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_vector));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_matrix);
    std::free(h_vector);
    std::free(h_output);
    std::free(h_reference);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
