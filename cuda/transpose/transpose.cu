
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

__global__ void transpose_v1(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

template <int TILE_DIM>
__global__ void transpose_v2_share(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int input_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int input_row = blockIdx.y * TILE_DIM + threadIdx.y;

    if (input_row < rows && input_col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }
    __syncthreads();

    int output_col = blockIdx.y * TILE_DIM + threadIdx.x;
    int output_row = blockIdx.x * TILE_DIM + threadIdx.y;

    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int TILE_DIM>
__global__ void transpose_v3_share(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int input_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int input_row = blockIdx.y * TILE_DIM + threadIdx.y;

    if (input_row < rows && input_col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[input_row * cols + input_col];
    }
    __syncthreads();

    int output_col = blockIdx.y * TILE_DIM + threadIdx.x;
    int output_row = blockIdx.x * TILE_DIM + threadIdx.y;

    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] = tile[threadIdx.x][threadIdx.y];
    }
}

template <int TILE_DIM, int BLOCK_ROWS> // 这个 block 在 y 方向启动的线程行数
__global__ void transpose_v4_share(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int input_col = blockIdx.x * TILE_DIM + threadIdx.x;
    int input_row = blockIdx.y * TILE_DIM + threadIdx.y;

#pragma unroll
    // 32 行，每次处理 8 行
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = input_row + j;
        if (row < rows && input_col < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * cols + input_col];
        }
    }
    __syncthreads();

    int output_col = blockIdx.y * TILE_DIM + threadIdx.x;
    int output_row = blockIdx.x * TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = output_row + j;
        if (row < cols && output_col < rows) {
            output[row * rows + output_col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

void transpose_host(const float* input, float* output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            output[col * rows + row] = input[row * cols + col];
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
    constexpr int ROWS = 1003;
    constexpr int COLS = 257;
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    constexpr int kWarmup = 10;
    constexpr int kIters = 100;

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(ceil_div(COLS, TILE_DIM), ceil_div(ROWS, TILE_DIM));

    const int element_count = ROWS * COLS;
    const size_t bytes = static_cast<size_t>(element_count) * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    float* h_output = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    if (h_input == nullptr || h_output == nullptr || h_ref == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_input);
        std::free(h_output);
        std::free(h_ref);
        return EXIT_FAILURE;
    }

    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < COLS; ++col) {
            h_input[row * COLS + col] = static_cast<float>((row * 17 + col * 13) % 97) * 0.25f;
        }
    }
    transpose_host(h_input, h_ref, ROWS, COLS);

    float* d_input = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    auto launch = [&]() {
        transpose_v4_share<TILE_DIM, BLOCK_ROWS><<<grid, block>>>(d_input, d_output, ROWS, COLS);
    };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_output, h_ref, element_count);
    float avg_ms = benchmark_kernel_ms(launch, kWarmup, kIters);
    double effective_bytes = static_cast<double>(bytes) * 2.0;
    double bandwidth_gbs = effective_bytes / (avg_ms * 1.0e6);
    bool pass = max_error < 1e-6f;

    std::printf("ROWS = %d, COLS = %d, BLOCK = %u x %u\n", ROWS, COLS, block.x, block.y);
    std::printf("time(ms)   = %.4f\n", avg_ms);
    std::printf("bandwidth  = %.2f GB/s\n", bandwidth_gbs);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_output);
    std::free(h_ref);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
