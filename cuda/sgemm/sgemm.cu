
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "utils.cuh"

__global__ void sgemm_v1(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;

    for (int index = 0; index < k; ++index) {
        float value_a = matrix_a[row * k + index];
        float value_b = matrix_b[index * n + col];

        sum = fmaf(value_a, value_b, sum);
    }

    matrix_c[row * n + col] = sum;
}

template <int TILE_SIZE>
__global__ void sgemm_v2(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    float sum = 0.0f;
    int tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tile_count; ++tile) {
        int matrix_a_col = tile * TILE_SIZE + tx;
        int matrix_b_row = tile * TILE_SIZE + ty;

        if (row < m && matrix_a_col < k) {
            tile_a[ty][tx] = matrix_a[row * k + matrix_a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        if (matrix_b_row < k && col < n) {
            tile_b[ty][tx] = matrix_b[matrix_b_row * n + col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int index = 0; index < TILE_SIZE; ++index) {
            sum = fmaf(tile_a[ty][index], tile_b[index][tx], sum);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        matrix_c[row * n + col] = sum;
    }
}

/*
    BM:M 维度的 block tile 大小 32
    BN:N 维度的 block tile 大小 32
    BK:K 维度每轮处理的长度
    TM:thread 级别的 tile       4
*/
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v3(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN * (BM / TM) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int thread_count = blockDim.x * blockDim.y;

    int col = blockIdx.x * BN + tx;
    int row_base = blockIdx.y * BM + ty * TM;

    float sums[TM];

#pragma unroll
    for (int index = 0; index < TM; ++index) {
        sums[index] = 0.0f;
    }

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        // 协作加载 A 的 BM × BK tile
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            if (global_row < m && global_col < k) {
                tile_a[tile_row][tile_col] = matrix_a[global_row * k + global_col];
            } else {
                tile_a[tile_row][tile_col] = 0.0f;
            }
        }

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            if (global_row < k && global_col < n) {
                tile_b[tile_row][tile_col] = matrix_b[global_row * n + global_col];
            } else {
                tile_b[tile_row][tile_col] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            float value_b = tile_b[inner][tx];
#pragma unroll
            for (int index = 0; index < TM; ++index) {
                float value_a = tile_a[ty * TM + index][inner];
                sums[index] = fmaf(value_a, value_b, sums[index]);
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int index = 0; index < TM; ++index) {
        int row = row_base + index;

        if (row < m && col < n) {
            matrix_c[row * n + col] = sums[index];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int thread_col = threadIdx.x;
    int thread_row = threadIdx.y;

    int tid = thread_row * blockDim.x + thread_col;
    int thread_count = blockDim.x * blockDim.y;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sums[TM][TN] = {};

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            tile_a[tile_row][tile_col] =
                global_row < m && global_col < k ? matrix_a[global_row * k + global_col] : 0.0f;
        } // 搬运a

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            tile_b[tile_row][tile_col] =
                global_row < k && global_col < n ? matrix_b[global_row * n + global_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sums[row][col] = fmaf(tile_a[thread_row * TM + row][inner],
                                          tile_b[inner][thread_col * TN + col], sums[row][col]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sums[row][col];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(const float* matrix_a, const float* matrix_b, float* matrix_c, int m,
                         int n, int k) {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert((BM / TM) * (BN / TN) <= 1024);

    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    int thread_col = threadIdx.x;
    int thread_row = threadIdx.y;
    int tid = thread_row * blockDim.x + thread_col;
    int thread_count = blockDim.x * blockDim.y;

    int row_base = blockIdx.y * BM + thread_row * TM;
    int col_base = blockIdx.x * BN + thread_col * TN;

    float sums[TM][TN] = {};
    float fragment_a[TM];
    float fragment_b[TN];

    int tile_count = (k + BK - 1) / BK;

    for (int tile = 0; tile < tile_count; ++tile) {
        for (int index = tid; index < BM * BK; index += thread_count) {
            int tile_row = index / BK;
            int tile_col = index % BK;

            int global_row = blockIdx.y * BM + tile_row;
            int global_col = tile * BK + tile_col;

            tile_a[tile_row][tile_col] =
                global_row < m && global_col < k ? matrix_a[global_row * k + global_col] : 0.0f;
        }

        for (int index = tid; index < BK * BN; index += thread_count) {
            int tile_row = index / BN;
            int tile_col = index % BN;

            int global_row = tile * BK + tile_row;
            int global_col = blockIdx.x * BN + tile_col;

            tile_b[tile_row][tile_col] =
                global_row < k && global_col < n ? matrix_b[global_row * n + global_col] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
#pragma unroll
            for (int row = 0; row < TM; ++row) {
                fragment_a[row] = tile_a[thread_row * TM + row][inner];
            }
#pragma unroll
            for (int col = 0; col < TN; ++col) {
                fragment_b[col] = tile_b[inner][thread_col * TN + col];
            }
#pragma unroll
            for (int row = 0; row < TM; ++row) {
#pragma unroll
                for (int col = 0; col < TN; ++col) {
                    sums[row][col] = fmaf(fragment_a[row], fragment_b[col], sums[row][col]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int row = 0; row < TM; ++row) {
#pragma unroll
        for (int col = 0; col < TN; ++col) {
            int global_row = row_base + row;
            int global_col = col_base + col;

            if (global_row < m && global_col < n) {
                matrix_c[global_row * n + global_col] = sums[row][col];
            }
        }
    }
}

void sgemm_host(const float* matrix_a, const float* matrix_b, float* matrix_c, int m, int n,
                int k) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int index = 0; index < k; ++index) {
                sum = fmaf(matrix_a[row * k + index], matrix_b[index * n + col], sum);
            }
            matrix_c[row * n + col] = sum;
        }
    }
}

float max_abs_error(const float* actual, const float* expected, int count) {
    float max_error = 0.0f;
    for (int index = 0; index < count; ++index) {
        max_error = fmaxf(max_error, std::fabs(actual[index] - expected[index]));
    }
    return max_error;
}

template <typename KernelLauncher>
float benchmark_kernel_ms(KernelLauncher&& launcher, int warmup, int iterations) {
    for (int index = 0; index < warmup; ++index) {
        launcher();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int index = 0; index < iterations; ++index) {
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
    constexpr int M = 1003;
    constexpr int N = 257;
    constexpr int K = 129;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;

    const int matrix_a_elements = M * K;
    const int matrix_b_elements = K * N;
    const int matrix_c_elements = M * N;
    const size_t matrix_a_bytes = static_cast<size_t>(matrix_a_elements) * sizeof(float);
    const size_t matrix_b_bytes = static_cast<size_t>(matrix_b_elements) * sizeof(float);
    const size_t matrix_c_bytes = static_cast<size_t>(matrix_c_elements) * sizeof(float);

    float* h_matrix_a = static_cast<float*>(std::malloc(matrix_a_bytes));
    float* h_matrix_b = static_cast<float*>(std::malloc(matrix_b_bytes));
    float* h_matrix_c = static_cast<float*>(std::malloc(matrix_c_bytes));
    float* h_reference = static_cast<float*>(std::malloc(matrix_c_bytes));
    if (h_matrix_a == nullptr || h_matrix_b == nullptr || h_matrix_c == nullptr ||
        h_reference == nullptr) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_matrix_a);
        std::free(h_matrix_b);
        std::free(h_matrix_c);
        std::free(h_reference);
        return EXIT_FAILURE;
    }

    for (int index = 0; index < matrix_a_elements; ++index) {
        h_matrix_a[index] = static_cast<float>((index * 17) % 101 - 50) * 0.02f;
    }
    for (int index = 0; index < matrix_b_elements; ++index) {
        h_matrix_b[index] = static_cast<float>((index * 13) % 79 - 39) * 0.015f;
    }
    sgemm_host(h_matrix_a, h_matrix_b, h_reference, M, N, K);

    float* d_matrix_a = nullptr;
    float* d_matrix_b = nullptr;
    float* d_matrix_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_matrix_a, matrix_a_bytes));
    CUDA_CHECK(cudaMalloc(&d_matrix_b, matrix_b_bytes));
    CUDA_CHECK(cudaMalloc(&d_matrix_c, matrix_c_bytes));
    CUDA_CHECK(cudaMemcpy(d_matrix_a, h_matrix_a, matrix_a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_b, h_matrix_b, matrix_b_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matrix_c, 0, matrix_c_bytes));

    dim3 block(BN / TN, BM / TM);
    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    auto launch = [&]() {
        sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(d_matrix_a, d_matrix_b, d_matrix_c, M, N, K);
    };

    launch();
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemcpy(h_matrix_c, d_matrix_c, matrix_c_bytes, cudaMemcpyDeviceToHost));

    float max_error = max_abs_error(h_matrix_c, h_reference, matrix_c_elements);
    float average_ms = benchmark_kernel_ms(launch, WARMUP, ITERATIONS);
    bool pass = max_error < 1e-5f;

    std::printf("M = %d, N = %d, K = %d, BLOCK = %u x %u\n", M, N, K, block.x, block.y);
    std::printf("time(ms) = %.4f\n", average_ms);
    std::printf("max_abs_err = %.8f\n", max_error);
    std::printf("correctness = %s\n", pass ? "pass" : "fail");

    CUDA_CHECK(cudaFree(d_matrix_a));
    CUDA_CHECK(cudaFree(d_matrix_b));
    CUDA_CHECK(cudaFree(d_matrix_c));
    std::free(h_matrix_a);
    std::free(h_matrix_b);
    std::free(h_matrix_c);
    std::free(h_reference);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
