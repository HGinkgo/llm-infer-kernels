#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

inline void cuda_check(cudaError_t error, const char* expr, const char* file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s failed with %s\n", file, line, expr,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK()                                                                        \
    do {                                                                                           \
        CUDA_CHECK(cudaGetLastError());                                                            \
        CUDA_CHECK(cudaDeviceSynchronize());                                                       \
    } while (0)
