# Reduce

CUDA reduction kernels and experiments.

## Roadmap

Recommended order:

```text
sum -> max -> softmax -> softmax_matrix
```

Directory layout:

```text
reduce/
├── sum/             # scalar/vector sum reduction
├── max/             # max reduction, prerequisite for numerically stable softmax
├── softmax/         # 1D softmax for one row/vector
├── softmax_matrix/  # row-wise softmax for M x N tensors
├── include/         # reduce-specific shared config
├── CMakeLists.txt
└── build.sh
```

## Build

This directory uses CMake. Every `.cu` file directly under `reduce/` or one of
its first-level subdirectories is compiled as an executable after it contains
`int main(...)`. Empty placeholder files are skipped.

```bash
cd /root/hpf/workspace/ai_infra/project/llm-infer-kernels/cuda/reduce
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

For RTX 3090, the default CUDA architecture is `sm_86`.

## Source Convention

Create your first source file here:

```text
sum/reduce_sum.cu
```

Then include the shared helpers if needed:

```cpp
#include "cuda_utils.cuh"
#include "utils.cuh"
```

Suggested implementation order:

```text
reduce_sum_v0: shared memory, one partial sum per block
reduce_sum_v1: dynamic shared memory
reduce_sum_v2: atomicAdd final accumulation
reduce_sum_v3: warp shuffle
reduce_sum_v4: float4 vectorized load
```

Profile example:

```bash
ncu --set full ./build/bin/sum_reduce_sum
```

## Benchmark

For each version, keep these fields:

```text
time(ms) | bandwidth(GB/s) | speedup | correctness
```

Speedup is measured against the baseline version in the same executable.
