# llm-infer-kernels

CUDA and Triton operator kernels for LLM inference experiments.

## Layout

- `cuda/`: CUDA kernel implementations grouped by operator.
- `triton/`: Triton implementations grouped by operator.

## Organization

This repository follows an operator-first layout similar to small CUDA learning
repos:

```text
cuda/
├── common/
├── reduce/
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── sum/
│   │   ├── README.md
│   │   └── reduce_sum.cu
│   ├── max/
│   │   ├── README.md
│   │   └── reduce_max.cu
│   ├── softmax/
│   │   ├── README.md
│   │   └── softmax.cu
│   └── softmax_matrix/
│       ├── README.md
│       └── softmax_matrix.cu
├── rmsnorm/
├── rope/
├── gemv/
├── sgemm/
└── flash_attention/
```

Each operator keeps its own:

- implementation versions
- `main()` for correctness and local benchmarking
- README notes for optimization records

There is no shared top-level `benchmarks/` or `docs/` directory. Benchmark logic
stays next to each operator.

## Requirements

- CUDA Toolkit with `nvcc`
- CMake 3.24 or newer
- NVIDIA GPU supported by the selected CUDA architecture

The CUDA CMake files currently default to `CMAKE_CUDA_ARCHITECTURES=86`, which
matches RTX 3090. Change that value when building for a different GPU.

## Build

Each CUDA operator directory owns its local build entry. For example:

```bash
cd cuda/reduce
./build.sh
./build/bin/sum_reduce_sum
```

## Code Style

- `.clang-format` defines the C/C++/CUDA formatting rules.
- `.editorconfig` defines basic editor behavior such as 4-space indentation,
  LF line endings, and final newlines.
- `.vscode/settings.json` only keeps repository-generic editor settings and
  intentionally avoids machine-specific absolute paths.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
