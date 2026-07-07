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
