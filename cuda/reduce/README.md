# Reduce

CUDA 归约算子及其在 softmax 中的应用。

## 学习顺序

```text
sum -> max -> softmax -> softmax_matrix
```

```text
reduce/
├── sum/             # 一维求和归约
├── max/             # 一维最大值归约
├── softmax/         # 单个向量的 softmax
├── softmax_matrix/  # M x N 矩阵逐行 softmax
└── include/         # reduce 专用配置
```

## 构建

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build/cuda -j
```

CMake 会为包含 `main()` 的 `.cu` 文件生成独立可执行程序。例如：

```bash
./build/cuda/bin/sum_reduce_sum
./build/cuda/bin/max_reduce_max
./build/cuda/bin/softmax_softmax
./build/cuda/bin/softmax_matrix_softmax_matrix
```

RTX 3090 默认使用 `sm_86`。可以在 CMake 配置阶段通过
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` 指定其他架构。

## Benchmark

各算子 README 记录以下指标：

```text
time(ms) | bandwidth(GB/s) | speedup | correctness
```

不同版本的 speedup 均以同一可执行程序中的基础版本为基准。
