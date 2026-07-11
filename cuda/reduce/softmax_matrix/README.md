# 矩阵 Softmax

对 `M x N` 的 `float` 矩阵按行计算 softmax。每一行独立完成 max、exp-sum 和归一化，是 attention score softmax 的基础形式。

## 优化路径

```text
v1: 一个 block 处理一行，覆盖 N <= 1024
v2: 一个 warp 处理一行；一个 block 同时处理八行，覆盖 N <= 128
v3: 按行宽在 v1 与 v2 之间分发
v4: 在 N 为 4 的倍数且指针 16-byte 对齐时使用 float4 读写；
    其余 shape 回退到 v2 或 v1
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

所有测试使用 `M=127`，通过 v4 dispatcher 发射。有效带宽按一次输入读取和一次输出写回计算。

| Shape | 分发路径 | 平均延迟 (ms) | 有效带宽 (GB/s) | 最大绝对误差 | 最大行和误差 | 正确性 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 127 x 127 | v2 warp-per-row | 0.0025 | 50.81 | 4.8e-7 | 9.5e-7 | pass |
| 127 x 128 | v2 warp-per-row | 0.0025 | 51.42 | 4.8e-7 | 8.3e-7 | pass |
| 127 x 256 | v4 float4 | 0.0027 | 96.95 | 7.2e-7 | 1.79e-6 | pass |
| 127 x 1024 | v4 float4 | 0.0032 | 327.74 | 5.1e-7 | 6.56e-6 | pass |
| 127 x 1003 | v1 标量回退 | 0.0032 | 321.02 | 6.3e-7 | 7.27e-6 | pass |

结论：短行由 v2 避免 block 级同步和闲置线程；满足对齐条件的长行可走 v4 的 float4 快路径；非 4 对齐的 `N=1003` 保持标量 v1 回退并通过正确性校验。

## 复现

```bash
cd /root/hpf/workspace/cuda/llm-infer-kernels/cuda/reduce
cmake -S . -B cmake-build-current -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-current -j
./cmake-build-current/bin/softmax_matrix_softmax_matrix
compute-sanitizer --tool memcheck ./cmake-build-current/bin/softmax_matrix_softmax_matrix
```
