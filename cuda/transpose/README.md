# 矩阵转置

将 row-major 输入矩阵 `input[rows, cols]` 转置为 row-major 输出矩阵 `output[cols, rows]`。

## 优化路径

```text
v1: 每线程处理一个 (row, col)，直接读输入并写转置输出
v2: shared-memory tile，合并读取输入并合并写回输出
v3: shared-memory padding，消除转置读取的 bank conflict
v4: 32 x 8 block，每线程循环搬运 4 行，降低 block 线程数
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

输入：`rows=1003`，`cols=257`。各版本的 block 配置见表格；有效带宽按一次输入读取和一次输出写回计算。

| 版本 | block | 平均延迟 (ms) | 有效带宽 (GB/s) | 相对 v1 | 最大绝对误差 | 正确性 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v1 | 16 x 16 | 0.0049 | 420.43 | 1.00x | 0.000000 | pass |
| v2 | 32 x 32 | 0.0062 | 334.52 | 0.790x | 0.000000 | pass |
| v3 | 32 x 32 | 0.0041 | 507.38 | 1.195x | 0.000000 | pass |
| v4 | 32 x 8 | 0.0030 | 682.66 | 1.624x | 0.000000 | pass |

结论：v1 的输入读取按行连续，但转置输出写入跨步，因而显著低于普通 elementwise add 的带宽。v2 使用 shared-memory tile，使全局内存读取和写回均为合并访问，但未 padding 的 `tile[32][32]` 在转置读取时产生 bank conflict，且 `32 x 32` block 的线程数较高；因此该 shape 下慢于 v1。v3 仅额外使用 128 B shared memory，将 tile 改为 `tile[32][33]`，消除转置读取的 bank conflict，并在该 shape 上达到 1.195x v1 的速度。v4 保留 padding，并将 block 从 1024 个线程减为 256 个线程；每个线程以 8 为步长搬运 4 个元素，在该 shape 上达到 1.624x v1 的速度。

## 复现

```bash
cd /root/hpf/workspace/cuda/llm-infer-kernels
nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 -I cuda/common \\
    cuda/transpose/transpose.cu -o /tmp/transpose
/tmp/transpose
compute-sanitizer --tool memcheck /tmp/transpose
```
