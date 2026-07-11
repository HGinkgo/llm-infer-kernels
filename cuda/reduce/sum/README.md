# 求和归约

对一维 `float` 数组做 block 级部分和归约。GPU kernel 输出每个 block 的部分和，CPU 用于汇总部分和并校验正确性。

## 优化路径

```text
v1: 每线程一个元素，shared memory 折半归约
v2: 每线程两个元素，减少 block 数量
v3: 最后一个 warp 改用 shuffle 归约，减少同步
v4: 每线程四个元素，结合 warp shuffle 与尾部处理
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

输入：`N=16,777,339`，`BLOCK_SIZE=256`。

| 版本 | 平均延迟 (ms) | 有效带宽 (GB/s) | 相对 v1 | 正确性 |
| --- | ---: | ---: | ---: | --- |
| v1 | 0.2811 | 238.72 | 1.00x | pass |
| v2 | 0.1369 | 490.14 | 2.05x | pass |
| v3 | 0.0925 | 725.68 | 3.04x | pass |
| v4 | 0.0868 | 772.84 | 3.24x | pass |

结论：v4 在该输入上最快。每线程多元素降低了归约 block 数，warp shuffle 进一步消除了最后阶段的 shared memory 同步。

## 复现

```bash
cd /root/hpf/workspace/cuda/llm-infer-kernels/cuda/reduce
cmake -S . -B cmake-build-current -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-current -j
./cmake-build-current/bin/sum_reduce_sum
```
