# 一维 Softmax

对单个 `float` 向量做数值稳定 softmax：先归约最大值，再计算指数和，最后归一化。

## 优化路径

```text
v1: 一个 block 处理一个向量，shared memory 折半归约
v2: 每线程处理多个元素；max/sum 保留在寄存器；
    warp shuffle 与少量 shared memory 完成 block 归约
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

输入：`N=1003`，`BLOCK_SIZE=256`，`ITEMS_PER_THREAD=4`。

| 版本 | 平均延迟 (ms) | 有效带宽 (GB/s) | 最大绝对误差 | 输出和 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| v2 | 0.0027 | 2.93 | 1.8e-7 | 1.00000334 | pass |

结论：v2 用寄存器保存每线程四个元素的中间值，避免将逐元素指数结果写入 shared memory；block 归约只交换各 warp 的局部结果。

## 复现

```bash
cd /root/hpf/workspace/cuda/llm-infer-kernels/cuda/reduce
cmake -S . -B cmake-build-current -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-current -j
./cmake-build-current/bin/softmax_softmax
```
