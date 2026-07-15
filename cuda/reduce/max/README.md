# 最大值归约

对一维 `float` 数组做 block 级部分最大值归约。该算子是数值稳定 softmax 的前置步骤。

## 优化路径

```text
v4: 每线程加载四个元素；shared memory 完成跨 warp 前归约；
    最后一个 warp 使用 shuffle 完成寄存器归约。
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

输入：`N=16,777,339`，`BLOCK_SIZE=256`。

| 版本 | 平均延迟 (ms) | 有效带宽 (GB/s) | 绝对误差 | 正确性 |
| --- | ---: | ---: | ---: | --- |
| v4 | 0.0941 | 713.05 | 0.000000 | pass |

结论：四元素寄存器局部归约结合 warp shuffle，在大数组上避免了最后一个 warp 的反复 block 同步。当前计时仅覆盖 GPU 的部分最大值 kernel；CPU 汇总部分结果只用于正确性校验。

## 复现

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda -j
./build/cuda/bin/max_reduce_max
```
