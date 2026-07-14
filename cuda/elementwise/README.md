# 逐元素算子

逐元素算子的每个输出只依赖同一位置的一个或多个输入元素，不需要跨线程归约。本目录从 `add` 开始。

## Add

```text
v1: 每线程处理一个元素，使用一维 grid/block 索引与尾部边界保护
v2: 每线程处理四个元素，一个 block 覆盖四倍输入数据
v3: float4 向量化访问完整四元素块，标量路径处理尾部
```

## 性能记录

测试环境：RTX 3090（driver 580.95.05）、CUDA 13.0、Release、`sm_86`。计时使用 CUDA event，预热 10 次、测量 100 次。

输入：`N=16,777,219`，`BLOCK_SIZE=256`。有效带宽按两次输入读取和一次输出写回计算。

| 版本 | 平均延迟 (ms) | 有效带宽 (GB/s) | 相对 v1 | 最大绝对误差 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 0.2394 | 840.85 | 1.00x | 0.000000 | pass |
| v2 | 0.2405 | 836.95 | 0.995x | 0.000000 | pass |
| v3 | 0.2389 | 842.77 | 1.002x | 0.000000 | pass |

结论：v1 的全局线程索引对应一个输出元素，尾部的 3 个元素由边界判断覆盖。v2 使用 `ITEMS_PER_THREAD=4`，将 grid 缩减为 `16,385` 个 block，同时在每一轮保持 warp 合并访问。v3 以 `float4` 读写完整四元素块，并用标量路径处理 3 个尾部元素；它要求输入和输出指针 16-byte 对齐。三个版本在该 shape 上性能接近，0.2% 级别差异需要多轮重复测量或 Nsight Compute 才能视为有效结论。

## 复现

```bash
cd /root/hpf/workspace/cuda/llm-infer-kernels/cuda/elementwise
cmake -S . -B cmake-build-current -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build cmake-build-current -j
./cmake-build-current/bin/elementwise_add
```
