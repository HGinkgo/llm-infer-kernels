# RMSNorm

逐行 RMSNorm 的渐进式 CUDA 实现。

## 优化路径

- v1：使用共享内存完成 block 归约。
- v2：使用 warp shuffle，减少共享内存访问与同步。
- v3：使用 `float4` 向量化读写。
- v4：一个 warp 处理一行，去掉跨 warp 归约和共享内存中转。

## 构建

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda -j
./build/cuda/bin/rmsnorm
```
