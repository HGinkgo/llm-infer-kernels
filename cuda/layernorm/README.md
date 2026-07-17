# LayerNorm

逐行 LayerNorm 的渐进式 CUDA 实现。

## 优化路径

- v1：一个线程串行处理一行。
- v2：一个 block 处理一行，使用 warp shuffle 完成 block 归约。
- v3：在 v2 基础上使用 `float4` 向量化读写。
- v4：一个 warp 处理一行，去掉跨 warp 归约与共享内存中转。

v3 和 v4 要求 `cols` 是 4 的倍数；其他 shape 使用 v2。

## 构建

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda --target layernorm -j
./build/cuda/bin/layernorm
```
