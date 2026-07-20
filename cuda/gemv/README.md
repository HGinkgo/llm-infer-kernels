# GEMV

矩阵向量乘法的渐进式 CUDA 实现。

## 优化路径

- v1：一个线程串行计算矩阵的一行点积。

## 构建

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda --target gemv -j
./build/cuda/bin/gemv
```
