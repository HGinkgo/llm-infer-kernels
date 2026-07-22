# SGEMM

矩阵乘法的渐进式 CUDA 实现。

## 优化路径

- v1：一个线程计算输出矩阵的一个元素。
- v2：使用 shared memory 对 A、B 进行 block tiling。
- v3：引入一维 thread tile，一个线程计算 `TM x 1` 个输出。
- v4：引入二维 thread tile，一个线程计算 `TM x TN` 个输出。
- v5：使用寄存器缓存从 shared memory 读取的 A、B fragment。

## 性能结果

测试环境为 RTX 3090、`M=1003`、`N=257`、`K=129`，每个版本均通过 CPU reference
正确性检查。

| 版本 | 时间（ms） | 说明 |
| --- | ---: | --- |
| v1 | 0.0373 | 每线程计算一个输出 |
| v2 | 0.0467 | shared memory 分块，小尺寸下同步开销较明显 |
| v3 | 0.0227 | `TM x 1` thread tile |
| v4 | 0.0168 | `TM x TN` thread tile |
| v5 | 0.0194 | 显式寄存器 fragment，当前参数下略慢于 v4 |

以上结果仅用于比较当前 shape 下的相对变化；后续需要增加大尺寸测试、GFLOPS、cuBLAS
baseline 和 Nsight Compute 分析。

## 构建

以下命令从仓库根目录执行：

```bash
cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/cuda --target sgemm -j
./build/cuda/bin/sgemm
```
