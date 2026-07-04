# Sum Reduce

First CUDA reduce operator.

Recommended source file:

```text
reduce_sum.cu
```

Suggested versions:

```text
v0: shared memory, each block writes one partial sum
v1: dynamic shared memory
v2: atomicAdd final accumulation
v3: warp shuffle block reduction
v4: float4 vectorized load with tail handling
```

Build target name:

```text
sum_reduce_sum
```
