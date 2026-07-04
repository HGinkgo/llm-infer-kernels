# Matrix Softmax

Row-wise softmax for `M x N` tensors.

Recommended source file:

```text
softmax_matrix.cu
```

This is the version closest to attention workloads, where each row often
corresponds to one query position over keys.

Suggested versions:

```text
v0: one block per row
v1: warp-level reduction for short rows
v2: block-level reduction for longer rows
v3: vectorized load/store where alignment allows it
```

Build target name:

```text
softmax_matrix_softmax_matrix
```
