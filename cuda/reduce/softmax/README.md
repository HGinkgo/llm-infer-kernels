# Softmax

One-dimensional softmax for a single vector/row.

Recommended source file:

```text
softmax.cu
```

Suggested versions:

```text
v0: separate max, exp+sum, normalize kernels
v1: one block handles one row/vector when N is small enough
v2: warp-level reduction for short rows
```

Build target name:

```text
softmax_softmax
```
