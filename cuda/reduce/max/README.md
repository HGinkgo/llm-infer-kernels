# Max Reduce

Maximum-value reduction.

Recommended source file:

```text
reduce_max.cu
```

Use this after `sum`, before `softmax`. Stable softmax needs a max pass:

```text
max_x = max(x)
exp_x = exp(x - max_x)
sum_exp = sum(exp_x)
y = exp_x / sum_exp
```

Build target name:

```text
max_reduce_max
```
