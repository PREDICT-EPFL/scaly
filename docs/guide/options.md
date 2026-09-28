# Options

Options set conventions that change the graphs Scaly builds: the derivative of nonsmooth operations
at ties, when linear algebra becomes straight-line code instead of loops, and how much reverse mode
may store for one loop.

```python
import scaly as sc

x = sc.sym("x", 3)
cost = sc.maximum(x, 0.0).sum()

with sc.options(nonsmooth="first"):
    grad = sc.gradient(cost, x)  # built under "first"

sc.set_options(nonsmooth="error")  # the process default, seen by every thread
print(sc.get_options().nonsmooth)
```

| `nonsmooth` | At a tie |
| --- | --- |
| `"split"` (default) | the derivative is shared equally among the tied arguments |
| `"first"` | the first tied argument takes all of it: the left operand of `maximum`, the lowest index of `max` |
| `"error"` | differentiating any of these operations raises `NotImplementedError` |

Away from ties every convention gives the same derivative. `abs` is differentiable with
`abs'(0) = 0` under every setting. `floor` and `ceil` have a zero derivative, exact everywhere but
at their jumps, under `"split"` and `"first"`, and `"error"` refuses them like the others.

| Option | Default | Meaning |
| --- | --- | --- |
| `dense_unroll` | 8 | the largest order at which `linalg.cholesky`, `ldl` and `solve_triangular` become straight-line code instead of loops |
| `sparse_unroll` | 1000 | the most multiply-adds and divisions a `linalg.SparseLDL` factorization may take and still become straight-line code (`schedule="auto"`) |
| `max_trajectory` | 50 000 000 | the most values reverse mode may store for the carries of one `scan` or `while_loop` |

The two unroll thresholds trade generation time for speed: straight-line code runs several times
faster than a loop at these sizes, but costs about a millisecond of generation per operation. A
reverse pass over a loop stores its carry at every step; building one that would exceed
`max_trajectory` values raises `ValueError` and names the loop. The usual fix is an implicit
derivative ([Custom derivatives](derivatives.md#custom-derivatives)), not a larger limit. These
three options take non-negative integers.

An option is read when a graph is built, not when it is compiled. A derivative built inside a
`with sc.options(...)` block keeps its convention after the block ends, because the convention is
part of that graph, and a graph built under a different setting generates different C and gets its
own JIT cache entry. `sc.options` changes the setting for the current thread or task only;
`sc.set_options` changes the default outside any block. An unknown option name or value raises at
the call.
