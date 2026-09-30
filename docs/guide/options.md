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
| `max_trajectory` | 50 000 000 | the most values reverse mode may store for the carries of one `scan` or `while_loop` |
| `linalg=dict(dense_unroll=...)` | None | the largest order at which `linalg.cholesky`, `ldl`, `lu` and `solve_triangular` become straight-line code instead of loops; None leaves the choice to the target the graph is rendered for |
| `linalg=dict(sparse_unroll=...)` | 1000 | the most multiply-adds and divisions a `linalg.SparseLDL` factorization may take and still become straight-line code (`schedule="auto"`) |

A reverse pass over a loop stores its carry at every step; building one that would exceed
`max_trajectory` values raises `ValueError` and names the loop. The usual fix is an implicit
derivative ([Custom derivatives](derivatives.md#custom-derivatives)), not a larger limit. The two
unroll thresholds trade generation time for speed: straight-line code runs several times faster
than a loop at these sizes, but costs about a millisecond of generation per operation. These three
options take non-negative integers, and `dense_unroll` also None. Without it, a dense factorization
or triangular solve is straight-line code while its body, counted in operations (`n^3 / 3` for a
Cholesky or `L D L^T` factor, `5 n^3 / 3` for `lu`, `n^2` per right-hand side of a solve), is under the target's
`Target.straight_line_ops`, and loops past it ([Code generation](codegen.md#tuning-for-a-processor)).

The unroll thresholds belong to the `linalg` namespace: `sc.options(linalg=dict(dense_unroll=4))`.
A package declares its own namespace with `scaly.utils.options.register_option_namespace` and says
whether its options can change a derivative. `scaly.linalg` declares `linalg` when it is imported,
and a namespace named after a package of `scaly` loads that package the first time `sc.options`
sees it, so the block can come before anything else touches `sc.linalg`. The `linalg` ones cannot: a derivative takes each
factorization's choice from the node it differentiates, so a gradient built under
`dense_unroll=0` is the default one, node for node.

An option is read when a graph is built, not when it is compiled. A derivative built inside a
`with sc.options(...)` block keeps its convention after the block ends, because the convention is
part of that graph, and a graph built under a different setting generates different C and gets its
own JIT cache entry. The derivative of a call, a map or a loop is built once per callee and per
setting of the options a derivative depends on (`nonsmooth`, `max_trajectory` and namespaces that
declare it), so one Function differentiated under two such settings gets two derivative bodies,
named apart, which can sit in one graph. `sc.options` changes the setting for the current thread or task only;
`sc.set_options` changes the default outside any block. An unknown option name or value raises at
the call.
