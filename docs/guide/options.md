# Options

Options set conventions that change the graphs Scaly builds. Today there is one, `nonsmooth`, the
derivative of `maximum`, `minimum`, `max` and `min` where arguments tie.

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
`abs'(0) = 0` under every setting.

An option is read when a graph is built, not when it is compiled. A derivative built inside a
`with sc.options(...)` block keeps its convention after the block ends, because the convention is
part of that graph, and a graph built under a different setting generates different C and gets its
own JIT cache entry. `sc.options` changes the setting for the current thread or task only;
`sc.set_options` changes the default outside any block. An unknown option name or value raises at
the call.
