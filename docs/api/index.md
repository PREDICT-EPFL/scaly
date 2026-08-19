# API reference

The public surface, grouped by what it is for and generated from the source, so it follows the
code.

A few names in the `alloy` namespace are aliases or constants that carry no documentation of their
own and so do not appear below: `al.sym` and `al.const` are `Expr.sym` and `Expr.const`, `al.scan`
is `al.map_`, and `al.C_API_SIGNATURE` is the ABI signature string. `alloy.__all__` is the
authoritative list of what is public.

For prose explanations rather than signatures, start with the [User Guide](../guide/getting_started.md).

| Page | Contains |
| --- | --- |
| [Core](core.md) | `Expr`, `Function`, the type vocabulary, and the expression builders |
| [Building functions](functions.md) | the `@function` decorator, derivative specs, and the named derivative wrappers |
| [Differentiation](ad.md) | forward and reverse mode, whole derivatives, sparsity and coloring |
| [Code generation](codegen.md) | rendering C, the ABI surface, the toolchain |
| [Solvers](solvers.md) | `qp`, `nlp`, `SolverFunction` and solve statistics |
| [Visualization](viz.md) | recording a compile and serving it |

Anything not on these pages is internal, and may move without notice.
