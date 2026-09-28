# API reference

Use this reference to look up signatures, arguments, return values, and errors. If you are building
your first model, start with [Getting started](../guide/getting_started.md). The reference assumes
that you already know which operation you need.

Most modelling code uses `import scaly as sc`. For example, `sc.sym` creates an unknown quantity,
`sc.function` defines a function, and `sc.gradient` constructs its derivative. The reference shows
the module where each object is defined. `sc.sym` and `sc.const` are the same methods as
`Expr.sym` and `Expr.const`.

| Page | What to look up |
| --- | --- |
| [Core](core.md) | Symbolic expressions, array operations, `Function`, and data types |
| [Building functions](functions.md) | Input and output declarations, the decorator, and derivatives of named functions |
| [Differentiation](ad.md) | Derivatives of expressions and the lower-level sparsity algorithms |
| [Code generation](codegen.md) | Writing C files, the generated calling convention, compilation, and caching |
| [Solvers](solvers.md) | Optimization problems, solver selection, solve statistics, and plugin interfaces |
| [Visualization](viz.md) | Recording what the compiler does to a function and browsing the recordings |

These pages are generated from source docstrings. They also include lower-level interfaces for
compiler extensions and integrations, which most models do not need. Names outside these pages
are internal and may move without notice.

`scaly.__all__` lists the names available directly under `scaly`. A few constants and module
aliases have no separate generated entry: `COMMON_OPS`, `OP_INFO`,
`SCALY_SOLVER_STATS_VERSION`, `factory`, `spec_expr`, `spec_expr_shared`, and `C_API_SIGNATURE`.
