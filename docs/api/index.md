# API reference

The public API, grouped by what it is for and generated from the source docstrings.

These pages are the public API. `scaly.__all__` is the subset re-exported in the `scaly` namespace;
a few of its members (`BACKEND_SUPPORT`, `COMMON_OPS`, `OP_INFO`,
`SCALY_SOLVER_STATS_VERSION`, `factory`, `spec_expr`, `spec_expr_shared`, `C_API_SIGNATURE`) have
no docstring of their own and are not listed here. `sc.sym` and `sc.const` are `Expr.sym` and
`Expr.const`, documented under [Core](core.md).

For prose explanations, start with the [User Guide](../guide/getting_started.md).

| Page | Contains |
| --- | --- |
| [Core](core.md) | `Expr`, `Function`, the type vocabulary, and the expression builders |
| [Building functions](functions.md) | the `@function` decorator, derivative specs, and the named derivative wrappers |
| [Differentiation](ad.md) | forward and reverse mode, whole derivatives, sparsity and coloring |
| [Code generation](codegen.md) | rendering C, the ABI, the toolchain |
| [Integrators](integrators.md) | explicit, implicit, adaptive and symplectic maps of a continuous-time model; shooting, collocation and pseudospectral transcriptions; exact discretization of linear systems; Butcher tableaus |
| [Interpolation](interp.md) | lookup tables and interpolating, smoothing and shape-constrained splines; the tensor-product B-spline they all are, its derivatives, integrals and inverse |
| [Model predictive control](mpc.md) | optimal control problems over a horizon, their controllers and control laws, closed-loop simulation |
| [Solvers](solvers.md) | typed problems, solver selection, quadratic proof, and solve statistics |
| [Visualization](viz.md) | recording a compile and serving it |

Anything not on these pages is internal and may move without notice.
