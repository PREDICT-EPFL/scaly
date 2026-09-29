# API reference

The public API, grouped by what it is for and generated from the source docstrings.

These pages are the public API. `scaly.__all__` is the subset re-exported in the `scaly` namespace;
a few of its members (`BACKEND_SUPPORT`, `COMMON_OPS`,
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
| [Export](export.md) | the C++ header, the CasADi layer, the acados drop-in |
| [Nonlinear equations](roots.md) | roots and least squares over typed unknowns, the Newton family that solves them, implicit derivatives of a root |
| [Integrators](integrators.md) | explicit, implicit, adaptive and symplectic maps of a continuous-time model; shooting, collocation and pseudospectral transcriptions; exact discretization of linear systems; Butcher tableaus |
| [Interpolation](interp.md) | lookup tables and interpolating, smoothing and shape-constrained splines; the tensor-product B-spline they all are, its derivatives, integrals and inverse |
| [Optimal control](ocp.md) | continuous and discrete OCPs, their transcriptions and their formulation as optimization problems, the methods that solve them, warm starts, terminal ingredients |
| [Sets](sets.md) | polytopes and ellipsoids, their linear programs, and the constraints that keep a point inside |
| [Neural networks](nn.md) | multilayer perceptrons, their activations and weight layouts, PyTorch checkpoints without torch (experimental) |
| [Geometry](geometry.md) | quaternions and SO(3), three-vectors, manifolds with a retraction and local coordinates (experimental) |
| [Solvers](solvers.md) | typed problems, solver selection, quadratic proof, and solve statistics |
| [Visualization](viz.md) | recording a compile and serving it |

Anything not on these pages is internal and may move without notice.
