# API findings from the documentation rewrite

These findings came from writing and executing examples for new users. They record the behavior
observed during that review. The solver-call and `vmap` changes are now implemented: solvers accept
parameters with optional initial guesses, and `vmap`
infers contiguous slices or broadcasting from input sizes.

## Python truth tests silently accept symbolic expressions

Follow-up: [#61].

`Expr` has no `__bool__` implementation, so Python treats an expression object as true. This can
record the wrong model without an error:

```python
import numpy as np
import scaly as sc

@sc.function(sc.L("x", ()), sc.L("y", ...))
def branch(x):
    return x if x else sc.const(10.0)

print(branch(np.array(0.0)))  # 0.0, although the intended false branch returns 10
```

The branch is chosen while the decorator traces the body. Numerical input values never revisit
that choice. Symbolic comparisons such as `x > 0` currently raise `TypeError`, and there is no
symbolic `where` operation. The silent truth test is a separate problem from the absence of
value-dependent control flow.

Evidence: the [`Expr` class](../../src/scaly/ir/expr.py) and the executed example above.
The functions guide now explains this limitation.

## Common clipping expressions cannot be differentiated

Follow-up: [#25] and
[#30].

`sc.minimum`, `sc.maximum`, `.floor()`, and `.ceil()` can appear in an evaluated model, but
requesting their gradient or Jacobian raises `NotImplementedError`. For example:

```python
x = sc.sym("x", 2)
sc.gradient(sc.maximum(x, 0.0).sum(), x)
```

This matters for models with saturation or piecewise costs. A reader familiar with numerical
libraries may expect an active-branch derivative away from a kink. The current implementation
has no such rule, even though the expression type marks these operations as non-differentiable.
The previous documentation incorrectly described zero derivatives for these operations.

Evidence: [`ExprOp` metadata](../../src/scaly/ir/expr.py),
[forward differentiation](../../src/scaly/ad/forward.py), and
[reverse differentiation](../../src/scaly/ad/reverse.py). Both derivative modes were exercised
for all four operations. The public derivative and IR pages now describe the errors.

## Array reductions have no axis argument

Follow-up: [#66].

`Expr.sum()` reduces the entire array to a scalar. `x.sum(axis=0)` raises `TypeError`, including
for a two-dimensional input. There is no `keepdims` argument either.

A model that represents stages or batches along an array axis therefore needs explicit slicing
and assembly to express ordinary row or column reductions. This is an API coverage gap relative
to the NumPy-like arithmetic used elsewhere in the library.

Evidence: [`Expr.sum`](../../src/scaly/ir/expr.py). The axis call was executed and the error
confirmed. The operation reference states the current whole-array behavior.

## PIQP option errors reach C compilation

Resolved by [#80].

PIQP option names are checked during Python solver construction, before any C generation,
including the sparse matrix-pattern probe. An unknown name raises `ValueError` identifying
all supported PIQP 0.6.4 settings. The plugin's accepted names match the vendored
`piqp_settings` struct. Scaly's `sparse` option is handled separately.
Numeric and Boolean values still become settings-struct assignments, and string-valued
settings still raise during rendering. [#112] will carry the name check into runtime options.

Evidence: `scaly_piqp._Backend.validate_options`, called at the start of
[`build_qp`](../../src/scaly/solvers/qp.py), and the regression tests in
[`test_qp_piqp.py`](../../plugins/scaly-piqp/tests/test_qp_piqp.py).
The solver-backend guide describes the construction-time error.

## Solver sensitivity has inconsistent failure behavior

Follow-up: [#16] and
[#61].

Differentiation through a solver is already an unsupported feature. The documentation review
also found that different ways of asking for it fail differently.

For the problem `minimize ||x - p||²`, a function wrapping the solve returns `x = p`. Its
Jacobian with respect to `p` evaluates to zero, while a central finite difference gives one.
Requesting `sc.jacobian(solve.function, "x", "p")` on the underlying function instead raises
`ValueError: replacement input tree does not match the Function graph`.

The zero result can be mistaken for a real sensitivity, and the direct-call error does not
explain the unsupported operation. Neither behavior supplies the derivative of the optimizer's
solution. The public guide now makes that limitation explicit.

Evidence: the `SOLVER_CALL` rules in
[forward](../../src/scaly/ad/forward.py) and [reverse](../../src/scaly/ad/reverse.py)
differentiation, [`_unseeded`](../../src/scaly/function/api.py), and
[`Function._with_trees`](../../src/scaly/function/model.py). Both the nested and direct requests
were reproduced using IPOPT.

[#61]: https://github.com/PREDICT-EPFL/scaly/issues/61
[#25]: https://github.com/PREDICT-EPFL/scaly/issues/25
[#30]: https://github.com/PREDICT-EPFL/scaly/issues/30
[#66]: https://github.com/PREDICT-EPFL/scaly/issues/66
[#80]: https://github.com/PREDICT-EPFL/scaly/issues/80
[#16]: https://github.com/PREDICT-EPFL/scaly/issues/16

[#112]: https://github.com/PREDICT-EPFL/scaly/issues/112
