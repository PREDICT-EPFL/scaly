# Linear algebra

`scaly.linalg` solves dense linear systems inside a function: triangular solves, the Cholesky
factorization and `solve` for symmetric positive definite matrices. They are ordinary Scaly
functions built from [loops](control_flow.md), so the generated C contains the factorization
itself and calls no BLAS or LAPACK library. All of them are differentiable.

## Solving a positive definite system

`solve(K, b)` returns the solution of \(K x = b\) together with a status, here for a
regularized least-squares system \(K = A^T A + \rho I\):

```python
import numpy as np
import scaly as sc
from scaly import linalg

status_type = sc.TensorType((), sc.dtypes.int64)

@sc.function(sc.arg("A", (3, 2)), sc.arg("y", 3),
             outputs=sc.group(sc.arg("x", 2), sc.arg("status", status_type)))
def least_squares(A: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    K = A.T @ A + 1e-3 * sc.const(np.eye(2))
    return linalg.solve(K, A.T @ y)

A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
x, status = least_squares(A, np.array([1.0, 2.0, 4.0]))
print(x, status)
# [0.83338868 1.49966686] 0
```

`assume="pos"`, the default, states that `K` is symmetric positive
definite. `solve` factors it as \(K = L L^T\) and solves the two triangular systems. `b` is a
vector of shape `(n,)` or a matrix of shape `(n, k)` whose columns are separate right-hand sides,
and `x` has the same shape as `b`.

`solve`, `cholesky` and `solve_triangular` are Python functions, not `Function`s. They are
called inside a function body on expressions and add calls of the library functions to its graph.

## Only the lower triangle is read

A symmetric matrix stores each off-diagonal value twice. `solve` and `cholesky` read only the
lower triangle, including the diagonal, and never look at the entries above it. An entry below
the diagonal stands for itself and its mirror above. A matrix whose upper triangle holds
anything at all gives the same result as its symmetric completion:

```python
@sc.function(sc.arg("K", (2, 2)), sc.arg("b", 2), outputs=sc.arg("x", 2))
def solve_only(K: sc.Expr, b: sc.Expr) -> sc.Expr:
    return linalg.solve(K, b)[0]

print(solve_only(np.array([[4.0, 99.0], [1.0, 3.0]]), np.array([1.0, 2.0])))
# [0.09090909 0.63636364]
```

This is the solution for \(K = \begin{bmatrix}4 & 1\\ 1 & 3\end{bmatrix}\). The same convention
fixes the derivative with respect to `K`, as [below](#derivatives) explains.

## The status of a factorization

Elimination divides by pivots, the diagonal values left after the earlier columns have been
eliminated. For a positive definite matrix every pivot is positive. A pivot is unusable when it
is not greater than machine epsilon times the diagonal entry of `K` it came from. That includes
zero, negative and NaN pivots, and pivots so small relative to their diagonal entry that the
cancellation behind them has left no accurate digits.

The status is 0 when every pivot is usable. Otherwise it is \(j + 1\) for the first unusable
pivot \(j\), counted from zero, which is how LAPACK numbers them. The factorization does not stop
there, so later entries of the factor and the whole solution are meaningless, but they cannot be
mistaken for a valid result as long as the status is checked:

```python
@sc.function(sc.arg("K", (3, 3)), outputs=sc.arg("status", status_type))
def pivot_status(K: sc.Expr) -> sc.Expr:
    return linalg.cholesky(K)[1]

print(pivot_status(np.diag([1.0, 2.0, 3.0])))  # 0
print(pivot_status(np.diag([1.0, -2.0, 3.0]))) # 2
print(pivot_status(np.ones((3, 3))))           # 2
```

The last matrix is singular: its second pivot is \(1 - 1 = 0\). `solve` returns the status of
its factorization, so a solver built on it can reject a step, add regularization or stop.

## Factors and triangular solves

`cholesky(A)` returns `(L, status)`, with `L` lower triangular and zero above the diagonal.
`solve_triangular(T, b)` solves \(T x = b\) for a lower triangular `T`. `lower=False` reads the
upper triangle instead, `trans=True` solves \(T^T x = b\), and `unit_diagonal=True` takes the
diagonal as ones without reading it. Like `solve`, it accepts one or several right-hand sides.

A factor computed once serves any number of solves. Two triangular solves with the factor give
\(A^{-1} b\):

```python
@sc.function(sc.arg("A", (2, 2)), sc.arg("b", 2), outputs=sc.arg("x", 2))
def factor_and_solve(A: sc.Expr, b: sc.Expr) -> sc.Expr:
    L, status = linalg.cholesky(A)
    y = linalg.solve_triangular(L, b)
    return linalg.solve_triangular(L, y, trans=True)

print(factor_and_solve(np.array([[4.0, 0.0], [1.0, 3.0]]), np.array([1.0, 2.0])))
# [0.09090909 0.63636364]
```

## Stacks of matrices

Every function accepts a stack of matrices over leading axes, such as one block per stage of a
horizon. A `K` of shape `(N, n, n)` takes a right-hand side of shape `(N, n)` or `(N, n, k)` and
returns a status of shape `(N,)`, one per matrix. The matrices are solved independently, as a
[mapped call](functions.md#regular-repetition-vmap), so the generated code does not grow with `N`:

```python
@sc.function(sc.arg("K", (4, 2, 2)), sc.arg("b", (4, 2)),
             outputs=sc.group(sc.arg("x", (4, 2)), sc.arg("status", sc.TensorType((4,), sc.dtypes.int64))))
def stage_solves(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return linalg.solve(K, b)

K = np.stack([np.eye(2) * s for s in (1.0, 2.0, -1.0, 4.0)])
x, status = stage_solves(K, np.ones((4, 2)))
print(status)
# [0 0 1 0]
```

## Derivatives

The solution is differentiable in `K` and `b`. Reverse mode, which computes gradients, uses
the implicit derivative of \(K x = b\):

\[
\bar b = K^{-1} \bar x, \qquad \bar K = -\bar b\, x^T.
\]

It reuses the factor of the solve instead of factoring `K` again or differentiating the steps of
the factorization. A function that returns a value and its gradient factors `K` once.

Because only the lower triangle is read, an entry below the diagonal receives the derivatives of
both entries it stands for, \(\bar K_{ij} + \bar K_{ji}\), and the entries above the diagonal
receive zero:

```python
@sc.function(sc.arg("K", (2, 2)), sc.arg("b", 2), outputs=sc.arg("cost", ()))
def cost(K: sc.Expr, b: sc.Expr) -> sc.Expr:
    return linalg.solve(K, b)[0].sum()

print(sc.gradient(cost, "cost", "K")(np.array([[4.0, 0.0], [1.0, 3.0]]), np.array([1.0, 2.0])))
# [[-0.01652893  0.        ]
#  [-0.14049587 -0.17355372]]
```

Forward mode differentiates the loops of the factorization and the substitutions, carrying the
tangents along with the values. Second derivatives, such as `sc.hessian`, combine the two and
also hold for `K` and `b` active together. `cholesky` and `solve_triangular` follow the same
rules: the reverse derivative of a triangular solve is one solve with the transposed matrix, and
that of `cholesky` is two.

!!! warning "A failed factorization has no meaningful derivative"
    The derivative is computed from the factor whatever the status. When the status is nonzero,
    the derivative is as meaningless as the solution, and it may hold NaN or infinite values.

## Generated code

Each factorization and substitution is a `scan` over the columns or rows of the matrix, with
inner loops whose trip counts depend on the step, so the loops cover only one triangle. The
factor and the solution are updated in place. The size of the generated code does not depend on
the order of the matrix, while the work grows with its cube for a factorization and with its
square for each right-hand side of a substitution.
