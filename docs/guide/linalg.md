# Linear algebra

`scaly.linalg` solves dense linear systems inside a function: triangular solves, the Cholesky
factorization and `solve` for symmetric positive definite matrices. They build calls of Scaly
functions written with [loops](control_flow.md), so the generated C contains the factorization
itself and calls no BLAS or LAPACK library. Their results are differentiable.

## Solving a positive definite system

`solve(K, b, assume="pos")` returns the solution of \(K x = b\) together with a status, here for
a regularized least-squares system \(K = A^T A + \rho I\):

```python
import numpy as np
import scaly as sc
from scaly import linalg

status_type = sc.TensorType((), sc.dtypes.int64)

@sc.function(sc.arg("A", (3, 2)), sc.arg("y", 3),
             outputs=sc.group(sc.arg("x", 2), sc.arg("status", status_type)))
def least_squares(A: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    K = A.T @ A + 1e-3 * sc.const(np.eye(2))
    return linalg.solve(K, A.T @ y, assume="pos")

A = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
x, status = least_squares(A, np.array([1.0, 2.0, 4.0]))
print(x, status)
# [0.83338868 1.49966686] 0
```

`assume="pos"` states that `K` is symmetric positive definite. `solve` then factors it as
\(K = L L^T\) and solves the two triangular systems. `b` is a vector of shape `(n,)` or a matrix
of shape `(n, k)` whose columns are separate right-hand sides, and `x` has the same shape as `b`.

`solve`, `cholesky` and `solve_triangular` are Python functions, not `Function`s. They are
called inside a function body on expressions and add calls of the library functions to its graph.

Because they are not `Function`s, they do not work everywhere a `Function` does. Called on NumPy
arrays outside a function body, they return expressions holding the calculation, not numbers,
so a numerical result needs an `@sc.function` around the call, as every example on this page has.
They cannot be passed to `sc.gradient`, `sc.jacobian` or `sc.vmap` by name either, which raise
`TypeError` for a Python function. Differentiate the `@sc.function` that calls them instead. A stack
of matrices needs no `vmap`, since every routine maps over leading axes by itself, as
[Stacks of matrices](#stacks-of-matrices) shows.

## Differences from NumPy and SciPy

The names follow SciPy's, but three differences could give a wrong result in code ported from
NumPy or SciPy without raising an error. Two of the arguments have no default to prevent that:

- `np.linalg.solve` and `scipy.linalg.solve` take a general matrix by default. `linalg.solve`
  needs the class of the matrix stated with `assume`, and `"pos"` is the only class it accepts.
  A matrix that is not symmetric gives the solution for its symmetric completion, as the next
  section shows.
- `scipy.linalg.solve_triangular` reads the upper triangle by default. `linalg.solve_triangular`
  needs `lower=True` or `lower=False`.
- `solve` and `cholesky` return a status alongside their result instead of raising on a singular
  or indefinite matrix.

All three take float64 matrices and right-hand sides, and raise `TypeError` for float32 or sparse
ones.

## Only the lower triangle is read

A symmetric matrix stores each off-diagonal value twice. `solve` and `cholesky` read only the
lower triangle, including the diagonal, and never look at the entries above it. An entry below
the diagonal stands for itself and its mirror above. Whatever the upper triangle holds, the result
is that of the symmetric matrix with the same lower triangle:

```python
@sc.function(sc.arg("K", (2, 2)), sc.arg("b", 2), outputs=sc.arg("x", 2))
def solve_only(K: sc.Expr, b: sc.Expr) -> sc.Expr:
    x, status = linalg.solve(K, b, assume="pos")
    return x

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

The status is 0 when every pivot is usable. Otherwise it is the position of the first unusable
pivot, counted from one as LAPACK counts them. The factorization does not stop there, so the
later entries of the factor and the whole solution are meaningless. Only the status says so,
since they need not be NaN:

```python
@sc.function(sc.arg("K", (3, 3)), outputs=sc.arg("status", status_type))
def pivot_status(K: sc.Expr) -> sc.Expr:
    L, status = linalg.cholesky(K)
    return status

print(pivot_status(np.diag([1.0, 2.0, 3.0])))  # 0
print(pivot_status(np.diag([1.0, -2.0, 3.0]))) # 2
print(pivot_status(np.ones((3, 3))))           # 2
```

The last matrix is singular: its second pivot is \(1 - 1 = 0\). `solve` returns the status of
its factorization, so a solver built on it can reject a step, add regularization or stop.

## Factors and triangular solves

`cholesky(A)` returns `(L, status)`, with `L` lower triangular and zero above the diagonal.
`solve_triangular(T, b, lower=True)` solves \(T x = b\) reading the lower triangle of `T`, and
`lower=False` reads the upper triangle instead. `trans=True` solves \(T^T x = b\), and
`unit_diagonal=True` takes the diagonal as ones without reading it. Like `solve`, it accepts one
or several right-hand sides.

A factor computed once serves any number of solves. Two triangular solves with the factor give
\(A^{-1} b\):

```python
@sc.function(sc.arg("A", (2, 2)), sc.arg("b", 2), outputs=sc.arg("x", 2))
def factor_and_solve(A: sc.Expr, b: sc.Expr) -> sc.Expr:
    L, status = linalg.cholesky(A)
    y = linalg.solve_triangular(L, b, lower=True)
    return linalg.solve_triangular(L, y, lower=True, trans=True)

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
    return linalg.solve(K, b, assume="pos")

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
    x, status = linalg.solve(K, b, assume="pos")
    return x.sum()

print(sc.gradient(cost, "cost", "K")(np.array([[4.0, 0.0], [1.0, 3.0]]), np.array([1.0, 2.0])))
# [[-0.01652893  0.        ]
#  [-0.14049587 -0.17355372]]
```

Forward mode differentiates the substitutions step by step. For the factor it uses the closed
form \(dL = L\, \Phi(L^{-1}\, dK\, L^{-T})\), where \(\Phi\) keeps the lower triangle and halves
the diagonal. That rule factors `K` once more, so a forward derivative and a Hessian with respect
to `K` cost an extra factorization. Second derivatives, such as those of `sc.hessian`, hold with
`K` and `b` varying together, and a gradient may itself be differentiated in reverse mode.

`cholesky` and `solve_triangular` follow the same rules. The reverse derivative of a triangular
solve is one more triangular solve with the transposed matrix, and that of `cholesky` takes two.

!!! warning "A failed factorization has no meaningful derivative"
    The derivative is computed from the factor whatever the status. When the status is nonzero,
    the derivative is as meaningless as the solution, and it may hold NaN or infinite values.

## Generated code

Each factorization and substitution is a `scan` over the columns or rows of the matrix, with
inner loops whose trip counts depend on the step, so the loops cover only one triangle. The
factor and the solution are updated in place. The loops themselves have the same size at every
order of the matrix, and only constant tables grow with it, while the work grows with the cube of
the order for a factorization and with its square for each right-hand side of a substitution.
