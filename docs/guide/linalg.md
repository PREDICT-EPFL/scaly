# Linear algebra

`scaly.linalg` solves linear systems inside a function: triangular solves, the Cholesky,
\(L D L^T\) and LU factorizations, and `solve` for positive definite, symmetric quasi-definite and
general matrices. Cholesky, \(L D L^T\) and their solves also take a
[sparse matrix](#sparse-matrices). They build calls of Scaly
functions written with [loops](control_flow.md), so the generated C contains the factorization
itself and calls no BLAS or LAPACK library. Their results are differentiable.

## Solving a positive definite system

`solve(K, b, assume="pos")` returns the solution of \(K x = b\) together with a status, here for
a regularized least-squares system \(K = A^T A + \rho I\):

```python
import numpy as np
import scaly as sc
from scaly import linalg

status_type = sc.TensorType((), sc.dtypes.int32)

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

`solve` and the other routines on this page are Python functions, not `Function`s. They are
called inside a function body on expressions and add calls of the library functions to its graph.

## Quasi-definite and general matrices

`assume` names the class of `K` and with it the factorization:

| `assume` | Class of `K` | Factorization | Triangle read |
|---|---|---|---|
| `"pos"` | symmetric positive definite | \(K = L L^T\), `cholesky` | lower |
| `"sym"` | symmetric quasi-definite | \(K = L D L^T\) without pivoting, `ldl` | lower |
| `"gen"` | general | \(P K = L U\) with partial pivoting, `lu_factor` | all |

A quasi-definite matrix has the form \(\begin{bmatrix}H & A^T\\ A & -R\end{bmatrix}\) with \(H\)
and \(R\) positive definite, as in the regularized optimality conditions of an equality
constrained quadratic program. It is indefinite, so `"pos"` fails on it, but every symmetric reordering
of its rows and columns has an \(L D L^T\) factorization with a diagonal \(D\), so it needs no pivoting.
It has as many negative pivots as \(R\) has rows:

```python
@sc.function(sc.arg("H", (2, 2)), sc.arg("A", (1, 2)), sc.arg("g", 2), sc.arg("c", 1),
             outputs=sc.group(sc.arg("z", 3), sc.arg("status", status_type)))
def kkt_step(H: sc.Expr, A: sc.Expr, g: sc.Expr, c: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    K = sc.concat([sc.concat([H, A.T], axis=1),
                   sc.concat([A, -0.5 * sc.const(np.eye(1))], axis=1)])
    return linalg.solve(K, sc.concat([-g, -c]), assume="sym")

z, status = kkt_step(np.diag([2.0, 1.0]), np.array([[1.0, 1.0]]), np.array([1.0, 0.0]), np.array([-1.0]))
print(z, status)
# [-0.125  0.75  -0.75 ] 0
```

A matrix without such structure takes `assume="gen"`. Its LU factorization exchanges rows to take,
in each column, the pivot of largest magnitude, so a zero on the diagonal does no harm:

```python
@sc.function(sc.arg("K", (2, 2)), sc.arg("b", 2),
             outputs=sc.group(sc.arg("x", 2), sc.arg("status", status_type)))
def general(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return linalg.solve(K, b, assume="gen")

x, status = general(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([1.0, 2.0]))
print(x, status)
# [-0.5  1. ] 0
```

!!! warning "`\"sym\"` does not pivot"
    Without pivoting, a symmetric matrix that is not quasi-definite can give a pivot that is small
    but usable, and a solution with no accurate digits and a status of 0. For
    \(K = \begin{bmatrix}10^{-20} & 1\\ 1 & 0\end{bmatrix}\) and \(b = (1, 2)\), `"sym"` returns
    \((0, 1)\) instead of \((2, 1)\). Such a matrix takes `"gen"`.

## Differences from NumPy and SciPy

The names follow SciPy's, but four differences could give a wrong result in code ported from
NumPy or SciPy without raising an error. Two of the arguments have no default to prevent that:

- `np.linalg.solve` and `scipy.linalg.solve` take a general matrix by default. `linalg.solve`
  needs the class of the matrix stated with `assume`. With `"pos"` or `"sym"`, a matrix that is
  not symmetric gives the solution for its symmetric completion, as the next section shows.
- `scipy.linalg.solve_triangular` reads the upper triangle by default. `linalg.solve_triangular`
  needs `lower=True` or `lower=False`.
- `scipy.linalg.solve(..., assume_a="sym")` pivots and accepts any symmetric matrix that is not
  singular. `linalg.solve(..., assume="sym")` does not pivot and is meant for quasi-definite
  matrices, as the warning above explains.
- `solve` and the factorizations return a status alongside their result instead of raising on a
  singular or indefinite matrix.

`lu_factor` returns the same `lu` and `piv` as `scipy.linalg.lu_factor`, followed by a status.
`scipy.linalg.ldl` pivots and returns `(lu, d, perm)` with `d` a block diagonal matrix, while
`linalg.ldl` returns `(L, d, status)` with `d` the vector of pivots. All of them take float32 or
float64 matrices, as [below](#float32-matrices) explains, and raise `TypeError` for integer ones.
`lu_factor`, `solve_triangular` and `solve(..., assume="gen")` take only dense matrices.

## Only the lower triangle is read

A symmetric matrix stores each off-diagonal value twice. `cholesky`, `ldl`, and `solve` with
`"pos"` or `"sym"` read only the lower triangle, including the diagonal, and never look at the entries above it. An entry below
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
eliminated. For a positive definite matrix every pivot is positive, and a Cholesky pivot is
unusable when it is not greater than the machine epsilon of `K`'s dtype times the diagonal entry
of `K` it came from.
That includes zero, negative and NaN pivots, and pivots so small relative to their diagonal entry that the
cancellation behind them has left no accurate digits.

The status is an int32, 0 when every pivot is usable. Otherwise it is the position of the first unusable
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

The pivots of `ldl` and `lu_factor` may have either sign, so they compare magnitudes: a pivot is
unusable when its magnitude is not greater than machine epsilon times the magnitude of the entry
of `K` it came from, or when it is infinite. For `lu_factor` that entry is the one in the row the
pivot search chose.

A status of 0 does not show that `K` is nonsingular or well conditioned. Rounding can leave the
last pivot of a singular matrix slightly away from zero, above the threshold.

## Factors and triangular solves

`cholesky(A)` returns `(L, status)`, with `L` lower triangular and zero above the diagonal.
`solve_triangular(T, b, lower=True)` solves \(T x = b\) reading the lower triangle of `T`, and
`lower=False` reads the upper triangle instead. `trans=True` solves \(T^T x = b\), and
`unit_diagonal=True` takes the diagonal as ones without reading it. Like `solve`, it accepts one
or several right-hand sides.

`ldl(A)` returns `(L, d, status)`, with `L` unit lower triangular, ones on its diagonal and zeros
above it, and `d` the vector of pivots, so that \(A = L\, \mathrm{diag}(d)\, L^T\).
`lu_factor(A)` returns `(lu, piv, status)`. The matrix `lu` holds \(U\) on and above its diagonal
and the entries of the unit lower triangular \(L\) below it, and the int64 vector `piv` says
that row `k` was exchanged with row `piv[k]` at step `k`.

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
             outputs=sc.group(sc.arg("x", (4, 2)), sc.arg("status", sc.TensorType((4,), sc.dtypes.int32))))
def stage_solves(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return linalg.solve(K, b, assume="pos")

K = np.stack([np.eye(2) * s for s in (1.0, 2.0, -1.0, 4.0)])
x, status = stage_solves(K, np.ones((4, 2)))
print(status)
# [0 0 1 0]
```

## Sparse matrices

A matrix with a [stored pattern](sparse_matrices.md) passes through the same functions:
`cholesky`, `ldl`, and `solve` with `"pos"` or `"sym"`. The factorization then computes only the
entries of the factor that the pattern allows, and its work grows with those entries rather than
with the cube of the order. The arrow matrix below couples its first unknown to every other one,
as a constraint couples the variables of a quadratic program:

```python
from scipy import sparse

K_value = sparse.csc_array(np.array([
    [-2.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 4.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 4.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 4.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 4.0],
]))
K_type = sc.TensorType((5, 5), pattern=sc.const(K_value).type.pattern)

@sc.function(sc.arg("K", K_type), sc.arg("b", 5),
             outputs=sc.group(sc.arg("x", 5), sc.arg("status", status_type)))
def arrow_solve(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return linalg.solve(K, b, assume="sym")

print(arrow_solve(K_value, np.ones(5)))
# (array([0.  , 0.25, 0.25, 0.25, 0.25]), array(0, dtype=int32))
```

`b` and `x` stay dense. As for a dense matrix, only the lower triangle is read, but the pattern
may store the lower triangle, the upper one or both. An entry stored in one triangle stands for
its mirror, and of a pair stored in both, the lower entry is read.

Elimination can create nonzeros where the matrix has none, called fill. Which entries fill depends
on the order of the unknowns, so the sparse factorization first reorders them symmetrically,
factoring \(P K P^T\) for a permutation \(P\). `ordering` chooses it when the function is
traced, from the pattern alone. `"amd"`, the default, is approximate minimum degree, which keeps
the fill small on general patterns. `"rcm"`, reverse Cuthill-McKee, keeps a banded matrix banded,
such as one whose unknowns are ordered stage by stage. `"natural"` keeps the given order. Eliminating
the arrow's first unknown first fills the whole lower triangle, while minimum degree eliminates it
near the end and leaves no fill:

```python
for ordering in ("natural", "amd"):
    @sc.function(sc.arg("K", K_type),
                 outputs=sc.group(sc.arg("L"), sc.arg("d", 5), sc.arg("status", status_type)))
    def arrow_factor(K: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
        return linalg.ldl(K, ordering=ordering)

    L, d, status = arrow_factor(K_value)
    print(ordering, L.nnz)
# natural 15
# amd 9
```

`solve` applies the permutation itself, so its result does not depend on the ordering beyond
rounding. The factors do. `L` is a sparse matrix of \(P K P^T\), in the reordered numbering, whose
pattern holds the entries the factorization may fill and the diagonal. `d` is a dense vector in the
same numbering, and so is the position of the first unusable pivot in the status. `ordering` with a
dense matrix raises `TypeError`, and a sparse matrix takes no stack over leading axes.

The derivatives of a sparse solve are those of a dense one, with \(\bar K\) on the entries the
factorization reads and zero on the mirrored entries it does not. Forward mode solves
\(dx = K^{-1}(db - dK\, x)\) with the factor of the solve instead of differentiating the steps of
the factorization. The sparse `L` and `d` are meant to be read, not differentiated, and a
derivative should go through `solve`.

## Float32 matrices

Every routine computes in the dtype of its matrix, float32 or float64, and returns factors and
solutions of that dtype. A function built from a float32 matrix uses single-precision arithmetic
throughout, which is what a microcontroller with a single-precision floating-point unit, such as
a Cortex-M4, needs. The right-hand side has the matrix's dtype too. A Python number or list takes
it, while a float64 array beside a float32 matrix raises `TypeError`, as any expression mixing the
two does, and asks for `sc.cast`.

The pivot test uses the machine epsilon of the matrix's dtype, about \(1.2 \times 10^{-7}\) for
float32 against \(2.2 \times 10^{-16}\) for float64. A matrix that factors in float64 can
therefore report an unusable pivot in float32. Here the off-diagonal entry rounds to 1 in float32,
which leaves a zero second pivot:

```python
@sc.function(sc.arg("K", (2, 2), dtype=...), outputs=sc.arg("status", status_type))
def cholesky_status(K: sc.Expr) -> sc.Expr:
    L, status = linalg.cholesky(K)
    return status

K = np.array([[1.0, 0.0], [0.99999999, 1.0]])
print(cholesky_status(K), cholesky_status(K.astype(np.float32)))
# 0 2
```

`dtype=...` leaves the dtype of `K` to the call, as described in
[Leaving the dtype open](functions.md#leaving-the-dtype-open), so the same function runs in both
precisions.

## Derivatives

The solution is differentiable in `K` and `b`. Reverse mode, which computes gradients, uses
the implicit derivative of \(K x = b\):

\[
\bar b = K^{-1} \bar x, \qquad \bar K = -\bar b\, x^T.
\]

It reuses the factor of the solve instead of factoring `K` again or differentiating the steps of
the factorization. A function that returns a value and its gradient factors `K` once. With
`"gen"`, \(\bar b = K^{-T} \bar x\), solved with the transposed factors.

With `"pos"` and `"sym"`, only the lower triangle is read, so an entry below the diagonal receives the derivatives of
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

Forward mode differentiates the substitutions step by step. For the factor it uses a closed form.
For Cholesky it is \(dL = L\, \Phi(L^{-1}\, dK\, L^{-T})\), where \(\Phi\) keeps the lower triangle
and halves the diagonal. For \(L D L^T\), with \(F = L^{-1}\, dK\, L^{-T}\), the pivots change by
the diagonal of \(F\) and \(dL = L X\), where \(X\) is the strict lower triangle of \(F\) with each
column divided by its pivot. For LU, with \(F = L^{-1} P\, dK\, U^{-1}\), \(dL\) is \(L\) times the
strict lower triangle of \(F\) and \(dU\) is the upper triangle of \(F\) times \(U\), the row
exchanges staying fixed. Each rule factors `K` once more for every direction, so a forward
derivative and a Hessian with respect to `K` cost one factorization per entry of `K` they
differentiate in, and generate correspondingly more code. Second derivatives, such as those of `sc.hessian`, hold with
`K` and `b` varying together, and a gradient may itself be differentiated in reverse mode.

`cholesky` and `solve_triangular` follow the same rules. The reverse derivative of a triangular
solve is one more triangular solve with the transposed matrix, and that of `cholesky` takes two.
`ldl` and `lu_factor` have only the forward rule. A gradient through their factors differentiates
the loops of the factorization, which costs far more than the reverse rule of `solve`, so a
derivative should go through `solve`.

!!! warning "A failed factorization has no meaningful derivative"
    The derivative is computed from the factor whatever the status. When the status is nonzero,
    the derivative is as meaningless as the solution, and it may hold NaN or infinite values.

## Generated code

Each factorization and substitution is a `scan` over the columns or rows of the matrix, with
inner loops whose trip counts depend on the step, so the loops cover only one triangle. A sparse
factorization reads its trip counts and the rows it updates from constant tables of the pattern,
column by column: it loads a column of the matrix into a dense work vector, subtracts the earlier
columns that reach it, and stores the column of the factor. The LU
factorization computes each column from the earlier ones, searches it for the pivot and exchanges
two rows of the matrix at the pivot's run-time position. The
factor and the solution are updated in place. The loops themselves have the same size at every
order of the matrix, and only constant tables grow with it, while the work grows with the cube of
the order for a factorization and with its square for each right-hand side of a substitution.
This holds for a solution and its gradient, not for forward derivatives in `K`, as the section on
derivatives explains.
