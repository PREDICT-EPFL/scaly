# Sparsity

Sparse derivative functions evaluate a compact vector of potentially nonzero
entries, accompanied by a fixed matrix pattern. This is useful for multistage
optimal control models, where each dynamics constraint depends on only a small
part of the decision vector. The pattern comes from symbolic dependencies, not
from numerical values observed during evaluation.

## Structural and numerical zeros

Consider a function from a four-element vector to a three-element vector:

```python
import numpy as np
import scaly as sc

@sc.function(sc.arg("x", 4), outputs=sc.arg("y"))
def model(x: sc.Expr) -> sc.Expr:
    return sc.stack([x[0] * x[1], x[2], x[3] * x[3]])

sparse_jac = sc.sparse_jacobian(model, "y", "x")
pattern = sparse_jac.sparsity()
assert pattern is not None
print(pattern.shape)  # (3, 4)
print(pattern.nnz)    # 4
print(pattern.rows)   # (0, 0, 1, 2)
print(pattern.cols)   # (0, 1, 2, 3)
```

`Function.sparsity` returns `None` for an ordinary dense output and a
`SparsityPattern` for a sparse derivative output. For a function with several
outputs, select one by name with `of="spjac_y_x"`.
The assertion makes that distinction explicit to a type checker.

The pattern uses coordinate format, abbreviated COO. The paired `rows` and
`cols` arrays give the zero-based coordinates of each stored value, together
with the full matrix `shape`.

The full Jacobian is

\[
J(x)=\begin{bmatrix}
x_1 & x_0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 2x_3
\end{bmatrix}.
\]

`nnz` is the number of entries that may be nonzero. Scaly determines the pattern
from the calculation, before you provide input values. An entry remains in the
pattern even if it happens to evaluate to zero at a particular input. Some
operations produce a conservative pattern containing additional possible
nonzeros.

```python
print(sparse_jac(np.zeros(4)))  # [0. 0. 1. 0.]
print(pattern.nnz)             # 4
```

Three stored entries happen to be zero here. They are still present because
other inputs make them nonzero. The pattern therefore stays valid across calls.
Dropping entries based on one numerical evaluation would lose that property.

## Compact values and matrix reconstruction

A sparse derivative function returns a one-dimensional array of values. The
pattern tells you where those values belong:

```python
values = sparse_jac(np.array([2.0, 3.0, 4.0, 5.0]))
print(values)  # [ 3.  2.  1. 10.]

J = np.zeros(pattern.shape)
J[np.asarray(pattern.rows), np.asarray(pattern.cols)] = values
print(J)
# [[ 3.  2.  0.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0. 10.]]
```

Keep the pattern alongside the values when passing them to another library.

For a function with open shapes, pass the same arguments to the query as to
evaluation. The query traces that binding if needed, but does not compile or
evaluate it. Symbolic arguments also work:

```python
@sc.function(sc.arg("x"), outputs=sc.arg("y"))
def squares(x: sc.Expr) -> sc.Expr:
    return x * x

jac = sc.sparse_jacobian(squares)
point = np.arange(3.0)
pattern = jac.sparsity(point)
values = jac(point)
assert pattern is not None
print(pattern.shape)  # (3, 3)
print(pattern.nnz)    # 3
```

Different shapes can have different patterns. The query always selects the
binding from its arguments, rather than the most recent call. A fully specified
function can omit the arguments. A plain sparse derivative's
[generated C header](codegen.md#sparse-output-patterns) contains the same index
tables, so a C caller can reconstruct the matrix without Python.

For a mapped sparse derivative, the query returns the per-iteration matrix
pattern. Apply it separately to each leading output slice; nested maps use the
same pattern for every iteration. The mapped concrete graph and its C header
do not carry these per-iteration tables. Keep the source pattern when passing
mapped compact values through another function or a factory.

For direct expression construction, the sparse result bundles the two parts:

```python
x = sc.sym("x", 4)
y = model(x)
sj = sc.sparse_jacobian(y, x)
compact_expression = sj.values
matrix_expression = sj.to_dense()
pattern = sj.sparsity
```

`sc.jacobian_sparsity(y, x)` obtains only the pattern. `pattern.to_mask()` returns
a dense Boolean array for inspection.

## The ordering rule

Entry `values[k]` belongs at `(pattern.rows[k], pattern.cols[k])`. Do not assume
the coordinates are sorted. Mapped functions can produce entries in stage order
rather than matrix row order.

Compressed sparse row and column formats, abbreviated CSR and CSC, require a
specific ordering. Scaly returns the required permutation with the index arrays:

```python
row_ptr, col_ind, permutation = pattern.to_csr()
values_csr = values[np.asarray(permutation, dtype=int)]

col_ptr, row_ind, permutation = pattern.to_csc()
values_csc = values[np.asarray(permutation, dtype=int)]
```

The `values` and `pattern` above can be converted to a SciPy matrix without
forming a dense intermediate:

```python
from scipy.sparse import csr_matrix

matrix = csr_matrix((values_csr, col_ind, row_ptr), shape=pattern.shape)
```

The permutation may be trivial for a small example, but a caller must apply
it for general patterns. Generated headers
expose the corresponding
`_csr_val_perm` and `_csc_val_perm` tables. See
[generated sparse-output patterns](codegen.md#sparse-output-patterns).

## Sparse Hessians

Hessians are symmetric. If a solver needs only one triangle, request that
triangle when constructing the derivative:

```python
@sc.function(sc.arg("x", 4), outputs=sc.arg("cost"))
def cost(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x) + x[0] * x[1]

sparse_hess = sc.sparse_hessian(cost, "cost", "x", triangle="lower")
hess_values = sparse_hess(np.ones(4))
hess_pattern = sparse_hess.sparsity()
assert hess_pattern is not None
lower = np.zeros(hess_pattern.shape)
lower[np.asarray(hess_pattern.rows), np.asarray(hess_pattern.cols)] = hess_values
H = lower + lower.T - np.diag(np.diag(lower))
print(H)
# [[2. 1. 0. 0.]
#  [1. 2. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 2.]]
```

The choices are `"full"`, `"lower"`, and `"upper"`. The pattern still has the
full matrix shape, but only contains entries in the selected triangle. To
reconstruct a full symmetric matrix from one triangle, reflect off-diagonal
entries and keep diagonal entries once.

Adding `lower + lower.T` alone would double the diagonal. A matrix built from
the compact lower triangle is not yet the full symmetric Hessian.

`sc.sparse_lagrangian_hessian` supports the same choices. Built-in solver
interfaces select the triangle their solver needs.

## Sparse derivatives and mapped structure

For independent stage calculations \(r_k=f(z_k)\), stacking the inputs and
outputs gives a block-diagonal Jacobian:

\[
R(Z)=\begin{bmatrix}f(z_0)\\ f(z_1)\\ \vdots\\ f(z_{N-1})\end{bmatrix},
\qquad
\frac{\partial R}{\partial Z}=
\begin{bmatrix}
J_f(z_0)&0&\cdots&0\\
0&J_f(z_1)&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&J_f(z_{N-1})
\end{bmatrix}.
\]

[`vmap`](functions.md#regular-repetition-vmap) can be used to express this repetition directly:

```python
N = 4

@sc.function(sc.arg("z", 2), outputs=sc.arg("residual"))
def stage(z: sc.Expr) -> sc.Expr:
    return sc.stack([z[0].sin() * z[1], z[0] + z[1] ** 2])

@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("residuals"))
def stages(zs: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, N)(zs).vec()

stage_jac = sc.sparse_jacobian(stages, "residuals", "zs")
stage_pattern = stage_jac.sparsity()
assert stage_pattern is not None
print(stage_pattern.shape)  # (8, 8)
print(stage_pattern.nnz)    # 16: four 2-by-2 blocks
```

Scaly constructs the stage derivative and evaluates it in a mapped loop. It
stores the entries of those blocks without filling the zero blocks between
them. The [mapped derivative example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/mapped_derivatives.py)
also exports the function and Jacobian so you can inspect their generated C.

Multiple-shooting constraints couple adjacent stages. For a defect
\(d_k(z_k,z_{k+1})\), let \(A_k=\partial d_k/\partial z_k\) and
\(B_k=\partial d_k/\partial z_{k+1}\). Their stacked Jacobian has the form

\[
\frac{\partial D}{\partial Z}=
\begin{bmatrix}
A_0&B_0&0&\cdots&0\\
0&A_1&B_1&\cdots&0\\
\vdots&&\ddots&\ddots&\vdots\\
0&\cdots&0&A_{N-1}&B_{N-1}
\end{bmatrix}.
\]

Overlapping input slices in `vmap` express these neighboring dependencies, as in
the [multiple-shooting example](getting_started.md#bonus-repeated-stages-with-vmap).
Each defect can be evaluated independently at the candidate states, even though
the constraints couple them. Shared parameters can add columns spanning all
stages, so a mapped calculation does not always have a block-diagonal Jacobian.

The repeated derivative formula can remain in one loop body as the horizon
grows. Evaluation work, stored values, and the
[generated sparsity tables](codegen.md#sparse-output-patterns) still grow with
that horizon. [Scalability results](../benchmarks/scalability.md) measure these
costs separately.

## No sparse arithmetic

Compact derivatives and sparse solver matrices do not provide general sparse
arithmetic inside an `Expr` graph. A `SparseJacobian` holds a pattern and an
expression for its values. It does not support matrix multiplication directly.
Converting it with `to_dense()` creates a dense matrix expression:

```python
def chain(x: sc.Expr) -> sc.Expr:
    return x[:-1].sin() * x[1:]

@sc.function(sc.arg("x", 6), sc.arg("v", 6), outputs=sc.arg("jv"))
def via_matrix(x: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.sparse_jacobian(chain(x), x).to_dense() @ v

@sc.function(sc.arg("x", 6), sc.arg("v", 6), outputs=sc.arg("jv"))
def via_product(x: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.jvp(chain(x), x, v)

data = (np.linspace(0.1, 0.6, 6), np.arange(1.0, 7.0))
np.testing.assert_allclose(via_matrix(*data), via_product(*data))
```

Both compute \(J(x)v\). The second requests a Jacobian-vector product directly,
without constructing the matrix. The
[sparse JVP example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/sparse_jvp.py)
exports both versions for comparison. Compiler simplification can remove some
intermediates, but `to_dense() @ v` is not a sparse matrix multiplication API.

!!! warning "Sparse constants"

    `sc.const` does not accept SciPy sparse matrices. A sparse matrix accepted
    by a numerical solver call cannot automatically be embedded in a symbolic
    call. The [fixed-matrix control example](solvers.md#fixed-sparse-matrices-without-sparse-constants)
    shows how to express such a problem without sparse constants.

Scaly has no general sparse arithmetic. Sparse derivative construction and the
solver interfaces still support complete optimal control workflows, including the [benchmark problems](../benchmarks/index.md).
