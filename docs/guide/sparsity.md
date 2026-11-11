# Sparse derivatives

A sparse Jacobian or Hessian stores only its potentially nonzero entries,
together with a fixed matrix pattern. This is useful for multistage optimal
control models, where each dynamics constraint depends on only a small part of
the decision vector. The pattern comes from symbolic dependencies, not from
numerical values observed during evaluation.

A sparse derivative is a [sparse matrix](sparse_matrices.md): its type carries
the pattern, and a sparse derivative `Function` returns it as a SciPy sparse
array whose values follow a declared [layout](sparse_matrices.md#layouts).

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

`Function.sparsity` returns the stored pattern of a sparse output and `None`
for a dense one. For a function with several outputs, select one by name with
`of="spjac_y_x"`. The assertion makes that distinction explicit to a type
checker. The pattern lists the zero-based `rows` and `cols` of the stored
entries in row-major order, together with the full matrix `shape`.

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
print(sparse_jac(np.zeros(4)).data)  # [0. 0. 1. 0.]
print(pattern.nnz)                  # 4
```

Three stored entries happen to be zero here. They are still present because
other inputs make them nonzero. The pattern therefore stays valid across calls.
Dropping entries based on one numerical evaluation would lose that property.

## Sparse results and their layout

A sparse derivative function returns a SciPy sparse array. By default it is a
`csc_array`, whose stored values run column by column:

```python
J = sparse_jac(np.array([2.0, 3.0, 4.0, 5.0]))
print(type(J).__name__)  # csc_array
print(J.data)            # [ 3.  2.  1. 10.]
print(J.toarray())
# [[ 3.  2.  0.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0. 10.]]
```

The derivative matrix's logical size, the product of its dimensions, must fit
in a signed 64-bit integer. Construction raises `ValueError` otherwise, even
when the stored pattern has only a few entries.

`layout=` chooses the order of the stored values, with the same choices as any
[sparse output](sparse_matrices.md#layouts): `"csc"`, `"csr"`, or a
`SparsityPattern` listing the derivative's stored coordinates in the order a
solver reads its triplets. The values arrive in that order, in Python and in
the [generated C](codegen.md#sparse-output-patterns), with no permutation to
apply:

```python
by_rows = sc.sparse_jacobian(model, "y", "x", layout="csr")
print(by_rows(np.array([2.0, 3.0, 4.0, 5.0])).data)  # [ 3.  2.  1. 10.]

triplets = sc.SparsityPattern((3, 4), (2, 1, 0, 0), (3, 2, 1, 0))
listed = sc.sparse_jacobian(model, "y", "x", layout=triplets)
print(listed(np.array([2.0, 3.0, 4.0, 5.0])).data)  # [10.  1.  2.  3.]
```

A coordinate layout must list exactly the derivative's pattern. A different one
raises `ValueError: coordinate layout has a different stored pattern` when the
function is built.

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
assert pattern is not None
print(pattern.shape)  # (3, 3)
print(pattern.nnz)    # 3
```

Different shapes can have different patterns. The query always selects the
binding from its arguments, rather than the most recent call. A fully specified
function can omit the arguments.

Mapping a sparse derivative over `N` independent points gives a batch of
matrices that share one pattern. The query returns that pattern, and a
numerical call returns a SciPy `coo_array` of shape `(N, *pattern.shape)`:

```python
batched = sc.vmap(jac, 2)
print(batched(np.arange(6.0).reshape(2, 3)).toarray()[1])
# [[ 6.  0.  0.]
#  [ 0.  8.  0.]
#  [ 0.  0. 10.]]
```

For direct expression construction, the result is a sparse `Expr` with the
pattern in its type:

```python
x = sc.sym("x", 4)
sj = sc.sparse_jacobian(model(x), x)
print(sj.shape)                 # (3, 4)
print(sj.type.pattern.nnz)      # 4
stored_expression = sc.values(sj)
matrix_expression = sc.to_dense(sj)
```

The stored values of a sparse `Expr` are in row-major order, and the layout of
a `Function` output decides the order the caller receives.
`sc.jacobian_sparsity(y, x)` obtains only the pattern. `pattern.to_mask()`
returns a dense Boolean array for inspection.

## Sparse Hessians

Hessians are symmetric. If a solver needs only one triangle, request that
triangle when constructing the derivative:

```python
@sc.function(sc.arg("x", 4), outputs=sc.arg("cost"))
def cost(x: sc.Expr) -> sc.Expr:
    return sc.sumsqr(x) + x[0] * x[1]

sparse_hess = sc.sparse_hessian(cost, "cost", "x", triangle="lower")
lower = sparse_hess(np.ones(4)).toarray()
H = lower + lower.T - np.diag(np.diag(lower))
print(H)
# [[2. 1. 0. 0.]
#  [1. 2. 0. 0.]
#  [0. 0. 2. 0.]
#  [0. 0. 0. 2.]]
```

The choices are `"full"`, `"lower"`, and `"upper"`. The matrix keeps its full
shape but stores only the entries in the selected triangle. To reconstruct a
full symmetric matrix from one triangle, reflect off-diagonal entries and keep
diagonal entries once. Adding `lower + lower.T` alone would double the diagonal.

`sc.sparse_hessian(f, of, wrt, layout="csr", triangle="upper")` combines a
triangle with a layout. `sc.sparse_lagrangian_hessian` takes the same choices.
Built-in solver interfaces select the triangle and the order their solver needs.

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
    return sc.vmap(stage, N)(zs.reshape((N, 2))).vec()

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

## Sparse derivatives as matrices

A sparse derivative takes part in arithmetic like any other
[sparse matrix](sparse_matrices.md#operations-on-sparse-values). A product with
a dense vector multiplies only the stored entries, and `sc.to_dense` creates a
dense matrix expression:

```python
def chain(x: sc.Expr) -> sc.Expr:
    return x[:-1].sin() * x[1:]

@sc.function(sc.arg("x", 6), sc.arg("v", 6), outputs=sc.arg("jv"))
def via_matrix(x: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.to_dense(sc.sparse_jacobian(chain(x), x)) @ v

@sc.function(sc.arg("x", 6), sc.arg("v", 6), outputs=sc.arg("jv"))
def via_sparse_matrix(x: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.sparse_jacobian(chain(x), x) @ v

@sc.function(sc.arg("x", 6), sc.arg("v", 6), outputs=sc.arg("jv"))
def via_product(x: sc.Expr, v: sc.Expr) -> sc.Expr:
    return sc.jvp(chain(x), x, v)

data = (np.linspace(0.1, 0.6, 6), np.arange(1.0, 7.0))
np.testing.assert_allclose(via_matrix(*data), via_product(*data))
np.testing.assert_allclose(via_sparse_matrix(*data), via_product(*data))
```

All three compute \(J(x)v\). The first multiplies a dense matrix, the second
multiplies only the stored entries, and the third requests a Jacobian-vector
product directly, without constructing the matrix. The
[sparse JVP example](https://github.com/PREDICT-EPFL/scaly/blob/main/examples/sparse_jvp.py)
exports the first and the last for comparison. Compiler simplification can
remove some intermediates, but `sc.to_dense(J) @ v` is a dense product.

A sparse derivative keeps its pattern through other operations too. Its
transpose `J.T` stores the transposed pattern, so a wrapper `Function` that
returns it reports that pattern from `Function.sparsity` and in its header.
