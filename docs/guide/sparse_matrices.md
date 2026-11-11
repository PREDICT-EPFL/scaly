# Sparse matrices

A sparse matrix in Scaly is a value whose type lists the entries it stores. Every
other entry is identically zero, so generated code stores, passes and computes
only the stored entries. This page covers sparse inputs, outputs and constants,
the operations that keep a pattern, and what reaches derivatives and generated C.

This is a different use of sparsity from [sparse derivatives](sparsity.md). There,
Scaly finds which entries of a Jacobian or Hessian may be nonzero and returns
them as a compact vector. Here, you choose the pattern and it is part of the
matrix's type.

## Stored patterns

Consider the tridiagonal matrix

\[
A = \begin{bmatrix} 4 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 4 \end{bmatrix}.
\]

Its pattern is the seven coordinates on and beside the diagonal. A
`SparsityPattern` lists coordinates in any order. `canonicalize` sorts them into
row-major order and returns the shared pattern object that a `TensorType` holds,
together with the permutation from the original order:

```python
import numpy as np
from scipy import sparse
import scaly as sc

A_value = sparse.csc_array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 4.0]])
pattern, _ = sc.SparsityPattern.from_csc(A_value.shape, A_value.indptr, A_value.indices).canonicalize()
A_type = sc.TensorType((3, 3), pattern=pattern)

print(pattern.nnz)           # 7
print(pattern.rows)          # (0, 0, 1, 1, 1, 2, 2)
print(pattern.cols)          # (0, 1, 0, 1, 2, 1, 2)
print(A_type.shape)          # (3, 3)
print(A_type.storage_shape)  # (7,)
```

`shape` keeps its mathematical meaning. `storage_shape` is the shape of the
stored values. A stored entry may hold zero and still counts as stored. An entry
outside the pattern is a structural zero: it has no storage and no operation
reads it. Like a shape, the pattern is fixed when the type is built and never
depends on numerical values. Equal coordinates give the same canonical pattern
object, so `is` compares two patterns.

Leading axes are batch axes. `sc.TensorType((5, 3, 3), pattern=pattern)` is five
matrices sharing one pattern, with stored values of shape `(5, 7)`.

## Sparse inputs and outputs

Passing the type to `sc.arg` declares a sparse input:

```python
@sc.function(sc.arg("A", A_type), sc.arg("x", 3), outputs=sc.arg("y"))
def apply(A: sc.Expr, x: sc.Expr) -> sc.Expr:
    return A @ x

x_value = np.array([1.0, 2.0, 3.0])
print(apply(A_value, x_value))       # [ 2.  4. 10.]
print(apply(A_value.data, x_value))  # [ 2.  4. 10.]
```

The numerical argument is either a SciPy sparse array with exactly the declared
pattern, in any SciPy format, or the vector of stored values in the input's
layout, described below. Scaly copies a SciPy argument, sums duplicate entries
and checks every stored coordinate. A dense array is refused:

```python
apply(np.eye(3), x_value)
# ValueError: expected a sparse array of shape (3, 3) or stored values of shape (7,), got (3, 3)
```

So is a SciPy array with a different pattern, which raises
`ValueError: sparse input has a different stored pattern`. A symbolic call needs
an argument with exactly the declared pattern too.

A sparse result comes back as a SciPy sparse array:

```python
@sc.function(sc.arg("A", A_type), outputs=sc.arg("B", A_type, layout="csr"))
def scale(A: sc.Expr) -> sc.Expr:
    return 2.0 * A

B_value = scale(A_value)
print(type(B_value).__name__)  # csr_array
print(B_value.data)            # [ 8. -2. -2.  8. -2. -2.  8.]
```

An output declared without a type comes back in compressed sparse column
format, CSC. A batched result comes back as an N-dimensional `coo_array`. Each
call returns a new array of values, but the index arrays are shared between calls
and read-only.

## Layouts

The layout of a sparse input or output is the order of its stored values in a
buffer. It applies to a vector of values passed from Python, to the buffers of
the generated C and to the tables its header publishes. There are three:

| `layout=` | Value order |
| --- | --- |
| `"csc"`, the default | Column by column, as in a SciPy `csc_array` |
| `"csr"` | Row by row, compressed sparse row (CSR) order |
| a `SparsityPattern` | The order of its coordinates, for a solver that reads triplets in its own order |

The layout belongs to the input or output, not to the pattern, so two inputs with
one pattern can use different layouts. A coordinate layout lists the pattern's
coordinates in the caller's order:

```python
triplets = sc.SparsityPattern((3, 3), rows=(2, 1, 0, 1, 2, 0, 1), cols=(2, 1, 0, 2, 1, 1, 0))

@sc.function(sc.arg("A", A_type, layout=triplets), outputs=sc.arg("B", A_type, layout=triplets))
def scale_triplets(A: sc.Expr) -> sc.Expr:
    return 2.0 * A

B_value = scale_triplets(np.arange(1.0, 8.0))
print(B_value.data)    # [ 2.  4.  6.  8. 10. 12. 14.]
print(B_value.coords)  # (array([2, 1, 0, 1, 2, 0, 1]), array([2, 1, 0, 2, 1, 1, 0]))
```

The stored value `1.0` belongs at `(2, 2)`, the first coordinate of `triplets`,
and the result keeps that order. Inside an expression graph, stored values are
always in row-major order. Scaly reorders them at the function boundary.

## Leaving the pattern open

`pattern=...` leaves an input's pattern to the call, as a
[shape hole](functions.md#leaving-shapes-out) leaves its shape. Each distinct
pattern traces and compiles its own instance:

```python
@sc.function(sc.arg("A", (3, 3), pattern=...), sc.arg("x", 3), outputs=sc.arg("y"))
def apply_any(A: sc.Expr, x: sc.Expr) -> sc.Expr:
    return A @ x

print(apply_any(A_value, x_value))                      # [ 2.  4. 10.]
print(apply_any(sparse.csr_array(np.eye(3)), x_value))  # [1. 2. 3.]
```

A SciPy array binds its own pattern. A dense array binds an instance without a
stored pattern. A vector of stored values carries no coordinates, so it cannot
bind a pattern hole. `sc.arg("A", pattern=...)` leaves the shape open as well.
An open pattern takes the layout `"csc"` or `"csr"`, since a coordinate layout
names its coordinates. Before export, bind the pattern with a type:
`apply_any.instantiate((A_type, (3,)))`.

## Sparse constants

`sc.const` accepts a SciPy sparse array. It copies the array, sums duplicate
entries, sorts the indices and keeps explicitly stored zeros:

```python
A_const = sc.const(A_value)
print(A_const.type.pattern is pattern)  # True
```

`sc.const(values, pattern=triplets)` builds the same kind of constant from a
vector of stored values in the order of a `SparsityPattern`.

## Stored values and dense matrices

Five functions convert between sparse and dense expressions:

| Call | Result |
| --- | --- |
| `A.values()` or `sc.values(A)` | The stored values as a dense array of the storage shape, in row-major coordinate order |
| `sc.pack(values, pattern)` | A sparse matrix holding `values`. With a `SparsityPattern`, the values are in its coordinate order |
| `A.to_dense()` or `sc.to_dense(A)` | A dense matrix with zeros outside the pattern |
| `sc.from_dense(D, pattern)` | A sparse matrix holding the entries of `D` inside `pattern`. The others are dropped |
| `sc.sparsify(D)` | A sparse matrix over the entries of `D` that may be nonzero |

`sc.sparsify` relies on `Expr.support`, the entries of an expression that may be
nonzero, which Scaly derives from constants, products and the other operations
that preserve zeros:

```python
z = sc.sym("z", 3)
D = sc.stack([z, z * sc.const([0.0, 1.0, 1.0]), sc.const([0.0, 0.0, 2.0])])
D_sparse = sc.sparsify(D)
print(D_sparse.type.pattern.rows)  # (0, 0, 0, 1, 1, 2)
print(D_sparse.type.pattern.cols)  # (0, 1, 2, 1, 2, 2)
```

Expression listings, such as those of `sc.render_expr_assembly`, show these
conversions as the operations `pack` and `values`. A sparse constant is a `pack`
of a dense constant holding its stored values.

## Operations on sparse values

The pattern of a result follows from the patterns of the operands, never from
their values:

| Operation | Pattern of the result |
| --- | --- |
| `-A`, `abs`, `sqrt`, `floor`, `ceil`, `sin`, `tan`, `asin`, `atan`, `sinh`, `tanh`, `erf`, `sc.stop_gradient` | The pattern of `A` |
| `A + B`, `A - B` | The union of the patterns. A dense operand stores every entry |
| `A * B` | The intersection. With a dense operand, the pattern of the sparse one |
| Slicing, transposition, reshaping, `sc.stack`, `sc.concat`, `sc.gather`, `sc.scatter` | The stored entries, moved. A result with fewer than two axes is dense |
| `A @ B` | See [products](#products) |
| `A.sum()` | A dense scalar |

A dense operand counts as stored everywhere, so `A_const + 0.0` stores all nine
entries while `A_const * 0.0` keeps seven. The unary operations in the table map
zero to zero. Any other elementwise operation raises as the expression is built,
including `exp`, `log`, `cos`, powers, division, `sc.atan2`, `sc.minimum` and
`sc.maximum`:

```python
A_const.exp()
# ValueError: exp has no stored-pattern rule for sparse operands with output shape (3, 3); use to_dense() explicitly
```

`A_const.to_dense().exp()` computes the dense result. Division by a scalar
refuses a sparse operand too, while multiplication by its reciprocal,
`A_const * (1 / s)`, keeps the pattern.

## Structural zeros

A product never reads an entry outside the pattern, even where `0 * inf` would
give NaN:

```python
@sc.function(sc.arg("A", A_type), sc.arg("x", 3), outputs=sc.group(sc.arg("stored"), sc.arg("dense")))
def compare(A: sc.Expr, x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return A @ x, A.to_dense() @ x

print(compare(A_value, np.array([1.0, 2.0, np.inf])))
# (array([  2., -inf,  inf]), array([ nan, -inf,  inf]))
```

Row 0 of `A` stores no entry in column 2, so the sparse product never multiplies
by `inf`. The dense product computes `0 * inf` and gets NaN. A stored zero is
still multiplied. In the same way, `A * D` with a dense `D` keeps the
structural zeros of `A` whatever `D` holds.

!!! note "Zeros in dense expressions"

    Only a stored pattern guarantees that an entry is skipped. In a dense
    expression, simplification folds `x * 0.0` to zero and removes a NaN in `x`,
    but `x * sc.const([1.0, 0.0])` multiplies both entries, so an `inf` in
    `x[1]` gives NaN. [Loops or scalar code](../how_it_works/lowering.md#loops-or-scalar-code)
    describes a related difference between the two forms of generated code.

## Products

A product with a sparse operand multiplies only stored entries. Its pattern
depends on the other operand:

| Product | Result |
| --- | --- |
| Sparse matrix times dense vector, or the reverse | A dense vector |
| Sparse `A` times dense matrix | Sparse, storing every column of each row where `A` stores an entry |
| Dense matrix times sparse `B` | Sparse, storing every row of each column where `B` stores an entry |
| Sparse times sparse | Sparse, with the Boolean product of the two patterns |

The Boolean product stores entry \((i, j)\) of \(AB\) whenever some \(k\) has
both \(A_{ik}\) and \(B_{kj}\) stored, even if the products cancel numerically.
Scaly computes this pattern when the expression is built. A weighted normal
matrix \(G^\top W G\), with \(W\) diagonal, is a typical case:

```python
G_value = sparse.csr_array([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0], [4.0, 0.0, 0.0, 5.0]])
G_type = sc.TensorType((3, 4), pattern=sc.const(G_value).type.pattern)

@sc.function(sc.arg("G", G_type), sc.arg("w", 3), outputs=sc.arg("H"))
def normal_matrix(G: sc.Expr, w: sc.Expr) -> sc.Expr:
    return G.T @ (w.reshape((3, 1)) * G)

H_value = normal_matrix(G_value, np.array([1.0, 2.0, 3.0]))
print(H_value.nnz)  # 8
print(H_value.toarray())
# [[49.  0.  2. 60.]
#  [ 0. 18.  0.  0.]
#  [ 2.  0.  4.  0.]
#  [60.  0.  0. 75.]]
```

`w.reshape((3, 1)) * G` scales the rows of `G` and keeps its pattern. A row of
`G` with \(r\) stored entries contributes up to \(r^2\) entries of the result, so
a single dense row makes \(G^\top W G\) dense.

## Derivatives

A gradient or adjoint with respect to a sparse input is sparse, with the input's
pattern and layout. For \(f(A, x) = x^\top A x\),
\(\partial f / \partial A_{ij} = x_i x_j\) at each stored \((i, j)\):

```python
@sc.function(sc.arg("A", A_type, layout="csr"), sc.arg("x", 3), outputs=sc.arg("f"))
def quadratic(A: sc.Expr, x: sc.Expr) -> sc.Expr:
    return x @ (A @ x)

grad_A = sc.gradient(quadratic, "f", "A")
print(grad_A(A_value, x_value).toarray())
# [[1. 2. 0.]
#  [2. 4. 6.]
#  [0. 6. 9.]]
```

The entries outside the pattern are not variables, so the gradient has none. A
forward derivative of a sparse output takes the output's pattern and layout.
Derivatives are computed on the stored entries, without a dense copy of `A`.

`sc.jacobian` with respect to a sparse input has one column per logical entry,
nine here, and the columns of entries outside the pattern are zero.
`sc.sparse_jacobian` and `sc.sparse_hessian` index the same logical entries.

!!! note "Sparse derivative outputs keep their own value order"

    `sc.sparse_jacobian` and `sc.sparse_hessian` return compact values with a
    separate pattern, as described in [the ordering rule](sparsity.md#the-ordering-rule),
    even when an input or output is a sparse matrix. They do not return a sparse
    output with a declared layout, and `Function.sparsity` returns `None` for a
    sparse matrix output, whose pattern is in its type.

## Generated C

The [pointer interface](codegen.md#the-pointer-entry) passes only stored values,
each buffer in its input's or output's layout. For `scale` above, the generated
header sizes both buffers by the seven stored entries and publishes each
pattern in its declared layout:

```c
typedef struct { SCALY_ALIGNAS(16) double data[7]; } scale_A_t;
typedef struct { SCALY_ALIGNAS(16) double data[7]; } scale_B_t;
#define scale_A_NROW 3
#define scale_A_NCOL 3
#define scale_A_NNZ 7
#define scale_A_NDIM 2
#define scale_A_STORAGE_SIZE 7
static const int scale_A_shape[2] = {3, 3};
static const int scale_A_csc_col_ptr[4] = {0, 2, 5, 7};
static const int scale_A_csc_row_ind[7] = {0, 1, 0, 1, 2, 1, 2};
/* The same five constants and the shape for B, then: */
static const int scale_B_csr_row_ptr[4] = {0, 2, 5, 7};
static const int scale_B_csr_col_ind[7] = {0, 1, 0, 1, 2, 1, 2};
```

A CSC input or output publishes `csc_col_ptr` and `csc_row_ind`, a CSR one
`csr_row_ptr` and `csr_col_ind`, and a coordinate layout `rows` and `cols` in its
own order. Each name starts with the function and the input or output name. A
batched matrix publishes its shared pattern once, and its `STORAGE_SIZE` counts
the values of every batch entry. A C header omits the tables of an empty pattern,
and an input or output with no stored values accepts a null pointer. The C++
header publishes the same tables as `constexpr std::array` constants in a
namespace per function and per input or output.

CasADi reads sparse values in CSC order only, so `casadi=True` accepts CSC
sparse inputs and outputs without batch axes and refuses any other layout. The
tables of sparse derivative outputs are described in
[sparse output patterns](codegen.md#sparse-output-patterns).
