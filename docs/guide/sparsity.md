# Sparsity

A constraint Jacobian in optimal control is mostly zeros, and the zeros have structure. Computing
and storing the dense matrix wastes time and space, and the waste grows with the horizon. Scaly
works out where the nonzeros can be, colors them and generates code that computes only those.

## Where the nonzeros are

```python
x = sc.sym("x", 4)
y = sc.stack([x[0] * x[1], x[2], x[3] * x[3]])

sp = sc.jacobian_sparsity(y, x)
sp.shape      # (3, 4)
sp.nnz        # 4
sp.rows       # (0, 0, 1, 2)
sp.cols       # (0, 1, 2, 3)
```

`jacobian_sparsity` is symbolic and value-independent. It answers where a nonzero can appear from
the shape of the graph alone, so a structural zero is one that no input can make nonzero.

It covers the structural and arithmetic operations exactly, `matmul` conservatively, and a `call`
by chain rule through the callee. Propagation uses compressed sparse row (CSR) Boolean arrays
internally, so horizon-shaped analysis stores dependencies instead of a dense output-by-input mask.
Public patterns are `SparsityType` coordinate lists.

This sparsity is used to construct and evaluate compact Jacobians and Hessians. Ordinary
expression operations such as `matmul` and `dot` lower as dense arithmetic even when an operand
carries a pattern; for sparse linear algebra use [`SparseMatrix`](#sparse-matrices-as-values). A conservative local `matmul` pattern can add
possible nonzeros, but keeping each stage as a `vmap` callee keeps the assembled optimal-control
derivative block sparse. Solver wrappers consume those compact patterns without densifying them.

`SparsityType` holds the pattern as coordinates plus a shape, and converts:

```python
sp.to_mask()   # a dense boolean array, for inspection
sp.to_csr()    # (row_ptr, col_ind, val_perm)
sp.to_csc()    # (col_ptr, row_ind, val_perm)
```

The third element of `to_csr` and `to_csc` is a permutation, explained under
[the ordering rule](#the-ordering-rule).

## Compact values

```python
sj = sc.sparse_jacobian(y, x)
sj.sparsity        # the pattern
sj.values          # an Expr of shape (nnz,) holding only the nonzeros
sj.to_dense()      # scatter them back into a dense matrix expression
```

At the function level the pattern travels with the result:

```python
spj = sc.sparse_jacobian(fn, "y", "x")
spj.output_sparsities[0].nnz
```

The pattern also reaches the generated C as static index tables in the header. See
[the generated interface](../how_it_works/generated_interface.md#sparse-outputs).

## The ordering rule

The compact value buffer is in `(rows, cols)` order, and that order is not necessarily sorted.
The structured path emits nonzeros one `VMAP` piece at a time, so the coordinate list, not
row-major order, says which value belongs where.

To pair values with a sorted structure, use the permutation:

```python
row_ptr, col_ind, val_perm = sp.to_csr()
values_csr = [values[k] for k in val_perm]
```

The C header carries the same tables (`_csr_val_perm`, `_csc_val_perm`), so Python and C agree by
construction. Comparing two backends' compact buffers element by element without going through the
pattern gives a false mismatch when the values are right and the orders differ.

## How it stays cheap

A dense Jacobian pushes the whole identity through forward mode, either as one batched pass
carrying `x.size` seeds or as one pass per seed for the few operations that have no multi-seed
rule. Either way the work scales with the number of inputs. A colored Jacobian scales with the
number of colors: two columns share a color if no row has a nonzero in both, so their contributions
cannot collide and one seed recovers both.

```python
colors = sc.column_coloring(sp)   # one color per column
groups = sc.color_groups(colors)  # the column indices belonging to each color
```

For a square symmetric pattern, `sc.star_coloring(sp)` uses the symmetry needed to recover a
Hessian from compressed forward products. It is a proper coloring with no two-colored path of
three edges. It is not a distance-2 coloring, so leaves around a shared variable can reuse a color.

The default path does better than coloring the global pattern when it can. If the output is a
`VMAP` node, or a concatenation of them, over exactly the input, each piece is handled on the
callee: compute the small local pattern, color it, push a constant seed matrix through one
derivative of the callee and wrap the result back in a `VMAP`. The work is proportional to the
callee, not to the number of iterations, so a hundred-stage constraint costs about what a
one-stage constraint costs.

This is why keeping repetition as [`vmap`](functions.md#regular-repetition-vmap) instead of a
Python loop matters for anything horizon-shaped, and most of why scaly's generated sources stay
small as problems grow. See [the numbers](../results/scalability.md).

Anything that is not a `VMAP` piece falls back to coloring the global pattern, which is still far
cheaper than dense. `sc.sparse_jacobian_reference` computes the dense Jacobian and gathers from it.
It is slow and obviously correct, and it is what small tests check the fast paths against.

## Hessians

The same machinery gives compact Hessians, including Lagrangian ones:

```python
sc.sparse_hessian(fn, "f", "x")
sc.sparse_lagrangian_hessian(fn, "x")
```

The Hessian path symmetrizes its structural pattern and uses one global star coloring. This keeps
the color count constant when a formal is shared across every iteration of a `VMAP` with stride 0
(a global parameter), even though the Hessian then has a dense row and column. The structured
`VMAP` Jacobian path stays one-sided and colors each local tile with `column_coloring`.

## Declaring sparsity on an input

A `TensorType` can carry a pattern, which the decorator accepts:

```python
@sc.function(sc.TensorType((3, 4), sc.dtypes.float64, sparsity=sp))
def f(J):
    ...
```

The pattern's shape must match the tensor's exactly. This tells the graph about structure it could
not otherwise infer, most usefully that a matrix coming in from outside is sparse.

## Sparse matrices as values

`sc.linalg.SparseMatrix` is a matrix whose pattern is fixed when the graph is built and whose values are an
expression. The pattern is compressed sparse column (CSC) with sorted row indices; `values` has one
entry per stored element, in that order.

```python
P = sc.linalg.SparseMatrix.symbol("P", p_pattern)   # values are the input "P", shape (nnz,)
A = sc.linalg.SparseMatrix.symbol("A", a_pattern)
rho, delta = sc.sym("rho", ()), sc.sym("delta", ())

K = sc.linalg.SparseMatrix.block([[P.add_diagonal(rho), A.T], [A, sc.linalg.SparseMatrix.identity(m) * (-delta)]])
r = K @ z                                    # a dense vector expression
H = sc.linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_hessian(f, x))
```

A pattern can be a `SparsityType`, a boolean mask or a SciPy sparse matrix. The constructors are:

- `symbol`: values are a new input.
- `from_pattern`: values given in the pattern's own order.
- `from_coo`: coordinate triplets; repeated coordinates are summed.
- `from_dense`: the entries of a dense expression on a pattern.
- `from_scipy`: a constant matrix.
- `from_sparse_jacobian`: the result of `sparse_jacobian` or `sparse_hessian`.
- `diag`, `identity` and `zeros`.

The operations are:

- `+`, `-` (the union of the patterns), scaling by a scalar, and `*` between two matrices (the
  intersection).
- `scale_rows`, `scale_cols`, `add_diagonal` and `.T`.
- `@` with a dense vector or matrix on either side.
- `@` between two sparse matrices, whose pattern is worked out at build time.
- `block`, `tril`, `triu`, `diagonal`, `select`, `with_pattern` and `to_dense`.

Every result is again a `SparseMatrix` with a static pattern, and its values are ordinary
expressions (static gathers, products and segment sums), so everything is differentiable through
the values.

Across a `Function` boundary a sparse matrix is its values vector, and `sc.linalg.S(pattern)` declares
it with its pattern in the signature. The body then receives a `SparseMatrix`, and calls
are refused unless the matrix passed has exactly the declared pattern. See
[Sparse matrices](functions.md#sparse-matrices) in *Building functions*:

```python
@sc.function(sc.linalg.S(p_pattern), sc.linalg.S(a_pattern), (), ())
def kkt(P, A, rho, delta):
    return sc.linalg.SparseMatrix.block([[P.add_diagonal(rho), A.T], [A, sc.linalg.SparseMatrix.identity(m) * (-delta)]])
```

The result is a `SparseMatrix`, so the output is sparse too, with the pattern the body built.

The C signature is unchanged by this: `symbol`'s values vector in, and the output's `values` out,
with the output's `sparsity` as the header metadata.
