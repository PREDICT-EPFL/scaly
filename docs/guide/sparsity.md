# Sparsity

A constraint Jacobian in optimal control is mostly zeros, and the zeros have structure. Computing
and storing the dense matrix wastes both time and space, and the waste grows with the horizon.
Alloy treats sparsity as a first-class property: it works out where the nonzeros can be, colors
them, and generates code that computes only those.

## Where the nonzeros are

```python
x = al.sym("x", 4)
y = al.stack([x[0] * x[1], x[2], x[3] * x[3]])

sp = al.jacobian_sparsity(y, x)
sp.shape      # (3, 4)
sp.nnz        # 4
sp.rows       # (0, 0, 1, 2)
sp.cols       # (0, 1, 2, 3)
```

`jacobian_sparsity` is symbolic and **value-independent**: it answers where a nonzero *can* appear,
from the shape of the graph alone, with no numbers involved. That is what makes it safe — a
structural zero is one no input can make nonzero, not one that happened to be zero this time.

It covers the structural and arithmetic operations exactly, `matmul` conservatively, and a `call`
by chain rule through the callee.

`SparsityType` holds the pattern as coordinates plus a shape, and converts:

```python
sp.to_mask()   # a dense boolean array, for inspection
sp.to_csr()    # (row_ptr, col_ind, val_perm)
sp.to_csc()    # (col_ptr, row_ind, val_perm)
```

The third element of `to_csr` and `to_csc` is a permutation — more on that below.

## Compact values

```python
sj = al.sparse_jacobian(y, x)
sj.sparsity        # the pattern
sj.values          # an Expr of shape (nnz,) — only the nonzeros
sj.to_dense()      # scatter them back into a dense matrix expression
```

At the function level, ask for it through the factory and the pattern travels with the result:

```python
spj = al.spjacobian(fn, "x", "y")
spj.output_sparsities[0].nnz
```

The pattern also reaches the generated C, as static index tables in the header. See
[the ABI](../how_it_works/c_abi.md#sparse-outputs).

## The ordering rule

**The compact value buffer is in `(rows, cols)` order, and that order is not necessarily sorted.**

This is the one thing to get right when consuming a compact derivative. The structured path emits
nonzeros piece by piece — one `map` piece at a time — so the coordinate list, not row-major order,
says what value belongs where.

To pair values with a sorted structure, use the permutation:

```python
row_ptr, col_ind, val_perm = sp.to_csr()
values_csr = [values[k] for k in val_perm]
```

The C header carries the same tables (`_csr_val_perm`, `_csc_val_perm`), so Python and C agree by
construction. Comparing two backends' compact buffers element by element without going through the
pattern is the classic way to get a false mismatch — the values are right and the orders differ.

## How it stays cheap

A dense Jacobian pushes the whole identity through forward mode — one batched pass carrying
`x.size` seeds, or one pass per seed for the few operations that have no multi-seed rule. Either
way the work scales with the number of inputs. A colored Jacobian scales with the number of
*colors* instead: two columns share a color if no row has a nonzero in both, so their contributions
cannot collide and one seed recovers both.

```python
colors = al.column_coloring(sp)   # one color per column
groups = al.color_groups(colors)  # the column indices belonging to each color
```

Alloy's default path does better than coloring the global pattern, when it can. If the output is a
`map` node — or a concatenation of them — over exactly the input, each piece is handled on the
*callee*: compute the small local pattern, color that, push a constant seed matrix through one
derivative of the callee, and map the result back. The work is proportional to the callee, not to
the number of iterations, so a hundred-stage constraint costs about what a one-stage constraint
costs.

That is why keeping repetition as [`map_`](functions.md#regular-repetition-map_) rather than a
Python loop matters for anything horizon-shaped, and most of why alloy's generated sources stay
small as problems grow. See [the numbers](../results/scalability.md).

Anything that is not a `map` piece falls back to coloring the global pattern, which is still much
better than dense. `al.sparse_jacobian_reference` computes the dense Jacobian and gathers from it:
slow, obviously correct, and what small tests check the fast paths against.

## Hessians

The same machinery gives compact Hessians, including Lagrangian ones:

```python
al.sphessian(fn, "x", "f")
al.sparse_lagrangian_hessian(fn, "x", ["f", "g"])
```

One case degrades. A formal shared across every iteration of a `map` with stride 0 — a global
parameter riding along — makes the Hessian dense in that row and column. One-sided column coloring
then needs one color per iteration, and the generated source for that Hessian grows with the map
length. The values stay exact; it is the code size and the compile time that suffer. Symmetric or
star coloring would fix it and is not implemented.

## Declaring sparsity on an input

A `TensorType` can carry a pattern, which the decorator accepts:

```python
@al.function("f", {"J": al.TensorType((3, 4), al.dtypes.float64, sparsity=sp)})
def f(J):
    ...
```

The pattern's shape must match the tensor's exactly. This tells the graph about structure it could
not otherwise infer — most usefully, that a matrix coming in from outside is sparse.
