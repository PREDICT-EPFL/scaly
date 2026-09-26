# Linear algebra

`scaly.linalg` holds linear algebra written as Scaly expressions. Like the rest of a graph, it is
generated code: no BLAS, LAPACK or sparse library is called. It has two layers: dense
factorizations and solves as expression ops, and sparse matrices with patterns fixed when the
graph is built ([Sparse matrices as values](sparsity.md#sparse-matrices-as-values)).

## Dense factorizations and solves

```python
from scaly import linalg

L = linalg.cholesky(A)                       # A = L L^T, lower; reads the lower triangle of A
F = linalg.ldl(K)                            # K = L D L^T without pivoting, packed in one matrix
x = linalg.solve_triangular(L, b)            # L x = b
y = linalg.solve_triangular(L, B, trans=True)  # L^T Y = B, B a matrix of right-hand sides

linalg.cho_solve(L, b)                       # A^{-1} b from the Cholesky factor
linalg.ldl_solve(F, b)                       # K^{-1} b from the packed LDL^T factor
unit_l, d = linalg.ldl_unpack(F)
linalg.solve(A, b)                           # Cholesky; assume="sym" uses LDL^T
```

**Which triangle is read.** `cholesky` and `ldl` read only the lower triangle of their argument.
`solve_triangular(T, B, lower=True, trans=False, unit_diagonal=False)` reads only the triangle it
is told to, and reads its diagonal only when `unit_diagonal` is false.

**The packed `ldl` result.** It holds the unit lower factor below the diagonal and `D` on it. It is
meant for quasi-definite matrices: a positive definite block and a negative definite block, as in
a regularized KKT system, where every leading pivot is nonzero without pivoting.

**Errors are not checked.** A zero pivot gives inf or NaN, and a Cholesky factorization of an
indefinite matrix gives NaN.

**Derivatives.** All three ops are differentiable in forward mode (one seed or many), in reverse
mode and to second order.

- A factorization's derivative treats the lower triangle it reads as that of a symmetric matrix.
  With `X = L^{-1} dA L^{-T}`:
  - Cholesky: `dL = L (tril(X) - diag(X)/2)`.
  - `L D L^T`: `dD = diag(X)` and `dL = L stril(X) D^{-1}`.
- A solve differentiates as `op(T) dX = dB - op(dT) X`.
- Their sparsity is conservative: the factor's lower triangle may depend on all of the lower
  triangle it reads, and each column of a solution on its whole column of right-hand sides.

**Generated code.** Orders up to `DENSE_UNROLL` (8) become straight-line code, which scalar
expansion keeps in registers. Larger orders become loops with triangular bounds that do not grow
with the order:

- The factorizations run row by row, taking dot products of contiguous rows.
- Dot products use four interleaved partial sums.
- A solve with the matrix transposed sweeps rows of the triangle, which are contiguous.
- Several right-hand sides are handled a row of `X` at a time, four rows of the triangle per pass.

**Speed.** Against OpenBLAS LAPACK at orders 32–64 (measured on aarch64 Linux, gcc 11, in
`internal/notes/tier2_pr6_report.html`):

- At `-O3`, the Cholesky, `L D L^T`, triangular solve and matrix-product kernels are within 1.5×.
- Below order 32 the generated code is faster, because it has no library call overhead.
- At gcc's `-O2`, which does not vectorize before gcc 12, the multi-right-hand-side solve and the
  matrix product fall to 3× (see `SCALY_CC_OPT` in [Environment variables](env_vars.md)).

## Sparse `L D L^T`

```python
K = sc.SparseMatrix.symbol("K", kkt_pattern)   # lower triangle, upper, or both
fact = linalg.SparseLDL(K)                     # analysis now, factorization loops in the graph
x = fact.solve(b)                              # b a vector or a matrix of right-hand sides
y = fact.solve(c)                              # the same factorization, reused
```

**Which matrices.** `SparseLDL` factors a symmetric quasi-definite matrix without pivoting, for
example a regularized KKT system `[[P + rho I, A^T], [A, -delta I]]`.

**Symbolic analysis.** When the graph is built, `linalg.analyze` chooses the ordering, the
elimination tree and the pattern of `L`. `ordering="auto"`, the default, keeps whichever of natural,
reverse Cuthill–McKee or minimum degree needs the least work. A stage-ordered MPC matrix usually
keeps its own order.

**Generated code.**

- The factorization is one `scan` per segment of columns. Each step is a left-looking column update
  that reads the analysis tables by the step number.
- The step's inner loop is a `ragged_add`: for each column `k` in row `j` of `L`, one loop of
  run-time length over the contiguous part of column `k` below row `j`. This is the loop hand-written
  sparse factorizations have, and it keeps the tables the size of `K` and `L`.
- Updates go to a single carry vector, which the loop proves safe to overwrite in place, so no step
  copies anything.
- The loops index without bounds checks.
- `solve` is a permutation, two in-place triangular sweeps and a diagonal scaling.

**Derivatives.** `fact.solve` carries the implicit derivative. In forward mode,
`dx = K^{-1}(db - dK x)` is one more solve with the same factor. In reverse mode, `bbar =
K^{-1} xbar` and `Kbar = -bbar x^T` on the entries the factorization reads (a mirrored pair
contributes through its lower entry). The factorization loops themselves are never differentiated.
Second derivatives use the same rules, and multi-seed forward mode maps the rule over the seeds.

**Speed.** Against an up-looking C factorization of the QDLDL kind on the same matrix and analysis,
MPC, random QP and grid systems factor within 1.0–1.4× and solve within 0.5–1.6×
(`internal/notes/tier2_pr8_report.html`).

**Generation cost.** Generation grows with `nnz(L)`, not with the work of the factorization. The
analysis and the in-place proof keep column runs as ranges instead of enumerating them. At
`nnz(L)` = 153 k with 25 million multiply-adds, generating and compiling take 2.5 s.
