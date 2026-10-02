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
linalg.solve(A, b)                           # Cholesky; assume="sym" uses LDL^T, assume="gen" LU
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

**Generated code.** Small orders become straight-line code, which scalar expansion keeps in
registers: without an option, those whose body is under the target's `Target.straight_line_ops`
operations (`n^3 / 3` for a factorization, `n^2` for each right-hand side of a solve; on Apple
silicon, a Cholesky factor up to order 23 and a solve of one vector up to 63), decided when the op
is rendered; with `sc.options(linalg=dict(dense_unroll=...))`, the orders up to it, decided when
the op is built. Larger orders become loops with triangular bounds that do not grow with the order:

- `cholesky` runs by 4×4 tiles of `L`. A tile's dot products over the columns to its left run
  together, each row entry loaded once for the whole tile and the sums kept in registers, each dot
  product in four partial sums over contiguous quarters of its columns. From six blocks of the
  processor's register tile (`Target.product_tile`, order 48 on Apple silicon) it goes by blocks of
  columns instead. The columns to a block's left update it in register tiles, the matrix product's
  code, with the dot products in the same four quarters added pairwise, and the block is then
  factored and solved as before, dividing by the diagonal.
- `cholesky(A, sums="running")` and `solve_triangular(T, B, sums="running")` add each dot product
  to one sum in order, as a BLAS does, and multiply by the reciprocal of a diagonal entry where the
  default divides by it. They go by blocks from two blocks of the register tile on, a solve for a
  tile's width of right-hand sides or more. With fewer right-hand sides a solve takes the
  reciprocals alone, and a single one is solved as the default solves it. A substitution that runs
  backward (`lower` with `trans`, or upper without) from four blocks on is the default one, which
  is the faster there. The default, `"pairwise"`, is the more accurate in the last pivots of an
  ill-conditioned matrix, which an interior-point solver's iteration count can hang on; the running
  factorization takes 0.62 to 0.85 of its time from 24 to 200 rows, and `linalg.blocks` uses it.
- `ldl` runs row by row, taking dot products of contiguous rows in four interleaved partial sums.
- A solve with the matrix transposed sweeps rows of the triangle, which are contiguous.
- A solve for several right-hand sides takes a row of `X` at a time, four rows of the triangle per
  pass. With a tile's width of them or more, from four blocks of the tile, it goes by blocks of rows
  the same way.

**Speed.** Against OpenBLAS LAPACK at orders 32–64 (measured on aarch64 Linux, gcc 11, in
`internal/notes/tier2_pr6_report.html`):

- At `-O3`, the Cholesky, `L D L^T`, triangular solve and matrix-product kernels are within 1.6×.
- Below order 32 the generated code is faster, because it has no library call overhead.
- At gcc 11's plain `-O2`, which does not vectorize, the multi-right-hand-side solve and the
  matrix product fell to 3–3.5×. The JIT therefore adds `-ftree-vectorize` at `-O2` for GCC before
  version 12, which vectorizes at `-O2` by itself from 12 on (see `SCALY_CC_OPT` in
  [Environment variables](env_vars.md)).

## Dense LU with partial pivoting

```python
F = linalg.lu(A)                      # P A = L U, packed with the permutation in an (n + 1, n) array
x = linalg.lu_solve(F, b)             # A^{-1} b; b a vector or a matrix of right-hand sides
y = linalg.lu_solve(F, c, trans=True) # A^{-T} c, with the same factor
linalg.solve(A, b, assume="gen")      # any nonsingular A, differentiable in A and b
```

`lu` factors any square matrix, choosing in each column the row of largest magnitude at or below
the diagonal, the first one on a tie, as LAPACK does. The packed result holds `L` (unit lower)
below the diagonal of its first `n` rows and `U` on and above it. Its last row holds the
permutation as floats, `perm[i]` being the row of `A` that became row `i`, so `L U = A[perm]`. A
singular matrix gives a zero pivot, and inf or NaN in what follows it.

**Derivatives.** The factorization has none, and differentiating through it raises.
`solve(A, b, assume="gen")` is the differentiable form. Its derivative is implicit,
`dx = A^{-1} (db - dA x)`, and in reverse mode it is one transposed solve and an outer product,
all with the one factorization. Its second derivatives are implicit too. `lu_solve` is
differentiable in `b`.

**Generated code.** While its `5 n^3 / 3` operations, the eliminations and the row swaps' selects, are
under the target's `straight_line_ops` (up to order 13 on Apple silicon), or up to
`sc.options(linalg=dict(dense_unroll=...))`, the factorization is straight-line code
with every access at a fixed address, and the row swap selects on the run-time pivot. Larger
orders loop, swapping through the pivot row's run-time address. Against LAPACK's `dgesv` from
Apple's Accelerate, the generated solve takes 0.12 of its time at order 4, 0.51 at order 8 and
about the same from order 12 to 40 (Apple M3 Max, `internal/notes/integrators_i2_report.html`).

## Sparse `L D L^T`

```python
K = sc.linalg.SparseMatrix.symbol("K", kkt_pattern)   # lower triangle, upper, or both
fact = linalg.SparseLDL(K)                     # analysis now, factorization loops in the graph
x = fact.solve(b)                              # b a vector or a matrix of right-hand sides
y = fact.solve(c)                              # the same factorization, reused
```

**Which matrices.** `SparseLDL` factors a symmetric quasi-definite matrix without pivoting, for
example a regularized KKT system `[[P + rho I, A^T], [A, -delta I]]`. Every diagonal entry must be
stored, even where its value is zero: a zero block of `SparseMatrix.block` stores nothing, and
`add_diagonal` stores the diagonal.

`SparseLDL(matrix, *, ordering="auto", schedule="auto", symbolic=None, cost=None, name=None)`:

- `symbolic` reuses an analysis (`linalg.analyze`, or another factorization's `.symbolic`); it is
  checked against the matrix's pattern.
- `cost` is the `CostModel` that chooses the loop segments of the `scan` schedule.
- `name` prefixes the generated procedures, and must differ between factorizations in one graph.
- `fact.l_values` and `fact.d` are `L` below the diagonal (CSC of the permuted matrix) and `D`, in
  the order of `fact.symbolic.perm`.

**Symbolic analysis.** When the graph is built, `linalg.analyze` chooses the ordering, the
elimination tree and the pattern of `L`. `ordering="auto"`, the default, keeps whichever of natural,
reverse Cuthill–McKee or minimum degree needs the least work. A stage-ordered MPC matrix usually
keeps its own order.

**Generated code.** `schedule="loop"` generates the factorization as one loop nest,
`schedule="scan"` as `scan`s over its columns, and `schedule="unroll"` as straight-line code. The
default, `"auto"`, unrolls when the factorization takes at most `sc.options(linalg=dict(sparse_unroll=...))`
multiply-adds and divisions (1000 by default; `fact.work` has the count) and loops otherwise.
Straight-line code has no loop overhead, which dominates small systems:

| KKT system | n | work | loops: factor / solve | straight-line: factor / solve | generation (straight-line) |
| --- | --- | --- | --- | --- | --- |
| MPC, 2 stages | 24 | 288 | 0.58 / 0.33 µs | 0.10 / 0.10 µs | 0.4 s |
| random QP 20 + 10 | 30 | 836 | 1.45 / 0.44 µs | 0.24 / 0.23 µs | 1.1 s |
| MPC, 10 stages | 104 | 1746 | 3.7 / 1.8 µs | 0.64 / 0.76 µs | 2.7 s |

(aarch64 Linux, gcc, per factorization; `internal/notes/tier2_pr9_report.html`; the loops there are
the `scan` schedule.) The loop nest of `schedule="loop"` (`linalg.ops.sparse_ldl_factor`):

- A left-looking factorization over a dense work column. The updates of column `j` come in chunks
  of up to eight columns whose rows from `j` down are the same, as a supernode's are; one pass over
  those rows applies all of them, the sum held in a register, so the work column is read and
  written once per chunk instead of once per column.
- Each column's updates keep the order of one column at a time, so the factor is the `scan`
  schedule's (to the last bit, but for the sign of a zero or of a NaN), 1.5 to 2.4 times faster
  (chunks of up to eight columns; `internal/notes/ipm_speed_report.html`).
- The solve (`linalg.ops.sparse_ldl_solve`) takes the forward sweep by chains of columns (a
  supernode's): each row below a chain is updated once for the whole chain, and the diagonal and
  the output permutation fold into the backward sweep. It too gives the `scan` schedule's values.
- The factor has no derivative of its own: `solve` differentiates implicitly and never needs one,
  and `schedule="scan"` differentiates the factorization through its loops. The looped solve is
  differentiable in its right-hand side.

The `scan` schedule:

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
contributes through its lower entry). The factorization loops themselves are never differentiated
to first or second order: second derivatives use the same rules, one level deeper. Third
derivatives in the matrix reach the factorization itself: the `loop` schedule (the default above
`sparse_unroll`) refuses them, and `schedule="scan"` takes them through its loops, where on a large
system they can exceed `sc.options(max_trajectory=...)` ([Options](options.md)). Multi-seed forward mode maps the rule over
the seeds for a vector right-hand side; with a matrix of right-hand sides it runs one seed at a time.

**Speed.** Against an up-looking C factorization of the QDLDL kind on the same matrix and analysis,
the `scan` schedule factors MPC, random QP and grid systems within 1.0–1.5× and solves within
0.5–2.1× (`internal/notes/tier2_pr8_report.html`, `tier2_review_report.html`); the `loop` schedule
factors 1.5 to 2.4 times faster than the `scan` schedule. The slowest solves are the
smallest, where two permutation gathers and the diagonal scaling are a large share of the work.

**Generation cost.** Generation grows with `nnz(L)`, not with the work of the factorization. The
analysis and the in-place proof keep column runs as ranges instead of enumerating them. At
`nnz(L)` = 153 k with 25 million multiply-adds, generating and compiling take 2.5 s.

**Refinement.** `fact.solve(b, refine=k)` adds `k` steps of iterative refinement,
`x += K^{-1}(b - K x)`, each one more solve and one product with `K`. With `tol`, the steps run in
a `while_loop` only while `||b - K x||_inf > tol * max(1, ||b||_inf)`, at most `k` of them. The
derivative is the implicit one either way. On a random QP with `delta = 1e-10`, one step takes the
residual `||b - K x||_inf` from 2e-5 to 8e-11 and two to 4e-15. The adaptive loop carries the
factor, `K` and `b` with the solution, so it copies them in once per solve; it then updates the
solution and residual in place.

**Health.** No pivoting happens, so a factorization of a matrix that is not quasi-definite, or not
regularized enough, can have a zero, tiny or wrong-signed pivot. Two checks run in the generated
code next to the factorization:

- `fact.inertia()`: the numbers of positive, negative and other (zero or NaN) pivots, as a
  `float64` vector of 3. By Sylvester's law of inertia these are the signs of the eigenvalues of
  `K`, so an SQP or interior-point step checks it against `(n, m, 0)` and raises its regularization
  when it differs (see `examples/linalg/sqp_newton_sparse.py`).
- `fact.health(*, signs=None, pivot_tol=0.0, x=None)`: a bool, true when every pivot is finite with
  `|D[j]| > pivot_tol`. With `signs` (`+1`/`-1` per row of `K`), each pivot must also have the
  expected sign, the quasi-definite pattern. With `x`, every entry of the solution must be finite.

**Sparsity.** A solve declares its own pattern ([Custom derivatives](derivatives.md#custom-derivatives)):
entry `i` of the solution depends on `b[j]` and on the entries of `K` exactly when they are in the
same connected component of `K`'s graph, and not on the factor. `sparse_jacobian` of a solve of a
block-diagonal system then colors each block separately.

**Examples.** `examples/linalg/sqp_newton_sparse.py` takes Newton steps on the KKT conditions of an
optimal-control problem, with the Hessian and the constraint Jacobian as sparse matrices and
inertia correction. `examples/linalg/kalman_update.py` updates a spatial field with a sparse information
prior through a quasi-definite system and differentiates the update: its Jacobian in the
measurements is the Kalman gain. `examples/linalg/heat_control.py` steps a heat equation implicitly with one
factorization and optimizes the heating through the solves' implicit rules;
`examples/linalg/truss_sizing.py`, `examples/linalg/lasso_admm.py`, `examples/linalg/lqr_tuning.py` and
`examples/roots/hanging_chain.py` use the dense kernels. `examples/README.md` lists them all.

## Banded systems

`linalg.banded` solves tridiagonal systems whose matrix is known when the graph is built, as a
spline fit's is: `solve_tridiagonal(lower, diag, upper, b)` with `lower[i] = A[i, i-1]`,
`diag[i] = A[i, i]` and `upper[i] = A[i, i+1]` as NumPy arrays, and
`solve_cyclic_tridiagonal(...)` with `lower[0] = A[0, n-1]` and `upper[-1] = A[n-1, 0]` in the
corners. The Thomas algorithm's pivots are computed when the graph is built and the generated code
is two `scan`s over the rows of `b`, which may have any trailing shape. Both are linear in `b` and
differentiable in it; neither pivots, so the matrix should be diagonally dominant. The cubic
interpolants of `scaly.interp` use them for Expr data along an axis of more than 40 sites.

## Stage-structured problems

`linalg.stagewise.Riccati` factors the KKT system of a linear-quadratic problem over `N` stages,

```text
minimize    sum_k 1/2 x_k' Q_k x_k + u_k' S_k x_k + 1/2 u_k' R_k u_k + q_k' x_k + r_k' u_k  +  1/2 x_N' Q_N x_N + q_N' x_N
subject to  x_{k+1} = A_k x_k + B_k u_k + c_k,   x_0 given,
```

by the backward Riccati recursion, one `scan` over the stages, and solves it for any initial state
and linear terms:

```python
from scaly.linalg.stagewise import Riccati

fact = Riccati(A, B, Q, R, QN, N=40)       # one matrix for every stage, or a stack of N
x, u, lam = fact.solve(x0, q=q, r=r, c=c)  # (N + 1, nx), (N, nu), (N + 1, nx)
fact.gains, fact.cost_to_go                # K_k with u_k = K_k x_k + k_k, and P_0 .. P_N
```

Each of `A`, `B`, `Q`, `R` and the cross term `S` (zero when omitted) is one matrix shared by every
stage or `N` of them; `Q`, `R` and `QN` enter through their symmetric parts, and each stage's
`R_k + B_k' P_{k+1} B_k` must be positive definite. `lam_k = P_k x_k + p_k` are the multipliers of
the dynamics and of the initial state, the gradient of the optimal cost-to-go.

**Derivatives.** The solution is differentiable in every matrix and linear term by the implicit
rule on the KKT system: the system is symmetric, so a tangent or a cotangent is one more solve with
the same factorization, and the derivatives in the matrices are outer products of that solve with
the solution, summed over the stages for a matrix they share. The recursion itself is never
differentiated for the solution; `gains` and `cost_to_go` differentiate through it like any other
loop. The factorization of the TinyMPC example's Riccati cache is the recursion from `P = rho I`,
and `tests/linalg/test_stagewise.py` checks the two agree.

## Block-tridiagonal systems

`linalg.blocks.BlockTridiagonalCholesky` factors a symmetric positive definite matrix of `K`
diagonal blocks of order `B`, each coupled with the next through its last `c` columns:

```text
[ D_0  E_0'            ]
[ E_0  D_1  E_1'       ]      E_k is zero outside its last c columns
[      E_1  D_2  ...   ]
```

That is the shape of the normal equations of a problem with stages once its variables are in
stage order, where a stage couples with the next through its states and not its inputs.

```python
from scaly.linalg.blocks import BlockTridiagonalCholesky

fact = BlockTridiagonalCholesky(diag, below)   # (K, B, B) and (K - 1, B, c)
x = fact.solve(rhs)                            # rhs of K * B values
fact.values, fact.diagonal                     # the factor, flat, and its diagonal
```

Only the lower triangles of the diagonal blocks are read. One step is a Cholesky of a block, a
triangular solve of the block below against its trailing `c` by `c` triangle, and a product that
updates the next block; the factorization is that step in a `scan`, and so are the two passes of
a solve, so the generated code does not grow with `K`. `solve_with(values, rhs)` solves with a
factor kept from another call. A matrix that is not positive definite leaves an entry of
`diagonal` that is not positive, or not a number, from the failing block on. The generated
interior-point solver's `"stagewise"` backend is built on it ([Solvers](solvers.md)).
