# Tier 2 — implementation status (done, tag `tier2-complete`)

Copied from the claude.ai project doc of the same name, so that Claude Code sessions can read it.
Branch `claude/tier2-sparse` (on `claude/tier1-primitives`), todo items C-101 … C-113. Reports:
`tier2_pr{1..9}_report.html` and `tier2_review_report.html` (the Tier 2 summary). Timings:
`perf_2026_09_26_tier2/`.

## What Scaly can do now (all generated code)
- Loop bodies can read the step number (`scan`/`while_loop(index=True)`).
- `take`/`put`/`put_add` index with run-time int64 indices; `ragged_add`/`ragged_dot` run loops of run-time length.
- In-place loop carries are proven at the loop from its index tables. Since T2-R, this also covers bodies whose indices are all constants.
- `scaly.linalg.SparseMatrix`: a static CSC pattern with `Expr` values, supporting algebra and block/KKT assembly.
- `linalg.analyze`: orderings (natural, RCM, MMD, auto), etree, the pattern of L, and the segment DP.
- `linalg.SparseLDL`: quasi-definite LDLᵀ without pivoting.
  - `schedule="scan"` gives loops; `"unroll"` gives straight-line code, up to `sc.options(sparse_unroll=1000)`.
  - `solve(b, refine=k, tol=)`, with implicit derivatives to second order.
  - `inertia()`, `health(signs=, pivot_tol=, x=)`, and a declared per-component sparsity.
- Dense: `cholesky`, `ldl`, `solve_triangular`, `cho_solve`, `ldl_solve`, `solve`.
- `sc.options(dense_unroll=, sparse_unroll=, max_trajectory=)`; `custom_derivative(sparsity=)`.
- Examples: `examples/sqp_newton_sparse.py` (Newton steps with inertia correction) and `examples/kalman_update.py`.

## Gates
| Gate | Result |
|---|---|
| Accuracy vs SciPy across δ = 1e-4 … 1e-13 | met; refinement reaches rounding at δ = 1e-10 |
| ≤ 2× QDLDL-class C | met: 1.04–1.5× with loops; 4–5× faster than C unrolled on small KKTs |
| Generation < 10 s at nnz(L) 50 k | met: 1.2 s at 46 k |
| Dense ≤ 2× LAPACK at n = 32–64 | met at -O3; 3–3.5× at gcc -O2 (C-107) |
| Two non-PIQP examples | met |

## Review round (T2-R, C-113)
Five agents reviewed the Tier 2 work and found 10 defects, all fixed with regression tests:
- generated names could shadow inputs;
- `-0.0` was lost in constant tables and C literals;
- the `vmap` sparse Jacobian bypassed a forward rule;
- `put`'s reverse rule was wrong with repeated indices;
- second derivatives of `x**p` were NaN at 0;
- `symbolic=` was never checked against the matrix;
- solve variants collided on names;
- hoisting separated a while loop's count from its loop;
- `jvp` formed the factorization's tangent;
- a declared sparsity pattern outlived the rules it described.

Optimizations: a JIT call went from 4.2 to 3.2 µs, adaptive refinement now runs in place, and the
quadratic parts of `fuse_elementwise` and `pack_workspace` are gone. At the end of Tier 2, 1 460
tests pass in the Linux VM.

## Open items handed on
- **C-107** (decision): whether `-O3` becomes the default.
- **C-111 / C-115 / C-116:** generation and compile time of unrolled graphs.
- **C-112:** loop-invariant `while_loop` inputs, which the Tier 3 outer loops need.
- **C-114:** unpadded factor updates (1.6–1.7× faster in a prototype).
- **C-117:** dense `double2` kernels and small-solve micro-fixes.
