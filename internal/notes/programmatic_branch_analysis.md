# Programmatic scaly: gap analysis (historical, 2026-09-25)

Copied from the claude.ai project doc of the same name, so that Claude Code sessions can read it.
It is the gap analysis that led to the PIQP plan. Where it differs from `piqp_plan.md`, the plan
wins. In particular, Tiers 1–2 took the pure-`Expr` route: `scan`/`while_loop`, run-time-index ops
and in-place carries proven at the loop. They did not take the "authored procedures" route of
Revision 2.

## Revision 5: what a full QP solver (PIQP-style IPM) needs beyond LDL^T
This revision assumes Revision 4: a fixed pattern, `scatter_add`, `scan` and a sparse tensor library.

### Per-iteration numerics (all pure `Expr`)
- Sparse matvecs `Px`, `Ax` and `Aᵀy` for the residuals.
- KKT assembly with an iteration-varying diagonal (`Σ = Z S⁻¹ + ρI`, prox terms), then a numeric LDL^T refactorisation each iteration.
- A factor/solve split: the predictor and the corrector reuse one factorisation. Add 1–3 steps of iterative refinement.
- Max-reductions (`reduce_max`, `segment_max`) for inf-norm residuals and Ruiz scaling.
- A branch-free fraction-to-boundary step: α = min(1, τ / max(ε, max_i(−ds_i/s_i), max_i(−dz_i/z_i))).
- Ruiz equilibration: a fixed number of sweeps of `segment_max`, `sqrt` and rescaling.

### Control
- Compare, select and a bool dtype, for:
  - convergence tests;
  - adaptive μ/ρ/δ updates;
  - NaN guards;
  - infeasibility certificates.
- The outer loop, in one of two forms:
  - a bounded `scan` whose `done` flag freezes the carry;
  - a functional `while_loop`. The carry is O(n) vectors plus scalars, so copying it each iteration is negligible next to the factorisation.
- Integer outputs (status, iteration count), or doubles as a stop-gap.

### Interface and verification
- Replace `SOLVER_CALL` to the vendored PIQP with an authored Function of the same signature, plus warm-start inputs.
- Optional: implicit differentiation at the solution.
- Differential tests against the vendored PIQP and against a NumPy eager run.

## Revision 4: minimal extensions for a fixed-pattern LDL^T
With the pattern fixed at generation time, the symbolic phase (ordering, etree, patterns, update
tables) runs in Python. The numeric phase is dataflow over constant index tables.

### The numeric phase
Left-looking: for each `k` in order,
1. `L[i,k] -= L[i,j] D[j] L[k,j]` over `(i,j)` in a static list `T_k`;
2. `D[k] = L[k,k]`;
3. `L[i,k] /= D[k]`.

### What was missing, in order
0. `scatter_add` / `segment_sum`.
1. A `scan` op.
2. Ragged steps: padding, or inner bounds read from a `ptr` table.
3. In-place carry lowering.
4. Pivots: static regularisation needs nothing more.
5. Triangular solves on the same machinery.
6. A custom derivative on the solve.
7. The rest is a library, not IR: a sparse tensor type, ordering, etree, column counts and table generation.

All of these landed in Tiers 1–2.

## Revision 3: scaly-authored algorithms against hand-written solvers
The gain from specialising to a pattern alone is modest, because the numerical factorisation is memory-bound.

### The real advantages
- Optimisation across the solver–problem boundary, with constants folded.
- Features the problem does not use disappear at compile time.
- Variants of structure-exploiting algorithms are cheap to write.
- One source serves Python debugging, generic C and specialised C.
- Sensitivities come through AD and implicit differentiation.
- Memory is static.

### Costs
- You maintain a compiler.
- The generated C can differ from the eager run.
- Only a restricted Python subset is available.
- Every structure needs a code-generation step.
- There is little benefit for large n or for heuristic-heavy code.

## Revision 2: real loop algorithms, not unrolled code
Colin rejected CVXGEN-style unrolling. The target is general loop algorithms (QDLDL/PIQP style),
where specialising to a pattern means partial evaluation of the integer arrays.

- **Hazard: hash-consed program loads across stores.** This hazard was real; the in-place proofs are what handle it now.
- **Keep `Expr` pure.** This remains the rule.

## Revision 1: unrolled tracing (superseded)
Tracing LDL^T over scalars compiled correctly, but rendering cost about 350 µs per scalar node and grew super-linearly. It is useful for tiny dense blocks only. Today this is the `schedule="unroll"` path, whose generation cost is about 1.2 ms per operation (C-111).
