# Section 2: Sparse kernels

Packet of section 2 of the [compiler run](../plan.md), parent #292. Built by #315 (#45, carrying
#349, the forward half of #46), #367 (the derivative half of #46), #369 (#47), #406 (#359) and
#382 (#307). Still open when this was written: #433 (#420) and the simplification pass. CSC and
CSR are compressed sparse column and row storage, nnz the number of stored entries, AD automatic
differentiation, KKT the Karush-Kuhn-Tucker matrix of a quadratic program (QP).

## Where this section sits

- A user multiplies, adds, slices, concatenates, selects and sums sparse matrices, multiplies two of them (`A.T @ A`), and differentiates all of it; the generated C works on stored entries only, with no buffer of the matrices' full size.
- Reductions and dense products go through one accumulator schedule (`passes/lowering/schedule.py`), which the sparse kernels reuse.
- The feature is published: `docs/guide/sparse_matrices.md` and the API pages (#382).
- Section 3 makes derivative outputs sparse leaves (#48) and lowers seed-axis tangents over their nonzero pairs (#225). Section 7 built axis reductions on the accumulators (#66).
- Section 8 fuses kernels at lowering time (#69) and adds lanes over loaded bounds (#71) and demanded entries (#75); [What fusion should know](#what-fusion-should-know) is written for it.

## Needs the maintainer's decision

1. Public interface, published by #382: sparse leaves and layouts, pattern holes, `sc.pack`, `sc.values`, `to_dense`, `from_dense`, `sc.sparsify`, the pattern rule of each operation, and the header tables. Recommend reading the guide page first; everything below is internal.
2. Documented behaviour: a product never reads an entry outside the pattern, so an absent zero times infinity contributes nothing, while a run-time zero at a stored entry is evaluated and may give NaN. Decided by the roadmap; recommend keeping.
3. Roadmap departure, applied: a movement with a scalar or vector result, and `A @ x`, `x @ A` with a dense vector, have dense storage. Recommend keeping.
4. Roadmap departure, applied: reverse mode through sparse-sparse products uses an internal `sampled` pattern attribute on `MATMUL` instead of a new op. Recommend keeping; it is the one place an attribute changes an op's pattern rule.
5. Roadmap departure, applied in this pull request: lanes are written out at lowering, without `UNROLL` ranges; the strided schedule carries partial sums in the destination. Recommend keeping until #71.
6. Roadmap departure, applied in this pull request: region forms are one size threshold plus `LowerCtx.index_at`; dense outputs with a layout and call, map and loop results go through a temporary; small products use constant row bounds. Recommend keeping.
7. Roadmap departure, applied in this pull request: sparse tangents are one kernel per seed, not one kernel over `(nseed, nnz)`. Recommend deciding with #225.
8. Roadmap departures, applied earlier: max and min over stored entries moved to #66 (delivered), sparse `SELECT` to #359 (delivered).
9. Process: #382's review ran at low effort instead of Fable's default; #45 and #349 landed as one commit (#315) after #349 merged into its base branch. Recommend one closer read of the guide page.

## Expected gaps

- A select mask is lost through the reverse rule of a sparse-sparse product, giving NaN gradients from an inactive branch: #420 (#433, implemented, awaiting review).
- Tangent code grows with the number of seeds: #225.
- Derivative outputs keep their own value order: #48.
- A sparse producer is never fused into its consumer: #69. Loaded-bound reductions have no lanes: #71.
- Outside the sampled adjoints, entries that no consumer reads are still computed: #75.
- Dense blocks, bands and supernodes inside a pattern are not detected: #91, not started, since its roadmap question waits for a workload.
- The simplification pass over this section's code is not landed; its line counts are missing below.

## Unowned

- #45's run-time gate (npmpc N=12, unbumpercars C=8 within 5%) was never measured and is in no milestone check: filed as #436.
- No test covers fusion inside `LowerCtx` against the sparse kernels; told to section 8 for #69, not filed.

# Reference

Checked on main at `6c29ef45`. Paths are under `src/scaly/`. Functions are cited by name, since
main moves. Files: `qp_kernels.py` and `qp_kernels_output.txt` (the demo), and
`milestone_checks_output.txt`. The implementing threads' notes, with their perturbation checks
and counts, are in `internal/notes/sparse_kernels_2026_11_10/` (`forward.md`, `ad.md`,
`products.md`).

## Decisions

Hardest to undo first.

1. **The published sparse interface** (#382). Once released, the names and the header tables are
   what users write and link against. The page states the limits next to the behaviour.
2. **Structural zeros** (#349; `sparse_kernels_2026_11_10/forward.md`). Absent entries are never
   read. Scalar and block lowering agree on this, with a regression test for absent zeros beside
   infinity and for run-time stored zeros.
3. **Dense storage for scalar and vector results** (#349; the pattern rules in `ir/expr.py`). A
   rank-reducing result is dense, since a stored pattern covers the last two axes.
4. **`sampled` on `MATMUL`** (#369; `ir/expr.py::matmul`, `plans::SampledProductPlan`). The
   adjoint of `C = A @ B` with respect to `A` is `Cbar @ B.T` restricted to `A`'s pattern. The
   attribute carries the requested pattern through differentiation and lowering, so the full
   Boolean product is never built. Rejected: a new op, against decision 2's closed language.
5. **One schedule for reductions** (#315; `schedule::emit_reduction`). Unit stride and constant
   bounds: four lanes written out, one accumulator each, plus a tail. Strided access: partial sums
   carried in the destination, reduce ranges outermost after batch axes. Loaded bounds: outputs
   outside, one ordered accumulator, no lanes. Each output keeps the reference summation order.
   Operation counts are unchanged in every C snapshot; run time is unmeasured (#436).
6. **Compact maps partitioned at build time** (#349; `plans::map_regions`, `plans::emit_map`). The
   union and intersection regions are computed in NumPy; generated code compares no indices.
7. **Layout permutations folded into tables** (#349; `ctx::LowerCtx.physical_slots`). Sparse
   outputs are written in their declared layout. Dense outputs with a layout keep main's
   temporary-and-reorder path: the first version aliased them and wrote row-major values into
   column-major buffers, which main's tests caught when #315 was rebased.
8. **Constant row bounds for small products** (#349). Below the automatic scalarization budget a
   row uses its own slice of the stored entries, so small products still become straight-line
   code. Found by the Opus review as a regression against section 1's lowering.
9. **Column accumulator for sparse-sparse products** (#369; `plans::lower_sparse_product`). One
   private column of `m` entries, cleared per column on statically known rows; tables grow with the
   stored entries of the operands and the product, never with product pairs.
10. **Sampled adjoint of mixed products** (#367; `reverse::_stored_matmul_vjp`). The cotangent is
    computed on the sparse operand's pattern; the dense outer product is never built.
11. **Sparse `SELECT`** (#406). The union of the branches' patterns, narrowed per entry under a
    constant condition; masked partials from #30 are kept on stored coordinates.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/sparse_kernels/qp_kernels.py`, on the
Maros-Meszaros problem CVXQP1_S with `P` and `A` as CSC inputs. One `Function` assembles the KKT
matrix and computes `P x`, `A x`, `Aᵀ y` and `Aᵀ diag(w) A`; a second gives the gradient of
`yᵀ A x` with respect to `A`. Output on main at `6c29ef45`:

```text
CVXQP1_S: P (100, 100) nnz 672, A (150, 100) nnz 248
KKT (250, 250) nnz 1318 (dense would be 62500), max |K - SciPy| = 0.0e+00
P x    max error 0.0e+00
A x    max error 0.0e+00
A^T y  max error 0.0e+00
A^T diag(w) A: nnz 392 (SciPy 392), max error 1.4e-14
d(y'Ax)/dA: nnz 248 (A has 248), max error 0.0e+00
directional derivative in A's values: finite difference 15.88958521, gradient 15.88958520
qp        largest private buffer 920 entries (m*n = 15000, n*n = 10000); C source 49407 bytes
gradient  largest private buffer 0 entries (m*n = 15000, n*n = 10000); C source 6675 bytes
```

The two Milestone 1 checks this section completes pass on the same commit
(`milestone_checks_output.txt`): KKT assembly and matrix-vector products on all 48 problems of
devrush's Maros-Meszaros subset with no dense buffer and Jacobian patterns equal to devrush's
(`tests/integration/test_sparse_kkt.py`, 48 passed), and sparse algebra against SciPy with finite
differences (215 passed across the five sparse test files of `tests/ad` and
`tests/passes/lowering`).

## What fusion should know

For #69, which decides fusion in `LowerCtx.read`.

- Sparse kernels do not go through `LowerCtx.read` or `write`: both raise on a sparse operand.
  Kernels load through `raw_buf_of`, `read_stored` and `stored_view`. A hook placed only in `read`
  never reaches a sparse consumer, so every sparse producer is materialized today.
- A sparse traversal is a range from `nest` or `reduce` whose bounds may be loaded (`ptr[row]` to
  `ptr[row + 1]`). Every index map goes through `LowerCtx.index_at`. One map node is several
  loops, one per region, and an absent operand reads the constant 0.
- `widen_ranges::_ordered_reduction` matches only a `REDUCE` range whose single statement is an
  undeclared `acc = acc + t`. `hoist_invariant::_split` moves the statements that touch a scalar
  `ASSIGN` local together or not at all. `fuse_ranges::_loops` refuses `REDUCE`, carried and
  non-constant ranges.
- Two kernels keep scratch state across a SERIAL range that must never be split, reordered or
  fused: the column of `plans::lower_sparse_product`, and the slot row of
  `plans::_lower_sampled_product`, which is -1 between rows.
- Inlining a producer into a sparse consumer that does not share its traversal breaks four
  things. Zeros: the producer must yield exactly 0 off its own pattern, and region membership is
  fixed at build time. Layout: an intermediate has no entry in `_orders`, so slot maps must be
  composed. Work: the value is recomputed per stored entry times the dense width in a row loop,
  and per contributing product in the column accumulator. Lanes: a reduce body that is no longer
  one `acc = acc + t` stops matching `_ordered_reduction`.
- The three contract tests of #46 (`test_accumulating_scatter_range_is_never_widened`,
  `test_carried_range_is_never_split_into_lanes`,
  `test_loaded_bound_reduce_is_never_fused_or_widened`, in
  `tests/passes/lowering/test_sparse_plans.py`) build programs by hand and call `widen_ranges`
  and `fuse_ranges` directly. They say nothing about fusion inside `LowerCtx`.

## Reading list

About 285 lines, in reading order.

1. `passes/lowering/ctx.py::LowerCtx.index_at` (about 48 lines): constant, affine arithmetic or a
   residual table for each index map.
2. `passes/lowering/plans.py::stored_slots` (11): logical coordinate to stored slot, -1 for a
   structural zero.
3. `plans.py::map_regions` (11): the build-time split by which operands are present.
4. `plans.py::emit_map` (31): straight-line code or one loop per region.
5. `passes/lowering/schedule.py::emit_reduction` (63): the three accumulator schedules.
6. `passes/lowering/contraction.py::_lower_stored_matmul` (50): the CSR and transposed
   traversals, constant or loaded row bounds.
7. `plans.py::lower_sparse_product` (32): the column accumulator.
8. `ad/reverse.py::_stored_matmul_vjp` (38): the sampled adjoint.

Second: `ctx.py::LowerCtx.register_outputs` and `physical_slots` (layouts),
`plans.py::_lower_sampled_product`, `passes/program/widen_ranges.py::_ordered_reduction`, and the
`MATMUL` branch of `ad/forward.py::_pushforward_rule`.

## Source lines

Under `src/`, by squash commit: #315 +500 −217, #367 +117 −26, #406 +144 −36, #369 +180 −11;
941 added and 290 removed in all. After the simplification pass: not available. The pass runs on
`claude-fable-5-2` (thread "Section 2 simplification pass"); its first batch leaves all recorded
programs and C sources identical to main's, and it had not opened a pull request when this was
written, because usage limits stopped the run's threads twice on 2026-11-11. This section and
#433's state are to be updated when both land.

## Least sure of

- **The run-time effect of the schedule.** #315 changed the emitted C of every dense reduction and
  product, and only operation counts, source size and compile time were compared (#436).
- **Interactions with features that landed in parallel.** Two real defects came from them, both
  found late: dense column-major layouts against direct output writes (found by main's tests at
  a rebase), and select masks against sparse-sparse adjoints (#420, found by the #47 thread at a
  rebase). Nobody has tested the sparse kernels inside `LOOP`, with float32 leaves, or through
  `sc.extern`.
- **The rebases.** #315, #367 and #369 were each rebased several times over conflicting changes
  in `ad/reverse.py`, `ad/forward.py`, `passes/expr.py` and `hoist_invariant.py`. Each push had a
  full suite, and the code-level resolutions I followed had a short Opus check; I did not verify
  that for two pushes of #315 (heads `7218623d` and `ab4129cd`). The original reviews read
  earlier heads.
- **Per-seed tangent kernels.** They are correct and share tables, but generated code grows with
  the number of seeds, and the roadmap's `(nseed, nnz)` form was not built.
- **Duplicated helpers.** The stored-slot lookup is written five times and the row-pointer table
  builder five times across `plans.py`, `contraction.py`, `reduce.py`, `ad/` and `ir/support.py`.
  The simplification pass has the list.
