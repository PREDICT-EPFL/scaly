# #48: sparse derivative outputs as sparse leaves

Design: Expr-level `sparse_jacobian`/`sparse_hessian` return a PACK over the canonical pattern
(internal `_Compact` keeps coloring width and the compressed batch). Function-level builders take
`layout=` ("csc" default, "csr", or a coordinate `SparsityPattern`) and the factory puts a
`SparseLayout` on the output leaf. `output_sparsities`, `SparseJacobian`, the `_val_perm` tables,
the CasADi gather and `casadi_output_sparsities` are gone. Solver oracles declare their computed order and
`_value_order` turns the leaf layout into the descriptor's COO pattern, so plugins are unchanged.
A map of independent stages keeps the 2-D Jacobian; its batched pattern comes from `sc.vmap` of
the stage's derivative Function.

Recovered from the deleted plan before the flush. The original checked items are retained below.
The six originally unchecked items are checked only after verification below. Suite and flush rules
are `/home/ted/.scaly-suite/RULES.md` and `/home/ted/.scaly-suite/super/flush-common.md`.
Work only in this worktree. Do not push, open a pull request, or start a review. The coordinator
will arrange a Fable review of the held head.

- [x] ad/sparse.py, factory, concrete, digest, model, api, ctx, solvers, codegen
- [x] tests ad, function, codegen; examples; benchmarks harness
- [x] remaining test groups (solvers, passes, viz, typing, benchmarks, integration)
  Covered by the focused checks and full hook, with the dependency and platform skips reported below.
- [x] rebase onto main (#382, #314 landed)
  Rebased onto current origin/main, including external C libraries and seed-pair lowering.
  Resolved all six named files. Retained typed entries and external solver preparation.
- [x] differential test: SciPy CSC/CSR/COO x full/upper/lower, top level and mapped
  Three complementary cases cover CSC/full, CSR/upper and shuffled COO/lower at the top level
  and under vmap, plus all three Jacobian layouts and a numerical transpose-through-call check.
  This covers the layout and triangle choices without a Cartesian test grid, as RULES.md asks.
  The focused layout and snapshot run passed all 19 tests. Each of the six differential cases
  rejects an in-memory mutation that reverses the sparse result's stored values.
- [x] docs (write-docs): sparsity guide, codegen guide, generated_interface, versioning, ir.md, api/ad.md
  Reviewed against the implementation, corrected remaining stale descriptions, ran all 52 Python
  fences on the four affected guides, checked the sparse-header excerpt exactly, and built the site.
  The coordinator's Fable review still needs to cover the prose as part of the branch review.
- [x] roadmap edit (batched pattern line), regenerate C snapshots + op counts/source size/compile time
  The prior commits already changed the batched-pattern decision and snapshots. No snapshots
  were regenerated during this flush. All snapshots render exactly as recorded after rebase.
  Only spjac.c and spjac.h differ from origin/main, as described below.
- [ ] ty, ruff, full suite via ~/.scaly-suite/run; PR; cross-review; follow-up issue (int64 logical size)
  The initial focused run passed 274 tests. Initial ty and both Ruff checks passed.
  The first full hook stopped at collection because this branch had not updated the node-ID
  baseline. The safe collection recipe now validates 6413 IDs; its diff contains only the eight
  sparse-layout cases and the renamed mapped-Jacobian test. The detached rerun uses hook-final.log.
  Verification is complete: the rerun exited 0, with 6357 tests passed and 56 skipped in 214.94s.
  All 13 smoke checks passed, ty and both Ruff checks passed, the docs built with no issues in
  2.80s, and the private-name check passed. This checkbox remains open only for the held PR and
  coordinator-owned review. No implementation or verification item remains open.
  PR and review remain with the coordinator under the flush instructions.
  The int64 logical-size follow-up already exists as #447, verified through REST. No duplicate
  issue is needed. Its choice between supporting huge logical sizes and rejecting them clearly
  remains outside this branch.

Verification order: finish focused coverage and documentation checks, inspect and measure the
two changed snapshots, then run the full hook detached through the shared queue. Record each
result here and commit only the task's own changes before holding the head for review.

Implementation and verification are complete. The temporary plan stays because the PR and
review are deliberately held. Nothing has been pushed. Remaining design choice: #447's behaviour
for logical sizes beyond int64, already tracked outside #48. No further decision is needed for
this branch's completed scope.

## Verification evidence

The Python fences in sparsity, sparse_matrices, derivatives and codegen all ran, 52 fences in
total, including the two documented errors and loading the exported shared library with CasADi.
The mapped_derivatives, triangular_hess and sparse_jvp examples ran from temporary copies.
The focused viz, typing, benchmark and acados integration run passed 45 tests and skipped the
acados comparison because acados_template is unavailable. The existing acados example therefore
remains unverified in that environment, and no check is claimed for it.

Snapshot differences, relative to origin/main:

- `tests/baseline/c/spjac.c`: only the 36-entry k8 gather-index initializer changes. It recovers
  values in CSC order directly from the compressed derivative batch, rather than row-major
  coordinate order. All arithmetic, loops, signatures, buffers and workspace stay the same.
- `tests/baseline/c/spjac.h`: replaces the compact-derivative metadata comment with the sparse-leaf
  comment, moves NNZ after the dimensions, adds NDIM=2, STORAGE_SIZE=36 and shape={12,16}, removes
  rows, cols, csr_row_ptr, csr_col_ind, csr_val_perm and csc_val_perm, and keeps csc_col_ptr and
  csc_row_ind unchanged. A sparse output publishes only its declared layout, whose values need
  no caller-side permutation. No other snapshot differs, including main's external solver files.

Small snapshot diagnostics on la015, using main's archived sources and this worktree separately:

| Metric | origin/main | Branch |
| --- | ---: | ---: |
| C source bytes | 10625 | 10625 |
| Header bytes | 3802 | 2920 |
| Mapped floating operations per stage | 22 | 22 |
| Stage trip count | 3 | 3 |
| Mapped workspace | 0 | 0 |
| Median generation time, five renders, ms | 29.988 | 30.484 |
| C kernel compile time, ms | 69.649 | 97.177 |
| Compiled C runtime, ns | 45.010 | 44.082 |

Generation includes lowering, rendering the C body and rendering its header. C compilation and
runtime use the repository's Google Benchmark driver and verify each output against the dense
Jacobian. These are single small diagnostics under shared machine load, with no performance
claim. The source diff independently proves that arithmetic is unchanged. Raw samples, compiler
commands, environment, and the detached hook log are under the gitignored directory
`benchmarks/results/issue48-flush.9sPVJl/`.

The same directory holds one correctness-checked Google Benchmark cell per revision for the
race-car constraint Jacobian and lower Lagrangian Hessian at N=5, with scalar libm selected.
These provide the benchmark-problem source, operation, generation, compilation and runtime
evidence required by the roadmap. They are diagnostics under shared load, not a performance study
or claim. The problem source sizes and arithmetic counts are identical between revisions.
Both problem C sources are byte-identical, independently checked after the runs.

| N=5 problem and revision | Source bytes | Mapped operations per iteration | Generation ms | Kernel compile ms | Compiled runtime ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constraint Jacobian, main | 33911 | 470 | 711.2 | 260.4 | 1473.6 |
| Constraint Jacobian, branch | 33911 | 470 | 386.8 | 154.7 | 1014.0 |
| Lower Lagrangian Hessian, main | 68848 | 1188 | 1228.7 | 292.9 | 1267.3 |
| Lower Lagrangian Hessian, branch | 68848 | 1188 | 967.0 | 258.8 | 1261.0 |
