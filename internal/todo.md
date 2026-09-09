# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why we are writing what we write** — [`internal/paper.md`](paper.md): thesis, scope, narrative,
  outline, claim gates, objections.
- **Why a number is or is not admissible** — [`docs/results/fairness.md`](../docs/results/fairness.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **What shape a refactoring should take** — [`internal/notes/refactorings.md`](notes/refactorings.md):
  one `#` section per refactoring, kept until that refactoring lands.
- **Library-internal phases** — [`internal/roadmap.md`](roadmap.md).

Reorganized 2026-09-07. Sections are themes that outlive the first release. Inside each section,
**Now** holds what is actively worked on or next in line, and **Deferred** holds what is
intentionally low priority: the reasoning is still good, nothing depends on it yet. A finished item
stays in place with its box checked until it is flushed out by hand; git history and the frozen
notes hold the record after that.

### Identifiers

Every item has an identifier `<PREFIX>-<n>`. The prefix names the section the item sits in; the
number comes from one counter shared by the whole file, which only ever grows.

**Next id: 56**

| Prefix | Section |
|---|---|
| API | API |
| C | Compiler internals |
| S | Solvers |
| BH | Benchmark harness |
| BP | Benchmark problems |
| L | Licensing |
| D | Documentation |
| R | Release |

Rules:

- A new item takes the next id and bumps the counter. A deleted item never frees its number.
- Moving an item to another section changes its prefix and keeps its number. Grepping the number
  alone finds the item, or proves it is gone.
- Other documents cite the full id and the title, `S-15 Replace METIS 4 with METIS 5`, so the
  reference survives both a move and a retitle.
- A new section adds a row with a prefix that is not in the table and never was.

Why identifiers at all: they give the short stable handle that Linear or GitHub issues give, while
the list stays git-tracked, lives next to the code, and changes per branch, so a worktree can add,
close and reorder its own items and the merge carries them. A global counter rather than one per
section because items move between sections more often than expected, because eight counters are
eight places to get wrong once completed items are deleted, and because the letters then carry
only the theme and nothing else has to stay stable.

Ordering constraints across sections, the only sequencing that matters:

- C-44, C-45, C-8 to C-10 and BP-23 come before BH-20, which re-decides every claim gate in
  paper.md §8.
- L-28 to L-31 come before any wheel or tag is public, even on test PyPI.
- R-37 comes before any merge of dev into main.

## API

### Now

- [ ] **API-1. Implement `FunctionTemplate` over concrete Functions**, including specialization,
      deterministic C names, one trace per instance, and lifted derivatives. Design:
      refactorings.md "Function templates" and `typing_playground/README.md`.
- [ ] **API-2. Preserve declared trees through `vmap` and Function-level differentiation.** Settle
      the mapped input convention and retain runtime and static acceptance tests. Rationale:
      refactorings.md "`vmap` and the AD entry points erase the callee's declared trees".
- [ ] **API-3. Decide the zero-input Function contract and reduce the private flat call path.**
      Preserve legitimate parameterless solver oracles. Rationale: refactorings.md
      "Zero-input `Function`s, and the flat call seam that survives because of them".
- [ ] **API-4. Finish the npmpc `FunctionTemplate` example** after API-1. The public
      typed decorators, exact `Function` annotations, and shared Alloy/CasADi runtime parameters
      landed first. Replace the remaining decoder-architecture builders with `FunctionTemplate`.
      This benchmark may use the packed parameter length as its specialization key because it does
      not add more MLP layouts; a general template must distinguish individual layer shapes because
      equal parameter counts do not prove equal architectures.
- [x] **API-5. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.
      Lives in `tests/ad/test_vmap.py` (duality) and `tests/integration/test_vmap.py` (pairwise Hessian).

### Deferred

- **API-6. A QP-subproblem contract so alloy-sqp can use other QP plugins.** Today `alloy-sqp`
  imports only `include_dir`/`lib_dir` from `alloy_piqp` and its C template calls
  `piqp_setup/update/solve` and reads `qp->result` directly, so a future OSQP, ProxQP or HPIPM
  plugin would be a standalone solver but not an SQP backend. The contract is narrower than
  `render_wrapper`: set up a QP with fixed sparsity, refill values, solve, read the step and
  multipliers in one sign convention, report status and iteration count, clean up. The hard part is
  form reconciliation (two-sided rows and box bounds versus OSQP's single `l <= Ax <= u`, and
  stage-structured solvers) the way CasADi's `conic` layer does it. Documented as a limitation in
  `docs/guide/solver_backends.md`.
- **API-7. Specialized OCP problem/solver tier** in alloy (structured staged OCP lowering to general
  form), then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as
  claims demand.

## Compiler internals

The 2026-09-06 [sweeps](../docs/results/scalability.md#extended-hessian-sweeps-2026-09-06) fail claim
gates 2 and 3 on race-car, cap unbumpercars at C=32 on static metadata, and win npmpc and
unbumpercars at range only because the faster CasADi encodings stop compiling inside the budget.
The 2026-09-07 investigation ([`notes/perf_2026_09_07/`](notes/perf_2026_09_07/README.md)) took the
generated kernels apart with drivers and ablations. The glue between stage calls is 22% of race-car,
3% of npmpc and 26% of chain; the stage kernels carry the loss. Race-car pays for tensor-shaped code
on a scalar body and for 0/1 seeds multiplied at runtime; npmpc and unbumpercars pay for
matrix-vector products lowered as serial reductions; chain pays for 24 dense tangent vectors and
1624 loops where straight-line scalar code is 7.7 times faster. Rationale and gate status:
paper.md §8. Every item under **Now** is a general compiler change with a `tests/` reproduction,
per the 2026-08-25 decision in paper.md §10; after they land, BH-20 re-decides the gates.

Start each compiler item by reading how the tools that shaped Alloy solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Alloy's are modelled on,
expands small tensor ops into scalar UOps and simplifies them symbolically (C-44), and has a
scheduler that fuses elementwise producers into their consumers and a symbolic index arithmetic
that turns strided views into closed-form index expressions (C-8 and C-9). MLIR's affine dialect and
its loop-fusion, affine-map and memref-normalization passes are the standard treatment of exactly
the loops we emit, and their design notes state the legality conditions we would otherwise
rediscover. JAX's `vmap` batching rules are the reference for what a mapped derivative rule should
produce without materializing per-trip index tables. The goal is to port the smallest idea that
fits Alloy's two dialects, not to adopt a framework; write down what was read and what was rejected
in `internal/notes/refactorings.md` before the implementation.

### Now

Ordered by measured payoff. The numbers are the 2026-09-07 note's, on the reference machine at the
protocol's compile flags.

- [x] **C-43. Lower matmul by layout.** `_lower_matmul` emits every product as
      `for i { out[i] = 0; for k out[i] += A[i,k] v[k] }`, a serial add chain per output that the C
      compiler cannot break without reassociation; `casadi_mtimes_dense` has the same shape, which is
      why both are slow. Emit the reduction loop outermost when the reduction axis is the matrix's
      slow axis (`v @ A`, and `A.T @ v` after the fold below), so the inner loop runs over independent
      outputs and vectorizes; emit a blocked dot with four accumulators as four unrolled statements
      when the reduction axis is contiguous (`A @ v`), because a four-trip inner loop becomes gathers
      under `-march=native`. Add `A.T @ v -> v @ A` and `v @ A.T -> A @ v` to `passes/expr.py` so the
      adjoint products materialize no transpose. Each output keeps its summation order, so results
      are bit-identical. Measured with the throwaway patch: npmpc N=12 83.2 to 45.7 µs (MX 51.1),
      unbumpercars C=8 3485 to 827 µs (MX 9950). Tests: a fixture per shape class against NumPy,
      and the C snapshots updated. The patch is `notes/perf_2026_09_07/experiments.patch`. This is
      a stopgap: the two rules are the special case of a range split with one accumulator per lane
      and a choice of outermost range, which C-8 provides generically; when C-8 lands, C-43's rules
      are deleted, not kept beside it.
- [x] **C-44. Scalarize small stage bodies, driven by the `lowering` hint.** For a callee whose
      body is marked `Expr.scalar()`, or fits a conservative scalar-operation budget under `auto`, and never under
      `block`: unroll to scalar SSA, hash-cons, fold `0`, `1` and constant arithmetic, and render
      expression trees (not one op per statement, which is what stops clang from forming FMAs in
      SX's output). The hint is now active. `notes/perf_2026_09_07/scalarize_stage.py` does the
      transform after the fact on the generated C: race-car eq stage 23.0 to 18.1 µs, chain eq
      stage 1910 to 247 µs, both bit-identical. The only item that moves chain, and it subsumes the
      seed half of C-9 and most of C-10. tinygrad's expand-then-devectorize is the reference
      (`notes/perf_2026_09_07/tinygrad_rangeify.md` §4). Gate: the two stage kernels above within
      10% of the hand-scalarized time. Implemented 2026-09-08 in `passes/program/scalarize.py`; the
      [validation](notes/perf_2026_09_07/README.md#c-44-validation-2026-09-08) separates runtime
      seeds from the constant-seed control. Review follow-up replaces the tensor-width heuristic
      with limits on scalar operations after folding and sharing, expansion work, and aggregate
      generated-code growth. The arithmetic contract and exceptional constant cases are documented
      and tested. The [closeout](notes/perf_2026_09_07/README.md#c-44-closeout) records minimal
      GCC/Clang compilation and runtime checks; the full benchmark rerun follows more Track C work.
      Automatic seed specialization remains C-45.
      Design: [arithmetic policy](notes/algebraic_simplification_2026_09_08.md#proposed-alloy-arithmetic-policy);
      rationale: [paper §8](paper.md#8-blocking-work-before-the-paper-can-be-written), [measurement protocol](../docs/results/fairness.md#the-measurement-protocol).
- [x] **C-52. Split program passes into an explicitly ordered package.** Implemented in `alloy.passes.program`, with shared helpers and an explicit pipeline in place of registration side effects; pass order, observer events, and behavior are preserved. [Design](notes/algebraic_simplification_2026_09_08.md#the-architectural-decision), [rationale](paper.md#8-blocking-work-before-the-paper-can-be-written).
- [x] **C-55. Preserve intended lowering hints through derivative Function construction.** Implemented 2026-09-08: every derived `Function` built in `ad/` takes the primal callee's effective hint (`block`/`opaque` -> `block`, `scalar` -> `scalar`, `auto` inherits nothing) on its output root, through `Function._effective_lowering`; the chain check `hinted_stage_hessian` and `tests/ad/test_lowering_hints.py` pin selection. The chain benchmark stage now carries `.scalar()` (decided 2026-09-08: the comparison is against each side's best formulation, and this is ours); the M=5 Hessian kernel runs at 835 µs against 1769 µs without. Race-car gets nothing from the hint because the automatic policy already selects its stage ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)). [Observed hint loss](notes/perf_2026_09_07/README.md#c-44-closeout), [rationale](paper.md#8-blocking-work-before-the-paper-can-be-written).
- [x] **C-53. Share arithmetic simplification across both dialects and program forms.** Implemented 2026-09-08 in `passes/arith.py` (one adapter per dialect, rules for neutral elements, zero annihilation, self-cancellation, negation normalization, bounded constant powers, dtype-checked constant evaluation) and applied through `passes/expr.py`, `scalarize`, and the new `fold_arith` loop-body pass after fusion; `tests/passes/test_arith.py` runs the same cases in all three forms. Left open: `_h{n}` renderer temporaries have no collision guard and deep index expressions are not hoisted, both unobserved in practice. [Design and validation](notes/algebraic_simplification_2026_09_08.md#a-small-common-implementation), [rationale](paper.md#8-blocking-work-before-the-paper-can-be-written).
- [x] **C-45. Bake stage-invariant constant tangents into the VMAP forward callee.** Implemented
      2026-09-08 in `ad/forward.py`: a constant `jvp_many` tangent whose per-iteration tiles repeat
      with period `k <= 8` (and at least twice, so short horizons of distinct tiles are not unrolled)
      is baked into one const-seed callee per tile, each mapped over its residue class and assembled
      with stack/transpose/reshape, so no seed table or gather is emitted; other constant patterns
      keep the local-coloring and runtime-seed paths. `tests/ad/test_const_seed_bake.py` pins equal,
      periodic, and fallback tiles. Race-car N=50 measured 26.0 to 24.0 µs together with C-53/C-10,
      static metadata 107 to 90 KB ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)).
- [ ] **C-46. Hoist stage-invariant callee work out of the mapped loop.** The npmpc stage kernels
      transpose the three weight matrices on every call, 3.5 of 47 µs, because the callee cannot know
      the argument is the same at every iteration; the `u` kernel also recomputes the primal and
      adjoint passes the `x` kernel already did, because the forward rule emits one mapped callee per
      formal. A program-dialect hoist of loop-invariant statements, and one callee for all formals of
      one VMAP, or one `W @ [v1 v2 v3]` product for batched seeds. With the row-major products then
      in column-sweep form, npmpc reaches 38.6 µs under `-march=native` by hand against MX's 41.9.
- [ ] **C-47. One accumulation buffer for a sum of scatters.** Chain's entry point zero-fills 43
      buffers of 23,544 doubles and scatters 576 values into each before summing them: 8 MB of memset
      per call and the 1,075,248-double workspace. `scatter(a) + scatter(b) -> scatter_add` at the
      expression level, or the fusion in C-8. Arithmetic identities such as `0 / x` belong to C-53.
- [ ] **C-49. Audit the AD rules for operation count, starting from the measured chain split.**
      Lean division rules, `(dx - f dy) / y` forward and `q = cot / y; adj_y = -q f` reverse with `f`
      the primal quotient, took race-car N=50 from 31.4 to 28.0 µs and chain M=5 from 2458 to 2277
      with the harness check passing; that was one afternoon. What remains on chain: the tangent of
      the adjoint costs 1,100 operations per unit seed after folding, more than the whole primal
      plus adjoint (993), against at most about 400 per seed in SX. Build the same stage in CasADi
      SX as an explicit forward-over-reverse with the same 24 unit seeds and compare per-operation
      histograms to separate rule quality (2,749 negations per stage is the next suspect) from
      composition. Evidence: `notes/perf_2026_09_07/README.md`, follow-up section.
- [x] **C-50. `-march=native` and `-fno-math-errno` in the JIT.** `codegen/jit.py` compiles with
      `-O2` (or `ALLOY_CC_OPT`) and no target flag, so every JIT kernel is SSE2 scalar code without
      fused multiply-adds on a machine that has them; measured on race-car N=50, `-mfma` alone is
      32.9 to 27.8 µs. Add the two flags to the JIT compile line, check that the solver plugins'
      compile paths (alloy-sqp's wrapper, the PIQP and IPOPT hooks) still link, and keep the plugin
      wheels themselves at the portable baseline. One line if it plays well with the solvers.
- [ ] **C-51. Coalesce consecutive scalar loads and stores into vector accesses in the C renderer.**
      After C-44 scalarizes a body, adjacent `buf[i], buf[i+1], ...` accesses can be emitted as one
      clang `ext_vector_type` load or store; tinygrad's `memory_coalescing` does this in about 60
      lines (`tinygrad_rangeify.md` §6) and it is the difference between scalar code and visible
      SIMD on clang. After C-44.
- [ ] **C-8. A range-based loop compiler for the program dialect, in the shape of tinygrad's
      rangeify.** Today every mapped op materializes an `N × width` intermediate: the race-car
      Hessian is about 120 consecutive full-length loops, the chain entry point zero-fills 43 buffers
      of 23,544 doubles per call, and the caller workspace grows with N on race_cars and npmpc. The
      2026-09-08 study (`notes/perf_2026_09_07/tinygrad_rangeify.md`) says how the reference design
      gets fusion without a dependence analysis: loop variables (ranges) are first-class values;
      views become index expressions over them; a producer with one consumer inherits the consumer's
      ranges, which is fusion by construction; a producer whose consumers disagree on an axis is
      materialized on that axis only; a reduce becomes `acc init / acc op= x / END(range)`; and one
      substitution `r -> r_outer * amt + r_inner` expresses tile, unroll and upcast, with an
      accumulator per upcast lane. Port that shape, not the framework, in this order, each step with
      a `tests/` fixture: (1) ranges and the three-case propagation rule over the lowered loops, which
      is the fusion pass and the workspace fix; (2) the accumulator lowering of reductions with a
      per-lane split, which supersedes the layout-specific matmul rules C-43 landed in
      `_lower_matmul` (delete them then) and generalizes them to `W @ [v1 v2 v3]`
      and to matrix-matrix products; (3) the reduce-under-broadcast rule so a value is never
      recomputed under an expand. Gates: race-car `workspace` fixed across N in the sweep CSV, chain
      workspace under 100k doubles at M=5, npmpc within 5% of today's kernel with those rules removed.
      Measured share of runtime today: 22% of race-car, 3% of npmpc, 26% of chain, so this is the
      workspace fix and the general form of the matmul fix; C-44 and C-45 are what narrow the
      race-car and chain ratios.
- [ ] **C-9. Affine index maps instead of materialized tables.** The VMAP multi-seed forward rule in
      `ad/forward.py` builds `gather` index arrays of size `nseed × length × slice` (`flat_idx`,
      `tile_indices`), and the reverse rule in `ad/reverse.py` builds per-iteration index lists; the
      renderer emits each as a `static const int64_t` table. Their contents are affine in the trip
      index (`start + it * stride + j`), and the coloring seed tile repeats one stage-invariant 0/1
      pattern per stage. These tables are the static metadata that grows with N on race_cars and
      reaches 54 MB at unbumpercars C=32. Represent affine gathers and scatters structurally
      (a strided window or an index expression) and broadcast stage-invariant constants instead of
      tiling them. Gate: `static_metadata_bytes` fixed across N on race_cars, and the unbumpercars
      C=32 cell compiles under the 50 MiB cap. The affine folder that makes index expressions stay
      small is tinygrad's `uop/divandmod.py` (109 lines: gcd factoring, congruence folding, nested
      div/mod) plus the `(x%c) + (x//c)*c -> x` recombination; see `tinygrad_rangeify.md` §3. With
      C-45 folding the seed tiles, what remains of C-9 is the index arithmetic, and it is a
      prerequisite for C-8's views-as-index-expressions.
- [x] **C-10. Fold the identities the AD rules introduce, at the expression level.** Implemented
      2026-09-08 in `passes/expr.py`: `v @ ones -> sum(v)`, identity-index gathers become reshapes,
      uniform 0/1 masks of the result shape fold, each pinned in `tests/passes/test_expr.py`. The
      matrix forms `A @ ones` and `ones @ A` were tried as stacked row sums and reverted: in loop
      form they lower to one loop per row and lose the fused producer, slower than the matmul. They
      wait for an axis reduction in the IR, which is C-8's accumulator lowering.
- [ ] **C-11. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
      42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
      computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
      body, then a scatter. Internal workload, so lower priority than C-8 to C-10; it
      decides whether chain can ever enter the long paper's tables. The 2026-09-07 note measured the
      stage kernel at 1920 of 2581 µs and the same kernel scalarized at 247 µs with the 24 unit seeds
      folded, so C-44 goes first; what remains after it is 2.6 times SX's operation count, of which
      the per-seed division in the DIV tangent rule (1738 divisions per stage against SX's 674) is
      the one identified piece.
- [x] **C-12. One matcher and iterative rewrite driver for both dialects.** Implemented 2026-09-08: `ir/match.py` is generic over both node types with an iterative driver (`fixpoint`, `revisit`, `max_steps`), `rebuild_program` in `passes/program/_common.py` is the program adapter, and `_transform` is gone. No nested patterns or captures: no call site needed them. C-13 closed with it. [Updated design](notes/refactorings.md#shared-compiler-rewrites), [rationale](paper.md#8-blocking-work-before-the-paper-can-be-written).

### Deferred

- [ ] **C-54. Add memory-aware program common-subexpression elimination and dead-code cleanup.** Build on C-12/C-53 with definition/use tracking and conservative read/write/alias handling; retain required calls and output stores, and test repeated loads across writes. Broader loop motion follows demonstrated workload need; load-node interning alone is not a current stale-value bug. [Design](notes/algebraic_simplification_2026_09_08.md#separate-value-cleanup-from-memory-optimization), [rationale](paper.md#8-blocking-work-before-the-paper-can-be-written).
- [x] **C-13. Make the Program IR passes iterative instead of recursive.** Done 2026-09-08 with
      C-12: the program passes, `scalarize`, and the C renderer no longer recurse per expression node,
      and the renderer hoists subtrees deeper than `MAX_SCALAR_DEPTH` into temporaries so clang's
      bracket limit is not hit. Witnesses in `tests/passes/test_program.py`: left folds at 400 and
      3000, the NPMPC-shaped flat per-stage reduction at N=100, and a hinted scalar fold, each
      compiled and checked against NumPy. Recursion proportional to statement nesting (loop and call
      depth) remains and is documented in `docs/how_it_works/lowering.md`.
- **C-14. Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Alloy already wins.

## Solvers

### Now

### Deferred

- **S-16. Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with
  our hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: paper.md §5.4.
- **S-17. Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library
  search paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".
- **S-18. CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its
  globalization, regularization and QP path differ from `alloy-sqp`. Add only if review asks for it.

## Benchmark harness

### Now

- [ ] **BH-19. Harness gaps.** `dispatch_trip_count`, `dispatch_workspace` and `dispatch_arithmetic`
      are empty for the race_cars and unbumpercars Alloy cells and filled only for npmpc and chain,
      so Table 2 cannot be built from the CSV yet. Gate 4's causal claim needs a same-protocol
      pre-port control: either measure the unrolled pair rows behind a flag or drop the causal
      wording and keep the descriptive one.
- [ ] **BH-20. Rerun the study** after C-8 to C-10 and BP-23. `uv run benchmarks/run.py study
      --out-dir benchmarks/results/followup/<date>`, then paste `report.md` into the results pages
      and re-decide every gate in paper.md §8. The npmpc and unbumpercars range wins are
      compile-budget wins today; after the rerun they are either real wins against a completed
      encoding or they are labeled as budget wins in the paper.
- [x] **BH-48. Adopt `-march=native` in the benchmarks and the AOT guidance; keep distributed
      binaries portable.** Decided 2026-09-08: the sweep and closed-loop harnesses compile both
      providers with `-march=native` (and `-fno-math-errno`), fairness.md states the rule and why;
      AOT users are told in the docs to pass it and it goes in the suggested CFLAGS; the solver
      plugin wheels stay at the portable baseline. The JIT side is C-50. Measured reason: on
      race-car the gain is FMA contraction (`-mfma` alone: Alloy 32.9 to 27.8 µs, SX 21.3 to 20.7,
      because SX's one-op-per-statement code never contracts), not vector width. Record both flag
      sets in fairness.md until BH-20 reruns. Evidence: `notes/perf_2026_09_07/README.md`.
- [ ] **BH-21. Add an immutable publication mode**: clean release candidate, every raw run retained,
      and an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.

### Deferred

- **BH-22. Embedded hardware benchmarks** (Raspberry Pi / Jetson).

## Benchmark problems

### Now

- [ ] **BP-23. Vmap the unbumpercars wall rows.** `filters.py` still builds the four wall barriers
      per car in a Python loop after the pair rows were mapped, which is why Alloy's executable
      source still grows with C (145 KB at C=2 to 835 KB at C=32) and why gate 1 fails on that
      problem. Same port shape as the pair-row port, with the existing unbumpercars gates retained;
      extend `pair_jac_codegen_growth` so it fails while any row family still unrolls.

### Deferred

- **BP-24. Replace the race-car tracking NMPC with the MPFC distillation**, in the same `race_cars`
  package, and **laopt as an external baseline** for it once laopt is published.
- **BP-25. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- **BP-26. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
  authors so the width study becomes a measured closed-loop column instead of an extrapolation.
  Also worth telling them their released episode's reported cost metric cannot be reproduced from
  the trajectory it ships with.
- **BP-27. Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.**

## Licensing

Alloy and the three plugins are BSD-2-Clause. The plugin wheels also ship other people's binaries,
so each wheel carries its dependencies' license texts the way CasADi does
(`casadi/include/licenses/<dep>/LICENSE`), except that CasADi's `mumps-external` and
`metis-external` entries are the COIN-OR wrapper's EPL text rather than the real MUMPS and METIS
licenses, which we do not copy. Surveyed 2026-09-07. What we ship and what it asks of us:

| Package | Component | License | Obligation |
|---|---|---|---|
| alloy, alloy-sqp | our code | BSD-2 | none |
| alloy-piqp | PIQP, BLASFEO | BSD-2 | notice |
| | Eigen | MPL-2.0 | notice; the build must define `EIGEN_MPL2_ONLY` |
| | LDL inside PIQP (`piqp/sparse/LDL_License.txt`) | LGPL-2.1 | notice; check how it is used |
| alloy-ipopt | IPOPT | EPL-2.0 | notice, upstream source of the pinned version; a separate dynamically loaded module, so our BSD-2 is unaffected |
| | MUMPS | CeCILL-C | notice |
| | OpenBLAS (static, Linux) | BSD-3 | notice |
| | libgfortran, libquadmath | GPL-3 + GCC runtime exception | notice; the exception covers this use |
| | METIS 5.2.1 | Apache-2.0 | notice |
| | GKlib | Apache-2.0, plus two glibc-derived headers under LGPL-2.1-or-later and one BSD-3-Clause file, per its `LICENSES.md` | notice for each |

### Now

- [ ] **L-28. Root `LICENSE` (BSD-2-Clause, Tudor Oancea, 2026)**, `license = "BSD-2-Clause"` and
      `license-files = ["LICENSE"]` in the root `pyproject.toml`, and the README "License" section
      replaces "TBD".
- [ ] **L-29. Copy the same `LICENSE` into each of `plugins/alloy-{sqp,piqp,ipopt}/`** with the same
      two pyproject fields. Each plugin is its own sdist and wheel, so each needs the file in its
      own tree; a copy, not a symlink, so sdists stay correct.
- [ ] **L-30. Third-party notices generated by the build hooks.** `hatch_build.py` in `alloy-piqp`
      and `alloy-ipopt` copies each dependency's license text from the already-cloned
      `third_party/` sources into `src/alloy_{piqp,ipopt}/licenses/<dep>/` and writes a short
      `THIRD_PARTY_NOTICES.md` listing name, pinned version from `build_config.json`, license and
      upstream URL; the directory joins the wheel `artifacts`. Generating at build time keeps the
      notices from drifting from the pins. libgfortran and libquadmath are not cloned, so their
      GPL-plus-runtime-exception text is vendored once or taken from the GCC install.
- [ ] **L-31. Close the two PIQP questions** in the table: confirm `EIGEN_MPL2_ONLY`, and how the
      LDL code is linked.
- [ ] **A test that the license directory exists for every dependency named in
      `build_config.json`** in each plugin wheel's file list, shown to fail when an entry is
      removed. Plus one paragraph in `docs/dev/contributing.md`: a new vendored dependency needs a
      `build_config.json` entry and a license copied by the hook.

## Documentation

The same category of debt the fairness audit found in the results pages: prose that outran what
the code does.

### Now

- [ ] **D-32. Rewrite the user-facing documentation**, the index and the guide first. Scoped to what
      is stable today: install, core concepts, the derivative API, Functions and solvers. Leave a
      marked gap where `FunctionTemplate` (API-1) will go, and quote no numbers until BH-20.
- [ ] **D-33. Rework `docs/how_it_works/comparison.md`.** A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering-hint discussion subsequently removed from the published docs. The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [ ] **D-34. Audit the whole of `docs/` for claims that outran the implementation**, the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached.
      The lowering-hint claims have been removed. Check any surviving timing that predates the
      reference-machine rule in `AGENTS.md`.
- [ ] **D-35. Reconcile the problem READMEs with the audit.** `benchmarks/problems/*/README.md`
      still describe the CasADi columns as "same NLP, same IPOPT, same options, only the oracle
      provider differs", which the audit disproved on two counts.

### Deferred

- **D-36. GPU backend milestone definition** in `internal/roadmap.md`, the prerequisite for the
  paper's outlook becoming a claim in any later paper.

## Release

These steps make the tree public and permanent, and each is cheap to do once and expensive to redo.

### Now

- [ ] **R-37. Move `internal/paper.md` out of this repository before merging to main.** Blocking,
      and enforced: `.config/wt.toml` has a `pre-merge` check that fails while the file is tracked.

      Why it is urgent rather than tidy: the note contains the "sell only if the reruns establish
      it" list, the "do not sell" list and the objections rehearsal, which are the three things a
      reviewer should least find in our own words. Deleting it at release time does nothing, because
      the content stays in every clone's history, and excising it afterwards means
      `git filter-repo --path internal/paper.md --invert-paths`, which rewrites every SHA from its
      first appearance onward and breaks any archive link or tag that references an old one.

      The file has never been on main, but it is tracked in dev's history. Fast-forwarding this
      API branch to dev preserves that history. Before merging dev into main, move the note out
      and squash the public changes, or remove the private path from the history being published.
      Deleting the file alone does not make a fast-forward to main safe.

      The design, agreed 2026-08-25:

      - `~/dev/alloy-notes/` as its own git repo, with its own private remote for backup, holding
        `paper.md` and any later private notes. Keeping it under git matters: the note is a dated
        decision log and a plain untracked file would lose its history.
      - `notes -> /home/ted/dev/alloy-notes` as a gitignored symlink in every worktree, with an
        **absolute** target so the link keeps pointing at the one source of truth even if a tool
        copies rather than links it. `/notes/` goes in `.gitignore`.
      - A `[post-start]` step in `.config/wt.toml` that recreates the symlink, so the behaviour does
        not depend on what `wt step copy-ignored` does with symlinks.
      - A line in `AGENTS.md`: what `notes/` is, that nothing public may depend on it, and never
        `git add -f` under it.

      Properties this buys. One source of truth across every worktree and branch, which is correct
      for a planning document since the plan is not per-branch. The worst possible accident commits a
      path string, never content. And the public rationale stays public, because
      `docs/results/fairness.md` carries the methodology and contains no strategy.

      Rejected: a submodule leaks its existence and URL in a committed `.gitmodules` and is unpleasant
      with worktrees; an orphan branch leaves the objects in the same store, so a full clone still
      exposes them; `git-crypt` or `age` puts ciphertext in public history permanently, a poor risk
      profile for a document whose value is candour; an external tool loses grep-ability and
      proximity to the code, which is the whole reason the note works.

- [ ] **R-38. Windows support.** Decide the toolchain (MSVC or clang) and the target: the core JIT
      plus `alloy-sqp` and `alloy-piqp` first; `alloy-ipopt` on Windows is a separate later item
      because it drags in Fortran and its own licensing survey.
- [ ] **R-39. Finalize the name.** Decide whether `alloy` ships under that name; see
      `internal/notes/naming.md`.
- [ ] **R-40. Versioning policy.** What a minor bump promises about the generated C symbols, the
      sparsity-table prefixes and the plugin ABI; written into `docs/dev/contributing.md`.
- [ ] **R-41. Wheel building and publishing** for the workspace packages (cibuildwheel, per-package
      native builds), test PyPI first. After L-28 to L-31.
- [ ] **R-42. Freeze measurements on `0.1.0rc1`, publish `alloy-v0.1.0`** and a durable archive.
      After R-41.
