# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why a number is or is not admissible** — [`internal/notes/benchmark_protocol.md`](notes/benchmark_protocol.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **What shape a refactoring should take** — [`internal/notes/refactorings.md`](notes/refactorings.md):
  one `#` section per refactoring, kept until that refactoring lands.

Reorganized 2026-09-07. Sections are themes that outlive the first release. Inside each section,
**Now** holds what is actively worked on or next in line, and **Deferred** holds what is
intentionally low priority: the reasoning is still good, nothing depends on it yet. A finished item
stays in place with its box checked until it is flushed out by hand; git history and the frozen
notes hold the record after that.

### Identifiers

Every item has an identifier `<PREFIX>-<n>`. The prefix names the section the item sits in; the
number comes from one counter shared by the whole file, which only ever grows.

**Next id: 86**

| Prefix | Section |
|---|---|
| API | API |
| C | Compiler internals |
| S | Solvers |
| CAPI | C API |
| BH | Benchmark harness |
| BP | Benchmark problems |
| L | Licensing |
| D | Documentation |
| R | Release |

The Track C closeout at the end groups active tasks across sections without changing their identifiers.

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

Priority order from 2026-09-11, the road to a public repository and a first alpha. Each step is
cheap once and expensive to redo, so the order is the sequencing that matters:

1. R-40 versioning policy.
2. D-32 to D-34 documentation rewrite, D-68 acknowledgements and AI disclosure, D-65 the two
   docs deployments.
3. R-67 make the repository public, after running the suite, ruff and ty locally on the rewritten
   tree.
4. R-66 platform-only wheel tags, R-41 wheels on test PyPI, R-42 `0.1.0a1`.

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
      typed decorators, exact `Function` annotations, and shared Scaly/CasADi runtime parameters
      landed first. Replace the remaining decoder-architecture builders with `FunctionTemplate`.
      This benchmark may use the packed parameter length as its specialization key because it does
      not add more MLP layouts; a general template must distinguish individual layer shapes because
      equal parameter counts do not prove equal architectures.
- [x] **API-5. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.
      Lives in `tests/ad/test_vmap.py` (duality) and `tests/integration/test_vmap.py` (pairwise Hessian).

- [x] **API-70. Infer `sc.vmap` slicing from sizes.** A bare outer tensor of `length * formal.size`
      is cut into contiguous chunks and one of `formal.size` is broadcast; the user positions data
      with ordinary slicing and the dict form names each slice by callee input. The
      `(outer, start, stride)` tuple stays for overlapping windows. No axis convention: outers stay
      rank-1 until the IR carries arbitrary-rank tensors and the loop compiler (C-8) lands, at which
      point a rank-2 outer mapped over its leading axis is a strict extension.
- [x] **API-69. Shorter numerical solver calls.** `sc.solver` returns a `Solver` whose call is
      `solve(params, *, x0=None, warm=None)`: missing groups default to zeros, `warm` takes a
      previous result since the four outputs are the first four inputs. `.function` holds the
      plain `Function` for `write_module`, the `scaly_codegen` CLI, `input_names` and nested
      symbolic calls; `write_module` and the CLI accept either. The C ABI does not change.
- [x] **API-83. Retire `Function._from_exprs` from ordinary tests.** Converted 264 calls to
      `@sc.function` bodies. Nine remain for constructor checks, sparse or lowering metadata,
      and zero-input hosts. Tests use `walk_program`, `find_c_compiler`, and
      `sc.jacobian_sparsity` in place of the audited private helpers.

### Deferred

- **API-6. A QP-subproblem contract so scaly-sqp can use other QP plugins.** Today `scaly-sqp`
  imports only `include_dir`/`lib_dir` from `scaly_piqp` and its C template calls
  `piqp_setup/update/solve` and reads `qp->result` directly, so a future OSQP, ProxQP or HPIPM
  plugin would be a standalone solver but not an SQP backend. The contract is narrower than
  `render_wrapper`: set up a QP with fixed sparsity, refill values, solve, read the step and
  multipliers in one sign convention, report status and iteration count, clean up. The hard part is
  form reconciliation (two-sided rows and box bounds versus OSQP's single `l <= Ax <= u`, and
  stage-structured solvers) the way CasADi's `conic` layer does it. Documented as a limitation in
  `docs/guide/solver_backends.md`.
- **API-7. Specialized OCP problem/solver tier** in scaly (structured staged OCP lowering to general
  form), then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as
  claims demand.

## Compiler internals

The [completed study](../docs/benchmarks/index.md) supplies the current measurements, and §8 of
The 2026-09-07 investigation under
[`notes/perf_2026_09_07/`](notes/perf_2026_09_07/README.md) remains the rationale and validation
record for the completed compiler tasks below.

Start each compiler item by reading how the tools that shaped Scaly solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Scaly's are modelled on,
expands small tensor ops into scalar UOps and simplifies them symbolically (C-44), and has a
scheduler that fuses elementwise producers into their consumers and a symbolic index arithmetic
that turns strided views into closed-form index expressions (C-8 and C-9). MLIR's affine dialect and
its loop-fusion, affine-map and memref-normalization passes are the standard treatment of exactly
the loops we emit, and their design notes state the legality conditions we would otherwise
rediscover. JAX's `vmap` batching rules are the reference for what a mapped derivative rule should
produce without materializing per-trip index tables. The goal is to port the smallest idea that
fits Scaly's two dialects, not to adopt a framework; write down what was read and what was rejected
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
      Design: [arithmetic policy](notes/algebraic_simplification_2026_09_08.md#proposed-scaly-arithmetic-policy);
      rationale: [measurement protocol](notes/benchmark_protocol.md#measurement-protocol).
- [x] **C-52. Split program passes into an explicitly ordered package.** Implemented in `scaly.passes.program`, with shared helpers and an explicit pipeline in place of registration side effects; pass order, observer events, and behavior are preserved. [Design](notes/algebraic_simplification_2026_09_08.md#the-architectural-decision).
- [x] **C-55. Preserve intended lowering hints through derivative Function construction.** Implemented 2026-09-08: every derived `Function` built in `ad/` takes the primal callee's effective hint (`block`/`opaque` -> `block`, `scalar` -> `scalar`, `auto` inherits nothing) on its output root, through `Function._effective_lowering`; the chain check `hinted_stage_hessian` and `tests/ad/test_lowering_hints.py` pin selection. The chain benchmark stage now carries `.scalar()` (decided 2026-09-08: the comparison is against each side's best formulation, and this is ours); the M=5 Hessian kernel runs at 835 µs against 1769 µs without. Race-car gets nothing from the hint because the automatic policy already selects its stage ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)). [Observed hint loss](notes/perf_2026_09_07/README.md#c-44-closeout).
- [x] **C-53. Share arithmetic simplification across both dialects and program forms.** Implemented 2026-09-08 in `passes/arith.py` (one adapter per dialect, rules for neutral elements, zero annihilation, self-cancellation, negation normalization, bounded constant powers, dtype-checked constant evaluation) and applied through `passes/expr.py`, `scalarize`, and the new `fold_arith` loop-body pass after fusion; `tests/passes/test_arith.py` runs the same cases in all three forms. Left open: `_h{n}` renderer temporaries have no collision guard and deep index expressions are not hoisted, both unobserved in practice. [Design and validation](notes/algebraic_simplification_2026_09_08.md#a-small-common-implementation).
- [x] **C-45. Bake stage-invariant constant tangents into the VMAP forward callee.** Implemented
      2026-09-08 in `ad/forward.py`: a constant `jvp_many` tangent whose per-iteration tiles repeat
      with period `k <= 8` (and at least twice, so short horizons of distinct tiles are not unrolled)
      is baked into one const-seed callee per tile, each mapped over its residue class and assembled
      with stack/transpose/reshape, so no seed table or gather is emitted; other constant patterns
      keep the local-coloring and runtime-seed paths. `tests/ad/test_const_seed_bake.py` pins equal,
      periodic, and fallback tiles. Race-car N=50 measured 26.0 to 24.0 µs together with C-53/C-10,
      static metadata 107 to 90 KB ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)).
- [x] **C-47. One accumulation buffer for a sum of scatters.** Chain's entry point zero-filled 43
      buffers of 23,544 doubles and scattered 576 values into each before summing them: 8 MB of memset
      per call and the 1,075,248-double workspace. `passes/program/combine_scatter_sums.py` already
      accumulated such sums into one buffer but required every operand to have the sum's declared
      shape, and chain's scatters are flat `(23544,)` buffers read through a `(24, 981)` reshape.
      Comparing element counts instead (2026-09-09) lets the pass fire: chain M=5 Hessian workspace
      1,075,248 -> 109,944 doubles and 850 -> 230 µs; M=3 266,616 -> 26,028 and 162 -> 46 µs, single
      cells from the harness with `--repetitions 1`. `test_scatter_sum_combines_reshaped_scatters`
      pins the zero-fill count. Arithmetic identities such as `0 / x` belong to C-53.
- [x] **C-50. `-march=native` and `-fno-math-errno` in the JIT.** `codegen/jit.py` compiles with
      `-O2` (or `SCALY_CC_OPT`) and no target flag, so every JIT kernel is SSE2 scalar code without
      fused multiply-adds on a machine that has them; measured on race-car N=50, `-mfma` alone is
      32.9 to 27.8 µs. Add the two flags to the JIT compile line, check that the solver plugins'
      compile paths (scaly-sqp's wrapper, the PIQP and IPOPT hooks) still link, and keep the plugin
      wheels themselves at the portable baseline. One line if it plays well with the solvers.
- [x] **C-51. Coalesce consecutive scalar loads and stores into vector accesses in the C renderer.**
      After C-44 scalarizes a body, adjacent `buf[i], buf[i+1], ...` accesses can be emitted as one
      clang `ext_vector_type` load or store; tinygrad's `memory_coalescing` does this in about 60
      lines (`tinygrad_rangeify.md` §6) and it is the difference between scalar code and visible
      SIMD on clang. After C-44. Landed as store coalescing only (`codegen/c.py`, `_emit_body`),
      width 2 via `vector_size`: race-car N=50 24.4 to 23.2 µs on clang, 30.3 to 30.4 µs on GCC;
      chain M=5 within noise on both. Width 4 slowed GCC on chain by about 5 %, and loads were left
      scalar because scalarized bodies consume them lane by lane.
- [x] **C-9. Affine index maps instead of materialized tables.** Implemented 2026-09-09 in
      `passes/affine.py`: `affine_index_map` factors a concrete index array into ranges whose
      contribution is affine plus a residual table, greedily, outermost first, and
      `LowerCtx.index_at` emits one term per range over the trip index so a fully affine gather or
      scatter declares nothing. No AD rule changed; `ad/forward.py` and `ad/reverse.py` still build
      the arrays, and the structure is recovered at lowering, which also catches every other affine
      gather in the graph. `passes/program/_common.py` gained `_index_values`, the counterpart that
      reads the indices back out of the expression, so `combine_scatter_sums` still sees the
      destinations it needs. `tests/passes/test_affine_index.py` pins the factoring, the forward and
      reverse VMAP derivatives against NumPy and against the unrolled form, and the two gates; all
      three gates fail with the affine path disabled.
      Measured: unbumpercars C=32 source 52.30 MiB to 1.35 MiB and static metadata 53.9 MB to
      345 KB, so the cell compiles and passes its correctness check under the 50 MiB cap (kernel
      compile 49.7 s, 8011 µs). race_cars metadata 90,115 to 39,578 bytes at N=50 and 1,015,708 to
      420,681 at N=500; all eight `int64_t` index tables are gone, and the two 6-entry `unique_j`
      residuals that remain are constant in N.
      Runtime is unchanged: race_cars N=50 24.0 µs and N=500 240 µs against 24.0 and 243 to 247 for
      the tables, three repetitions each. Emitting each level as `(k // stride) % dim` did cost
      about 5% (25.4 µs and 257 µs), because the modulo is a second division. It is gone: since
      `k // stride[i] // dims[i] == k // stride[i - 1]`, the coordinates telescope and the index is
      a combination of the plain quotients `k // stride[i]` with coefficients
      `c[i] - c[i+1] * dims[i+1]`, which is the `(x % c) + (x // c) * c -> x` recombination applied
      once at emission rather than as a folding pass. Exact integer algebra for a non-negative trip
      index, so the indices are unchanged; `tests/passes/test_affine_index.py` gathers through every
      factored case and compares against NumPy to pin that. A single absolute size threshold was
      also tried and rejected: it makes the emitted shape depend on N, which
      `test_vmap_sparse_hessian_c_source_is_constant_in_length` correctly rejects.
      The one part of the gate not met is `static_metadata_bytes` fixed across N, and index
      arithmetic cannot make it so; C-57 owns what still grows.
      Design and what was rejected from tinygrad's `uop/divandmod.py`:
      [refactorings](notes/refactorings.md#affine-index-maps-for-gathers-and-scatters).
- [x] **C-10. Fold the identities the AD rules introduce, at the expression level.** Implemented
      2026-09-08 in `passes/expr.py`: `v @ ones -> sum(v)`, identity-index gathers become reshapes,
      uniform 0/1 masks of the result shape fold, each pinned in `tests/passes/test_expr.py`. The
      matrix forms `A @ ones` and `ones @ A` were tried as stacked row sums and reverted: in loop
      form they lower to one loop per row and lose the fused producer, slower than the matmul. They
      wait for an axis reduction in the IR, which is C-8's accumulator lowering.
- [x] **C-12. One matcher and iterative rewrite driver for both dialects.** Implemented 2026-09-08: `ir/match.py` is generic over both node types with an iterative driver (`fixpoint`, `revisit`, `max_steps`), `rebuild_program` in `passes/program/_common.py` is the program adapter, and `_transform` is gone. No nested patterns or captures: no call site needed them. C-13 closed with it. [Updated design](notes/refactorings.md#shared-compiler-rewrites).

### Deferred

The 2026-09-22 items below come from re-measuring a hand-optimized race-car Hessian on the M4 Max;
[`notes/perf_2026_09_22/`](notes/perf_2026_09_22/README.md) holds the numbers, the generality
analysis of each hand change, the `zig cc` and compiler-extension survey, and the rendering
decision. Like C-8 they are deferred and not required for 0.1.0: the current kernels already beat
CasADi on every benchmark cell, and the release work comes first. They are organized around C-8
rather than beside it. C-77 *is* C-8's step 0 (inline scalarized callees into their mapped loop,
which nothing in C-8 covered) and step 1 (range propagation); doing it lands the first third of
C-8 and the assembly fix at once. C-78 is three independent passes that need none of C-8 and can
go first if cheap wins are wanted. C-79 is one more rewrite over ranges, tinygrad's `shift_to`,
and sits after C-8's step 1 because a widened kernel that still writes 48 outputs to memory gains
little; it must not be built as a separate `VMAP`-loop transformation that C-8 would then delete,
the way C-43 is scheduled for deletion. C-80 is solver work. C-81 was the x86 re-run; it
confirmed the order C-77 then C-79 and settled C-79's lane width and loop shape. The same
evening the "libm only, everywhere" stance was revised after a three-model review: the
generated C stays portable by default, and a target-aware opt-in (`lanes`, `vector_libm`, a CPU
recipe) buys the measured 2× on x86; C-79, C-83 and R-71 record the result. When
C-8 is resumed, fold C-77 and C-79 into its step list and close them there.

- [x] **C-77. Inline scalar callees into their mapped loops and fuse the derivative assembly.**
      C-8 step 0 and step 1. Today `_lower_vmap` emits `FOR { CALL }` and every pass stops at the
      `CALL`, so the sparse-Hessian recovery (`gather(transpose(jvp_many(...)))`) runs as 15
      separate array passes: 35% of the race-car N=200 Hessian, and the reason inlining alone buys
      nothing (43.5 to 43.0 µs) while fusion lets 6 of 16 block entries die (28.4 to 18.0 µs).
      The triangle selection in `ad/sparse.py` already gathers straight from the compressed
      block, so the six upper entries per stage are dead only once the kernel body and its
      gather share a range; nothing before that can drop them. They are unread rows inside seed
      directions the star coloring needs anyway, not extra directions, so this is dead-store
      elimination, not an AD change; computing fewer sweeps is C-11's second-order reverse pass. Gates: no assembly loops in the
      generated C; no store of an upper-triangle compressed entry in the stage body; the
      workspace holds no colored or transposed intermediate, so `SZ_W` is zero at W = 1 and grows
      only with lane staging (the hand-written kernel needs no `w[]` at all); race-car hess lower
      N=200 under 22 µs on the M4 without vectorization.
      Implemented 2026-09-22: 957 tests pass, 3 skip; independent review passed. On the x86
      reference host, N=200 medians are 59.71 µs GCC and 25.36 µs Clang, with zero workspace.
      The M4 timing gate has not been remeasured.
- [x] **C-78. Constant-tile folding, invariant-divisor reciprocals, terminal-trip peeling.** Three
      small passes. A `static const` table that tiles a period P becomes `k[i % P]`, a scalar at
      P=1; `k0`, `k22`, `k26` from C-57 are exactly these. `x / y` with `y` loop invariant becomes
      `x * inv_y` hoisted, behind a policy flag because it moves the last bit; both race-car
      divisors are invariant and this is the `-ffast-math` gap (18.0 to 15.5 µs). Peeling the last
      trip of a mapped axis whose slice differs makes the recovery gather `k44` affine (period 24,
      residual 13, verified) so C-9's map replaces the table. Gates: no `double` table growing
      with N in the race-car header; no division in the stage body.
      Implemented 2026-09-22: periodic tables and terminal intervals fold during C-77 fusion;
      reciprocal multiplication is opt-in. No stage-body division remains when enabled.
      Independent review and 980 tests pass (3 skip); GCC N=200 improves 59.71 to 57.01 µs,
      while Clang remains at about 25.3 µs.
- [ ] **C-79. Explicit lanes on mapped ranges.** Port tinygrad's `shift_to`,
      `r -> r_outer * W + r_lane` with `r_lane` of `RangeKind.VECTOR`, applied to the mapped axis
      first (independent trips, no dependence analysis), then a contiguous output axis, then a
      reduction axis with terms staged per lane and accumulated in their original order. Under a widened range a unit-stride access is
      a vector load or store, a constant stride is a staging transpose at the ABI boundary (buffers
      created under the range are lane major; ABI arrays stay stage major), anything else is per
      lane; `minimum` and `maximum` render per lane. W comes from a `lanes` render option:
      `"auto"` (the AOT default) renders a preprocessor block that picks W from the compiler's
      target macros, 8 under `__AVX512F__`, 4 under `__AVX__` or 256-bit SVE, 2 under `__SSE2__`
      or `__aarch64__`, otherwise 1 (Cortex-M and ARMv7 have no double vectors), overridable with
      `-DSCALY_LANES=n`, so one generated file serves several CPU builds and cross targets need
      no target description; an integer renders a fixed W and drops the block, which is what the
      JIT passes because the host is known and the `.so` never moves. Python caps W per kernel by
      register pressure (`live values × W ≤ 4 × register file`, note §3.1), emitted as a cap on
      the macro. The fused stage body is rendered once as an `always_inline` function
      taking the stage index and a count of valid lanes; the main loop runs the `N / SCALY_LANES`
      full trips with the count fixed at W, and one remainder call under
      `#if (N % SCALY_LANES) != 0` handles the last partial vector with clamped loads and guarded
      stores. No scalar tail and no second copy of the body. Staging buffers are sized for W = 8 so
      the workspace size in the generated header does not depend on the macro. Two render modes,
      documented in `docs/api/codegen.md` with their differences and supported compilers when this
      lands: `gnu` (default; `vector_size` types, `v[i]`, `__builtin_shufflevector`,
      `__builtin_convertvector`, `restrict`; gcc ≥ 12, clang, `zig cc`, armclang) and `c` (opt in;
      the widened program as a scalar body inside an inner lane loop over the same staging buffers;
      any C99 compiler). Transcendentals are per-lane scalar libm calls by default, and a third
      render option `vector_libm="none" | "glibc"` (default `none`) turns on glibc libmvec: C-81
      measured 26.6 µs at 8 lanes against 12.7 with the `_ZGVeN8v_*` prototypes declared in the
      source, equal to the hand-written kernel on gcc, clang and `zig cc` alike, so declaring
      them ourselves is compiler-neutral and `-fveclib`, gcc's `simd` attribute and
      `__builtin_elementwise_*` all stay out. The `glibc` option renders a prototype block for
      the ops the program uses, guarded on `__x86_64__`, `__GLIBC__`, the matching width
      (`_ZGVeN8v_*` needs `__AVX512F__` and W = 8, `_ZGVdN4v_*` needs `__AVX__` and W = 4, never
      a wider W on a narrower build) and `__GLIBC_PREREQ(2, 35)` for `tanh`; a build that fails
      the guard hits an `#error` naming the option and the `-lmvec` link flag, so a wrong AOT
      build fails at compile time with a sentence rather than at link time with an undefined
      symbol. It moves results (glibc documents 4 ulp against scalar libm's under 1), so it is
      off in distributed AOT output, on in the JIT on a glibc x86-64 host once the version check
      passes, and off under `zig cc` cross builds, whose glibc stubs omit libmvec. Check whether
      `-lm` alone already pulls `libmvec` through glibc's `libm.so` linker script before
      documenting `-lmvec`. No vendored SLEEF, no Accelerate (vForce is array-based and would
      undo C-77's fusion; Apple's scalar `sin` is 2 ns so the M4 gain is bounded at 5.7 of
      14.2 µs), no own polynomials; `generic` means `lanes="auto"`, `vector_libm="none"`.
      Compiler and OS are not dimensions of the generated C, only of the build recipe: a small
      table in `codegen/toolchain.py` keyed by CPU level (`native`, `x86-64-v3`, `x86-64-v4`,
      `apple-m4`, `generic`) yields the gcc and clang flag spellings, defines and link flags, and
      the same `BuildRecipe` value is printed by `scaly_toolchain`, by the AOT CLI
      (`--cpu`, `--lanes`, `--dialect`, `--vector-libm`; no named OS × arch × compiler targets,
      which would render byte-identical C) and as a comment block at the top of the generated
      `.c` and `.h` with the exact build line, the CPU baseline and the libc requirement.
      Distributable AOT output requires an explicit CPU baseline; `native` is for host-local
      builds. The complete vector-math study finished on 2026-09-23 under the
      [shared policy](notes/benchmark_protocol.md#vector-math-study-policy). Retain the scalar-libm
      comparison separately. Merge approved with the performance follow-up deferred to a new
      branch. Keep C-79 open until the
      [remaining regressions](notes/benchmark_comparison_history.md#remaining-regressions-and-limits) are resolved. Gates: compile matrix gcc × clang × W ∈ {1, 2, 4, 8} × `vector_libm` on and
      off; byte-identical output between the two modes and across W at the same compiler and
      flags with `vector_libm="none"` (bitwise equality across targets never existed: FMA
      contraction and Apple versus glibc libm already move the last bits); an `nm` check that no
      `_ZGV*` symbol appears when off, perturbed to prove it can fail; on the x86 reference
      machine within 1.2× of `variant_w8_lane.c` (26.6 µs, gcc 13) when off and within 1.1× of
      the hand-written kernel (12.7 µs) when on; no regression on the M4 at W = 2.
      Implemented 2026-09-23 with original-order reduction accumulation. Independent review,
      all 1,178 tests, and the complete vector-math study pass. All 23 Scaly sweep cells completed
      five processes, and all 75 controller episodes succeeded. Latest five-process race N=200
      microbenchmark medians: scalar libm 27.21 µs GCC / 24.69 µs Clang; libmvec 12.34 µs GCC /
      12.23 µs Clang. Both x86 limits pass; M4 performance remains unmeasured.
      Follow-up on a new branch: isolate UB base/gradient/Jacobian/Hessian costs on retained
      canonical inputs, then compare compiler and lane-width effects without changing reduction
      order. Recover the September 10 closed-loop function-evaluation baseline of 16.419 ms IPOPT
      and 9.513 ms SQP; the latest study measures 17.022 and 10.045 ms. Investigate UB C=16/C=32
      slowdowns against the intermediate scalar-policy study separately. Preserve the artifacts
      under `benchmarks/results/study-2026-09-23-c77-c79-libmvec` and
      `benchmarks/results/study-2026-09-23-c77-c79-final`. Recheck the race microbenchmark and
      targeted UB comparisons before a final full study. The user approved merging the current
      implementation on 2026-09-24 with these measured regressions deferred, not resolved.

- [ ] **C-80. Parameter-only oracle prologue.** An oracle whose subgraph depends only on `p`
      runs once per solve and its result is reused across SQP iterations (the race-car cost block,
      and the 402 `cos`/`sin` of the reference heading that the cost tangent and adjoint callees
      each recompute per stage; they depend on `p` alone). Function or solver level, not a program
      pass; belongs with the S items once C-77 lands.
- [x] **C-81. Re-run the 2026-09-22 variants on the x86 reference machine.** Done 2026-09-22,
      section 5 of `notes/perf_2026_09_22/README.md`, `x86_variants.sh` reproduces it. C-77 stays
      first (108.5 to 59.4 µs on gcc). A native `zig cc` links `-lmvec`. Declared
      glibc `_ZGV*` prototypes reach the hand-written kernel (12.7 µs) and gcc's `simd` attribute
      does so from scalar source; the prototypes became C-79's opt-in `vector_libm="glibc"` the
      same evening, the `simd` attribute and `__builtin_elementwise_*` stay out because clang
      ignores the former and LLVM's libmvec table stops at 4 lanes without `tanh` for the
      latter. Side finding: gcc 13, the JIT's `cc` here, is 40% slower than clang 20 on today's
      generated code at the protocol flags, one reason R-71 puts `zig cc` first.
- [ ] **C-83. Fingerprint the host in the JIT cache key.** `_compute_cache_key` in
      `codegen/jit.py` hashes the flag string `-march=native`, not the CPU it resolves to, nor the
      compiler identity. A cache directory shared across machines (an NFS home on a mixed
      cluster) can hand an AVX-512 `.so` to a node without AVX-512, and a compiler upgrade reuses
      stale objects. Add the resolved CPU features (`sysctl` or `/proc/cpuinfo`), the compiler
      command and its version string, and, once C-79 lands, the render options and the glibc
      version to the key. Gate: a test that changing any of them misses the cache.
- [ ] **C-82. A frame budget in `pack_workspace` instead of the per-buffer spill threshold.**
      Fusion (C-77) and lane staging (C-79) move memory from full-length intermediates into
      per-stage locals, and inlined callees add their locals to the caller's frame; the
      hand-written kernel uses no `w[]` and about 40 kB of stack. Today a slot spills to `w[]`
      only when it alone reaches 1024 doubles, so nothing bounds the frame. Replace it with a
      per-procedure estimate (local buffers plus inlined callees' locals plus a fixed allowance
      for scalar spills) against a budget, default about 64 kB on hosts and a compile option for
      embedded builds, where zero sends everything to a caller-provided `w[]`; spill the largest
      buffers until the estimate fits; report the estimate beside `SZ_W` in the header. Verify
      the estimate with the compiler: a test builds a generated module with
      `-Wframe-larger-than=<budget> -Werror` on gcc and clang and fails if the estimate was
      optimistic. After C-77, since that is what shifts memory onto the stack.

- [ ] **C-8. A range-based loop compiler for the program dialect, in the shape of tinygrad's
      rangeify.** Caller workspace still grows with N on race_cars and npmpc. C-47 removed chain's
      43 separate scatter accumulation buffers and reduced M=5 workspace to 109,944 doubles.
      Defer the general loop compiler until after the closeout study and documentation work. The
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
      recomputed under an expand. Gates: race-car `workspace` free of colored and transposed
      intermediates (zero at W = 1, lane staging only otherwise; see C-77 and C-82), chain
      workspace under 100k doubles at M=5, npmpc within 5% of today's kernel with those rules removed.
      The pre-optimization caller shares were 22% of race-car, 3% of npmpc, and 26% of chain.
      Re-measure the remaining costs before resuming this work.

- [ ] **C-57. The static metadata that still grows with N after C-9.** With every affine index
      table gone, race_cars metadata is 39,578 bytes at N=50 and 420,681 at N=500, so it still
      grows roughly linearly. Three things are left, none of them index arithmetic. The generated
      header's sparsity tables are `O(nnz)` by construction (23,677 bytes at N=50: rows, cols, the
      CSR and CSC pointers and both value permutations) and the question is whether a banded or
      per-stage-block encoding can describe them in closed form for a multistage problem instead of
      listing them. The sparse-assembly gather is a genuine `nnz`-length permutation with no affine
      structure (`k44`, 657 entries at N=50), and would need the assembly itself restructured, not
      its index compressed. And three `double` constant tables that C-45's periodic-tile bake did
      not reach (`k0` 1200, `k22` and `k26` 1224 entries at N=50) grow with N; find out which
      tangent or weight each one is and whether the bake's period test is simply too narrow.
      Deferred 2026-09-09: growing metadata is reported separately from executable code and
      artifacts must stay within the compile cap. Revisit if measured artifact size becomes a
      deployment limit.

- [ ] **C-11. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
      42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
      computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
      body, then a scatter. Deferred beyond this closeout. The
      [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#what-is-inherent-to-forward-over-reverse-here)
      finds composition accounts for most excess operations, with symmetry the remaining
      second-order opportunity. Reassess only if a current workload justifies the work. Rediscovered
      empirically on 2026-09-22 while hand-optimizing the race-car Hessian kernel
      ([note](notes/perf_2026_09_22/README.md)): each of the 4 seed directions yields all 4 rows of
      the block, 16 entries where the lower triangle needs 10. There the 6 are dead rows that C-77's
      fusion deletes for free; only where the coloring width itself grows does a second-order
      reverse sweep (edge pushing) that touches each nonzero once pay for its complexity.

- [ ] **C-58. Inline small pure callees before differentiation.** Revisit a bounded expansion policy
      if measured workloads justify it. Excluded from C-49 closeout to preserve mapped structure
      without introducing a new expansion policy. Diagnosis: [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#ranked-rule-and-composition-edits-for-an-implementer).

- [ ] **C-54. Add memory-aware program common-subexpression elimination and dead-code cleanup.** Deferred, like all compiler work, unless a measured workload requires it. Build on C-12/C-53 with definition/use tracking and conservative read/write/alias handling; retain required calls and output stores, and test repeated loads across writes. Broader loop motion follows demonstrated workload need; load-node interning alone is not a current stale-value bug. [Design](notes/algebraic_simplification_2026_09_08.md#separate-value-cleanup-from-memory-optimization).
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
  width-axis result Scaly already wins.
- [x] **C-84. Remove the unused `TensorType.sparsity` field.** Done, together with the unused
      `ProgramOp.PARAM`. No expression constructor sets it,
      and lowering, codegen and AD never read it. Real patterns live on `Function.output_sparsities`.
      Drop the field and its `__post_init__` check in `ir/types.py`, the `sparsity-shape-matches`
      rule in `ir/expr_spec.py`, the ` sparse` suffix in `ir/text.py`, and the positional copies in
      `function/model.py`, `function/tree.py`, `solvers/problem.py`, `solvers/nlp.py`,
      `solvers/qp.py` and `tests/solvers/problem_helpers.py`. Delete the test in
      `tests/ir/test_types.py` and the `sparsity` line in `tests/ir/test_verifier.py`.
- [ ] **C-85. Put the compiler's identity in the JIT cache key.** `_compute_cache_key` in
      `codegen/jit.py` hashes the flags but not the compiler, so pointing `SCALY_CC` at another
      compiler with the same flags reuses artifacts built by the first. Hash the resolved compiler
      path and its `--version` output, and bump the cache version.

## Solvers

### Now

### Deferred

- **S-16. Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with
  our hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: [protocol](notes/benchmark_protocol.md).
- **S-17. Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library
  search paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".
- **S-18. CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its
  globalization, regularization and QP path differ from `scaly-sqp`. Add only if review asks for it.

## C API

The generated C, C++ and CasADi-compatible interface. [Design](notes/generated_interface_2026_09_18.md).

### Now

- [x] **CAPI-72. Native entry cleanup.** `mem` becomes `int`; the `f_sz_*()` functions and the
      `alloc_mem/init_mem/free_mem` stubs go from `codegen/c.py` and `codegen/aot.py`; the JIT
      reads `module.workspace_size` instead of calling `f_sz_w()`; the ABI doc follows. Touches
      every rendered header, so the whole suite runs and the `benchmarks/results/smoke/**`
      fixtures are regenerated if compared textually. [Design](notes/generated_interface_2026_09_18.md#the-pointer-entry-both-languages-always).
- [x] **CAPI-73. C header with a caller-owned workspace.** After CAPI-72. Buffer structs become
      `f_x_t` (no `_in`/`_out`), 16-byte aligned; `f_workspace_t` is passed to `f_call` instead of
      being stack-allocated inside it. Update the two C++ smoke tests in `tests/codegen/test_c.py`.
      [Design](notes/generated_interface_2026_09_18.md#the-c-header-langc).
- [x] **CAPI-74. C++ header.** After CAPI-73, independent of CAPI-75. `lang="cpp"` renders `f.hpp`
      beside the same `f.c`: a guarded `Buffer<T, Ns...>` with inline aligned storage and a
      `constexpr shape`, a namespace per function with `x_t`/`workspace_t` aliases, `constexpr`
      sparsity tables, `call(..., workspace_t&)`; no enclosing `scaly` namespace. Smoke test: a C++
      caller against a `(N, nx)`-shaped function and a sparse Jacobian, reading a value through
      `csc_val_perm`. [Design](notes/generated_interface_2026_09_18.md#the-c-header-langcpp).
- [x] **CAPI-75. CasADi 3.8 compatible symbols.** After CAPI-73, independent of CAPI-74. `casadi=True`
      adds the guarded `casadi_int`/`casadi_real` typedefs, the query set (`_n_in`, `_n_out`,
      `_name_in`, `_name_out`, `_default_in`, `_sparsity_in`, `_sparsity_out`, `_work`,
      `_work_bytes`, `_checkout`, `_release`, `_incref`, `_decref`), compressed-column sparsity
      tables, and an entry that gathers compact sparse outputs through `csc_val_perm` (adding
      `nnz` to `f_SZ_W`). Dense matrix inputs or outputs are rejected at render time. Tests: load
      the library with `casadi.external` and compare against `numerical_call`; assert the six
      symbols acados needs resolve through `ctypes`. [Design](notes/generated_interface_2026_09_18.md#the-casadi-layer-casaditrue-either-language).
- [x] **CAPI-76. Document the generated interface.** After CAPI-74 and CAPI-75. Rename the ABI page to
      "The generated interface": the pointer ABI, then the C, C++ and CasADi layers; say plainly
      that both header languages compile the same kernel. Update `guide/codegen.md` and the CLI
      help (`--lang`, `--casadi`). [Why ABI and API are both right](notes/generated_interface_2026_09_18.md#what-this-is-called).

## Benchmark harness

### Now

- [x] **BH-48. Adopt `-march=native` in the benchmarks and the AOT guidance; keep distributed
      binaries portable.** Decided 2026-09-08: the sweep and closed-loop harnesses compile both
      providers with `-march=native` (and `-fno-math-errno`), fairness.md states the rule and why;
      AOT users are told in the docs to pass it and it goes in the suggested CFLAGS; the solver
      plugin wheels stay at the portable baseline. The JIT side is C-50. Measured reason: on
      race-car the gain is FMA contraction (`-mfma` alone: Scaly 32.9 to 27.8 µs, SX 21.3 to 20.7,
      because SX's one-op-per-statement code never contracts), not vector width. Record both flag
      sets in fairness.md until BH-20 reruns. Evidence: `notes/perf_2026_09_07/README.md`.
- [ ] **BH-21. Add an immutable publication mode**: clean release candidate, every raw run retained,
      and an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.

### Deferred

- **BH-22. Embedded hardware benchmarks** (Raspberry Pi / Jetson).

## Benchmark problems

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

Scaly and the three plugins are BSD-2-Clause. The plugin wheels also ship other people's binaries,
so each wheel carries its dependencies' license texts the way CasADi does
(`casadi/include/licenses/<dep>/LICENSE`), except that CasADi's `mumps-external` and
`metis-external` entries are the COIN-OR wrapper's EPL text rather than the real MUMPS and METIS
licenses, which we do not copy. Surveyed 2026-09-07. What we ship and what it asks of us:

| Package | Component | License | Obligation |
|---|---|---|---|
| scaly, scaly-sqp | our code | BSD-2 | none |
| scaly-piqp | PIQP, BLASFEO | BSD-2 | notice |
| | Eigen | MPL-2.0 | notice. PIQP does not define `EIGEN_MPL2_ONLY` itself, so `hatch_build.py` passes it through `CMAKE_CXX_FLAGS`; PIQP 0.6.2 compiles under it, which proves no LGPL Eigen file reaches the library |
| | LDL inside PIQP (`piqp/sparse/LDL_License.txt`) | LGPL-2.1-or-later | notice plus the LGPL-2.1 text. PIQP's `sparse/ldlt` is a modified LDL, instantiated in `ldlt.cpp` and compiled into `libpiqpc`, so the shared library we ship contains LGPL code. That is allowed: the LGPL text travels with it, the modified source is PIQP's public tag, and `libpiqpc` is a separately loaded shared library the user can replace |
| scaly-ipopt | IPOPT | EPL-2.0 | notice, upstream source of the pinned version; a separate dynamically loaded module, so our BSD-2 is unaffected |
| | MUMPS 5.8.2, via COIN-OR `ThirdParty-Mumps` 3.0.12 | CeCILL-C, EPL-2.0 for the wrapper | notice for each |
| | OpenBLAS (static, Linux) | BSD-3 | notice |
| | libgfortran, libquadmath | GPL-3 + GCC runtime exception | notice; the exception covers this use |
| | METIS 5.2.1 | Apache-2.0 | notice |
| | GKlib | Apache-2.0, plus two glibc-derived headers under LGPL-2.1-or-later and one BSD-3-Clause file, per its `LICENSES.md` | notice for each |

### Now

- [x] **L-28. Root `LICENSE.md` (BSD-2-Clause, copyright EPFL, 2026)**, the holder the lab's other
      projects name, with the author only in the pyproject `authors` entry, `license = "BSD-2-Clause"` and
      `license-files = ["LICENSE.md"]` in the root `pyproject.toml`, and the README "License" section
      replaces "TBD".
- [x] **L-29. Copy the same `LICENSE.md` into each of `plugins/scaly-{sqp,piqp,ipopt}/`** with the same
      two pyproject fields. Each plugin is its own sdist and wheel, so each needs the file in its
      own tree; a copy, not a symlink, so sdists stay correct.
- [x] **L-30. Third-party notices generated by the build hooks.** `hatch_build.py` in `scaly-piqp`
      and `scaly-ipopt` copies each dependency's license text from the already-cloned
      `third_party/` sources into `src/scaly_{piqp,ipopt}/licenses/<dep>/` and writes a short
      `THIRD_PARTY_NOTICES.md` listing name, pinned version from `build_config.json`, license and
      upstream URL; the directory joins the wheel `artifacts`. Generating at build time keeps the
      notices from drifting from the pins. libgfortran and libquadmath are not cloned, so their
      GPL-plus-runtime-exception text is vendored once in `plugins/scaly-ipopt/licenses/`, as is
      the LGPL-2.1 text for LDL in `plugins/scaly-piqp/licenses/`. `scaly-piqp` got its own
      `build_config.json`, which also pins BLASFEO to `0.1.4.3`; it built from an unpinned
      `master` before. A missing notices file counts as "not built", so libraries from before this
      change are rebuilt once.
- [x] **L-31. Close the two PIQP questions** in the table: confirm `EIGEN_MPL2_ONLY`, and how the
      LDL code is linked. Both answered in the table.
- [x] **A test that the license directory exists for every dependency named in
      `build_config.json`** in each plugin wheel's file list, shown to fail when an entry is
      removed. Plus one paragraph in `docs/dev/contributing.md`: a new vendored dependency needs a
      `build_config.json` entry and a license copied by the hook. `plugins/*/tests/test_*_notices.py`.

## Documentation

The same category of debt the fairness audit found in the results pages: prose that outran what
the code does.

### Now

- [x] **D-32. Rewrite the user-facing documentation.** Done 2026-09-16: every page under `docs/`
      and the README rewritten for register and accuracy, the wordmark added to both entry points,
      the single status admonition in `docs/index.md` (remove it at 0.1.0), and the plain-speech
      rules recorded in `docs/dev/conventions.md`. `FunctionTemplate` (API-1) is not mentioned in
      the docs; add its page when it lands. Wheel installation instructions belong to R-41.
- [x] **D-33. Rework `docs/how_it_works/comparison.md` (now `influences.md`).** Done 2026-09-16: every Taken/Changed row
      checked against code; `sc.problem` returns a `Problem`, only one upward import is checked. A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering-hint discussion subsequently removed from the published docs. The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [x] **D-34. Audit the whole of `docs/` for claims that outran the implementation.** Done
      2026-09-16; every guide snippet executed, two were broken and are fixed. Originally: the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached.
      The lowering-hint claims have been removed. Check any surviving timing that predates the
      reference-machine rule in `AGENTS.md`.
- [x] **D-68. Acknowledgements and AI disclosure in the README and `docs/index.md`.** Done 2026-09-16. An
      acknowledgement of NCCR Automation, which funded the research, and a disclosure that AI coding
      agents (Claude, Codex and others) were used to write parts of the code and documentation.
      Before R-67, so the first public snapshot carries both.
- [ ] **D-65. Two documentation deployments.** GitHub Pages publishes the user-facing docs from
      `main` on each version tag, through a `docs.yml` job with `pages: write` and `id-token: write`
      permissions on a tag trigger. Cloudflare Pages publishes the latest docs from `main` and a
      preview per branch, which the existing every-branch build already produces. Each site carries
      a banner or version switcher saying which one it is.
- [x] **D-35. Reconcile the problem READMEs with the audit.** Completed 2026-09-11. The problem
      READMEs now describe only the current formulations and link measured comparisons to the
      canonical result pages.

### Deferred

- **D-36. GPU backend milestone definition**: what a first accelerator target must demonstrate before
  any backend work starts.

## Release

These steps make the tree public and permanent, and each is cheap to do once and expensive to redo.

### Now

- [x] **R-37. Move the paper design note out of this repository.** Done 2026-09-16; it goes to
      the paper's own repository. The note was never on main but is in dev's history, which R-62
      rewrites.
- [ ] **R-38. Windows support.** Decide the toolchain (MSVC or clang) and the target: the core JIT
      plus `scaly-sqp` and `scaly-piqp` first; `scaly-ipopt` on Windows is a separate later item
      because it drags in Fortran and its own licensing survey. Candidate toolchain: make `ziglang`
      (R-71) a required dependency on Windows through a `sys_platform == 'win32'` marker, and build
      the Windows `scaly-piqp` wheel with `zig cc` as the CMake C and C++ compiler, so the JIT and
      the solver library share one toolchain and no MSVC-versus-MinGW runtime mismatch can arise.
- [x] **R-39. Finalize the name.** Decided 2026-09-14: `scaly`. `scali` was the first choice, but
      PyPI refused it as too similar to `scaii`, an abandoned 2019 project: PyPI treats `l`, `i`
      and `1` as one character when comparing names, and nobody can override that check. `scaly`
      is pronounced the same, is a real word people spell right after hearing it, and matches the
      scale-filled S of the logo.
- [x] **R-60. Reserve the PyPI names** `scaly`, `scaly-sqp`, `scaly-piqp` and `scaly-ipopt`: a
      placeholder package per name at version `0.0.0a0` whose description says what it will become,
      built with `uv build` and uploaded with `uv publish` from the gitignored
      `package-placeholders/`. A pre-release version so `0.1.0a1` stays free; PyPI never lets a
      version be reused. `scaly` is reserved as of 2026-09-14; the three plugins wait on PyPI's
      new-project rate limit. The `scali-sqp`, `scali-piqp` and `scali-ipopt` placeholders published
      before the name changed stay up as tombstones, since deleting a project frees its name for
      anyone; point their descriptions at the `scaly` packages once those exist.
- [x] **R-59. Rename everything to scaly.** Done 2026-09-16: the package, the three plugin
      distributions and directories, the `scaly_*` modules, `SCALY_*` environment variables, the two
      console scripts, the JIT cache directory, CI cache paths, `.config/wt.toml`, `zensical.toml`,
      `docs/` and `internal/`, and the `sc` import alias. The Foxglove layouts never contained the
      old string. The GitHub repository was renamed in place and this
      clone's `origin` now points at `PREDICT-EPFL/scaly`; any other clone needs
      `git remote set-url origin git@github.com:PREDICT-EPFL/scaly.git` and, if its directory was
      renamed too, a recreated `.venv` since uv's console scripts hardcode the venv path.
- [x] **R-61. Remove the pre-extraction history from file contents.** Done 2026-09-16; the
      library roadmap went with it, since everything it planned is either implemented or too
      vague to keep.
- [x] **R-63. Audit `internal/` for publication.** Done 2026-09-16 with an independent second
      pass. It stays in the public repository and off the documentation site, which Zensical
      guarantees because it builds `docs/` only. The `no-private-names` `pre-merge` hook in
      `.config/wt.toml` keeps the gate alive after the audit.
- [x] **R-62. Rewrite history and force-push the renamed repository.** After R-37, R-59, R-61 and
      R-63 (all done): drop the local `refs/t3/checkpoints/*` refs, `git filter-repo --invert-paths` on
      the paths R-37 removed, re-add `origin`, force-push every branch and
      tag, delete stale remote branches, and hard-reset or re-clone every other clone and worktree
      rather than pulling. Do this while the repository is still private, since GitHub keeps
      unreachable commits fetchable by SHA until its garbage collection.
- [x] **R-64. Publication metadata and hygiene.** A secrets scan over the rewritten history,
      `CITATION.cff`, a real pyproject description, and ruff's `target-version` aligned with `requires-python`.
- [ ] **R-67. Make the repository public.** After R-62, R-64 and the merge into main, with the suite,
      ruff and ty green locally. The first CI run happens here because the month's Actions minutes
      are spent, and both workflows will consume them once they refill.
- [x] **R-40. Versioning policy.** What a minor bump promises about the generated C symbols, the
      sparsity-table prefixes and the plugin ABI; written into `docs/dev/versioning.md`, with the
      plugins pinning `scaly>=0.1.0a1,<0.2`.
- [x] **R-66. Platform-only wheel tags for the plugins.** Nothing in the plugins touches the Python
      C API, the solvers load through ctypes, yet `hatch_build.py` in `scaly-piqp` and `scaly-ipopt`
      sets `infer_tag = True`, which stamps the running interpreter's `cpXY-cpXY-<platform>` tag.
      Set the tag to `py3-none-<platform>` explicitly instead, so one wheel per OS and architecture
      serves every Python version and no per-interpreter build matrix or stable ABI is needed.
- [ ] **R-41. Wheel building and publishing.** cibuildwheel with one matrix entry per OS and
      architecture, solvers built natively on each runner as CI already does. `scaly`, `scaly-sqp`
      and `scaly-piqp` first; `scaly-ipopt` follows once its Fortran runtime licensing (L-30) is
      settled. Test PyPI first. After L-28 to L-31 and R-66. Decide the compiler the wheels are
      built with at that point; the system compiler on each runner is the default. `cmake` comes
      from PyPI as a build requirement of both plugins; a C++ compiler and gfortran stay system-wide
      prerequisites for anyone building the wheels themselves.
- [ ] **R-71. `zig cc` as the JIT's preferred compiler.** Order: `SCALY_CC`, then
      `python -m ziglang cc` when `ziglang` is importable, then `CC`, then `cc` on `PATH`. It wraps
      clang, so the JIT always speaks one flag dialect and gcc-only behaviour (the sincos merge
      that blocks vectorization, the 40% gap to clang measured in C-81) leaves the JIT path; on
      the x86 reference machine zig's clang 21 was within 10% of clang 20 on the baseline and
      linked `-lmvec` natively (11.6 µs on the 8-lane libmvec variant). The compiler becomes a
      command list rather than a binary path, which `scaly_toolchain` must print. Before flipping
      the default, check that the solver plugins' JIT paths (the `scaly-sqp` wrapper, the PIQP and
      IPOPT hooks) link against the vendored libraries with zig's driver on all three operating
      systems, and measure cold compile latency, since zig builds its own libc on first use.
      Expose it as the `scaly[toolchain]` extra, required on Windows (R-38).
- [ ] **R-42. Freeze measurements on `0.1.0a1`, tag `v0.1.0a1`** and publish the wheels and a durable
      archive. After R-41.

## Track C closeout before merging to dev

- [x] **C-59. Complete the approved optimization cleanup.** Follow the
      [review and plan](notes/optimization_cleanup_2026_09_10.md), preserving the arithmetic and
      measurement contracts in [protocol](notes/benchmark_protocol.md).
  - [x] Pin regressions and capture the baseline.
  - [x] Centralize Program analyses, names, and procedure reachability.
  - [x] Normalize compilation expressions and consolidate derivative helpers.
  - [x] Close fusion, arithmetic, and scalarization interactions.
  - [x] Move store coalescing and scalar scheduling into Program passes.
  - [x] Validate the combined compiler and refresh documentation.
  - [x] Address Fable's review and repeat the controlled comparison. The remaining generation overhead is measured and accepted for this cleanup.

Complete the implementation and independent reviews, then run BH-20 on the combined tree.
C-56 and BP-23 can run independently. C-46 and C-49 share derivative construction and run together.
After integration, refresh the test baselines and run the full suite, formatting, lint, and strict
type checks. Run the study without competing tests or compilation. Remove the temporary
implementation worktrees after integrating and reviewing their changes.
Keep metadata and caller-workspace growth as measured limitations. After updating the results,
merge into dev and prioritize documentation.

- [x] **C-56. Prevent workspace slot names from colliding with user buffers.** `pack_workspace`
      allocates scratch slots around every existing parameter, output, and private buffer name.
      The regression in `tests/passes/test_program.py` reproduces silent wrong results with outputs
      named `s1` and `s2`. Implemented and independently reviewed 2026-09-09.

- [x] **C-46. Share stage-invariant and cross-formal derivative work.** The program pass hoists
      invariant buffers before mapped loops. Forward differentiation now combines active formals
      and packs compatible specialized results into one mapped callee output, sharing primal and
      adjoint expressions. This does not coalesce arbitrary original Function outputs or batch
      every runtime seed into a matrix product. Implemented and independently reviewed 2026-09-09.
      [Controlled closeout probes](notes/perf_2026_09_07/README.md#c-46-and-c-49-closeout-2026-09-09).

- [x] **C-49. Reduce derivative operation count without pre-differentiation inlining.** Joint
      propagation across active formals, negation identities, lean division rules, vector self-dot
      and elementwise-square rules, and shared square-root reciprocals are implemented. Tests cover
      seed layouts, caches, overlapping slices, logical input names, and finite-scale derivatives.
      Independently reviewed 2026-09-09. The [audit](notes/perf_2026_09_07/c49_ad_op_audit.md)
      records the diagnosis; [closeout probes](notes/perf_2026_09_07/README.md#c-46-and-c-49-closeout-2026-09-09)
      record the combined C-46/C-49 results. Pre-differentiation inlining is deferred as C-58.

- [x] **BP-23. Vmap the unbumpercars wall rows.** The four wall barriers per car now use one
      mapped Function, preserving constraint order and all existing numerical gates. The growth
      check covers Jacobian source and retained pair/wall call families through the full exact
      Lagrangian Hessian. The old inline-wall formulation fails the growth check. Outer Hessian assembly
      still grows with car count and remains part of deferred C-8. Independently reviewed 2026-09-09.

- [x] **BH-19. Harness gaps.** `dispatch_trip_count`, `dispatch_workspace` and `dispatch_arithmetic`
      were empty for the race_cars and unbumpercars Scaly cells because `_dispatch_metrics` gave up
      on any kernel mapped over two axes (`N` and `N+1` stages; cars and pairs). It now reports the
      dispatch-loop family carrying the most arithmetic per call and the workspace over every
      dispatch, so all four problems fill the columns. The unrolled pair rows no longer exist in
      `filters.py` (removed by the pair-row port); we chose not to recover them from history for a
      same-protocol control, so the mapped-pair result is reported descriptively.
      Closeout follow-up implemented and independently reviewed 2026-09-09: hoisted procedures
      retain their original callee identity for dispatch metrics. Unit-trip and longer maps exclude
      the one-time prologue. Disabling the identity lookup makes the regression fail.

- [x] **BH-20. Complete the closeout study.** Completed 2026-09-11
      in `benchmarks/results/study-2026-09-10`. The [results overview](../docs/benchmarks/index.md)
      and [scalability tables](../docs/benchmarks/scalability.md) contain the completed study.
      The interrupted 2026-09-09 attempt
      remains preserved in its original result directory and frozen investigation note.
