# C77–C79 implementation record

Archived on 2026-09-24 for merge to dev. The user approved deferring the remaining performance
investigation to a new branch. [#71] owns the follow-up; unchecked items
below record the original acceptance requirements, not an active plan for this worktree.


The original acceptance requirements were full regression checks, independent review, and a
measured improvement in the race-car N=200 Hessian microbenchmark. Do not mark a task complete from timing
alone. Preserve scalar arithmetic by default and measure reciprocal and vector-libm policies separately.

Sources: `docs/how_it_works/architecture.md`, `docs/dev/codebase.md`, `docs/dev/conventions.md`,
`docs/dev/contributing.md`, `docs/benchmarks/fairness.md`, `internal/notes/perf_2026_09_22/`,
`internal/notes/perf_2026_09_07/tinygrad_rangeify.md`, and the existing Program pass pipeline.

## Sequence

- [x] Establish unchanged full-suite and race-car microbenchmark baselines on the reference host.
- [x] C77: record the range propagation design, implement scalar callee inlining and derivative
  assembly fusion, and add independent numerical and structural regression fixtures. Review,
  run required checks and isolated timings, then commit only if the gates pass.
- [x] C78: fold periodic constants, peel differing terminal trips, and add an explicit reciprocal
  policy. Verify numerical policy boundaries and structural gates, review, measure, and commit.
- [ ] C79: widen mapped ranges, stage accesses, render GNU vectors and C lane loops, and expose
  lane, dialect, vector-libm, and CPU recipe controls. Verify tails, widths, compiler matrix,
  scalar equality, libmvec accuracy and symbols. Review, measure, and commit.
- [ ] Run the complete five-process benchmark study without competing builds or tests. Investigate
  regressions in chain, neural-process MPC, and unbumpercars and repair in-scope causes.
- [ ] Update published results from retained artifacts, run documentation and regression checks,
  review, commit, and remove this temporary plan.

C77 precedes C79. C78 analysis and independent review may run alongside implementation, with
separate file ownership. All timing runs are sequential. Implementation delegates use low or
medium reasoning, or Astra low. Review delegates inspect without edits.

## Evidence

The host matches the published Ryzen 9 7940HS reference. The governor is `performance` and boost
is disabled. The initial working tree was clean. C77 passed review, 957 tests (3 skipped), Ruff, strict types, and isolated timing gates.

The initial editable solver build hit a self-copy of its generated Fortran runtime. Moving that
generated copy aside allowed `uv sync` to complete without changing the build hooks. The unchanged
source is retained under `/tmp/scaly-c77-baseline-tree`, and its N=200 generated kernel and timings
are under `/tmp/scaly-c77-baseline`. The baseline workspace contains 36,084 doubles.

C77 N=200 medians (three processes, pinned CPU 3): GCC 109.46 → 59.71 µs; Clang
76.99 → 25.36 µs. Generated workspace is zero, source is 871 lines. Artifacts are retained in
`benchmarks/results/compiler-c77-c79-2026-09-22/c77-accepted`. M4 timing was not available on this host.

C78 passed independent review, 980 tests (3 skipped), Ruff and strict types. With reciprocal
multiplication enabled, GCC improves from 59.71 to 57.01 µs; Clang is effectively unchanged
(25.36 to 25.32 µs). No divisions remain in mapped stage bodies. The policy remains opt-in.

C79 passed independent review, all 1,098 tests, Ruff, strict types, and the documentation build.
The scalar-libm compiler/dialect/width matrix is bitwise equal, including ordered reductions.
Clean five-process microbenchmark medians: scalar libm GCC 29.05 µs, Clang 24.71 µs;
libmvec GCC 12.29 µs, Clang 12.07 µs. Workspace remains zero. Both x86 targets pass;
M4 performance remains unmeasured. Earlier C79 timings overlapped another session’s OpenBLAS
build and are retained as diagnostic evidence only.

The detached study runner is `benchmarks/results/compiler-c77-c79-2026-09-22/run-study.sh`.
It writes the full five-process scalar-libm study to `benchmarks/results/study-2026-09-23-c77-c79`.
Launch with nohup and setsid; retain `study.log`, `study.pid`, and `study-exit-status` beside the runner.
Do not run competing tests or builds during the study. Inspect all failures and compare the four
workloads with the previous published tables before updating the three results pages.

The first detached full study completed with exit status 1 after all 690 sweep attempts.
Race-car and neural-process MPC sweep timings improved, as did chain M=3/M=5. However,
chain M=9 exceeded the 300-second code-generation cap. All unbumpercars Scaly sweep cells
and the neural-process MPC/unbumpercars closed loops failed to compile because a sanitized
lane name was used as an internal substitution key. Only 35 of 75 closed-loop episodes completed.
C79 is reopened pending the renderer correction, chain diagnosis, full regression/performance
gates, and a fresh detached study. Preserve the failed study artifacts.

The lane-name correction passes the unbumpercars C=2 and C=32 numerical gates and both
neural-process MPC closed-loop smoke runners. Query-local sparsity reuse and shared Program
folding reduce chain M=9 generation to about 52 seconds in a diagnostic run, and its full sweep
correctness check now passes. Independent review found that a general expression rewrite memo
would change lowering hints. Limit shared memo support to Program graphs instead.

The first follow-up suite passed 1,114 tests, Ruff, strict types, and the documentation build.
The narrowed memo requires a fresh full run. Race N=200 generated source and headers remain
byte-identical to the accepted C79 artifacts for scalar libm and libmvec.

Clean C=32 unbumpercars comparisons expose a widening regression: five alternating pinned
processes average 7.322 ms with auto lanes and 6.922 ms with lane 1. The previous published
mean is 6.872 ms. Investigate narrower vectors and standalone ordered-reduction overhead before
accepting another compiler milestone or launching the replacement full study.

The narrowed Program-only memo passed independent review and the complete 1,110-test suite,
Ruff, formatting, and strict types. Fresh five-process race medians remain inside the x86
acceptance limits: scalar libm GCC 27.49 µs and Clang 24.72 µs, libmvec GCC 12.30 µs and
Clang 12.51 µs. Retained under `compiler-c77-c79-2026-09-22/followup-micro`.

Unbumpercars width diagnostics retain the original summation order. Width 4 averages 5.407 ms;
removing standalone reduction widening makes it slower. A four-lane cap on non-reduction
helpers averages 5.372 ms, whereas capping only ordered reductions averages 8.528 ms.
The next diagnostic separates the two helpers with serial inner loops from flat output helpers.
Do not accept a workload-specific width override.

The blanket cap for nested loops is rejected: neural-process MPC N=200 slows from about 300 to
336 µs, though all numerical outputs remain bitwise equal. Race source and header are unchanged.
Further inspection found that four-lane execution still strides private scratch by eight, so it
uses half of every cache line. Implement active-width private indexing while reserving eight-lane
allocation capacity. Resolve private aliases to owner indices before widening, preserve external
layouts, and use the effective helper width for both full and remainder chunks. Then retest
unbumpercars and neural-process MPC before selecting any default width cap. No width heuristic
has been accepted or added to production yet.

Active-width private packing passed independent review, all 1,121 tests, Ruff, strict types, and
the documentation build. Tests cover chained private aliases, external alias refusal, tails,
fixed eight-lane allocation capacity, and differently capped helpers reusing one workspace slot.
The race source and header remain byte-identical to the accepted scalar-libm and libmvec kernels.
All four unbumpercars/neural-process MPC diagnostic candidates match control output bytes.

No new lane-width cap is accepted. In the final paired run, the unchanged unbumpercars control
shifted from about 7.33 ms to 5.33 ms despite unchanged object code, governor, and boost setting.
The cause is unresolved. New default-width packing is neutral in that paired run: unbumpercars
5.343 ms versus control 5.361 ms, neural-process MPC 298.921 µs versus control 299.298 µs.
Treat earlier absolute unbumpercars timings as diagnostic, not a published regression claim.
Preserve every timing run. A complete replacement study is required before publishing results.

The replacement runner is `compiler-c77-c79-2026-09-22/run-study-final.sh` and its output directory
is `benchmarks/results/study-2026-09-23-c77-c79-final`. It uses the unchanged five-process study
protocol with scalar libm, performance governor, and boost disabled. Keep `study-final.log`,
`study-final.pid`, and `study-final-exit-status` beside the runner. Launch under nohup and setsid.

## Next study policy, user decision on 2026-09-23

The user requests the fastest validated tested configuration for each provider. The decision is
recorded in `docs/benchmarks/fairness.md`, under "Next study policy". Do not alter code, compiler
flags, environment, or the running scalar-libm study before its runner has actually exited.
Afterward, enable Scaly AOT/JIT glibc vector math, apply the supported compiler libmvec option
and link requirements to both Scaly and CasADi, retain numerical/reduction policy checks, and
record actual vector symbols and compiler options. Keep scalar-libm artifacts separate.
Run a new complete five-process study under nohup with the new policy before updating headline
results. No reciprocal or broad fast-math/reassociation change is authorized by this decision.

## Scalar-libm study completed

`study-2026-09-23-c77-c79-final` finished with exit status 0. All 12 commands succeeded,
all 690 sweep attempts ran, and all 75 closed-loop episodes completed. Sweep totals are 510
successful timings, 15 compilation timeouts, and 165 skips after earlier backend failures.
Every Scaly sweep cell completed five processes. Large-case Hessian means are race N=200
29.286 µs, neural-process MPC N=200 307.330 µs, chain M=9 282.777 µs, and unbumpercars
C=32 5,361.081 µs. Unbumpercars C=2 regressed from the previous published 207.925 to
345.619 µs. Scaly closed-loop function evaluation also regressed for neural-process MPC
and unbumpercars. Keep these regressions visible and investigate them before final conclusions.

The user-authorized next vector-math policy may now be implemented. Retain the compiler already
used within each comparison: Clang sweep builds receive its supported libmvec selection flag
for both providers. GCC closed-loop builds have no equivalent selection flag; both providers
receive the same applicable flags and link libraries. Do not imply that linking alone generates
vector calls. Preserve explicit scalar-libm reproduction and record actual symbols. No broad
fast-math or reassociation is authorized. Compiler overrides must not silently give providers
different flags. Benchmark implementation and compiler-regression diagnosis run separately;
coordinate all accepted timing runs to avoid competing builds.

The short-loop diagnosis found that count-two helpers still ran eight lanes. Cap each helper at
the next power of two of its static iteration count, combined with the existing register cap.
Five alternating scalar-libm C=2 processes improve from 471.207 to 186.143 µs, with identical
output bytes. Paired race N=200 is neutral at 29.146 versus 29.014 µs, also byte-identical.
Retain these paired diagnostics separately from the full-study absolute timings.

The cap and benchmark vector-math policy passed independent review, all 1,178 tests, Ruff,
strict types, and the documentation build. Four generated-C snapshots changed only lane-cap
macros. Fresh five-process race medians are scalar libm GCC 27.21 / Clang 24.69 µs and
libmvec GCC 12.34 / Clang 12.23 µs. All remain within the accepted x86 limits. Evidence is
under `compiler-c77-c79-2026-09-22/tripcap-micro` and `tripcap-diagnostics`.
A one-process canonical NPMPC/unbumpercars closed-loop check precedes the new full study.

All eight canonical vector-math diagnostic episodes completed. NPMPC function evaluation is
0.968 ms with IPOPT and 0.478 ms with SQP, below the old published 1.006/0.503 ms. UB remains
17.114/10.311 ms, above the old 16.419/9.513 ms. All oracle counts agree; UB SQP has one
iteration mismatch, with maximum state/control differences 4.58e-9/3.69e-9. This is one process,
not a headline replacement. C79 remains open for the full study and remaining UB investigation.
The largest UB mapped Jacobian/Hessian helpers perform matrix arithmetic without transcendental
calls, so vector math cannot fix their main cost. Next isolate base/gradient/Jacobian/Hessian
costs on retained canonical inputs before considering a generic width policy change.

A two-provider race N=5 sweep passed the independent numerical checks. Both compile logs
contain Clang's `-fveclib=libmvec`. Both objects reference actual vector sin/cos symbols; Scaly
also references vector tanh. Retain the evidence in `c79-vector-sweep-check`.

The next detached runner is `compiler-c77-c79-2026-09-22/run-study-libmvec.sh`, writing to
`benchmarks/results/study-2026-09-23-c77-c79-libmvec`. It sets `SCALY_VECTOR_LIBM=glibc` and
`SCALY_CC=gcc`, with the unchanged full five-process protocol. Launch under nohup/setsid and
retain `study-libmvec.log`, `study-libmvec.pid`, and `study-libmvec-exit-status`. Do not run
competing builds or tests, or change benchmark/compiler code, while it is running.

## Completed vector-math study and publication

The detached libmvec study finished with exit status 0 on 2026-09-23, after 21,600 seconds.
All 12 commands succeeded: 690 sweep attempts, 520 timings, 15 compile timeouts, 155 skips,
and 75 closed-loop episodes. Every Scaly cell completed five successful processes.
Race N=200 is 12.502 µs versus best completed CasADi SX 81.206 µs and previously published
Scaly 89.722 µs. All Scaly Hessian means improve against September 10.

The results-page update reports this complete study, including scalar comparisons and remaining
regressions. UB closed-loop function evaluation is still 3.7%/5.6% slower than September 10
for IPOPT/SQP. UB C=16/C=32 Hessians are slower than the intermediate scalar-policy study,
which also predates the short-loop cap. NPMPC IPOPT total rises 1.5% despite faster evaluation.
Do not claim that libmvec alone caused any between-study change. C79 stays open under the
user's no-regression requirement; keep this plan until the remaining investigation is complete.

Publication validation: independent artifact and documentation review confirms all 138 sweep
rows and all closed-loop/agreement rows match the report. All 34,690 recorded control steps
succeeded. Documentation builds and whitespace checks pass. No compiler code changed for this
publication update; the prior 1,178-test validation still describes the measured implementation.

## Integration with dev

The merge incorporates dev's inferred vmap slicing, parameter-only solver calls, public
`walk_program`, and `load_library` APIs. The resolved integration passed 1,180 tests and
independent review. Race N=200 scalar-libm and libmvec source/header pairs remain byte-identical
to the timed candidates.

The merge smoke checks exposed two stale structural assumptions. The chain scalar-hint check
now observes the per-stage procedure immediately after scalarization, before fusion removes it,
and selects by hoisting provenance. Perturbing scalarization metadata makes the check fail.
The typed UB Hessian check explicitly uses scalar lanes to isolate accidental CALL-boundary
seed materialization from intentional vector staging. Its unchanged 200 KB / 100,000-double
budgets pass at 101,795 bytes / zero doubles with scalar lanes. Auto lanes use 1,546,549 bytes
and 122,880 doubles. A reintroduced CALL boundary with scalar lanes produces 1,852,520 bytes
and 1,601,141 doubles, failing both budgets. Default vector artifact sizes remain measured by
the study, not bounded by this scalar structural check.

## The C-79 todo entry at closure

Moved here verbatim from `internal/todo.md` on 2026-09-29, when C-79 was closed and its open
performance regressions moved to [#71]. It holds the full design and gates that
`perf_2026_09_22/README.md` refers to.

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

[#71]: https://github.com/PREDICT-EPFL/scaly/issues/71
