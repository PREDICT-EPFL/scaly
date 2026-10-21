# Forward traversal refresh, October 21

After the prerequisite squash merges and #189/#158, #166 now targets main in stack #192. The normal hook passes with the preferred zig JIT compiler: 2446 tests pass and only the two existing power cases remain expected failures. No #188 expected-failure markers remain. Correctness smoke accepts the new compiler while measured closed-loop runs retain their compiler guard; a short independent review approved that integration fix.

The measurements below are retained Clang 20.1.8 C sweep results, not zig JIT timings. The current protocol still uses Clang for sweeps and GCC for measured vector-math closed loops. No new timing study was run for the rebase. The later section about #188 records the historical pre-fix validation limit; the complete-pipeline hook result above supersedes that blocker.

This refresh measures #166 after main's compact lane helpers and bit-packed seed selectors. The control is main `fd427d2811935f6cb1478304b53990db21621031`. The stack base is the refreshed #18 pull request, #165. [comparison.md](comparison.md) contains generated-source bytes, generation time, native runtime and workspace for current main, the rebased traversal and the retained fixes, plus all rejected candidates. Run `uv run internal/notes/perf_2026_10_21_forward/compare.py` to reproduce that table from the raw rows.

## Method and retained attempts

Measurements use la015 and [the benchmark protocol](../benchmark_protocol.md): performance governor, boost off, glibc vector math, Clang compiling C, identical samples and the existing native correctness gate. Environment is `PYTHONHASHSEED=0 SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=clang CXX=clang++`. The harness uses one fresh process per cell; uncertain comparisons were repeated as separate invocations. The final table combines four current-main processes and three final processes per workload. Generation includes derivative construction and C rendering. CV is the sample coefficient of variation; a lone exploratory process supplies no dispersion estimate.

The grid is `chain_jac:33`, `npmpc_jac:100`, `chain:3`, `unbumpercars:2`, backend `scaly`. Every timing invocation first runs `pgrep -af 'pytest|wt hook|benchmarks'`, waits for other heavy work and records an idle audit. The audit keeps full process command lines private and records only PID, program and working directory. A competing run appeared during the last candidate7 confirmation batch; all four cells of `candidate7-repeat-3` are retained but excluded conservatively. These are separate from the historical October 9 contaminated cells, which remain in the earlier diagnosis evidence.

The exact timing command is:

```sh
uv run benchmarks/run.py sweep --workloads "$workload" --sizes "$size" \
  --backends scaly --repetitions 1 --headline --boost off --order-seed 0 \
  --out "$output/$workload/$workload.csv"
```

Each candidate was pushed, checked out cleanly in `/tmp/scaly19-refresh-fixN`, synced with all groups and extras, then passed the full benchmark smoke and a small chain preflight before its headline cells. Main and the unmodified rebased traversal are `/tmp/scaly19-refresh-main` and `/tmp/scaly19-refresh-after`. Full generated C, native compiler logs, correctness records, executables and provenance remain under their `benchmarks/results/issue19-*` directories. `raw/` retains the CSV rows, summaries and provenance. The persistent launch scripts are `/tmp/scaly19-refresh-study.sh`, `/tmp/scaly19-fix-study.sh`, `/tmp/scaly19-repeat-study.sh` and `/tmp/scaly19-final-study.sh`; their logs are `/tmp/scaly19-refresh.log`, `/tmp/scaly19-fixN.log`, `/tmp/scaly19-confirm.log` and `/tmp/scaly19-final-study.log`.

All 72 headline attempts passed native correctness and timing, with no compile failures, timeouts or skipped cells. Four conservatively excluded candidate7 attempts leave 68 rows in the harness [report](report.md). Tracked provenance copies omit unstable task-branch commit hashes; the original provenance files retain them. Phase names identify each code variant, and the final head is recorded on the pull request. [idle-audit.log](idle-audit.log) records the per-invocation checks.

Against current main, the final means are 25.1% faster for chain Jacobian and 1.8% faster for chain, with residual increases of 2.3% for neural MPC Jacobian and 5.7% for unbumpercars. The two residual regressions remain maintainer open points. These measured improvements do not establish that all remaining costs are explained or that the neural network's zero seed column has been packed out.

## Measured changes

The earlier [C diagnosis on #166](https://github.com/PREDICT-EPFL/scaly/pull/166#issuecomment-6085304894) is adapted to main's bit selectors. These are lowering and schedule changes around the decided traversal. There is no seed-by-seed AD fallback, new partial rule or changed arithmetic reduction order.

- Candidate1 tried narrow private accumulators and explicit concat axes without enough index folding. Neural MPC improved, but chain Jacobian worsened. Candidate2 used one private dot at every output width; unbumpercars became much slower. Both forms were discarded.
- Candidate3 retained narrow private accumulators and disjoint constant-color selection. Chain Jacobian improved from 3118 to 2798 microseconds. General and nonunit coefficients keep the existing selector path.
- Candidate4 uses four-output-column private matrix panels, explicit concat axes and bounded integer quotient/remainder folding. It reduces chain Jacobian to 2379 microseconds and unbumpercars to 239. The private panel keeps each output's ordered reduction in scalars and stores once. No alias or widening limit changes.
- Candidate5 adds invariant reuse after scalarization and exact splitting of inner copy rows. Chain falls to 23.99 microseconds. Candidate6's view-fusion exception and alternate scalar store order did not help and were removed. Candidate5 and candidate6 neural MPC sources are byte-identical despite different single-process times, so that difference is measurement noise, not a compiler improvement.
- Candidate7 keeps exported invariant coefficients named, permits duplication of only two-use scalar products with constant/variable operands, and leaves tiny structural rows to scalar C under automatic widening. Chain's existing register estimate selects four lanes again instead of two. The budgets are unchanged, and scalar work counts actual expression occurrences.
- The final candidate chooses the smallest exact divisor when exposing an inner flat row. For 93 entries this yields 31 masses by three components, rather than three blocks of 31. It removes the remaining component division and gathers in the chain contraction. The flattened coordinate regression passes and fails when the old largest-divisor choice is restored in-process.

Scalarization now handles the private accumulator ASSIGN statements, including its forced-scalar public contract. The focused lowering, program-pass, scalarization and constant-seed checks pass, 196 tests. Claude Opus 5.5 at xhigh approved the performance diff. Its optional empty-selection and name-reservation notes were fixed: an empty selection no longer keeps a dead mapped derivative, and loop names use the shared `name_scope`. A low-effort delta check approved both fixes and the final divisor choice. No further review round was run.

## Snapshot review

The conflicted `jac.c` was taken from main during the rebase and regenerated, never hand-merged. Final differences against current main are exactly:

- `forward.c`, 54 lines on both: concat loop variables gain an axis suffix. Expressions and loop bounds are identical.
- `jac.c`, 206 to 138 lines: broadcast reads share `vel` and `2 * vel` instead of copying them four times. Four copy helpers and an unused two-lane type disappear. Packed buffers shrink from 8+8+16+1 to 4+16+1 doubles. Explicit seed/component concat loops replace flat division/modulo indices. Other edits rename buffers and helpers. The reduction order and final output transpose remain the same.
- `spjac.c`, 150 to 206 lines: disjoint constant-color selection writes mapped partials, scatters the selected entries into structural zeros, subtracts the residual identity and gathers sparse outputs. This adds two 48-double local arrays, static source/target indices and three movement helpers. Previously these operations fused into the output-writing helper. Scalar temporary names also change with scheduling. The derivative formulas and sparse output order are unchanged. This is an explained source-size increase, not snapshot churn to discard.

Every header and every other C snapshot is byte-identical to current main, including `vmap.c`, `wide.c` and `workspace.c`. Main's compact helpers and packed-mask pass remain in place. The snapshot suite and numerical lowering tests validate the regenerated corpus.

## Maintainer open points and validation limits

The rewritten chain gate still guards the original intent: the forward mass helper must retain its mass loop. At 33 and 65 masses its combined helper source is 592 and 592 lines. A real perturbation forces scalar lowering of those derivative helpers; source becomes 10452 and 21044 lines and the unchanged 1.2 ratio gate fails. Whole-kernel source is 1973 and 3061 lines because primal and boundary code also grows; the gate rewrite requires maintainer acceptance. The old 5/33 whole-source comparison crossed the automatic scalarization cutoff. The first scratch gate runners had import-path setup errors, and the first coordinate perturbation omitted the competing divisor; these attempts remain in `/tmp` and are not counted as gate evidence.

Neural MPC workspace is still 6144 doubles at horizon 6 and 7744 at horizon 100, versus main's 0 and 1600. The six fixed 1024-double product/transpose buffers are 32 hidden units times four seeds times eight horizon lanes across three network layers. The tested folds and schedules did not recover this fixed storage. The gate remains tight at 6144 fixed, 7744 total and 1600 horizon growth, as before this refresh. This workspace increase and gate change require maintainer acceptance; it is not claimed to be inherent to the decided design. Removing the nested transpose copies remains lowering work associated with #69.

The normal hook is **blocked by #188**, as requested by the coordinator. The final full suite reports 2365 passed, 57 expected failures, three strict XPASS cases and ten empty-loop errors. The strict XPASS cases are periodic empty maps whose work is now eliminated by constant selection; their upstream markers remain unchanged. The ten errors hit the upstream empty-loop mask-packing crash. None of the affected expected failures counts as an empty-shape pass. The 57 expected failures comprise 55 affected empty cases and the two existing power cases.

To exercise the traversal independently, `/tmp/scaly19-check-empty.py` removes only `pack_masks` from the in-process program pipeline, selects the 68 retained affected AD cases and runs `pytest --runxfail -n0`: 68 actual passes. This does not validate the complete pipeline and changes no production code or markers. The ordinary hook's failure log is `/tmp/scaly19-final-hook-v2.log`. Ruff formatting/lint, types, clean collection of 2435 node IDs, the focused checks, snapshots and the unchanged benchmark gates pass. CI remains blocked until the upstream fix is supplied through the stack.

Existing follow-ups remain #20 for shared elementwise rules and reverse cleanup, #21 for intermediate `wrt`, #22 for custom rules, #67 for batched matrix products, #69 for movement/fusion and #167 for the pre-existing mixed-zero movement dependency. No published benchmark page is updated from this focused study.
