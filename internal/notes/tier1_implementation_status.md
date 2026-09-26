# Tier 1 primitives: implementation status (2026-09-26)

Copied from the claude.ai project doc of the same name, so that Claude Code sessions can read it.
Branch `claude/tier1-primitives` (on `t3code/plan-compiler-performance-passes`), tag
`tier1-complete`. Plan: `tier1_primitives_plan_2026_09_25.html`. Reports: `tier1_pr1_report.html` …
`tier1_pr9_report.html` and `tier1_summary_report.html`. Timing scripts: `perf_2026_09_25_tier1/`.

Scope decisions (Colin):
- One generated solver per fixed sparsity structure.
- Fix `scatter` to accumulate.
- `sc.options(nonsmooth=...)` with both tie conventions; the default is `"split"`.
- `while_loop` is differentiable.
- No integer ABI: float64 counters, and bool as 0/1 across the double ABI.
- Reverse mode stores the carry trajectory.
- No Hypothesis for now.

| PR | todo | Capability |
| --- | --- | --- |
| 1 | C-82 | compare/logic, `sc.where`, `isfinite`, `copysign`, `cast`; `Expr` has no truth value |
| 2 | C-83 | `sc.options`; differentiable `maximum`/`minimum`/`abs`; `reduce_max`/`reduce_min`/`norm_inf`/`norm_1` |
| 3 | C-84 | accumulating `scatter`; `segment_sum`/`segment_max`/`segment_min`; linear gather adjoint |
| 4 | C-85 | `sc.scan`: one SERIAL loop, ping-pong carry, trajectory for reverse mode |
| 5 | C-86 | `sc.while_loop`: BREAK_IF + exit_var trip count; masked backward scan |
| 6 | C-87 | `index_add`/`index_set`; in-place carry under `in_place_chain` |
| 7 | C-88 | `sc.custom_derivative(fn, jvp=, vjp=)` |
| 8 | C-93 | multi-seed forward mode through `scan`/`while_loop`: one tangent loop carrying all seeds |
| 9 | review | 4-agent review of C-93, fixes and optimizations |

## PR 9: the review round
Colin asked for an agent review, thorough tests, fixes, and ranked optimizations, implemented.

Fixes:
- Interned tangent-symbol names gave a silently wrong Jacobian; fixed with `claim_name`.
- Tangents were mixed up between inlined calls; the memo is now keyed by id, and the graph is freed.
- A loop Function called K times grew the code as K²; calls are inlined only when seeds cannot be pruned.
- Forward over forward crashed on duplicate names.
- Two different Functions with one name are now refused at lowering.
- int64 inputs to callees, scan and vmap were read as raw doubles.
- The generated C depended on `PYTHONHASHSEED` (`pack_workspace`).
- The `cast` multi-seed rule, and a cache bypass in strict mode.

Optimizations:
- Per-step Jacobian tangents.
- One backward scan per scan (C-91).
- The `delinearize_loops` pass, giving division-free index maps.
- The JIT passes raw addresses.

Results against PR 8:

| | PR 8 | PR 9 |
| --- | --- | --- |
| MPC Hessian, N = 100 | 109 µs | 58 µs |
| RK4 Hessian, N = 100 | 1079 µs | 93 µs |
| Trivial JIT call | 6.5 µs | 4.7 µs |

Open items: C-89, C-90, C-92, C-94 … C-100 in `internal/todo.md`. Pending question to Colin: generalize
`segment_sum`/`segment_max` into one `segment` op with a combine function.
