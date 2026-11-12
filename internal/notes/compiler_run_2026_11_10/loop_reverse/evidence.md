# Reverse LOOP evidence

The #33 change is based on the verdict of #32. `tests/function/test_loop_reverse.py`
checks scans against unrolled graphs, forward/reverse duality, trajectory cotangents,
run-time trips, reverse scans, overlapping windows, step numbers and mixed carries.
While-loop checks include early and zero exits, reaching the bound, Newton iteration
and forward-over-reverse second derivatives. Nested loops also check reverse over reverse.

The problems ported from frozen devrush
`tests/core/integration/test_loop_edge_cases.py` into `tests/function/test_loop_reverse.py` cover zero/one-step scans, window and
step-number derivatives, mixed nonfloating carries, while-loop derivative boundaries,
and per-loop trajectory limits with nested loops. The port uses main's Function API,
nonnegative windows and a per-loop argument. Global options, strict-mode switches,
negative strides and per-step Jacobian compression were not ported.

The initial tests failed with the existing reverse-mode rejection. Removing the
trajectory-cotangent addition temporarily made the unrolled comparison fail in
`test_trajectory_windows_index_and_mixed_carries[3-False]`. The rule was restored.

## Milestone loop measurement

`measurements.json` records a single RK4 value-and-gradient function with outputs for
the scalar cost, initial-state gradient and per-step-control gradient, at N=20, 200
and 2000 on la015. The workload is the `rk4` in `tests/function/test_loop.py`, with
cost equal to terminal squared norm plus the sum of per-step costs. Both gradients
come from a single `sc.vjp` request.

After integration with the refreshed forward-loop base, every horizon produces 551
program nodes. C size is 13,933, 13,960 and 14,017 bytes. C compilation took
0.0848, 0.0785 and 0.0786 seconds using
`/usr/bin/cc`, the module's recipe and the JIT compile command. These are C compile
times under concurrent compiler-run work, not run-time performance evidence.
Bounds and storage literals account for the source-size changes. The constant
cotangent of the stage-cost sum is passed whole to each backward step, avoiding a
horizon-length table of ones.

The test checks NumPy rollout values at all three lengths, finite differences of
both gradients at N=20, and control-gradient duality at all three lengths. It pins
constant program size and less than 300 bytes of source-size variation. Operation
counts include integer indexing as well as floating operations, and are recorded
in full in the JSON. No C snapshots were regenerated.

## Publication and decisions

`docs/guide/control_flow.md` documents scan, while_loop, their differentiation and
storage limits, and the print rule. Its Python fences were executed together.
The API adds both loop builders and removes the LOOP enum filter. The cond issue #53
and pull request #417 have merged. The guide and API now publish cond and remove its
enum filter, while preserving the sparse-operation entries already published on main.

The backward loop has no condition and takes the primal count as its trip operand.
This needs no relaxation of #31's ban on conditional reverse loops. The roadmap
clarifies that generated derivative loops do not inherit a user loop's trajectory limit,
so second derivatives work at the primal trajectory limit. Stacked-output sparsity remains conservative. Per-step
Jacobian compression, checked builds, in-place carries and unrolling stay outside
this change.

## Validation before review

The detached `~/.scaly-suite/run` completed successfully: 4,818 passed and 56 skipped,
benchmark smoke passed, documentation built without warnings, and the private-name
check passed. Ruff formatting, lint and ty passed. C snapshots passed unchanged.
The complete collection contains 4,874 tests. Focused loop/map/snapshot runs passed,
including all 28 new reverse-loop cases.

README Python fences and the function_composition, mapped_derivatives and
function_with_print examples ran successfully. A separate finite-difference check
of a while loop inside a map also passed, with different trip counts per entry.

## Review fixes

Both independent reviews requested changes. `review.md` records each finding and
its fix. The final focused loop/map/snapshot run passed 148 tests, including 39
reverse-loop cases. Ruff lint, formatting and ty passed, and collection contains
4,885 tests. The revised guide fences ran with their stated second-derivative and
printing outputs, and the documentation build passed without warnings.

The new map regression keeps multi-output adjoints mapped. The trajectory-limit
port now checks Hessians and reverse over reverse at the primal trajectory limit.
The mapped while-loop finite-difference check is a permanent test. The four sol
findings and the neighboring step-number cache case have failing regressions that
pass with their fixes. C snapshot files remain byte-identical. The RK4 source sizes
and program counts remained identical after these fixes.

## Final validation after review fixes

The final detached `~/.scaly-suite/run` exited successfully: 4,829 tests passed,
56 skipped, benchmark smoke passed, documentation built without warnings and the
private-name check passed. Ruff formatting, lint and ty passed. The 4,885-test
collection includes all review regressions. C snapshots remained unchanged.

Both independent short delta checks approved the final fixes. The correctness
checker independently reran all 39 reverse-loop cases. The main checker confirmed
the technical fixes and every documentation note. No review finding remains open.
The final changes after the checked source are validation and review records only.

## Integration with the refreshed loop stack

The rebase retained the index-operand gather/scatter implementation, conditional
rules and in-place proofs from the new base. Loop adjoints use `_scatter_zero` for
parameter and sliced-input cotangents and `_accumulate` for sibling seeds and
returned cotangents. Carried boundaries still drop select activity, consistently
with forward mode. Independent maps keep their existing mask protocol.

The loop builders retain strict `in_place` requests. Lowering excludes a carry
whose trajectory is used from reuse, and a strict request on it remains refused.
A runtime-index scatter regression compares reverse derivatives with an unrolled
reference and checks that the trajectory-bearing primal has no in-place procedure.
A second regression checks that incoming select activity is dropped at the carried
reverse boundary. No new select-mask protocol or in-place policy was added.

The refreshed focused loop, map, conditional, in-place and snapshot run passed 225
tests. C snapshots remain byte-identical to the new base. The refreshed RK4
measurements above retain identical program-node counts across all three horizons.
The guide's Python fences, including the newly published cond example, ran with
the stated outputs. The earlier validation counts describe the pre-rebase tree.
