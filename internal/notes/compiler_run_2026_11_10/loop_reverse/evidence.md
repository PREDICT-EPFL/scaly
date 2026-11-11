# Reverse LOOP evidence

The #33 change is based on the verdict of #32. `tests/function/test_loop_reverse.py`
checks scans against unrolled graphs, forward/reverse duality, trajectory cotangents,
run-time trips, reverse scans, overlapping windows, step numbers and mixed carries.
While-loop checks include early and zero exits, reaching the bound, Newton iteration
and forward-over-reverse second derivatives. Nested loops also check reverse over reverse.

The problems ported from frozen devrush are in
`tests/core/integration/test_loop_edge_cases.py`: zero/one-step scans, window and
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

At every horizon, the generated program contains 536 nodes. C size is 13,272,
13,299 and 13,369 bytes. C compilation took 0.0844, 0.0785 and 0.0721 seconds using
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
The API adds both loop builders and removes the LOOP enum filter. #53 was still
open when this was written, so cond remains unpublished.

The backward loop has no condition and takes the primal count as its trip operand.
This needs no relaxation of #31's ban on conditional reverse loops. No roadmap
departure was needed. Stacked-output sparsity remains conservative. Per-step
Jacobian compression, checked builds, in-place carries and unrolling stay outside
this change.
