# Section 9: Control flow

Packet of section 9 of the [compiler run](../plan.md), parent #23. Built by #411 (#31), #416
(#50), #417 (#53), #454 (#35, #36, #441), #425 (#34), #426 (#32) and #428 (#33). Open when this
was written: #78, which the plan schedules last, and the gaps below. The section had no
simplification pass (#467). AD is automatic differentiation and IR the intermediate
representation. A carry is the value a loop passes from one step to the next, and a trajectory
is the stack of the carries that entered each step.

## Where this section sits

- A user writes `sc.scan` for a fixed number of steps, `sc.while_loop` for a condition under a static bound, and `sc.cond` for a call to one of several Functions, and differentiates all three in forward and reverse mode, second derivatives included.
- A user reads and writes tensors at indices computed at run time (`sc.gather`, `sc.scatter`, `take`, `dynamic_slice`, `index_add`, `index_set`, `segment_sum`, `segment_max`), and a loop carry that a step updates at a few entries is updated in place when lowering can prove it safe.
- Generated code does not grow with the number of steps: an RK4 rollout with both its gradients is 18 kB of C at 20, 200 and 2,000 steps. `sc.vmap` is now a loop without carries, which is the one invocation model #23 was opened for.
- Published in `docs/guide/control_flow.md`, *Runtime indexing* of `docs/guide/functions.md` and the API page. The section's three Milestone 3 rows pass in `demo.py`, without run times.
- Section 10 writes the dense factorizations in these loops (#51, #52), section 11 the generated solvers (#205, #206) and small carries in C locals (#74), and section 8 the fusion and lane rules for loop bodies (#69, #71).

## Needs the maintainer's decision

1. Public interface, published by #428 and #454 although the roadmap places it in Milestone 3: the signatures of `scan`, `while_loop` and `cond`, the eight indexing builders and their `mode` defaults. It ships with the next release unless filtered again. Recommend reading `docs/guide/control_flow.md` first and deciding whether 0.2.0 carries it.
2. Published behaviour: `gather` and `scatter` default to `mode="promise_in_bounds"`, where a bad index is undefined behaviour, and the checked builds that would catch it are not built (#78). Recommend keeping the default and starting #78 before a release.
3. Published behaviour: a derivative holds the number of steps fixed, reaching `max_iter` raises nothing, and a negative index is out of range instead of counting from the end. Recommend keeping.
4. Roadmap departures, applied by #411: a loop with a condition cannot run in `reverse`, a negative `trip` counts as zero, and the fusion and lane rules for loop bodies moved to #69 and #71. Recommend keeping.
5. Roadmap departure, applied by #416: a loop with narrow stores, kept call boundaries, nested run-time ranges or prints stays a loop even under a scalar hint. Recommend keeping.
6. Roadmap departures, applied by #454: the in-place proof evaluates indices with exact bounded integer and boolean arithmetic, where the roadmap had NumPy, and a constant-index add scatter onto zeros keeps its zero-fill loop so that the snapshots stay identical. Recommend keeping the first and revisiting the second at the next snapshot regeneration.
7. Roadmap clarification, applied by #428: `max_trajectory` counts a user loop's own trajectory, and the tangent and backward loops that differentiation creates do not inherit it. Recommend keeping, since a Hessian otherwise fails at the limit that accepted its gradient.
8. Select masks at loops, changed on 2026-11-12: forward loops carry a select's condition across iterations (#426) and reverse loops rely on the conditions inside each step (#428). The dtypes packet holds the rule. Recommend confirming it there.
9. IR change, applied by #425: `ExprOp.VMAP` is gone and printed listings say `expr.loop`. A loop's results are numbered carries, stacked outputs, custom-rule residuals, the trip count, then trajectories. Recommend keeping.

## Expected gaps

- Checked builds, for bounds and nonfinite carries: #78.
- A sparse carry in a layout other than the canonical one raises: #412.
- Work that does not change between steps is not hoisted out of a carried loop, and a one-trip loop inside one is not unrolled: #376.
- Small carries are not kept in C locals: #74. No fusion or lanes inside a loop body: #69, #71.
- A conditional call drops select activity in forward mode: #464. A fill-mode index update is never done in place: #465.
- A forward Jacobian by per-step inputs grows with steps times seeds (53 kB of C at 20 steps, 278 kB at 200, from #426). The roadmap leaves per-step compression as an open question with no issue. The guide says to use reverse mode.
- Nested loops each count their own `max_trajectory`, and the product is not bounded. The guide says so.
- The Jacobian pattern of a run-time index covers every entry the index might select. The guide says so.
- No run time was measured for any loop. Milestone 3 is not complete, so its Phase 4 checks have not run.
- The section's simplification pass did not run: #467.

## Unowned

- `sc.cond` loses a select's condition in forward mode, so forward and reverse disagree: filed as #464.
- `index_add` and `index_set` default to fill mode, which the in-place proof skips, and a strict `in_place=` then refuses without a reason. Filed as #465, together with two smaller reuse limits that #454 recorded.
- Sections 3, 4 and 9 closed without a simplification pass: filed as #467.

# Reference

Checked on main at `cd2b6c75` on 2026-11-17. Paths are under `src/scaly/` and line numbers are on
that commit. Beside this file: `demo.py` and `demo_output.txt`, and `loop_guards.py`. Two sibling
folders belong to this packet and are linked, not repeated: [`../loop_reverse/`](../loop_reverse/)
and [`../in_place/`](../in_place/).

## Decisions

Hardest to undo first.

### Public interface

1. **`sc.scan(body, init, xs=(), *, length, trip=None, index=False, reverse=False, in_place=(),
   max_trajectory=None)`** (#411, #428; `function/sugar.py:494-527`). The body takes its carries
   first and returns the next carries first, then per-step outputs, which are stacked. `xs` uses
   the slice rules of `vmap`: a leading axis, `sc.broadcast` or `sc.window`. `trip` is an int64
   expression clamped to `[0, length]`, and shapes stay those of `length`. `reverse` runs the
   steps in descending order and leaves the layout of the stacked outputs alone.
2. **`sc.while_loop(cond, body, init, *, max_iter, params=(), index=False,
   max_trajectory=None)`** (#411, #428; `function/sugar.py:530-550`). The condition is a Function
   of the carries and parameters, evaluated before each step. It returns the carry and the int64
   count of steps taken. Reaching `max_iter` returns the current carry and raises nothing.
3. **`sc.cond(selector, branches, *operands)`** (#417; `function/sugar.py:366-406`). A bool
   selects between two branches and an integer, clamped, among more. Only the selected branch
   runs, prints included. Sparsity and support are the union of the branches. There is no
   separate switch.
4. **The indexing builders** (#454; `ir/expr.py:1522-1650`). `gather` and `scatter` take an
   int64 index operand, a constant or a run-time value, with `axis=`. `scatter` combines by
   `add`, `set`, `max` or `min` in index order. `mode` uses JAX's names: `promise_in_bounds` is
   the default of `gather` and `scatter`, and `fill` the default of `take`, `dynamic_slice`,
   `index_add` and `index_set`. A negative index is out of range.
5. **`in_place=` is a strict request, and reuse without it is automatic** (#454). Semantics stay
   functional. A carry named in `in_place=` raises when lowering cannot prove the update, and the
   message names the carry. A carry whose trajectory reverse mode needs is never reused.
6. **Derivatives follow the steps the primal took** (#426, #428). The condition and the trip
   count have no derivative. Reverse mode stores the trajectory once and `max_trajectory=` bounds
   it per user loop, by the static `length` or `max_iter`, and names `sc.custom_derivative` when
   exceeded.

### IR

7. **`ExprOp.LOOP`** (#411; `ir/expr_spec.py::loop_result_type`). Arguments `(*inits, *outers[,
   trip])`, attributes `callee`, `cond`, `n_carry`, `length`, `starts`, `strides`, `index`,
   `reverse`, `in_place` and `output`. A stride of zero is a parameter every step reads whole.
   A scan has no condition and no trip operand, so every pass reads its static length from the
   attributes and it pays for no flag and no zero fill.
8. **`VMAP` is a `LOOP` without carries** (#425; `ir/expr.py:985-995`). One predicate, `is_map`,
   selects the map lowering and the mapped derivative rules, and generated C is unchanged. A map
   and its trip count share one invocation.
9. **`ExprOp.COND` and the program `IF`** (#417). The verifier wants a bool scalar and two
   blocks, and an integer selector lowers to a chain. The QP proof follows branch callees.
10. **Index operands replace the `indices` attribute** (#454). A constant index is the static
    case, and its C snapshots are byte-identical.

### Differentiation

11. **Forward mode is the same loop with tangent carries** (#426; `ad/calls.py:339-357`). The
    seed axis leads, the condition reads primal carries only, and integer and bool carries get
    no tangent. A flagged tangent has a boolean carry beside it. A loop without flags generates
    the C it generated before.
12. **Reverse mode is one backward loop over the stored trajectory** (#428;
    `ad/calls.py:734-766` and `800-854`). The rule asks the same primal invocation for its
    trajectories, so the primal runs once. The backward loop has no condition and takes the
    primal's trip count as its `trip`, so it visits exactly the steps taken. It carries the carry
    cotangent and one accumulator per parameter, and differentiates only the carries that a
    seeded output can reach. A trajectory result is differentiable, which is what reverse over
    reverse needs. [`../loop_reverse/review.md`](../loop_reverse/review.md) has the four defects
    the correctness review found here and their fixes.
13. **A branch derivative recomputes its primal** (#417; `ad/calls.py:117-162`). Forward is a
    `COND` over the branch tangents and reverse a `COND` over branch adjoints.

### Lowering

14. **One carried range per loop** (#411; `passes/lowering/calls.py:214-248`). `[call cond;
    break if not flag;] call body`. A carry lives in two alternating slots, in `length + 1` slots
    when its trajectory is used, or in one slot when the update is proved in place. Stacked
    outputs and trajectories are zero-filled past the trip count only when a condition or a trip
    operand can end the loop early.
15. **A static loop is unrolled within a budget** (#416; `passes/lowering/calls.py:375-387`).
    When the whole lowered body times the length is at most 256 right-hand sides
    (`STATIC_LOOP_BUDGET`), or the node has a scalar hint, the loop becomes straight-line code,
    with the summation order of the rolled loop. Block and opaque hints keep it. In #416's table
    a three-statement body unrolls up to 85 steps.
16. **The in-place proof has three rules** (#454; `passes/lowering/in_place.py:104-164`). Last
    use: nothing reads the old carry after the update. Same index: the update reads only what it
    writes, for any index. Disjoint indices: every index is evaluated for every step and the
    entries read later must differ from those written. The reused carry is one `inout` parameter
    of an `<body>_inplace` procedure, which scalarization and hoisting leave alone.
    [`../in_place/`](../in_place/) has the measurement: four entries of a carry of 100,000
    updated for 100 steps cost 4 stores and 8 loads per step with reuse, against 100,004 and
    100,012 without.
17. **The pass guards of section 5 hold on real loops** (#411; `loop_guards.py`). The script
    builds 228 configurations of `scan` and `while_loop` programs and removes each guard of #29
    in a scratch copy of the source. Three guards are reached (`scalarize.work`, the exit-loop
    rule of `unroll_unit_loops` and the carried-work rule of `hoist_invariant`, with 24, 24 and 6
    wrong results) and each has a test. The other eight give no wrong result, because a carried
    range whose body is a call is refused before them. That is #411's run. I did not rerun it
    after #452 and #454.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/control_flow/demo.py`. The three Milestone 3
rows the section owns, then a conditional call and a clipped rollout. It takes no run times.
Output on main at `cd2b6c75`, on la015 while other work ran:

```text
== 1. an RK4 rollout with its gradient, written with scan ==
    N   value against NumPy   C bytes of value + both gradients   generation
   20   3.6e-15                18025 bytes, 30 loops              0.10 s
  200   1.7e-13                18057 bytes, 30 loops              0.10 s
 2000   1.2e-11                18119 bytes, 30 loops              0.12 s
N = 20, d cost / d z0: max |scan - unrolled| 4.4e-16, max |scan - finite differences| 5.6e-09
N = 20, d cost / d us: max |scan - unrolled| 5.6e-17, max |scan - finite differences| 6.3e-09
N = 20, forward mode against reverse mode: 2.8e-14

== 2. a Newton iteration inside while_loop ==
a = 1: root 1.000000000000 after 0 steps (condition false at entry); error 0.0e+00
a = 5: root 1.709975946677 after 6 steps (early exit); error 2.2e-16
a = 1e+09: root 1000.000000000000 after 37 steps (early exit, far start); error 3.4e-13
a = 5 with max_iter = 3: 1.722002 after 3 steps (the bound is reached and nothing is raised)
d root / d a: 0.1139983964 (exact 0.1139983964)
second derivative, forward over reverse: -0.0151997862 (exact -0.0151997862, finite difference of the gradient -0.0151997862)

== 3. run-time indices, and a carry updated in place ==
take(x, [5, 9, 0], fill -1) = [ 6. -1.  1.]   dynamic_slice(x, 4, 4) = [5. 6. 0. 0.]
segment_sum = [ 4.  5. 12.  0.], np.add.at gives [ 4.  5. 12.  0.]
50 steps, each adding one to two entries, chosen at run time, of a carry of 100000 entries: max error 0.0e+00
reused carry (in_place=(0,), also the default): workspace 800000 bytes, 0 loops over all 100000 entries in the C
reuse switched off for comparison:              workspace 1600000 bytes, 0 loops over all 100000 entries in the C
the same step written with index_add (fill mode), in_place=(0,): LoweringError: cannot prove in_place carry 'carry' of 'bump_fill'
without in_place=, index_add:                   workspace 2400064 bytes, 2 loops over all 100000 entries in the C

== 4. a conditional call runs one branch ==
negative branch, x=4
value at -4: 2.0
negative branch, x=4
gradient at -4: -0.25

== 5. a clipped rollout: select masks across iterations ==
d y / d x: reverse scan [0.      0.00012]
           finite: True; max |reverse scan - unrolled| 0.0e+00; max |forward scan - unrolled| 2.7e-20
d y / d us: reverse scan [0.      0.      0.06634 0.      0.14263 0.      0.      2.     ]
           finite: True; max |reverse scan - unrolled| 0.0e+00; max |forward scan - unrolled| 0.0e+00
```

In part 1 the three sources differ only in the literals of bounds and storage. Part 3 counts
workspace and loops over the whole carry. The step operations are in
[`../in_place/measurement.json`](../in_place/measurement.json). Its last two lines are #465.
Part 5 is the rollout whose forward tangent was NaN before #426's fix: the step takes the square
root of a clipped carry, and the first entry starts below zero.

## Reading list

About 285 lines, in reading order.

1. `ir/expr.py:985-995`. `is_map`: which loop results are a map.
2. `passes/lowering/calls.py:214-248`. `_lower_loop`: the used results, the trajectory slots,
   the reuse proof and the strict refusal.
3. `passes/lowering/calls.py:263-282`. The clamped trip count and the carry slots.
4. `passes/lowering/calls.py:375-387`. The decision to expand a static loop.
5. `ir/support.py:228-266`. `_loop_support`: the fixed point over the carries.
6. `ad/calls.py:339-357`. The head of the forward rule: which carries get a tangent and which
   a flag.
7. `ad/calls.py:734-766`. The head of the reverse rule: the carries a seeded output reaches.
8. `ad/calls.py:800-854`. The trajectory limit, the arguments of the backward loop, and the
   cotangents it returns.
9. `passes/lowering/in_place.py:104-164`. `_safe_link` and `carry_reuse`.

Second: `function/sugar.py::_loop` (the builder both loops share),
`passes/program/scalarize.py::expand_static_loop`, `ad/calls.py::_cond_jvp_many` and `_cond_vjp`,
`passes/lowering/gather.py::_address` (the clamped address of fill mode), and
`ad/sparsity.py::_loop_mask`.

## Source lines

Under `src/`, from `git show --numstat` of each squash commit: #411 +601 −120, #416 +113 −8,
#417 +272 −25, #454 +937 −148, #425 +114 −137, #426 +158 −11, #428 +195 −20. That is 2,390 added
and 469 removed, a net 1,921. After the simplification pass:
not available, the pass did not run (#467).

## Least sure of

- **The last day's rebases.** #454, #425, #426 and #428 all landed on 2026-11-12, each rebased
  over #434, #455 and #452. #425 resolved 15 conflicted files. The Opus reviews read the heads of
  the day before, and each integration had a check of the difference only, by `gpt-6.1-sol` or by
  Fable.
- **In-place carries next to lowering fusion.** The integration check of #452 found a gather
  moved past a later overwrite of its source, with wrong results in five forms, all correct
  without #452. It is fixed, with six regressions. This is the pair of features I would read
  first.
- **Select masks in reverse loops.** #428 added no transport and relies on the conditions inside
  each step. Four regressions and this demo's fifth part match the unrolled graph. The derivative
  guide's table says activity is not passed between steps, and I found no failing case, which is
  not a proof that none exists.
- **Guards nobody reaches.** Eight of the eleven guards of #29 cannot be reached by a loop whose
  body is a call. They start to matter with #71's inlining and #74's C locals, and nothing but
  hand-built programs tests them.
- **What was published.** The control-flow guide shipped in the same pull request as reverse
  mode. Its `in_place=` paragraph does not say that the documented update builders are refused
  (#465), and the functions guide beside it still says inputs must be float64 (#466).
- **Run time.** Every claim here is a source size, an operation count or a generation time.
  Whether a rolled loop with a call per step is fast enough for #51 and #205 is not known.
- **A small one.** Reciprocal hoisting does not track assignments nested in a hand-built `IF`
  (#417). Lowering cannot emit that shape today, so there is no issue.
