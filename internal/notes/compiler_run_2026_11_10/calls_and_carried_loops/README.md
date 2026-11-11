# Section 5: Calls and carried loops

Packet of section 5 of the [compiler run](../plan.md), parent #294. Built by #313 (#28) and #331
(#29), then simplified by #379. `sc.extern` (#83, in #355), the solver plugins on externs
(#342) and linker namespace reuse (#357) are still in flight, and their part is the last heading.
IR is the intermediate representation, AD automatic differentiation.

## Where this section sits

- A user who maps a Function with several outputs over a batch gets one loop that computes every output the caller uses, and a print inside the mapped body runs once per trip.
- Nothing else is visible to a user: carried ranges, `BREAK_IF` and exit variables exist only in the program dialect, and every optimization pass leaves them intact.
- Section 9 (Control flow, #23) builds `LOOP`, `scan` and `while_loop` on carried ranges (#31), differentiates them (#32, #33) and folds `VMAP` into `LOOP` (#34).
- Section 8 (Fusion and lanes) designs fusion with carried ranges present (#69), and section 4 (Dtypes, #24) counts loop trajectories in the workspace (#76).
- The section completes none of the Milestone 1 checks in the roadmap's *Milestone checks* table.

## Needs the maintainer's decision

1. Public interface, lands with #355: `sc.extern` and `sc.CLibrary` are exported from `scaly`, filtered from the API page and kept out of the guide until you have reviewed the interface. Recommend reviewing it before 0.2.0 is tagged, then publishing it or removing the two exports.
2. Documented behaviour (#313): a mapped Function with several used outputs prints once per trip, where it printed once per used output. A call is shared only when its arguments are the same expressions, so `callee(x)` and `callee(sc.print(..., x))` run the callee twice. The functions guide says both. Recommend keeping.
3. Roadmap departure, applied (#313): `invocation_key` also leaves out `slice_size`, because the siblings of one map have different result sizes. Recommend keeping.
4. Re-cut by the coordinator: #83 keeps `sc.extern`, `sc.CLibrary`, `EXTERN` nodes and their builds. Moving the solver plugins onto them and removing `SOLVER_CALL` became #342, and #357 joined the section. Recommend keeping the cut.

## Expected gaps

- No user can write a carried loop yet: #31 for `LOOP`, `scan` and `while_loop`, #32 and #33 for their derivatives.
- `VMAP` stays its own op beside the future `LOOP`: #34.
- Fusion and lanes have no rule for loop bodies, the passes only refuse carried ranges: #69, #71.
- The workspace does not count loop trajectories, which do not exist yet: #76.
- Reverse mode through a map with several used outputs runs one adjoint helper loop per output: #73.
- The solver plugins still go through `SOLVER_CALL`: #342.
- Many distinct external libraries in one process exhaust glibc's linker namespaces: #357.
- The wrapper parameter shadowing that #313's compile measurements worked around was #311, fixed on main by #351.

## Unowned, filed by this section

- #375 forward mode through a map with several used outputs runs one helper loop per output.
- #376 invariant work is not hoisted out of a carried loop whose body is more than one call, and unit loops inside an exit loop are not unrolled.
- #377 call arguments that differ only by a reshape round trip are two invocations since #313, where they were one call.
- Filed and now owned by the section as sub-issues of #294, each started in its own thread: #374, two exit loops that declare the same variable in one procedure pass the verifier and their C does not compile, and #380, `hoist_invariant` can hoist a top-level scalar assignment away from the statement that reads it (older than this section, reachable only from hand-built programs today).

# Reference

Line numbers are on main at `d287c6c3`, after #379, and paths are under `src/scaly/`.
Beside this file: `demo.py` and `demo_output.txt`, `loop_shapes.py`, and `invocations/` with the
snapshot diff and the measurements that #313 recorded.

## Decisions

Hardest to undo first. "Left open" marks a choice the roadmap did not settle.

### Tracker and public interface

1. **#83 was cut in three.** #83 keeps the external-code interface: `sc.CLibrary`, `sc.extern`,
   `EXTERN` nodes grouped by invocation, and their JIT and ahead-of-time builds. #342 moves the
   PIQP and IPOPT plugins onto externs and removes `SOLVER_CALL`, blocked by #83. #357, found when
   #355's review loaded 15 distinct dependency sets and hit glibc's namespace limit, makes
   compatible libraries share a namespace. The cut lets the interface land and be reviewed alone.
2. **`sc.extern` and `sc.CLibrary` are exported but not published** (lands with #355).
   `scaly/__init__.py` exports both with docstrings. `docs/api/core.md` filters `EXTERN` from the
   op list and leaves both names out of its curated entries, and no guide mentions them. A user
   who reads the source can call them, so the names are a commitment once 0.2.0 is tagged.

### Documented behaviour

3. **A mapped Function with several used outputs is one loop** (#313;
   `passes/lowering/calls.py`). Every used result is written by the same call in the same trip,
   so a print in the mapped body runs once per trip instead of once per used output and trip.
   `docs/guide/functions.md`, *Printing from generated code*, says so. An unused result still gets
   a scratch buffer, because the callee's procedure writes all its results.
4. **Invocations are keyed by argument identity** (#313; `ir/expr.py`, `invocation_key`). Two
   calls share one emitted call only when the callee is the same object, the arguments are the
   same expressions, and for a map the length, starts and strides agree. The old key was the
   callee and the names of the argument buffers. `PRINT`, `RESHAPE`, `STOP_GRADIENT`, `PACK` and
   `VALUES` alias their operand's buffer, so the old key merged `callee(x)` with
   `callee(sc.print(..., x))`, and the new one runs the callee twice. The guide documents this.
   The cost is an extra call whenever simplification leaves two spellings of one value apart,
   which today happens for a reshape round trip (#377).

### IR and lowering

5. **`invocation_key` leaves out `slice_size`** (#313, roadmap departure, applied). The roadmap's
   key was the op, the callees by identity, the argument ids and the attributes except `output`.
   `slice_size` is the stored size of the selected result, so siblings of one map differ in it
   and would never group. The roadmap's [#28] paragraph has the replacement line.
6. **`callees_of` returns a tuple** (#313). It holds one callee today. `COND` will return two,
   and the eleven graph walks that used a hard-coded `{CALL, VMAP}` set already loop over it.
7. **Residuals are siblings of the primal** (#313). A custom forward rule's residual results
   select `callee.results` past the declared outputs. They group with the primal's invocation,
   and the callee's procedure writes them only when some call reads one.
8. **`carried` is an attribute of `RANGE`, `exit_var` an attribute of `FOR`** (#331;
   `ir/program.py`). Both are absent when false, so every program built before #331 interns and
   renders unchanged. A carried range keeps its `kind`, `SERIAL` by default. Left open.
9. **What the verifier forbids** (#331; `ir/program_spec.py`). A carried range of kind `VECTOR`.
   `exit_var` on a range that is not carried. An exit loop whose start is not the constant 0 or
   whose step is not the constant 1, since the exit variable is then not a trip count (left
   open). A `BREAK_IF` whose nearest enclosing loop is not a carried exit loop: a plain loop in
   between withdraws the permission, a `BLOCK` does not, and it never crosses a procedure (left
   open: the roadmap says "inside a carried range with `exit_var`", not "nearest"). A `BREAK_IF`
   without exactly one bool scalar. Markers that are not booleans.
10. **C spelling** (#331; `codegen/c.py`). An exit loop renders as `long long i = 0;` in the
    enclosing scope followed by `for (i = 0; ...)`, and `BREAK_IF` as `if (...) break;`. A
    `BLOCK` statement now renders its statements in place. Two exit loops with one name in one
    scope do not compile (#374).
11. **Pass guards** (#331). The roadmap listed five, and #331 added four where a reproduction
    failed.
    - `scalarize.work` returns no estimate for a carried range, so a procedure that holds one is never expanded, nor are its callers.
    - `unroll_unit_loops` keeps a loop with `exit_var` whatever its trip count, and does not visit its body. A carried loop of one trip without an exit variable is unrolled, which is correct.
    - `widen_ranges` never widens a carried range or a range that contains one.
    - `fuse_ranges` takes a carried range neither as producer nor as consumer.
    - `hoist_invariant` keeps the loop's attributes when it rewrites the call in its body. Inside a callee it leaves in the body every statement that names a scalar of a carried loop, the loop index included, and hoists the rest when it is pure.
    - Added: the shared producer matcher `inline_producer` (used by `fuse_elementwise`, `fold_arith` and `combine_scatter_sums`), the index splitting of `fold_arith`, the scatter matcher of `combine_scatter_sums` and `pack_masks` each skip a carried range.
    - `pack_workspace` is unchanged. The roadmap gives it the trajectories, which arrive with #31 and are counted by #76.

### Left as it is by the simplification pass

Judgement calls that #379 did not take.

- The carried-work rule of `hoist_invariant` is three cases over two set comprehensions. A shorter rule over scalar names was tried and withdrawn: the review showed that refusing more statements can leave a hoisted scalar without its reader (#380).
- `_lower_call` and `_lower_vmap` repeat seven lines that pick each result's layout and scratch buffer. Sharing them means a helper that returns five values, in a file #355 edits.
- `p.break_if` checks its argument and the verifier checks it again. The other builders do the same.
- The verifier checks that `carried` and `exit_var` are booleans, as it does for `ASSIGN`'s `declare`.
- The C renderer handles `BLOCK` only because one test wraps a `BREAK_IF` in one. No pass creates a `BLOCK` in a body.
- An exit loop's C initializes its variable twice.
- The key holds `id(callee)` although `_attrs_key` would key a Function attribute by identity too. The roadmap asks for identity by name, so it stays explicit.
- `_result_buffers` maps every sibling to its buffer in a second loop. It matters only when two nodes select the same result, which needs nodes that differ in their type alone.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/calls_and_carried_loops/demo.py` from a checkout.
It maps the `vmap_multi` snapshot's Function, with a print added, over five rows, then compares
the snapshot with the numbers #313 recorded, then sends a hand-built carried range with a
`BREAK_IF` through `optimize_program` with lanes and reciprocal hoisting on. `demo_output.txt` is the whole
output on la015, and this is the same output without the snapshot diff and the pass names.

```text
== #28: a mapped Function with two used outputs and a print ==
  for (long long it_t0 = 0; it_t0 < 5; ++it_t0) {
    pair_raw((arg[0] + (2 * it_t0)), (s0 + (it_t0 * 2)), (res[1] + it_t0), NULL);
  }
loops in the entry that call the mapped body: 1
trip x=[0, 1]
trip x=[2, 3]
trip x=[4, 5]
trip x=[6, 7]
trip x=[8, 9]
prints: 5 for 5 trips and two used outputs (before #28: 10)

[the snapshot diff, invocations/snapshots.diff]
                              before #28   after #28  this checkout
program nodes                         57          50             50
calls                                  2           1              1
loops                                  5           4              4
buffers                                8           7              7
generated C, bytes                  4889        4747           4720
cc -O3 -c, median of 3, ms          32.8        35.6           33.7
The first two columns are recorded in invocations/measurements.json (baseline 8d8c7870). Counts include the
mapped body's own procedure. The recorded renderings are 27 bytes longer than the snapshot file, for a reason the
record does not give. The diff above is the 142 bytes between them. The compile times come from a shared machine
and show no difference.

== #29: a hand-built carried range with BREAK_IF ==
program
  proc recurrence(flags:bool(6,), out:float64(1,), count:int64(1,)):
    out[0] <- 0
    for i in [0, 6) step 1 kind=serial, carried, exit_var:
      break_if flags[i]
      out[0] <- ((out[0] / 2) + 1)
    count[0] <- i

the loop keeps its six trips, exit variable and break after each of 16 passes (lanes=2, reciprocal=True):
[the sixteen pass names]

static inline void recurrence_raw(const uint8_t* flags, double* out, int64_t* count, double* w) {
  (void)w;
  out[0] = 0.0;
  double inv = (1.0 / 2.0);
  long long i = 0;
  for (i = 0; i < 6; ++i) {
    if (flags[i]) break;
    out[0] = ((out[0] * inv) + 1.0);
  }
  count[0] = i;
}
```

## Reading list

About 270 lines, in reading order, on main at `d287c6c3`.

1. `ir/expr.py:748-757`. `callees_of` and `invocation_key`.
2. `passes/lowering/ctx.py:244-247`. The used results of each invocation, collected once from the order lowering already walks. Then `661-667`, where a call that selects a residual marks its callee.
3. `passes/lowering/calls.py:28-117`. `_lower_call`, `_result_buffers` and `_lower_vmap`: one call writes every used sibling, and the map's loop holds that one call.
4. `ir/program.py:351-380`. The builders `range_(carried=)`, `for_(exit_var=)` and `break_if`.
5. `ir/program_spec.py:27-37`, the walk that places `BREAK_IF`, then the rules at `139-142`, `156-168` and `177-180`.
6. `codegen/c.py:278-296`. The C for a `BLOCK`, an exit loop and a `BREAK_IF`.
7. The guards: `passes/program/scalarize.py:145-150`, `unroll_unit_loops.py:30-32`, `widen_ranges.py:82-96`, `fuse_ranges.py:76-81`, `hoist_invariant.py:95` and `123-148`, `_common.py:36-42`, `fold_arith.py:139-147`, `combine_scatter_sums.py:70-71`, `pack_masks.py:100-101`.

## Source lines

Lines under `src/`, from `git show --numstat`.

- #313 added 107 and removed 87, in 11 files. #331 added 115 and removed 17, in 13 files.
- Before the simplification pass the section had added 222 lines, a net 118.
- #379 added 10 and removed 19, in five files. After it the section's net addition is 109 lines.

## What the coordinator was least sure of

**That leaving `slice_size` out of the key cannot group two invocations that should stay apart.**
It cannot. `ir/expr_spec.py:310-311` rejects a `VMAP` whose `slice_size` is not the
stored size of `callee.results[output]`, and `_mapped_call` is the only builder. The size is
therefore a function of the callee and the selected result, and two nodes that agree on the
rest of the key and on `output` have the same size. Nodes with different `output` are the
siblings the key is meant to group. Length, starts and strides stay in the key, and
`test_different_map_layouts_are_distinct_invocations` pins them.

**That every pass guard of #29 is reachable once `LOOP` emits carried ranges.** Not all of them,
if `LOOP` lowers to a call in a carried range. `loop_shapes.py` builds three such programs by
hand: a scan (`CALL` the step, copy the carry back), a while loop (`BREAK_IF`, `CALL`, copy,
exit variable) and a loop whose body is one call. It runs each through `optimize_program` for
three scalarization modes, both lowering hints, lengths 0, 1 and 5, lanes 1, 2 and auto, with
and without reciprocal hoisting, compiles the C and compares with NumPy. All 324 configurations
agree. I then removed the guards one at a time in a scratch copy of the source. Three removals
break these programs: `scalarize.work` (36 configurations), the exit-loop rule of
`unroll_unit_loops` (144) and the loop attributes kept by `hoist_invariant` (108). The other
eight change nothing, because a body that holds a call is already refused by the passes they
guard. They are reached only when the stores sit directly in the carried range, as in #331's
hand-built tests, which needs #31's lowering or #71's inlining to put a body there. The
carried-work rule of `hoist_invariant` is reached when a Function that holds a loop is itself
mapped with a broadcast argument, which #331's test builds and these shapes do not. #31 should
rerun the removal experiment on real `LOOP` programs before relying on the eight.

**That `sc.print` inside a callee keyed by argument identity behaves as the guide says.** It
does. The callee prints once for `pair(x)[0] + pair(x)[1]`, for one used output, and for two
identical printed arguments, which print once themselves. It prints twice for `pair(x)` with
`pair(sc.print(..., x))`. Mapped, it prints once per trip for two used outputs and for the same
map built twice, and twice per trip when one map's argument is printed. The gradient and the
Jacobian of a scalar of both outputs print once, or once per trip when mapped. "The same
expressions" is judged after simplification: `x * 1.0`, `x + 0.0` and `1.0 + x` against
`x + 1.0` share the call, and a reshape round trip does not (#377).

Two things I did not expect. The sizes in `invocations/measurements.json` (4,889 and 4,747
bytes) are 27 bytes above the snapshot file #313 committed (4,720), and the record does not say
why. The difference between them, 142 bytes, is exactly the snapshot diff. And no benchmark
shows #28's gain: the benchmark oracles select one result per map, and derivatives still run
one helper loop per used output (#73, #375).

## `sc.extern`, solver plugins on externs, linker namespaces (#83, #342, #357)

To be written by the coordinator when #355, #342 and #357 have merged. `extern.md` beside this
file is #355's own note.
