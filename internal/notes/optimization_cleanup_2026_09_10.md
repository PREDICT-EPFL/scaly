# Optimization cleanup before merging to dev

Approved for implementation, 2026-09-10. The initial review compared the branch with its merge
base with `dev`, including optimizations added outside the named pass pipeline. The findings below
record that review; execution is tracked under C-59 in `internal/todo.md`.

Keep the two intermediate representations, the explicit pipeline, and the measured lowering
choices. The cleanup should make each decision visible in the representation that can justify it,
share the analyses that passes already repeat, and test the boundaries between passes.
A replacement loop compiler is unnecessary for this merge.

## What was reviewed

The source paths below are relative to `src/scaly/`. Automatic differentiation is abbreviated AD;
a Jacobian-vector product is abbreviated JVP.

| Optimization | Current owner and branch change | Recommended disposition |
| --- | --- | --- |
| Joint tangents across active formals | `ad/forward.py`, single and multiple seeds | Keep in AD. The seed and formal-input structure is available here. |
| Constant and periodic seed specialization | `ad/forward.py`, repeated tiles of period at most eight | Keep in AD. Consolidate derivative-helper construction without losing periodic, local-coloring, or runtime-seed paths. |
| Packing compatible mapped derivative results | `ad/forward.py::_pack_jvp_maps` | Keep beside seed specialization. Preserve binding compatibility, result order, zero rows, and shared primal work. |
| Square, self-dot, division, and square-root derivative formulas | `ad/forward.py`, `ad/reverse.py` | Keep derivative formulas in AD. Ordinary arithmetic identities remain shared below. |
| Derivative lowering hints | `Function._effective_lowering`, `_inherit_lowering` in forward mode | Put the policy beside `Function._effective_lowering`; reverse mode should not import forward mode for this policy. |
| Arithmetic identities and constant evaluation | `passes/arith.py`, expression and Program adapters | Keep one identity implementation. Tensor evaluation and scalar evaluation still need different adapters. |
| Transpose-matmul, ones-vector, uniform-mask, and identity-gather simplification | `passes/expr.py` | Keep in the expression dialect; make their compilation entry point explicit. |
| Iterative rewriting | `ir/match.py` | Keep the shared driver and separate verifier. Do not merge them or add a pattern language. |
| Layout-sensitive matmul loops | `passes/lowering.py` | Keep here. Preserve four-row matrix-vector blocks, tails, and reduction-outermost products. |
| Affine gather/scatter indices | `passes/affine.py`, `LowerCtx.index_at` | Keep recovery at lowering. Preserve quotient telescoping and residual tables. |
| Invariant mapped-callee work | `passes/program/hoist_invariant.py` | Keep in Program form, before scalarization. Strengthen names, call safety, and reachability handling. |
| Bounded scalar expansion | `passes/program/scalarize.py` | Keep Program expansion and the existing budgets. Separate selection, value substitution, and scalar scheduling responsibilities. |
| Scatter-sum combination | Existing pass, extended for reshaped buffers and arithmetic indices | Keep in Program form, before fusion. Preserve grouping and source-read ordering. |
| Producer fusion | Existing pass, migrated to the shared iterative driver | Keep in Program form. Check the cost and safety of the expanded producer chain. |
| Loop arithmetic cleanup | New `passes/program/fold_arith.py` | Keep, including constant-buffer reads and constant fills. Add cleanup after unit-loop substitution. |
| Empty/unit loops | Existing pass, moved into the package | Keep after fusion. Share loop and substitution helpers. |
| Workspace packing | Existing pass, with generated-name collision protection | Keep late. Preserve alias lifetimes, nested workspace accounting, and the Apple-clang call-input workaround. |
| Two-lane stores and depth-bounded C expressions | `codegen/c.py`, with scheduling also in scalarization | Move the decisions into Program passes. Keep C syntax and vector typedefs in the renderer. |
| Native compilation flags | `codegen/jit.py`, reused by the benchmark harness | Keep at compilation. Preserve cache-key and invalidation agreement; this is not an IR pass. |

The package split is worthwhile. Its remaining problem is ownership within the package, not the
number of modules.

## Findings that determine the plan

### Generated names need a common rule

Two probes reproduce failures that the existing tests miss.

- `hoist_invariant._split` concatenates invariant positions without separators. A 13-input callee
  mapped with positions `(1, 2)` invariant and mapped again with `(12,)` invariant produces two
  procedures named `<callee>_hoist12` and two named `<callee>_hoisted12`. `verify_program` accepts
  that result. Name-keyed tables in later passes cannot distinguish the procedures.
- A block-form callee with an input named `_h0` and forty repetitions of `y = sin(y) + 0.1`
  fails C compilation. `_emit_scalar` declares `double _h0` while its initializer reads the
  input pointer `_h0[0]`. This reproduces the renderer-name limitation already recorded under C-53.

Allocate generated names against the containing namespace. Use an unambiguous positional suffix
for hoisted procedures, then check it against existing procedure names. Verify procedure-name
uniqueness before a later pass constructs a name-keyed table. Cover C identifier spelling when
checking local-name collisions.

### Expression rewrites have no uniform compilation boundary

`Function` construction and `_lower_to_proc` do not run `simplify`. Derivative builders and
explicit `sc.simplify` calls do. Consequently, an ordinary `A.T @ v` still materializes a
transpose, while its explicitly simplified equivalent uses the new layout-aware product.
The matmul tests explicitly simplify the expression, so they do not cover this difference.

Give compilation a private expression-normalization step per reached Function, before buffer
allocation. Use the existing expression rules instead of reconstructing transpose and gather
semantics from loads and stores. Preserve the user's Function graph, declared inputs, output
metadata, and effective lowering hint. Observe the normalized expressions as well as the original
ones. Do not inline calls before AD or change differentiation's input graph.

This is an intentional extension of where existing rules apply. Validate it separately from
mechanical file moves, especially for large graphs, typed constants, and opaque solver calls.

### Hints can disappear during simplification

The probe `sc.simplify((A.T @ v).block())` returns an `auto` expression. The same happens for
`.scalar()` and `.opaque()`, and for ones-vector, identity-gather, and neutral-element rewrites.
Some of these losses predate the branch; the branch now makes the hints operational.

Capture the effective Function hint before compile-time normalization. Also make rewrite
replacement policy explicit so an explicit hint is not accidentally dropped by a new rule.
An identity returning a declared input needs particular care: cloning that input with a different
hint creates a different expression identity. Retain a policy-carrying identity node when necessary
rather than introducing an undeclared input or discarding the hint. Test conflicting hints and
constant outputs as well as the ordinary derivative-helper path.

### Program analyses are scattered across passes

`fold_arith`, scatter combination, hoisting, and unit-loop removal import private helpers from
`fuse_elementwise`. `_stmt_refs`, hoisting's `_refs`, workspace packing's `calls_in`/`calls_out`,
and the renderer's `_reads` each recover related buffer information with different interfaces.

Move shared loop matching, substitution, and declaration pruning out of fusion. Provide one small
buffer-reference analysis that distinguishes loads, stores, call inputs, and call outputs, with
alias-owner resolution. Fusion must still pin every buffer passed to a call; distinguishing call
inputs from outputs must not weaken that rule. Passes should compute their own facts from this
shared analysis, without a persistent analysis cache or pass-manager framework.

Hoisting also needs an explicit boundary for opaque or solver-bearing calls. Its current split
analysis uses their reads and writes without proving their effects can move. Conservatively keep
such calls and dependent work in place. This is a source-inspection concern, not a reproduced
solver failure.

### Several interactions have direct reproductions

- For block-form `a = sin(x) + 1; y = a * a`, fusion emits `sin(x)` twice. It checks whether
  each original producer is expensive, but the cheap `+ 1` producer becomes expensive when its
  own producer is expanded. This is a source-level duplication; the downstream C compiler may
  recover sharing. Assess the expanded chain before accepting fusion, and test textual
  multiplicity as well as enclosing loop counts.
- A unit loop reading a nonuniform constant table at its loop variable keeps a `LOAD` after
  optimization, even though the index is then constant. `fold_arith` runs before the substitution
  and never runs again. Add a final arithmetic cleanup before workspace packing.
- An explicitly scalarized root that calls `sin(x)` leaves the callee procedure in the program
  even when no `CALL` remains. Scalarization removes calls but does not prune procedures.
  Hoisting has its own partial pruning rule. Use one reachability cleanup after call-changing
  passes, rooted at the entry and the solver-oracle relationships.

The fusion issue is inherited from `dev`; the branch's deeper expansion and shared rewriting make
it part of the cleanup review. The unit-loop gap is an interaction with the new arithmetic pass.

### The renderer now makes compiler decisions

`_emit_body` decides whether two stores can execute together, including an alias check.
`_emit_scalar` chooses temporary definitions and their scope. Neither decision appears in the
verified Program or its pass observations. Scalarization separately performs use counting and
depth-based scheduling.

Represent a paired store as one narrow Program statement containing a target and two scalar
values. This needs no vector expression dialect or general vectorizer. Choose pairs after
workspace packing, when physical buffer reuse is known. Preserve width two, double alignment,
`may_alias`, odd tails, and the rule against a later lane reading an earlier lane's destination.

Share scalar scheduling machinery between scalarization and final Program preparation.
Scalarization may share values across outputs because it has already resolved mutable buffers.
Final preparation must use statement-local scopes and preserve load timing across writes and calls.
Bound value and index expression depth, and reserve generated variable names. Do not turn this
into general memory-aware common-subexpression elimination.

### Some apparent duplication should remain

Arithmetic needs to run on expressions, during scalar substitution, and after fusion. Those
stages expose different operands. Affine recovery needs concrete index arrays at lowering;
`_index_values` answers a different question for the scatter pass after those arrays are gone.
Both should remain, with empty-range tests and consistent integer division semantics.

AD seed specialization is also distinct from Program scalarization. It specializes a mapped
callee without expanding the horizon. Hoisting shares work between trips; derivative packing
shares work between tangent contributions within a trip. Neither replaces the other.

Keep `cse_many`: expression interning does not subsume its commutative equivalence handling.
Do not run a whole pipeline to a fixed point. Workspace packing and scalarization have ordering
and budget constraints; only local canonicalization should repeat.

Within expression normalization, make repetition explicit. `simplify` has eight outer rounds,
`simplify_cse_fixpoint` has four, and the shared matcher and arithmetic folder also repeat at a
node. Some repetition handles newly created children; some handles commutative sharing. Use the
driver's replacement revisiting for the former, with termination and deep-graph tests, and retain
the latter only where common-subexpression elimination exposes another identity. Avoid silently
replacing all these bounds with an unbounded loop.

## Proposed pipeline

At each Function's compilation boundary, capture policy and normalize a private set of expression
outputs. Lower those outputs using the existing matmul and affine-index choices. Then run:

```text
hoist_invariant
scalarize
prune unreachable procedures
combine_scatter_sums
fuse_elementwise
fold_arith
unroll_unit_loops
fold_arith + prune unused buffer declarations
pack_workspace
coalesce adjacent stores
prepare scalar expressions for emission
verify and render
```

Keep the first arithmetic cleanup while adding the second. Folding can expose unit-loop bounds;
unrolling then exposes constant indices and more arithmetic. Give the two observations distinct
names. Remove an invocation only if interaction tests and measurements demonstrate that it is
redundant. Do not rerun hoisting, scalarization, or workspace packing to seek a fixed point.

The scalarizer's immutable output scheduling and final statement-local preparation should reuse
one scheduling implementation where their semantics agree. Preserve scalarization's budget
accounting over the scheduled candidate, including assignments and stores.

## Implementation phases

These are proposed phase contents and acceptance criteria, not a second active task list.
On approval, link the cleanup from `internal/todo.md` and track execution there.

### Phase 1: Pin the failures and pass contracts

Add small self-contained regressions for duplicate hoist names, `_h0` shadowing, hint loss,
transitive expensive fusion, post-unroll constant reads, and unreachable callees. Assert the
specific missing behavior so each regression fails on the current implementation.

Add tests that verify each observed Program stage, including global procedure-name uniqueness.
Record a compact baseline of source size, operation counts, workspace, and selected scalarized
procedures for the existing representative kernels. Preserve current public entry symbols,
output order, and arithmetic contract. Internal helper names may change.

### Phase 2: Centralize shared Program facts and generated names

Move common helpers into `passes/program/_common.py`; introduce another narrowly owned module
only if the resulting file warrants it. Replace pass-to-pass helper imports. Consolidate buffer
references without weakening alias or call protections. Allocate fresh local and procedure names,
and fix hoist suffixes. Keep workspace packing's call-input dependency protection intact.

Add one procedure-reachability cleanup, respecting normal entries, solver roots, external oracles,
and calls retained by other procedures. Keep callee-before-caller order and `hoisted_from`, which
the benchmark dispatch metrics use. Replace hoisting's private pruning logic with that cleanup.

### Phase 3: Define expression normalization and simplify AD helper construction

Normalize private compilation outputs and preserve policy before rewriting. Exercise ordinary
Function compilation without a manual `sc.simplify` call. Keep the original expressions available
to the recorder and keep output sparsity metadata attached to the correct outputs.

Move hint inheritance beside the existing Function policy. Consolidate the single-formal constant
JVP helper with the joint constant/runtime helper where their inputs and active-row behavior agree.
Keep periodic-tile selection, local coloring, and generic seeds as explicit branches with named
helpers in `ad/forward.py`. Preserve weak-key cache ownership and distinguish formal sets, seed
contents, output indices, and binding layouts. Do not unify forward and reverse derivative rules
or move their shape logic into the generic arithmetic adapter.
Simplify the expression-normalization loops using the shared driver's replacement revisiting,
while preserving commutative sharing and explicit termination limits.

### Phase 4: Close Program pass interactions

Evaluate fusion costs on expanded producer chains. Check source-buffer writes and aliases when
moving reads, using the shared analysis. Keep scatter grouping and the no-motion checks intact.
Add the post-unroll arithmetic cleanup and remove declarations it makes unused. Use the shared
reachability cleanup after scalarization and hoisting.

Keep automatic scalarization's operation, expansion-work, and aggregate-growth limits unchanged,
including explicit requests consuming capacity for later automatic candidates. Retain the
float64-only substitution boundary, block/opaque precedence, and auto prologues that expand only
inside an expanding caller.
Replace the boolean-or-string `scalarize` attribute with an explicit selection mode for disabled,
inline-only, and independently expandable procedures. Keep the original lowering hint separate
because it also determines whether automatic budgets apply.

### Phase 5: Make final scheduling and paired stores visible in Program form

Add the paired-store representation, verifier, text rendering, and C spelling together. Move
pair selection out of `_emit_body`. Extract shared scalar scheduling and remove renderer-generated
`_h` temporaries. Cover deep store values, load/store indices, and call offsets. Keep loop-dependent
calculations inside their loop and preserve the evaluation frequency of range expressions.

Update observers and metric walkers for the final Program form. Ensure paired stores do not
change arithmetic counts and one-time hoisted work remains excluded from per-trip dispatch cost.
Preserve solver-bearing translation-unit ordering, including procedures created by hoisting.

### Phase 6: Validate the combined compiler and refresh its documentation

Run the full suite and the repository's pinned format, lint, and strict type checks. Refresh
snapshots and the complete collection baseline only after examining the intended changes.
Compile representative scalar, loop, and solver-bearing artifacts with GCC and Clang where
available. Report unavailable platforms rather than claiming coverage.

Run controlled performance comparisons after tests finish, with the existing fairness protocol
and no competing compilation. Include race-car mapped derivatives, chain Hessians, npmpc
matrix-vector work, and the large unbumpercars affine-index case. Compare source size, compile
time, runtime, workspace, metadata, and dispatch arithmetic. Investigate regressions outside the
measurement noise before accepting the cleanup. Keep the existing interrupted study intact;
BH-20 remains separately tracked.

Update `docs/how_it_works/architecture.md`, `lowering.md`, `autodiff.md`, and Program reference
material to match the actual boundaries and observations. Add import-layer entries for any new
modules. Remove stale pass-number comments and obsolete helpers as their replacements land.

Phases 1 and 2 precede the semantic changes. Phase 3 and Phase 4 can be developed independently
after the shared interfaces settle, but integrate them before Phase 5. Phase 6 runs on the complete
result. One agent can execute the phases sequentially; parallel work is optional.

## Coverage to retain and extend

| Area | Existing evidence | Required additions |
| --- | --- | --- |
| Arithmetic and rewriting | Same identity cases across expression, scalar, and loop forms; invalid constants; integer truncation; deep rewrite tests | Policy-preserving replacements; float32/integer constant and store boundaries; constant-fill chains and refusal cases; cleanup after unit loops |
| Matmul and expression structure | All product shapes, row blocks/tails, exact integer data, transpose equivalence | Automatic compile normalization; cancellation-sensitive accumulation-order checks; empty reduction dimensions |
| Seeds and packed derivatives | Constant/runtime seeds, periodic tiles, zero rows, overlapping slices, mixed local/generic paths, Hessians | Period eight and rejection beyond eight; empty/single trips; all-zero and disjoint active rows; incompatible map bindings; cache reuse across layouts |
| Hoisting | Broadcast/varying inputs, two split variants, read between writes | Multi-digit position collisions; existing-name collisions; nested maps; zero/unit trips; opaque calls; alias writes; retained ordinary callers and solver oracles |
| Scalarization | Budget boundaries, aliases, repeated writes, nested calls, dtype rejection, reductions | Hoist-prologue interaction and reachability; shared scheduling keeps budgets and evaluation order |
| Fusion and scatters | Reduction fusion, matmul refusal, overlapping/shared scatters, grouping and alias-write guards | Transitive expensive work; repeated loads; mutated producer sources; empty arithmetic/table indices; verify that eligible cases still optimize |
| Final emission and workspace | Paired contiguous stores, odd tails, alias-dependent refusal, slot collisions, spill and call-input checks | Executed alias-dependent store cases; deep indices and generated-name collisions; paired stores after slot reuse; hoisted oracle translation units |
| Compilation and observations | Native flags, cache invalidation, dispatch metrics, source snapshots | Updated observer ordering; Program/C metric agreement; GCC/Clang runs with the same flags; no stale names or missing split procedures |

A numerical comparison alone does not prove an optimization ran. Pair it with a structural
assertion, then disable or perturb the optimization to prove that assertion fails. Likewise,
source-level operation counts do not prove a runtime improvement. Keep timing in the benchmark
harness and correctness reproductions in `tests/`.

## Review evidence and limits

The focused review run passed 305 tests in 19.61 seconds:

```sh
uv run pytest tests/passes tests/ad/test_joint_jvp.py tests/ad/test_const_seed_bake.py \
  tests/ad/test_lowering_hints.py tests/ir/test_match.py tests/codegen/test_c.py -q
```

At the time of the initial review, separate in-memory probes reproduced the naming, hint, fusion, unit-loop, and reachability findings
above. No regression tests or implementation changes had been added at that point. The full solver suite and
new performance comparisons were not run for this planning-only change.

The review used the [architecture](../../docs/how_it_works/architecture.md),
[conventions](../../docs/dev/conventions.md), [contributing checks](../../docs/dev/contributing.md),
[current lowering contract](../../docs/how_it_works/lowering.md),
[arithmetic investigation](algebraic_simplification_2026_09_08.md),
[refactoring notes](refactorings.md), [derivative operation audit](perf_2026_09_07/c49_ad_op_audit.md),
[closeout record](perf_2026_09_07/README.md), and [current task status](../todo.md).
The relevant implementations are [AD](../../src/scaly/ad/forward.py),
[expression rewriting](../../src/scaly/passes/expr.py), [shared arithmetic](../../src/scaly/passes/arith.py),
[lowering](../../src/scaly/passes/lowering.py), [Program passes](../../src/scaly/passes/program/),
[rendering](../../src/scaly/codegen/c.py), and [compilation](../../src/scaly/codegen/jit.py).

The range-based compiler, general memory-aware common-subexpression elimination, static metadata
redesign, and pre-AD inlining remain C-8, C-54, C-57, and C-58. This cleanup does not need them.


## Implementation record

The implementation retains the proposed pipeline and budgets. Shared buffer references now include
paired stores, aliases, and separate call inputs and outputs. Reachability runs after hoisting and
scalarization and retains solver oracles and callees reached from retained kernels. Generated pure
hoist procedures remain eligible for subsequent nested hoisting; automatic prologues only expand
inside an expanding caller.

Expression rewriting preserves effective subtree hints, including public `sc.simplify` calls made
before Function construction. The compiler also captures Function policy before private
normalization. Common-subexpression keys use canonical child identities so their size stays bounded
on deep graphs. Constant and runtime derivative helpers share cache construction while retaining
the specialized single-formal derivative path.

Store pairing and scalar preparation are visible Program passes. Scalar preparation reserves C
identifier spellings once per procedure and uses one increasing temporary counter. A counting test
pins that behavior: the first implementation repeatedly normalized the complete namespace for each
statement, making chain code generation much slower. Dispatch metrics now recognize the index
assignments that preparation can place before a mapped call, while still excluding the hoist
prologue and unrelated calls.

The depth bound covers store values, load/store indices, and call offsets. Range expressions remain
unchanged so their start, stop, and step retain their evaluation frequency. Hand-built deep range
expressions remain outside this bound; adding a new loop representation was unnecessary for this
cleanup.

The C snapshots deliberately change for private normalization, arithmetic cleanup, and explicit
scalar temporaries. Normalizing the wide corpus's constant transpose removes its 1,600-double
workspace. A separate 2,048-double shared-intermediate corpus retains workspace spill coverage.


### Initial controlled comparison

All 24 cells passed the existing correctness gate: three fresh processes per variant and workload,
with alternating before/after order. The baseline source was frozen before implementation. The
Google Benchmark harness used clang 20.1.8, `-O3 -march=native -fno-math-errno`, a 0.5-second minimum,
and separate output/cache directories. Recorded CPU settings were the performance governor with
boost off; no task tests or competing compilation ran during the comparison. This is a development
regression check, not a new headline library comparison.

Medians follow; arrows mean before → after. Generation includes derivative graph construction and
source rendering. Compile time includes the kernel, wrapper, and link steps.

| Workload and size | Runtime (µs) | Generation (s) | Compile (s) | C source (bytes) |
| --- | ---: | ---: | ---: | ---: |
| race_cars_50 | 23.458 → 23.459 | 0.434 → 0.525 | 0.861 → 0.859 | 39,972 → 39,935 |
| chain_5 | 201.932 → 201.984 | 35.901 → 36.833 | 3.175 → 3.195 | 492,819 → 473,295 |
| npmpc_12 | 34.713 → 34.665 | 1.082 → 1.167 | 1.187 → 1.185 | 47,011 → 46,917 |
| unbumpercars_32 | 6,933.389 → 6,940.400 | 4.514 → 5.903 | 18.878 → 19.072 | 903,324 → 896,320 |

Workspace, static metadata, and dispatch arithmetic are unchanged in every case. Procedure counts
show the removal of unreachable chain helpers; retained mapped work remains the same.

| Workload and size | Workspace (doubles) | Static metadata (bytes) | Arithmetic per dispatch trip | Procedures |
| --- | ---: | ---: | ---: | ---: |
| race_cars_50 | 8,472 | 39,578 | 1,129 | 6 → 6 |
| chain_5 | 109,944 | 474,100 | 21,780 | 16 → 3 |
| npmpc_12 | 0 | 22,150 | 28,138 | 5 → 5 |
| unbumpercars_32 | 1,285,536 | 345,288 | 908,094 | 16 → 16 |

Median runtime changes range from −0.14% to +0.10%, within the observed run-to-run variation. C
compilation changes range from −0.2% to +1.0%. Generation is slower by 2.6% to 30.8%; rendering alone
is slower by 10.0% to 60.8%. The additional normalization, reference analysis, post-unroll cleanup,
and scalar preparation have a measurable cost. The chain profile identified and removed repeated
namespace normalization as the largest avoidable regression: final preparation fell from 35.5 to
0.6 seconds under profiling. The remaining generation overhead is recorded as a cost of the cleanup,
not claimed as a speed improvement.

Raw samples, compiler logs, source fingerprints, environment records, and the reproducible runner
remain in `benchmarks/results/optimization-cleanup-2026-09-10/`; `summary.json` checks that source
fingerprints and structural metrics stayed identical across each variant's repetitions. Interrupted
exploratory samples are kept separately in `initial-comparison/`. The existing BH-20 study was not
changed.


### Validation

The final collection contains 909 tests; `uv run pytest -n=auto` passed all 909 in 50.52 seconds.
Pinned Ruff lint and format checks and strict `ty` checking passed with the unrelated, unfinished
`bruh.py` scratch file excluded. The full unfiltered type check fails on that file; it was preserved.
Separate GCC and Clang runs at `-O3` each passed 94 compiler, scalarization, and Program tests with
separate caches. Additional namespace regressions passed in the final suite. Both installed solver
plugins were exercised; macOS and other platforms were not tested.

The original hoist-pruning/prologue-policy and generated-pure-callee regression tests failed before
the corresponding fixes and passed afterward. Program observations verify the paired stores,
post-unroll constant reads, dispatch arithmetic, and unreachable-procedure removal. The scalar
namespace test counts normalizations to catch quadratic work without a timing threshold.

The first independent Fable review attempt was blocked by Claude's monthly spend limit. A later
retry completed and found shared output work duplicated by hint propagation, a stale chain
benchmark check, and smaller gaps in consumer-write checks and generic call and launch references.
The fixes preserve existing expression identities when rewrite replacements inherit hints, capture Function policy
separately during lowering, include consumer writes in fusion safety checks, and conservatively
treat invocation arguments with unspecified access modes as both read and written.

The follow-up also removes repeated namespace normalization in hoisting and repeated buffer
analysis in declaration pruning. Three compiled shared-output regressions and three Program
regressions failed before the corresponding fixes. Counting tests cover the two avoidable costs.
The final tree passed all 917 tests in 50.56 seconds. Separate GCC and Clang runs at `-O3` passed 101 tests each.
The chain `hinted_stage_hessian` check passed separately. Pinned lint, formatting, and strict types
passed with the unrelated `bruh.py` excluded.

Fable's first re-review identified an unnecessary output copy introduced by wrapping fresh
derivative-helper roots. Those roots now receive their hint directly. The existing scalar/block
helper test was extended to reject output-copy loops in adjoint, constant multi-seed, runtime
multi-seed, and single-seed helpers. It failed with four copy loops in the block case before the
fix and passes afterward. Fable's final review reports no open findings from its review.

### Final controlled comparison

The final tree was measured in 24 fresh cells under the same protocol: three repetitions per
variant and workload, alternating order, with no competing tests or compilation. All correctness
gates passed. Source fingerprints and structural metrics stayed identical across repetitions.
The original implementation baseline remains unchanged. The first follow-up comparison stopped
after one repetition to fix the derivative-helper copies and remains in `review-comparison/`.
The complete final samples and their runner are in
`benchmarks/results/optimization-cleanup-2026-09-10/final-comparison/`.

The following medians separate derivative construction from lowering and C rendering. Arrows
mean before → after. Generation total includes both stages, so a change in derivative-construction
time can mask the increase in lowering time, as it does for chain.

| Workload and size | Lowering and C rendering (s) | Generation total (s) | C compilation (s) | Runtime (µs) |
| --- | ---: | ---: | ---: | ---: |
| race_cars_50 | 0.191 → 0.271 | 0.430 → 0.504 | 0.864 → 0.861 | 23.450 → 23.495 |
| chain_5 | 2.350 → 2.657 | 36.446 → 36.217 | 3.175 → 3.255 | 201.592 → 201.823 |
| npmpc_12 | 0.919 → 1.002 | 1.083 → 1.167 | 1.186 → 1.202 | 35.021 → 34.691 |
| unbumpercars_32 | 2.282 → 3.410 | 4.490 → 5.628 | 19.015 → 19.206 | 6,986.507 → 6,917.560 |

Lowering and C rendering remain 9.0% to 49.5% slower, adding 0.080 to 1.129 seconds in these
workloads. Total generation changes range from −0.6% to +25.3%. The extra expression normalization,
arithmetic cleanup after loop removal, verification, and scalar preparation add Python graph
traversals and node construction. Profiling exposed repeated name normalization and repeated
reference analysis; those avoidable costs were removed. The remaining generation cost is a
measured tradeoff, accepted for this cleanup after review on 2026-09-10.

Runtime medians change by −1.0% to +0.2%, with overlapping before/after sample ranges in every
workload. C compilation changes range from −0.3% to +2.5%. Workspace, static metadata, dispatch
arithmetic, procedure counts, and C source sizes match the initial comparison's structural tables.
These three-repetition development checks establish no small runtime speedup claim.
