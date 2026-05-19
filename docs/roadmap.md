# Alloy roadmap

Alloy is an experimental, pure-Python symbolic IR package living beside `anvil`. It should stay isolated until the design proves that it can express CasADi-like modeling, AD, solvers, integrators, and code generation better than extending the current anvil/tinygrad stack directly.

The guiding idea is:

```text
named Function + sparse typed graph + derivative factory + call nodes + mixed scalar/block lowering
```

## Current experiment pivot

The IR-level proof of concept is in place. Tracking NMPC and unbumpercars showed that one Alloy graph — named `Function`s, MAP-based loop preservation, colored sparse AD, and the scalar C renderer — can match CasADi SX runtime within ~10% and emit 20-58× less C source than SX (see [`scalability.md`](scalability.md) and the Phase 1/3 progress below). The next pivot is from "prove the IR on Jacobian benchmarks" to "build a real control workload on Alloy."

### Driving application: CBF safety filter with neural dynamics

The concrete target is a CBF-based safety filter for a system whose continuous dynamics are a neural network. Two variants matter:

1. **Input-affine model** `xdot = f(x) + g(x) u`, with `f`, `g` neural networks. At each safety-filter call `x` is fixed, so `f(x)`, `g(x)`, and any required derivatives are evaluated once. The CBF condition is then affine in `u` and the filter is a **QP** with constant data for that step. PIQP is the candidate sparse-QP backend.
2. **Fully nonlinear model** `xdot = f(x, u)`, with `f` a neural network. The CBF condition is nonlinear in `u` and the network must be re-evaluated at every optimization step. The filter is an **NLP** with IPOPT as the candidate backend.

Both variants stress the capabilities the IR proof already covered (dense NN matmul/activation blocks + scalar/sparse barrier assembly + named-function reuse + colored sparse Jacobians) and force the two pieces still missing for a usable surface: a JIT execution path and first-class QP/NLP solver functions.

Success means a single Python file that declares the neural dynamics as an Alloy `Function` (loading weights from a `.pth` checkpoint), declares the CBF condition and slack/objective terms, assembles the filter through `al.qp(...)` or `al.nlp(...)`, and is callable through the universal ABI from a small C++ harness with realistic per-step runtime.

Historical benchmarks (tracking NMPC equality Jacobian and unbumpercars inequality Jacobian) remain in `tests/alloy/*_workload.py` and `benchmarks/scalability_sweep.py` as regression guards while the focus moves to the new workload.

## North star

Alloy should become a small CasADi-like core with a modern compiler architecture:

- **CasADi-like user model**: symbolic expressions, named functions, derivative factories, QP/NLP builders.
- **Unified IR**: no hard global SX/MX split; individual regions can lower as scalar code, block loops, kernels, or opaque calls.
- **AOT-friendly ABI**: universal CasADi-style C ABI plus optional typed wrappers for safer C++ use.
- **Control-first AD**: sparse Jacobians/Hessians, Lagrangian Hessians, graph coloring, and solver sensitivity support.
- **Backend-flexible codegen**: scalar C, C++ wrappers, loop-preserving kernels, and GPU code only once a workload demands it.

## Concept map

A useful way to keep Alloy aligned with compiler/tinygrad/anvil terminology:

| Alloy concept | Compiler role | tinygrad/anvil/CasADi analogy |
| --- | --- | --- |
| `Ops` | opcode enum | tinygrad `Ops` |
| `Expr` | immutable operation node / SSA value | tinygrad `UOp` |
| `TensorType` | value type metadata | shape/dtype/sparsity metadata around tinygrad buffers |
| `Function` | graph boundary and compilation unit | anvil `NumericalFunction`, CasADi `Function` |
| `CallOp` / `Ops.CALL` | function invocation node | CasADi call node, future external/solver/integrator calls |
| `Tape` / `Instruction` | linearized schedule | tinygrad linearized UOps, CasADi `algorithm_` |
| `Function.factory()` | derivative/helper function builder | CasADi factory request language |
| `alloy.api.gradient(fn, ...)` | human convenience layer | thin wrapper over factory requests |
| lowering hints | region/codegen policy | SX-like scalar vs MX-like block lowering choice |
| future `PatternMatcher` | graph/tape rewrite system | tinygrad `PatternMatcher`/`UPat` |

This map should stay visible in the design: `Expr` is not the high-level tensor API forever; it is the IR node. We can add nicer matrix/tensor facades later, but they should lower to `Expr`/`Ops` rather than hide a separate graph representation.

## Construction model

Alloy should support the low-level CasADi-like style of creating symbolic values and then wrapping them in a `Function`, because it is useful for tests, imports, debugging, and graph surgery. It should not be the only or preferred user model. Long-term user-facing modeling should look closer to anvil: a Python-scoped function/decorator/builder creates fresh symbolic inputs from a declared signature, traces the body, and returns a named `Function`.

Reasons to keep the explicit graph constructor:

- it is the smallest IR-level API for unit tests and compiler passes;
- it makes imported CasADi/anvil graphs easier to represent;
- it allows deliberately shared symbolic placeholders when constructing small graphs by hand.

Reasons to make scoped construction the primary API:

- accidental cross-function placeholder sharing is avoided;
- function boundaries and names are known at trace time;
- call nodes, derivative factories, and future JIT caches can hang off the named `Function` immediately;
- it matches the way anvil users already write numerical models.

Shared subgraphs across functions should normally be represented as named `Function.call(...)` nodes, not by leaking the same raw placeholder into several graph builders.

## Non-goals for the experiment

- Do not replace `anvil` yet.
- Do not pursue full CasADi API, factory-syntax, or C ABI compatibility before the IR proves itself. CasADi equivalence is a benchmark and semantic guide; exact surface compatibility is not central right now.
- Do not implement every special function or solver plugin early.
- Do not make splines/interpolants a quick hack; their semantics deserve a dedicated design.
- Do not force solvers into pure symbolic graphs when opaque call nodes are more honest.
- Integrators and broader CasADi/anvil interop are out of scope for this experiment; they are not on the roadmap.

## Phase 0 — Seed package and vocabulary

Status: started.

Implemented:

- `src/alloy` as a separate package.
- `Expr`, `Ops`, `Function`, `Tape`, `Instruction`.
- Basic NumPy evaluation.
- Basic symbolic JVP-based `jacobian`, `gradient`, `hessian`.
- CasADi-like `Function.factory()` for `jac:*`, `grad:*`, `hess:*`, `lam:*`, and aux Lagrangian outputs.
- Human-friendly wrappers like `gradient(fn, input_name, output_name)` built on top of factory requests.
- First-class `call` expressions.
- Lowering hints: `auto`, `scalar`, `block`, `opaque`.
- Universal C ABI spelling and optional typed buffer header generation.
- CasADi equivalence tests for simple gradients, Jacobians, and Hessian-of-Lagrangian cases.

Exit criteria:

- Unit and CasADi-equivalence tests pass.
- The spec and roadmap describe the experimental scope.

## Phase 1 — Solidify the symbolic core

Goal: make Alloy comfortable enough for small nonlinear models.

Status: in progress.

Progress:

- `Function` evaluation now runs through the topologically linearized `Tape` interpreter; recursive `Expr.eval` remains as a reference/debug path.
- `Tape.plan_workspace()` provides a deterministic lifetime-based temporary workspace layout for non-leaf instructions.
- Basic construction/execution errors are explicit for function signature arity, undeclared symbolic inputs, call argument shapes, missing inputs, input shape mismatches, invalid `matmul`, invalid transpose axes, and invalid stack/concat axes/shapes.
- Structural coverage expanded with `transpose`, `Expr.T`, `vec`, `slice`/indexing, `split`, flat `gather`, `scatter`, and `concat`; JVP rules and tape evaluation support them.
- Reduction helpers now include shape-checked `dot`, `sumsqr`, and `norm_2`.
- Named-function keyword evaluation now rejects missing and extra keyword inputs instead of silently ignoring extras.
- JVP can differentiate through `CallOp` by inlining the callee derivative graph; this is the first policy for nested named-function AD.
- Added explicit `ScalarType` and `SparsityType` vocabulary, plus symbolic `jacobian_sparsity(expr, wrt)` for the current structural/arithmetic subset, including call-node chain rule.
- Added compact `SparseJacobian` metadata/values wrapper, CSR/CSC conversion helpers on `SparsityType`, and greedy column coloring helpers for future sparse derivative evaluation.
- Added a first colored sparse-Jacobian value path: `sparse_jacobian_colored` uses `jacobian_sparsity`, `column_coloring`, compressed seed vectors, batched color JVPs through structural/nonlinear elementwise subsets and call structure, and simplification/CSE cleanup; `sparse_jacobian_reference` keeps the old dense-Jacobian-then-gather implementation for correctness checks. The default `spjac` factory path now uses the colored implementation. The tracking path preserves named stage derivative calls and color batches, but still lacks a true horizon loop IR.
- Added factory-level `spjac:out:in` and human `spjacobian(fn, input, output)` wrapper; sparse outputs carry `Function.output_sparsities` metadata.
- Added `sphess:out:in:in`, `sphessian`, and sparse Lagrangian-Hessian convenience wrapper over auxiliary factory outputs.
- C API header generation now emits sparse output metadata (`NNZ`, `NROW`, `NCOL`, COO `rows`/`cols`, CSR, and CSC arrays) for compact derivative buffers.
- Added structural equality/hash helpers separate from `Expr.id`, plus CSE over structurally equivalent subgraphs.
- Added stable topological debug printing (`Expr.debug()`, `format_expr`) for inspecting small graphs without relying on construction IDs.
- Added tape debug printing (`Tape.debug()`, `format_tape`) with explicit non-`auto` lowering annotations.
- Added `Tape.lowering_regions()` and `TapeRegion` as a first contiguous-tape grouping of lowering metadata.
- Added a tiny graph rewrite layer (`Pattern`, `PatternMatcher`, `rewrite`, `simplify`) with constant folding and first algebraic cleanups (`x + 0`, `x * 1`, `x * 0`, identity reshape/transpose). Rewrites and CSE now walk the DAG in topological order with replacement caches instead of recursive tree traversal.
- Added `.opaque()` and the `opaque` lowering hint to represent named call, solver, and integrator boundaries before full mixed-lowering region formation exists.
- `TensorType.diff` now propagates through expression constructors, structural ops, call outputs, and marks nonsmooth ops like `floor`/`minimum` as non-differentiable metadata.
- `TensorType`, `SparsityType`, and symbolic shape construction now reject negative dimensions, and tensor sparsity metadata must match tensor shape.
- Explicit split section lists now reject negative sizes before building slice expressions.
- `Function` now validates that sparse output metadata agrees with the compact output buffer size.
- Forward and reverse AD now prune subgraphs independent of the requested variables, so nonsmooth parameter-only terms do not block smooth derivatives.
- `SparsityType` now rejects duplicate COO coordinates so `nnz` and compact sparse buffers stay canonical.
- `Expr` now has method forms for binary nonlinear helpers: `.atan2(...)`, `.minimum(...)`, and `.maximum(...)`.
- `Function.call(...)` now normalizes raw constant-like arguments into `Expr` nodes before shape checking and call-node construction.
- Added an anvil-style `@alloy.function(...)` scoped construction helper that creates fresh symbolic inputs from a declared signature and returns the same IR-level `Function` object as explicit graph construction.
- JVP construction now memoizes subgraphs and substitutes call derivatives with a topological cache, which is required for named-call Jacobians such as the tracking NMPC stage dynamics.
- Added the tracking NMPC equality-Jacobian fixture with an opt-in `ALLOY_TRACKING_SWEEP=1` pytest harness for horizons `N=1,2,5,10`, recording graph size, sparsity nnz, color count, colored sparse-AD construction time, and generated source size while comparing compact values against CasADi.
- Added `benchmarks/alloy_tracking_eq_jac_benchmark.py`, a Google Benchmark C++ harness generator for Alloy vs CasADi SX/MX tracking equality Jacobians. After local-temporary codegen, slice aliasing, and constant-seed specialization, a local `N=50` run produced Alloy code at 364 KB / 13112 lines / 0 workspace and 5.1 us, CasADi SX at 466 KB / 25902 lines / 77 workspace doubles and 4.6 us, and CasADi MX at 2.0 MB / 63134 lines / 8046 workspace doubles and 8.4 us.
- Added an official-size unbumpercars inequality-Jacobian fixture in `tests/alloy/test_unbumpercars_workload.py`, using the example model dimensions (`256 -> 128 -> 3`) and official `model_kinematic_mlp.pth` weights in the parameter vector, plus scalar C3BF/wall/slack sparse assembly, CasADi MX dense/sparse structural checks, and lowering metadata assertions. Added `benchmarks/alloy_unbumpercars_ineq_jac_benchmark.py` for Google Benchmark comparisons against CasADi SX/MX. A local official-size `N=2` stats run after zero-matmul, zero-constant cleanup, and active color-lane compression produced Alloy at 61 KB / 2289 lines / 0 workspace, CasADi SX at 12.2 MB / 447759 lines / 36274 workspace doubles, and CasADi MX at 217 KB / 6689 lines / 141500 workspace doubles; the Alloy native run was 48 us and the latest Alloy-vs-MX run was 46 us vs 237 us. SX is now correctly enormous for the dense MLP case and is best inspected with `--stats-only` unless a long compile is desired.
- Fixed a C codegen correctness bug where `_contiguous_slice_offset` accepted column-style slices like `(slice(None), 1)` on a `(5, 7)` tensor and emitted a `v + 1` pointer alias that read flat indices `[1,2,3,4,5]` instead of the column `[1,8,15,22,29]`. The Python tape interpreter still produced the correct values via NumPy, so the bug only manifested in compiled C. Added a regression test that compiles a column slice and verifies its values.
- Simplify now iterates the rewrite loop to fixed point. Added two structural rewrites that benefit the benchmark paths: `slice(stack(args, axis=A), index)` with a full slice on `A` becomes a stack of per-arg slices, and `slice(slice(x, idx1), idx2)` with step-1 indices composes into a single slice. Together they eliminate intermediate stacked tensors that were materialized only to read individual rows/columns, and they collapse double pointer aliasing chains like `v9 = v8 + 0; v10 = v9 + 0` introduced by the JVP slicing path.
- Added several codegen cleanups: literal `(k % N)` / `(k / N)` coord expressions fold when the loop variable is a constant, sparse-constant matvec terms drop `1.0 * x` factors (and emit `-x` for `-1.0`), and single-use scalar (size-1) elementwise ops now inline their C expression at the consumer instead of materializing a 1-element buffer.
- Extended the gather-transposed-concat peephole to also recognize `gather(transpose(stack(args, axis=1)))` where each stack arg is rank-1, skipping the materialized stack+transpose buffer the colored sparse-Jacobian assembly used to leave behind.
- Generalized the inline table to single-use vector elementwise ops (`v_X[k] = expr(args[k])`) and threaded it through `STACK`/`CONCAT` arg reads and the gather peephole's `_read_element` recursion. Chains like `v115 = -v44 * v114; v116 = v37 * v115; v117 = v107 + v116; v118 = v117 * v42` collapse into one fused loop body. After these passes the official `N=2` unbumpercars Jacobian source dropped to 38 KB / 1235 lines (N=4 to 137 KB / 4251 lines) and the tracking `N=50` source dropped to ~340 KB / ~13 k lines, while Alloy's native runtime is ~94 us for the official `N=2` unbumpercars Jacobian (vs ~252 us for CasADi MX) and ~4.97 us for tracking `N=50` (vs ~4.6 us for CasADi SX and ~8.3 us for CasADi MX).
- Added a fail-fast correctness check to both benchmark harnesses. Each Python-side run scatters Alloy's Python compact output into a dense Jacobian, cross-checks it against the dense factory output, and writes the realistic sample input plus the dense reference to side-by-side binary files. The compiled binary loads these at startup, runs each backend, scatters its compact output into a dense matrix using that backend's own `(rows, cols)`, and reports an exact per-element diff (including NaN-vs-finite mismatches) before any benchmark runs. The earlier compact-vs-compact comparison was wrong: Alloy emits COO row-major nnz order while CasADi emits CSC column-major. This check also re-exposed a `~2x` runtime regression that the prior column-slice codegen bug had been silently masking on the unbumpercars workload (the buggy pointer alias was reading flat indices into the zero-padded stack rows, so most of the "fast 46 us" computation was effectively no-ops on zeros).
- Added lifetime-based buffer packing. Each storage-owning instruction is now assigned to a shared C local `sN` sized to the max instruction in the slot; aliases (RESHAPE / contiguous SLICE), inlined chains, and peephole-skipped intermediates propagate their last-use to the underlying storage so a reused slot is never overwritten while still observable. The MLP JVP callee's stack frame drops from ~25 KB / 16 disjoint 256/128-element buffers to ~14 KB across 9 shared slots, the unbumpercars `N=2` source drops to 34 KB / 1129 lines, and tracking `N=50` runtime improves from ~4.97 us to ~4.66 us (vs CasADi SX ~4.37 us, CasADi MX ~8.09 us). The benchmark correctness check guards every backend so any lifetime bug fails loudly.
- Added `benchmarks/scalability_sweep.py`, a unified per-cell sweep across tracking horizons `N ∈ {1, 5, 10, 25, 50, 100, 200, 500, 1000}` and unbumpercars car counts `C ∈ {2, 4, 8, 16, 32}` with all three backends. Each cell compiles its own Google Benchmark binary (with a configurable per-compile timeout and a max-source-size cap) so the existing dense-comparison correctness check guards every measurement. A backend that hits a compile timeout or size cap at one cell is short-circuited to `skipped_after_failure` for every larger cell of the same workload, because both source size and compile cost are monotonically increasing in the iteration count. Results live in `benchmarks/scalability_results.csv`; see `docs/alloy/scalability.md` for the summary tables and the comparison against the `tracking-nmpc-benchmarks` worktree experiments. On the tracking workload Alloy runtime sits within ~10% of CasADi SX from `N=1` through `N=200` and produces less C source than SX past `N=10`; on official-size unbumpercars Alloy beats CasADi MX by roughly 2.7× across the cells that compile, and CasADi SX is bounded by the 180 s compile timeout at the very first cell because of the dense MLP. Every backend still scales linearly in source size and runtime — closing that with constant-source loop preservation is the next concrete milestone (see Phase 3 below).

### Expression and type system

- Add explicit `ScalarType`, `TensorType`, and `SparsityType` concepts if current `TensorType` becomes too small.
- Track dtype, shape, sparsity, differentiability, and lowering policy consistently.
- Add shape checking for all ops with useful error messages.
- Decide scalar conventions: `()` vs `(1,)` vs `(1, 1)`.
- Add structural equality / hashing strategy separate from object identity.
- Add stable expression names and debug printing.

### Scoped function construction

Add an anvil-style user construction layer while keeping the current explicit `Expr.sym` + `Function(...)` IR constructor. Candidate API:

```python
@alloy.function("eq_interstage", inputs={"z": (6,), "znext": (6,), "p": (4,)})
def eq_interstage(z, znext, p):
  return {"eq": rk4_dynamics(z[:4], z[4:]) - znext[:4]}
```

or a builder form that avoids decorator typing complexity. This layer should create fresh symbolic inputs, trace the Python body, normalize outputs/names, and return a named `Function`. It should also make call-node composition natural.

### Operation coverage

MVP ops to support well:

- Structural: `input`, `const`, `reshape`, `vec`, `transpose`, `stack`, `concat`, `split`, `project`, `gather`, `scatter`, `call`.
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`.
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
- Hyperbolic: `sinh`, `cosh`, `tanh`.
- Elementary nonlinearities: `exp`, `log`, `sqrt`, `abs`.
- Reductions: `sum`, `dot`, norms, block reductions.
- Linear algebra: `matmul`, dense/sparse matvec, triangular solve, linear solve as an opaque/block op.
- Nonsmooth utilities: `minimum`, `maximum`, `floor`, `ceil`, with explicit AD behavior.

Deferred:

- `expm1`, `log1p`, special functions, matrix exponential.
- Splines/interpolants until there is a semantic spec for knots, extrapolation, differentiability, and codegen storage.

### Rewrite system and PatternMatcher

The current `Pattern`/`rewrite` layer is only a seed. It is too recursive and tree-like for the benchmark workloads, and CSE is a separate structural pass rather than part of a scalable graph rewrite pipeline. Replace it with a tinygrad-inspired system before adding many more optimization rules.

Design target:

- immutable DAG nodes remain the IR values;
- traversal is iterative/topological, not Python-recursive;
- replacements are cached per original node so DAG sharing is preserved;
- patterns are op-indexed with early rejection based on child op sets, like tinygrad's `PatternMatcher`;
- support bottom-up and top-down rewrites plus optional walk-only analysis passes;
- CSE/hash-consing becomes either construction-time interning or a graph rewrite with structural keys, not a late fragile cleanup;
- debugging/profiling can report which patterns fired, because AD/lowering cleanup will need real evidence.

The API can remain much smaller than tinygrad's `UPat`, but it should borrow the important mechanics: `UPat`-style variable binding, op groups, replacement caches, and non-recursive graph rewriting.

Planned passes once the infrastructure exists:

1. graph simplification: constant folding, identities, reshape/project/gather cleanup;
2. AD cleanup: remove zero tangents/adjoints and independent subgraphs without scalarizing whole tensors;
3. sparsity-aware simplification: eliminate structurally zero derivative blocks before codegen;
4. region/lowering cleanup: simplify scalar/block/opaque boundaries;
5. tape/codegen peepholes: dead instruction elimination, slot lifetime cleanup, and scalar expression CSE.

### Evaluation

- Keep the tape interpreter as a deterministic reference/debug evaluator.
- Long term, make `Function.__call__` JIT by default, following anvil's model: lazily render/compile/cache native code on first call, then dispatch through the compiled ABI.
- The interpreter should remain available for tiny graphs, diagnostics, and tests, but performance claims should be based on generated/JIT code.
- Add workspace planning for temporaries.
- Add deterministic topological ordering.
- Add better error handling for missing inputs, shape mismatches, and unsupported ops.

Exit criteria:

- Small dynamics and objective functions can be written without falling back to anvil/tinygrad.
- Tape interpreter and recursive evaluator agree.
- Unit tests cover all MVP ops.

## Phase 2 — AD and sparsity as first-class features

Goal: match the core derivative workflows that make CasADi useful for optimization.

Status: started.

Progress:

- Added initial graph-level reverse-mode `vjp(outputs, wrts, cotangents)` over `Expr` graphs, including adjoint accumulation for multiple outputs, broadcast unprojection, structural ops, matmul, and call-node inlining.
- Routed scalar `gradient(expr, wrt)` through reverse mode instead of constructing it as a full JVP-unrolled Jacobian.
- Added factory-level `fwd:out:in` requests and a human `forward(fn, input, output)` wrapper for seeded `J(out, in) @ fwd:in` products.
- Added factory-level `adj:out:in` requests and a human `adjoint(fn, input, output)` wrapper for seeded `J(out, in).T @ lam:out` reverse products.
- Added lower-level multi-seed `jvp_many` and `vjp_many` APIs using a leading seed axis.
- Added Alloy reverse-mode tests for scalar outputs, vector-seeded VJPs, broadcasted multi-output accumulation, structural/matmul paths, nested `CallOp` reverse AD, factory/API forward/adjoint directional derivatives, multi-seed AD, and clear errors for omitted directional seeds.
- Factory construction now reports unknown requested inputs, derivative-spec inputs/outputs, and aux outputs as explicit `ValueError`s instead of leaking raw mapping errors.
- Factory aux outputs now reject names that shadow real function outputs.

### Dense AD

- Replace naive JVP unrolling where necessary with graph-level forward and reverse passes.
- Expand and harden VJP/reverse mode beyond the initial `Expr` API.
- Treat multi-seed AD as a batched/looped derivative operation, not as Python cloning of one graph per seed. The current `jvp_many` / `vjp_many` leading-axis helpers are useful semantics, but their implementation must become loop-preserving before they are used for benchmark-sized sparse derivatives.
- Support nested `CallOp` AD through a policy: inline small callees, but call cached derivative functions for large or repeated callees. Repeated OCP stages and per-car dynamics should not be cloned blindly.
- Handle nonsmooth ops intentionally: reject, pick a convention, or mark nondifferentiable.

### Sparse AD

Sparse AD is the core of the next checkpoint. Do not keep `spjacobian` as dense-Jacobian-then-gather except as a reference path for tiny tests.

- Add symbolic sparsity propagation.
- Add Jacobian sparsity estimation independent of numerical values.
- Add graph coloring for sparse Jacobian/Hessian evaluation.
- Use coloring to build compressed seed matrices and evaluate compressed JVPs/VJPs, then uncompress compact nonzero values.
- Preserve seed/color axes as loops or batch dimensions through lowering; do not scalarize them in the AD pass.
- Add compact vs full sparse derivative representations.
- Add symmetric/upper-triangular Hessian conventions needed by sparse SQP.
- Expand CSC/CSR sparsity metadata beyond the initial `SparsityType` conversion utilities and generated C header arrays where useful.
- Learn directly from anvil's `spjacobian` experience: unrolled CONST basis construction is a scalability trap; vmap/per-block approaches are the useful patterns.

### Factory language

Expand and harden CasADi-like factory requests:

- Expand `fwd:*` beyond the initial single-output `fwd:out:in` JVP request.
- Expand `adj:*` beyond the initial single-output `adj:out:in` VJP request.
- `jac:out:in`.
- `grad:out:in`.
- `hess:out:in1:in2`.
- `lam:out`.
- auxiliary outputs like `gamma`.
- possibly named aliases and grouped outputs.

Human-facing convenience APIs should stay as wrappers over this language, not a second derivative implementation:

- `gradient(fn, input_name, output_name)` -> `fn.factory(..., [f"grad:{output_name}:{input_name}"])`
- `jacobian(fn, input_name, output_name)` -> `jac:*`
- `hessian(fn, input_name, output_name)` -> `hess:*`
- `lagrangian_hessian(fn, input_name, output_names)` -> `lam:*` inputs plus an aux output

Exit criteria:

- Dense and sparse Jacobians/Hessians match CasADi on a growing corpus.
- Lagrangian Hessian workflows needed by SQP are ergonomic and tested.
- AD through nested calls works for both inline and non-inline policies.

## Phase 3 — Mixed SX/MX-style lowering

Goal: make the central hypothesis real: one graph can contain scalar and block regions. Scalarization should happen here, not inside AD. AD should produce mathematical derivative graphs/linear maps with shape, sparsity, and seed metadata; lowering decides which pieces become scalar SX-like code, block loops, compact sparse assembly, or opaque calls.

**Status: loop-preserving lowering done; broader mixed lowering on standby.** The loop-preserving milestone below is complete on both benchmark workloads (constant-source primal and `spjac:*` on tracking, MAP-ified unbumpercars). The rest of this phase — explicit `scalar`/`block`/`opaque` region formation, boundary pack/unpack/project ops, real block-loop codegen, and automatic lowering heuristics — is deferred until the safety-filter workload demonstrates a concrete need. Today `block` is metadata: dense MLP matmuls are rendered as scalar C and still beat CasADi MX, so investing in real block lowering before there is evidence it pays off is premature. Revisit when the QP/NLP solver workloads or generated-code size for the neural-network blocks demand it.

Benchmark target for this phase:

- tracking NMPC `eq_constraints_jac`: discover or represent the repeated interstage block structure, preserve horizon loops, and produce compact sparse Jacobian values without cloning the whole horizon per seed/color;
- unbumpercars `ineq_constraints_jac`: keep MLP matmuls/softplus layers as dense block regions, scalarize C3BF/wall barrier residuals and sparse Jacobian assembly, and share the per-car dynamics call/region across pairwise constraints.

### Region formation

- Treat `lowering = auto | scalar | block | opaque` as real metadata.
- Partition graphs into scalar regions, block regions, and opaque calls.
- Insert boundary ops: pack, unpack, project, materialize, gather, scatter.
- Preserve sparsity/layout metadata across boundaries.

### Scalar lowering

- Lower scalar regions to SX-like tapes.
- Allocate scalar work slots.
- Generate simple scalar C.
- Add algebraic simplification and common-subexpression elimination.

### Block lowering

- Preserve dense vector/matrix operations as loops or kernel-level ops.
- Avoid scalarizing dense matmul-heavy regions unless explicitly requested.
- Introduce loop IR for map/reduce/scan patterns.
- Add local pattern rewrites.

### Loop-preserving lowering (done)

The scalability sweep showed Alloy already matches CasADi SX runtime on the tracking `eq:z` Jacobian and beats CasADi MX by ~2.7× on the official-size unbumpercars `ineq:u` Jacobian, while generating less C source than SX. Every backend grows source linearly in `N`/`C` because the user-side fixture spells out `N` separate `Function.call(...)` constructions — the IR has no concept of "the same callee N times with index-shifted slices." Anvil's `multistage` codegen produces an `O(1)`-source C `for` loop calling one shared per-stage kernel, which is what gets it to 1.5 µs at `N=50` (about 3× faster than us). The goal of this milestone is to give Alloy the same property: keep current runtime, drive source size down to `O(1)` in the iteration count.

Plan, in roughly the order that minimizes blast radius:

1. **`Ops.MAP` in the IR**. Hold `(callee, length, input_slice_specs, output_slice_spec)`. Each spec is `(outer_tensor, start, stride, slice_size)` and describes how the i-th iteration's argument is sliced out of an outer tensor and where the i-th output writes into the outer assembly. Shape inference yields `(length * out_slice_size, ...)` along the assembly axis. Start with a numpy-backed `Tape` evaluator so we can compare against the unrolled `concat`-of-call equivalent before touching codegen or AD. **Done**: `Ops.MAP` added with rank-1 inputs/outputs, `(start, stride)` per callee input (stride=0 broadcasts), recursive `Expr.eval` and tape evaluation paths, plus tape/expr pretty-printing. Tests in `tests/alloy/test_map.py` compare against the unrolled `concat`-of-call equivalent for fully-strided, overlapping-stride, zero-length, and broadcast cases.
2. **Scoped sugar `al.scan(...)` / `al.map(...)`**. One IR node instead of N call nodes:
   ```python
   parts = al.scan(eq_interstage, length=N,
                   inputs={"z":     (z, 0,  NZ),
                           "znext": (z, NZ, NZ),
                           "p":     (p, NX, NX)})
   ```
   **Done**: `al.map_` is exported from `alloy` and accepts either a sequence of `(outer, start, stride)` tuples ordered to match `callee.inputs`, or a `Mapping[str, ...]` keyed by callee input name; `al.scan` is an alias. The dict form errors loudly on unknown or missing callee input names. `slice_size` is taken from the callee input shape, so callers do not have to spell it out.
3. **Scalar codegen for `Ops.MAP`**. One new branch in `_render_instruction`: emit `for (int it = 0; it < N; ++it) { callee_raw(arg_bases + it*stride, ..., out_base + it*out_stride); }`. Lifetime treats the map output buffer as a single slot of size `length * out_slice_size`. Validate: rewrite the tracking fixture with `al.scan` and confirm the **primal** source is constant in `N` while the benchmark correctness check still passes. **Done**: the renderer emits the loop, declares scratch for any non-selected callee outputs once before the loop, and `_callees` now also walks `Ops.MAP` so the nested callee's raw body is emitted alongside the caller. On the tracking primal fixture the rendered C source is 76 lines from `N=10` through `N=500` (the unrolled-`concat`-of-call equivalent grows from 193 to 6073 lines), and the compiled output matches the unrolled reference.
4. **JVP rule for `Ops.MAP`**. `jvp_many(map(f, ...), wrt, seeds) = map(jvp_many_of_f, ...)`. The per-iteration JVP callee already exists in the `_call_jvp_many_function` / `_call_jvp_many_const_function` caches. After this step `spjac:eq:z` stays constant-source: the colored JVP graph collapses N calls into one map node whose body is the cached per-stage JVP function. **Done**: `_jvp_many_structural` has a MAP branch. Each per-input tangent is transposed and flattened so the per-iteration seed slice (`(formal.size, nseed)` row-major = `formal.size * nseed` contiguous doubles) is a leading-axis slice. A new `_call_jvp_many_flat_function` caches a thin wrapper around the existing leading-seed JVP callee that reshapes/transposes the flat seed at entry and the leading-seed output at exit; the wrapper itself contains a CALL to the inner JVP callee so codegen sees the loop body as a fixed function. The MAP-of-JVP output is then transposed back to the leading-seed convention. Also fixes a pre-existing CSE crash where children of `ADD`/`MUL` were sorted by raw tuple `<` which fails on heterogeneous attrs (e.g. SLICE index `(int, slice)` vs `(slice,)`); the sort key is now `hash`.
5. **Sparsity propagation through `Ops.MAP`**. `_jac_mask(map(f, ...), wrt)` produces a tile pattern repeated `length` times along the assembly axis. `jacobian_sparsity` should build the tile + length in `O(1)` time even though the materialized `SparsityType.rows / cols` arrays remain `O(N)` (they are still data, not source code). **Done**: `_map_mask` reuses the per-formal callee dependency tile across iterations and only slices a different range of the outer-tensor dependency mask each iteration. End-to-end `spjacobian` over a MAP-based tracking fixture matches the unrolled-concat equivalent numerically and has identical sparsity metadata; the new test fixture in `tests/alloy/test_map.py` exercises this with the full RK4 dynamics.
6. **Tile-strided gather lowering**. When the compact-value gather indices form a tile pattern (`base[i*nnz_per_stage..(i+1)*nnz_per_stage]` for `i in [0, length)`), emit a `for` loop instead of an O(N) unrolled index table. This is the last unrolled site that keeps the rendered C source from being `O(1)`. **Done**: two complementary paths land constant LOC. `_detect_tile` finds `(prefix, tile_size, length, stride, base)` patterns in the flat gather indices and emits a nested `for` loop reading from the gather source. The transposed-concat peephole now groups output entries by their CONCAT input block and, for groups above 32 entries, emits a single static `idx[]` table + a `for` loop whose body is built via `_read_element` — so the inline-through-ADD/SUB chains still apply when the block is itself a skipped sum of materialized tensors. With this, the realistic RK4 tracking spjac source stays at 724 LOC across `N=5` through `N=500` (the unrolled-call variant grows from 2579 to 32647 LOC). The byte count still grows linearly because the `idx[]` table is O(nnz) bytes of data, but that no longer feeds the C parser as code.

Replacing the JVP wrapper with a tile-strided gather is the matching fix on the IR side: the MAP-JVP rule now builds a per-iteration leading-seed seed buffer via `gather(actual_tan, tile_indices)` and calls the existing `_call_jvp_many_function` body directly, so each MAP iteration is just a function call with no transpose pre/post. The MAP constructor also accepts rank-N callee inputs/outputs as long as the outer tensor stays rank-1.

7. **Structured sparse Jacobian for MAP (per-formal local coloring + const-seed JVP)**. **Done**: the runtime gap is closed. `al.sparse_jacobian` now tries `_sparse_jacobian_structured` first: it splits the top-level expression along axis 0 (handling `Ops.CONCAT` as a piecewise decomposition), and for each piece that is itself an `Ops.MAP` whose outer tensors are exactly `wrt`, it dispatches to `_sparse_jacobian_map`. Per formal input of the callee, that routine (a) computes the local Jacobian sparsity tile of shape `(slice_size, formal.size)`, (b) greedily colors **just that tile**, (c) builds a constant local-color seed matrix, (d) routes it through the existing `_call_jvp_many_const_function` cache (which simplifies/CSEs the JVP graph with the seed embedded as a constant — so per-color DCE eliminates the entire dead tangent subgraph), (e) wraps that DCE'd per-formal JVP in `Ops.MAP` over the original slicing pattern, and (f) assembles the global compact nnz buffer via a constant-table `gather + scatter`. Multiple formals overlapping at the same `(row, col)` are summed. The fallback path (any formal whose outer tensor depends on `wrt` through computation rather than identity) is the old global-colored route, so nothing regresses. The result for tracking NMPC `spjac:eq:z`:

| N   | unrolled          | map (old, global)     | **map_structured (new)** | CasADi SX           | CasADi MX        |
| --- | ----------------- | --------------------- | ------------------------ | ------------------- | ---------------- |
| 10  | 878 ns / 125 KB   | 1309 ns / 31 KB       | **960 ns / 18 KB**       | 830 ns / 97 KB      | 1380 ns / 384 KB |
| 50  | 4448 ns / 184 KB  | 6559 ns / 47 KB       | **4623 ns / 23 KB**      | 4193 ns / 466 KB    | 7860 ns / 2.0 MB |
| 100 | 9249 ns / 259 KB  | 13727 ns / 69 KB      | **9557 ns / 29 KB**      | 8800 ns / 929 KB    | 17712 ns / 4.1 MB|
| 200 | 19084 ns / 416 KB | 29374 ns / 112 KB     | **20068 ns / 42 KB**     | 18411 ns / 1.85 MB  | 39185 ns / 8.7 MB|
| 500 | 51881 ns / 889 KB | 72958 ns / 243 KB     | **54112 ns / 80 KB**     | 44718 ns / 4.64 MB  | (skipped)        |

The previous ~48% map-vs-unrolled gap drops to ~3-5%; runtime is now within ~9-21% of CasADi SX (closer for medium N, slightly behind at the extremes) while emitting 20-58× less C source. The rendered C source stays at **481 LOC from N=2 through N=500** (the byte count grows only because of the gather/scatter index tables, which are data, not code). Codegen time at N=500 drops from 14.2 s (unrolled) and 13.5 s (old MAP path) to **252 ms** — the new path's AD construction is independent of `N`.

Comparison against CasADi's own `Function.map(N, "serial")` primitive on the same workload: CasADi SX with `.map` produces source byte-for-byte identical to fully unrolling (95 bytes of difference at N=200 in a 1.85 MB file) — SX collapses the loop at codegen time. CasADi MX with `.map` keeps a loop shape but its workspace grows linearly (`sz_w` = 80 506 doubles at N=200) and runtime is ~2× slower than SX/unrolled. Alloy's structured MAP path is the only configuration in the comparison that gives both constant source and runtime within ~10% of CasADi SX, with `SZ_W == 0`.

Anvil's per-stage `multistage` approach reaches 1.5 µs at N=50 on a simpler 4-state bicycle; the slip-angle + tanh-drag fixture here is about 3× heavier per stage, so the absolute speedup is workload-bound. The mechanism is now matched: per-stage coloring with const-seed JVP bodies, plus a constant-index assembly into the global compact buffer — without introducing a `multistage` primitive.

Each step is a separate commit with a benchmark guard. After step 3 we already have constant-source **primal** tracking. Steps 4-6 carry that property through `spjac:*`.

Risks to watch:

- An opaque per-iteration callee blocks the C compiler's loop optimizer from fusing/vectorizing across iterations. Right now the unrolled call sites let it inline the callee body once at each site. If we lose 10-20% runtime moving to a real loop, consider marking the callee body `__attribute__((always_inline))` for small callees, or have the codegen emit both an inlined unroll for small `N` and the loop form otherwise, picking per benchmark target.
- The lifetime-packing pass treats slot lifetimes per IR instruction. The map node holds a single instruction-level lifetime but its slot covers `length` iterations — sizing and reuse must allocate the outer assembly buffer once and not over-pack iteration-local scratch with anything outside the loop body.
- `SparsityType` continues to materialize the full COO triple. That keeps existing benchmark/correctness/sparsity-metadata paths working unchanged. A future move to a structured sparsity representation would also strip the `O(N)` data table, but is not required by this milestone.

Exit criteria:

- A tracking fixture rewritten with `al.scan` produces C source whose size is `O(1)` in `N` for both the primal and `spjac:eq:z` paths, with runtime within ~10% of the current unrolled form across the existing horizon sweep.
- The benchmark correctness check still passes on every cell in the sweep.

### Policy

- Start with manual hints: `.scalar()`, `.block()`, `.opaque()`.
- Add heuristics later:
  - sparse scalar expressions -> scalar lowering;
  - dense matmul/reductions -> block lowering;
  - repeated integrator stages -> loop lowering;
  - solver calls -> opaque by default.

Exit criteria:

- A single `Function` can mix scalarized first-principles dynamics with dense block linear algebra.
- The generated tape clearly shows region boundaries.
- Mixed lowering produces the same numerical results as all-scalar and all-block reference paths where both exist.

## Phase 4 — C ABI, codegen, JIT, and typed wrappers

Goal: make Alloy functions callable from generated C/C++ and compatible with CasADi-like consumers. The generated path should also become the default execution path: `Function.__call__` should eventually lazily JIT and cache native code, with the interpreter kept as a reference/debug mode.

Status: started.

Progress:

- C API header generation now emits `sz_arg`, `sz_res`, `sz_iw`, and `sz_w` helper declarations alongside the universal ABI.
- Added a standalone scalar C source renderer for the current tape subset: constants, inputs, elementwise ops, reductions, structural reshape/transpose/slice/gather/scatter/stack/concat, and rank-1/rank-2 matmul.
- Added a compiled `ctypes` smoke test that builds generated C with `cc`, calls the universal ABI, checks the `sz_*` helpers, and verifies numerical outputs.
- `render_c_source(outer)` now emits nested callee raw bodies before callers and lowers `CallOp` instructions by invoking those internal raw bodies directly; only the root function is exported through the universal ABI for a rendered translation unit.
- C codegen now emits tape temporaries as local C arrays with contiguous slice/reshape aliases instead of monotonically growing caller workspace. This fixed the tracking benchmark symptom where Alloy's reported workspace grew with horizon; pure generated functions now usually have `SZ_W == 0`, while the ABI still reserves `w` for future regions that need caller scratch.
- Added a compiled nested-call ABI test covering a callee with multiple outputs.
- Generated headers now expose compile-time ABI size macros and an inline C++ typed-buffer wrapper that calls the universal ABI internally.
- Added a C++ compile-and-run smoke test for the typed wrapper path.
- Generated headers/source now define named `ALLOY_*` ABI status codes for success, null ABI arrays, missing workspace, null result slots, and null input slots.
- Expanded the compiled ABI test to verify those null-pointer error paths.
- Added `render_c_module`, which returns a paired generated header/source module with explicit output filenames and the source including the generated header.
- Updated the C++ wrapper smoke test to compile the generated C source separately and link it into a C++ caller.
- Generated ABI headers/source now include stateless `alloc_mem`, `init_mem`, and `free_mem` hooks; the compiled ABI test verifies the exported hook symbols.
- Added a compiled C-module smoke test for `spjacobian(...)` factory output, verifying compact sparse derivative values through the universal ABI.
- Added a C++ typed-wrapper smoke test for `lagrangian_hessian(...)` factory output, including sanitized `lam:*` input names.
- The ABI spec now states caller ownership, row-major buffer layout, workspace/null-pointer rules, lack of default inputs, and alignment assumptions.
- Generated C++ typed wrappers now include `static_assert` checks that typed buffer structs have the expected flattened `double` size.

### Next concrete milestone: JIT as default execution path

**Status: done.** `Function.__call__` now lazily renders, compiles, caches, and dispatches through the universal ABI on first call; the interpreter is preserved as `Function.eval_interpreter(...)` and as the fallback when `ALLOY_DISABLE_JIT=1`, no C compiler is available, or codegen raises `NotImplementedError`.

Landed:

- `src/alloy/jit.py` owns the JIT pipeline: SHA-256 cache key over (`_JIT_CACHE_VERSION`, `C_API_SIGNATURE`, function name, generated C source), per-user cache directory (`$ALLOY_CACHE_DIR` overrides, otherwise `$XDG_CACHE_HOME/alloy/jit` or `~/.cache/alloy/jit`), `cc -O2 -fPIC -shared/-dynamiclib` invocation, and a `CompiledFunction` that wires `ctypes` against the universal ABI entry point and `_sz_w` helper.
- A process-local `_artifact_cache` lets multiple `Function` instances with identical generated source share the same `.so` after the first compile; the on-disk cache survives across processes.
- `Function._compiled` holds the per-instance handle. `Function.recompile()` drops the in-process handle and removes the cached source/library directory. `Function.eval_interpreter` (existing reference path), `Function.eval_list` (default JIT dispatcher), and `Function.__call__` are the public entry points.
- Internal callees inside `Tape.evaluate(...)` and `Expr.eval(...)` now route through `callee.eval_interpreter(...)` so the interpreter path remains a pure reference (no implicit JIT inside the reference path).
- Fixed a latent codegen bug uncovered by this work: `_skipped_instructions` was vacuously dropping output-only TRANSPOSE/ADD/SUB nodes because `all(...)` over an empty consumer list is `True`. Output instructions are now excluded from the skip set so the final copy loop always has a materialized buffer.
- Test coverage in `tests/alloy/test_alloy_jit.py` covers: JIT matches interpreter, `ALLOY_DISABLE_JIT=1` falls back to interpreter, cache keys are stable across `Function` instances of the same graph, `recompile()` invalidates both in-memory and on-disk caches, multi-output + keyword inputs, factory `spjacobian` outputs (sparse compact buffer), nested `CALL` nodes, and shape-mismatch validation.

Resolved open questions:

- Multiple `Function` instances with identical generated source share an `.so` via `_artifact_cache` keyed by the SHA-256 hash, but each `Function` still holds its own `ctypes.CDLL` handle (cheap relative to the compile itself).
- The JIT cache lives under a separate Alloy directory (`~/.cache/alloy/jit/<hash>/`) so it is independent of tinygrad's cache.
- The workspace buffer `w` is allocated per call as a ctypes array of size `sz_w`. Workspace sizes in benchmark workloads are zero or tiny, so per-call allocation is fine; revisit if a workload pushes `sz_w` high enough that allocation cost shows up in profiles.

### Universal ABI

Canonical C ABI:

```c
int f(const double** arg, double** res, int* iw, double* w, void* mem);
```

Tasks:

- Define ABI precisely: ownership, null pointers, default inputs, error codes, alignment, workspace sizes.
- Generate `sz_arg`, `sz_res`, `sz_iw`, `sz_w` helpers.
- Generate `alloc_mem`, `init_mem`, `free_mem` hooks where needed.
- Support nested calls through the same ABI.

### Typed wrappers

- Generate optional C++ `Buffer<dtype, shape...>` wrappers.
- Generate overloads that call the universal ABI internally.
- Add compile-time shape checks.
- Keep typed wrappers as sugar; the universal ABI remains canonical.

### Renderers

- Start with scalar C renderer.
- Add C++ header/source module renderer.
- Add block-loop C renderer.
- Later: CUDA/Metal for region types that actually benefit.

Exit criteria:

- Generated C compiles for representative functions.
- ABI tests execute generated functions through `ctypes` or compiled test binaries.
- Typed wrappers are tested for shape/signature correctness.

## Phase 5 — QP and NLP solvers as Alloy Functions

Goal: make `al.qp(...)` and `al.nlp(...)` return real callable Alloy `Function`s whose oracles are driven by Alloy's derivative factory, with PIQP and IPOPT as the first concrete backends. Driving workload: the CBF safety filter described in [Current experiment pivot](#current-experiment-pivot).

### Repository split

Phase 5 starts by lifting Alloy out of the anvil repository into its own project. Reasons:

- the IR no longer imports from `anvil`; the dependency tree is meaningfully smaller (no `tinygrad`, `jax`, `scipy`, `networkx`);
- the build hook is about to grow significantly to vendor IPOPT + MUMPS + BLAS/LAPACK, and that machinery is alloy-specific;
- independent versioning lets Alloy release without dragging anvil's experimental state along;
- the PIQP shared library built for anvil and the one Alloy needs are the same artifact; cleaner if each project owns its own copy.

The new `alloy` repository owns all `src/alloy/`, `tests/alloy/`, `docs/alloy/`, and `benchmarks/` artifacts from this worktree, plus its own `pyproject.toml`, `hatch_build.py`, `.github/workflows/`, `worktrunk` config, and `uv`/`ruff`/`ty` settings. The anvil repository keeps its current state; cross-references stay as documentation only.

### Vendored solver shared libraries

Both PIQP and IPOPT ship as redistributable shared libraries inside the alloy wheel (`src/alloy/lib/libpiqpc.{dylib,so}`, `src/alloy/lib/libipopt.{dylib,so}`), with C headers under `src/alloy/include/`. The same artifacts are used:

- by the Python runtime, loaded via `ctypes` from the JITed wrapper Functions;
- by AOT C++ consumers that link against `-lpiqpc` / `-lipopt` from `alloy/lib/`.

Build strategy:

- **PIQP**: reuse the existing anvil `hatch_build.py` pattern. Clone PIQP v0.6.2, Eigen 3.4.1, blasfeo; build `piqp_c` as a shared library; copy headers. Cold build ~1-2 min.
- **IPOPT**: source build via a coinbrew-style hook. Clone coin-or/Ipopt 3.14+, `ThirdParty-Mumps`, `ThirdParty-Metis`, and either OpenBLAS (Linux) or rely on Apple Accelerate (macOS). Build MUMPS (sequential, no MPI) and IPOPT against them. **Statically link `libgfortran`, `libgcc`, and `libstdc++` into `libipopt`** (`-static-libgfortran -static-libgcc -static-libstdc++`) so the resulting `.dylib`/`.so` has no runtime dependency on the host's Fortran toolchain. This is what makes the AOT story clean — external C++ links against `libipopt` as a normal library without rpath gymnastics or libgfortran ABI surprises. Cold build ~5-8 min.
- **Caching**: identical to the anvil hook — skip the rebuild if the lib + headers already exist in `src/alloy/lib/`.
- **Platform support**: macOS (arm64 + x86_64) and Linux (manylinux_2_28) for v1. Windows is deferred: MSVC has no Fortran, MinGW/intel Fortran would require its own build path. Document the limitation; revisit if a concrete Windows user appears.

Build requirements on the host: a C/C++ compiler, CMake, and `gfortran` (from `brew install gcc` on macOS, `apt install gfortran` on Linux). CI builds use these toolchains directly.

cyipopt was considered as a runtime dependency and rejected: although its wheel does bundle `libipopt`, the bundled library has a mangled SONAME, an `$ORIGIN`-relative RPATH pointing inside `cyipopt.libs/`, and no shipped C headers — none of which support the "external C++ links against our libipopt" use case. A source build with statically linked Fortran runtime is the only path that makes the same artifact usable from both the Python and the AOT entry points.

### Problem formulation: PIQP-style explicit constraints

Alloy deliberately departs from CasADi's bilateral `lba ≤ Ax ≤ uba` convention in favour of PIQP-style explicit constraint categories. Each category carries structural information the solver can exploit directly, without runtime detection of equalities from `lba == uba`.

QP shape:

```
min     0.5 xᵀ P x + cᵀ x
s.t.    A_eq x = b_eq
        l_ineq ≤ G_ineq x ≤ u_ineq   (two-sided general inequalities)
        x_lb  ≤ x ≤ x_ub              (box)
```

NLP shape:

```
min     f(x, p)
s.t.    h_eq(x, p) = 0
        l_ineq ≤ g_ineq(x, p) ≤ u_ineq   (two-sided general inequalities)
        x_lb   ≤ x ≤ x_ub                 (box)
```

Two-sided general inequalities are kept rather than reduced to one-sided form because PIQP supports them natively (band/range constraints appear frequently in MPC) and the implementation cost is small — one extra sparsity union and two bound vectors versus one.

PIQP path: pass each category through as the matching field on the PIQP C struct. No conversion.

IPOPT path: at solver build time, stack `[h_eq; g_ineq]` into IPOPT's `g(x)` with `g_L = [0; l_ineq]` and `g_U = [0; u_ineq]`. Sparsity unions stack vertically; the conversion is a one-shot at solver-construction, not per call. Box constraints pass directly to IPOPT's `x_L`/`x_U`.

### Oracle layer

Two schemas, both built on top of the existing `Function` API and factory language. Derivative helpers are generated through the factory and cached as named `Function`s so they take the JIT path the same way user code does.

**NLP oracle** for `al.nlp(...)`:

- inputs: `x`, `p`
- outputs: `f`, `h_eq`, `g_ineq`
- generated helpers (factory-built, JIT-cached):
  - `nlp_eval`: `x, p -> f, h_eq, g_ineq`
  - `nlp_jac`: `x, p -> grad:f:x, jac:h_eq:x, jac:g_ineq:x` (sparse Jacobians)
  - `nlp_hess_l`: `x, p, lam_eq, lam_ineq -> hess:lagrangian:x:x` (sparse Lagrangian Hessian)

**QP oracle** for `al.qp(...)`. The QP backend consumes the affine-in-`x` specialization of the same shape: at solver build time Alloy extracts `P = hess:f:x:x`, `c = grad:f:x` at `x=0`, `A_eq = jac:h_eq:x`, `b_eq = -h_eq(x=0)`, `G_ineq = jac:g_ineq:x`, and shifted bounds from `g_ineq(x=0)` via the factory. The resulting QP data are `Function`s of `p` only. This is the correct specialization for input-affine CBF filters, where `f(x)`, `g(x)`, and their Jacobians are evaluated once per step and become the QP's `p`.

Both schemas accept dict-form builders for ergonomics but normalize to the same internal `Oracle` object.

### Backends

A plugin-style registry similar to CasADi's `nlpsol` / `conic` plugins:

- **`piqp`** (QP backend). Sparse interior-point QP. Suitable for the input-affine CBF safety filter, where the QP data is rebuilt every step but the sparsity is fixed. Bind through PIQP's C interface; Alloy supplies sparsity patterns from `SparsityType` so PIQP's sparse path is used directly without conversion overhead. PIQP's native `A_eq`/`G_ineq`/`x_lb`/`x_ub` fields map one-to-one to the alloy QP schema.
- **`ipopt`** (NLP backend). Standard NLP backend for the fully nonlinear CBF safety filter. Bind through IPOPT's C interface (`IpStdCInterface.h`), with Alloy supplying eval callbacks via the JITed oracle functions. Sparse Jacobian/Hessian patterns come from `SparsityType`; coloring stays internal to Alloy since IPOPT consumes structured sparse triplets, not colored seeds. The eq/ineq stacking conversion happens at solver construction.

Each backend is a `Function` wrapper around the solver. From the outside, calling a QP/NLP solver `Function` follows the universal ABI and the JIT-as-default path from Phase 4. The memory hooks (`alloc_mem`, `init_mem`, `free_mem`) are no longer no-ops here — they own the solver workspace, factorizations, and cached warm-start state.

### Solver API

```python
import alloy as al

# QP — PIQP backend
qp = al.qp(
    P=P_expr, c=c_expr,
    A_eq=A_eq_expr, b_eq=b_eq_expr,
    G_ineq=G_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr, x_ub=ub_expr,
    solver="piqp",
)

# NLP — IPOPT backend
nlp = al.nlp(
    x=x_sym, p=p_sym,
    f=f_expr,
    h_eq=h_eq_expr,
    g_ineq=g_ineq_expr,
    l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr, x_ub=ub_expr,
    solver="ipopt",
)
```

- **QP inputs (call-time):** `x0`, `lam_eq0`, `lam_ineq0`, plus any free `p` parameters in the symbolic oracle.
- **QP outputs:** `x`, `cost`, `lam_eq`, `lam_ineq`, `lam_box`.
- **NLP inputs (call-time):** `x0`, `lam_eq0`, `lam_ineq0`, plus any free `p`.
- **NLP outputs:** `x`, `f`, `h_eq`, `g_ineq`, `lam_eq`, `lam_ineq`, `lam_box`.

Solvers are opaque `Function`s by default — their `.tape()` shows a single `opaque` call into the backend. Fixed-iteration unrolling stays available as an explicit advanced option but is not the default; it conflicts with backend-internal warm starting.

### Safety-filter assembly

The concrete API target for the driving workload looks like:

```python
import alloy as al

dynamics = al.load_nn_dynamics("model.pth")  # input-affine: returns (f, g) Functions

@al.function("safety_filter", {"x": (NX,), "u_ref": (NU,)})
def safety_filter(x, u_ref):
    f_x = dynamics.f(x)
    g_x = dynamics.g(x)
    h, h_grad = cbf(x)
    P = al.eye(NU)
    c = -u_ref
    G_ineq = h_grad @ g_x
    l_ineq = -(h_grad @ f_x + alpha * h)
    u_ineq = al.inf
    return {"u": al.qp(P=P, c=c, G_ineq=G_ineq, l_ineq=l_ineq, u_ineq=u_ineq, solver="piqp")}
```

The same shape carries over to the NLP variant with `al.nlp(...)` and `f(x, u)` evaluated inside the oracle.

Exit criteria:

- Alloy repository split out, with PIQP + IPOPT building and shipping in wheels for macOS (arm64 + x86_64) and Linux (manylinux_2_28).
- `libipopt.{dylib,so}` is statically linked against libgfortran/libgcc/libstdc++ and is linkable from a standalone C++ binary with no runtime Fortran dependency.
- Both safety-filter variants build, JIT, and execute through the universal ABI.
- PIQP and IPOPT bindings pass small NLP/QP test problems against reference solutions (CasADi+PIQP, CasADi+IPOPT).
- A C++ harness can call the generated safety filter through the universal ABI with realistic per-step runtime, linking against the vendored `libipopt` / `libpiqpc`.
- Two-sided general inequalities are exercised by tests on both backends.
- Lagrangian Hessian and sparse Jacobian factory paths are exercised end-to-end by the NLP variant.

## Benchmark-driven development plan

Before broadening solvers/integrators, add a small Alloy benchmark suite modeled on the `tracking-nmpc-benchmarks` worktree. The suite should be runnable in tiers: quick structural/codegen checks in normal tests, and slower compile/runtime sweeps behind an explicit benchmark flag.

### Workload A: tracking NMPC equality Jacobian

Source model: `examples/tracking_nmpc/nmpc_anvil.py` and `candidacy_experiments/experiment1/2` from the benchmark worktree.

What to build in Alloy:

- scoped-function versions of `eq_initial` and `eq_interstage`;
- a horizon-level equality constraint function using `CallOp` or a future map/loop region;
- `spjac:eq:x` over the flattened horizon decision vector;
- CasADi SX/MX and current anvil comparison scripts for small horizons first, then `N = 10, 20, 50, 100`.

What to learn:

- whether repeated stage structure survives AD and lowering;
- where graph size grows with `N`;
- whether coloring and sparse assembly happen without full scalar unrolling;
- whether mixed lowering can reproduce the per-block multistage result without hard-coding `MultistageProblem`.

### Workload B: unbumpercars inequality Jacobian

Source model: `examples/unbumpercars/safety_filter_anvil.py` and `candidacy_experiments/experiment3` from the benchmark worktree.

What to build in Alloy:

- scoped-function versions of per-car dynamics, MLP forward, global velocity, C3BF, wall residuals, and the full inequality vector;
- dense block lowering for MLP matmuls and activations;
- scalar/sparse lowering for pairwise C3BF, wall residuals, slack column, and compact Jacobian assembly;
- comparison against CasADi MX and current anvil for `N = 2, 4, 8, 16, 32` where feasible.

What to learn:

- how to merge dense neural-network blocks with sparse solver-facing derivatives;
- whether call-node derivative policy should inline or call per-car derivative helpers;
- whether automatic lowering heuristics can identify dense matmul regions and scalar sparse regions.

### Workload C: CBF safety filter with neural dynamics

This is the driving workload for the next phase pivot. Source model: a small input-affine and a small fully-nonlinear neural dynamics example, sized similarly to unbumpercars' MLP (`256 -> 128 -> {NX or NX*NU}`), loaded from a `.pth` checkpoint.

What to build in Alloy:

- scoped `f`, `g` (input-affine) or `f` (nonlinear) neural-network `Function`s;
- a CBF condition and its required Jacobian/Hessian through the factory;
- safety-filter assembly via `al.qp(...)` (PIQP) and `al.nlp(...)` (IPOPT);
- end-to-end calls through the JIT path and through a generated C++ harness.

What to learn:

- whether the existing IR (named functions + MAP + sparse AD + scalar C codegen) is enough to express both filter variants without ad-hoc IR additions;
- where solver-call boundaries should sit (opaque call vs. inlined unroll) and how warm-start state propagates through `mem`;
- whether the QP specialization (extract `H`, `q`, `A` symbolically from an `x`-affine objective/constraint) needs new IR support or fits inside existing factory passes;
- whether IPOPT's Jacobian/Hessian evaluator callback shape requires changes to the sparse derivative ABI.

Metrics specific to this workload:

- generated source size, compile time, JIT cache hit/miss behavior;
- per-step runtime of the filter for representative state samples;
- numerical agreement of `u*` with a reference CasADi+IPOPT and CasADi+PIQP build;
- sparse Hessian/Jacobian nnz, coloring count, and compact output order match between Alloy and the reference build.

Metrics for all workloads:

- expression/tape node counts before and after rewrites;
- AD construction time and graph size;
- sparsity pattern, nnz, coloring count, and compact output order;
- generated source size and compile time;
- runtime of generated/JIT code;
- numerical agreement with CasADi/anvil and finite differences on sampled inputs.

## Test strategy

### Unit tests

- Expression construction, shape inference, evaluation.
- Tape linearization and deterministic ordering.
- AD rules per op.
- Factory request parser and output naming.
- ABI/header generation.

### Equivalence tests

- Compare against CasADi SX for scalarized expressions.
- Compare against CasADi MX for nested calls and block operations.
- Compare Jacobians/Hessians against CasADi and finite differences.
- Compare integrators and NLP helper functions against CasADi.

### Generated-code tests

- Compile generated C/C++.
- Call through universal ABI.
- Call through typed C++ wrappers.
- Validate workspace size helpers and nested calls.

### Property/fuzz tests later

- Random expression trees within supported op subset.
- Random shapes for broadcasting/reshape/stack.
- Random sparse patterns for derivative sparsity.

## Design questions to resolve

- Should `Function` store graph outputs only, or an explicit mutable graph object?
- What is the minimal scoped-construction API that feels like anvil without hiding the IR?
- How should structural CSE/hash-consing interact with user-visible expression identity?
- What is the canonical sparse matrix format in the IR?
- Should `CallOp` AD default to inlining, derivative-function calls, or policy-based selection?
- How much CasADi factory syntax should be copied exactly?
- Where should solver memory live in Python vs generated code?
- What is the JIT cache key and lifecycle for `Function.__call__` once native execution becomes default?
- What is the first GPU-worthy workload: batched dynamics, map residuals, dense linear algebra, or something else?

## Milestone summary

1. **M0 Seed**: symbolic expressions, functions, tapes, simple AD, CasADi equivalence smoke tests. *Done.*
2. **M1 Core modeling + scoped construction**: robust op coverage, shape/type errors, low-level explicit graph API, and an anvil-style scoped function/decorator API. *Done.*
3. **M2 Scalable rewrites + sparse AD**: tinygrad-inspired graph rewrites, reverse mode, true colored sparse derivatives, no dense-Jacobian-then-gather for benchmark paths. *Done.*
4. **M3 Loop-preserving lowering**: MAP-based per-stage callee reuse and colored sparse Jacobians constant in `N`/`C`, demonstrated on tracking NMPC `eq_constraints_jac` and MAP-ified unbumpercars `ineq_constraints_jac`. *Done.* Broader mixed scalar/block/opaque region formation is on standby until the safety-filter workload requires it.
5. **M4 JIT as default**: `Function.__call__` lazily renders, compiles, caches, and dispatches through the universal ABI; interpreter preserved as debug fallback. *Done.*
6. **M5 QP and NLP solvers**: `al.qp(...)` with PIQP and `al.nlp(...)` with IPOPT, driven by the CBF safety-filter workload. *Bindings + IR nesting + C codegen for both backends landed*. Both builders ship, sparse Jacobian / sparse Lagrangian Hessian for IPOPT come through `Function.factory(...)`, and the new `Ops.SOLVER_CALL` op + `SolverFunction` subclassing `Function` lets a solver be embedded inside any other `@al.function` graph. **Nested QP and NLP both compile to a single `.so` that links directly against the vendored `libpiqpc` / `libipopt` — no Python in the hot path.** See [`solvers.md`](solvers.md). Still open: a C++ harness exercising the generated `.so`, sparse PIQP, warm-start handover, implicit-function AD through `SOLVER_CALL`.

GPU codegen and broader CasADi/anvil interop are deferred until a workload's CPU runtime or migration need actually justifies them.
