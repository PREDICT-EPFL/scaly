# Scaly roadmap

Scaly is a pure-Python symbolic compiler for optimal-control problems. This roadmap records
future library direction. Completed implementation history belongs in git and the frozen notes.

The guiding idea is:

```text
named Function + sparse typed graph + derivative factory + call nodes + mixed scalar/block lowering
```

## Lowering-hint prototype

`Expr.lowering`, `with_lowering()`, `.scalar()`, `.block()`, and `.opaque()` remain an internal
prototype. The lowering pipeline ignores all four hint values, including `opaque`. Solver opacity
comes from `ExprOp.SOLVER_CALL` and solver descriptors.

Hints participate in expression interning and structural keys, and graph rewrites and automatic
differentiation preserve them. The focused graph and substitution tests in `tests/ir/test_expr.py`
cover that metadata behavior. Ordinary models and published documentation do not use the hints.

The prototype can change or be removed without compatibility guarantees. Keeping it does not commit
Scaly to this interface or to implementing it. Future region-formation work must decide whether
per-expression hints are useful before adopting them. This status supersedes the historical hint
plans below.

## Current status

The compiler, sparse automatic differentiation, generated C path, and solver integrations are in
place. The [current benchmark results](../docs/results/index.md) measure the implementation. The
[actionable list](todo.md) owns the remaining work.

## North star

Scaly should become a small CasADi-like core with a modern compiler architecture:

- **CasADi-like user model**: symbolic expressions, named functions, derivative factories, QP/NLP builders.
- **Unified IR**: no hard global SX/MX split; individual regions can lower as scalar code, block loops, kernels, or opaque calls.
- **AOT-friendly ABI**: universal CasADi-style C ABI plus optional typed wrappers for safer C++ use.
- **Control-first AD**: sparse Jacobians/Hessians, Lagrangian Hessians, graph coloring, and solver sensitivity support.
- **Backend-flexible codegen**: scalar C, C++ wrappers, loop-preserving kernels, and GPU code only once a workload demands it.

## Concept map

A useful way to keep Scaly aligned with compiler/tinygrad/anvil terminology:

| Scaly concept | Compiler role | tinygrad/anvil/CasADi analogy |
| --- | --- | --- |
| `ExprOp` | opcode enum | tinygrad `ExprOp` |
| `Expr` | immutable operation node / SSA value | tinygrad `UOp` |
| `TensorType` | value type metadata | shape/dtype/sparsity metadata around tinygrad buffers |
| `Function` | graph boundary and compilation unit | anvil `NumericalFunction`, CasADi `Function` |
| `CallOp` / `ExprOp.CALL` | function invocation node | CasADi call node, future external/solver/integrator calls |
| `Program IR` | lowered executable schedule | tinygrad scheduled UOps, generated CasADi/anvil code |
| `Function.factory()` | derivative/helper function builder | CasADi factory request language |
| `scaly.function.api.gradient(fn, ...)` | human convenience layer | thin wrapper over factory requests |
| lowering hints | unused prototype metadata | intended scalar vs block preference, ignored by lowering |
| `PatternMatcher` | graph rewrite system | tinygrad `PatternMatcher`/`UPat` |

This map should stay visible in the design: `Expr` is not the high-level tensor API forever; it is the IR node. We can add nicer matrix/tensor facades later, but they should lower to `Expr`/`ExprOp` rather than hide a separate graph representation.

## Construction model

Scaly should support the low-level CasADi-like style of creating symbolic values and then wrapping them in a `Function`, because it is useful for tests, imports, debugging, and graph surgery. It should not be the only or preferred user model. Long-term user-facing modeling should look closer to anvil: a Python-scoped function/decorator/builder creates fresh symbolic inputs from a declared signature, traces the body, and returns a named `Function`.

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

- `src/scaly` as a separate package.
- `Expr`, `ExprOp`, `Function`.
- JIT execution through generated C.
- Basic symbolic JVP-based `jacobian`, `gradient`, `hessian`.
- CasADi-like `Function.factory()` for typed `Jac`, `Grad`, and `Hess` requests, `lam:*` inputs, and aux Lagrangian outputs.
- Human-friendly wrappers like `gradient(fn, of, wrt)` built on top of factory requests.
- First-class `call` expressions.
- Prototype lowering metadata: `auto`, `scalar`, `block`, `opaque`. See [current status](#lowering-hint-prototype).
- Universal C ABI spelling and optional typed buffer header generation.
- CasADi equivalence tests for simple gradients, Jacobians, and Hessian-of-Lagrangian cases.

Exit criteria:

- Unit and CasADi-equivalence tests pass.
- The spec and roadmap describe the experimental scope.

## Phase 1 — Solidify the symbolic core

Complete. The expression graph, construction model, operations, rewrite system, and evaluators are
covered by the sections below.

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
@scaly.function("eq_interstage", inputs={"z": (6,), "znext": (6,), "p": (4,)})
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
5. Program IR/codegen peepholes: dead instruction elimination, slot lifetime cleanup, and scalar expression CSE.

### Evaluation

- Keep Python execution on the generated-code path; use expression/Program IR dumps for debugging.
- Long term, make `Function.__call__` JIT by default, following anvil's model: lazily render/compile/cache native code on first call, then dispatch through the compiled ABI.
- The interpreter should remain available for tiny graphs, diagnostics, and tests, but performance claims should be based on generated/JIT code.
- Add workspace planning for temporaries.
- Add deterministic topological ordering.
- Add better error handling for missing inputs, shape mismatches, and unsupported ops.

Exit criteria:

- Small dynamics and objective functions can be written without falling back to anvil/tinygrad.
- JIT execution and generated-source checks agree with explicit NumPy/CasADi references where available.
- Unit tests cover all MVP ops.

## Phase 2 — AD and sparsity as first-class features

Goal: match the core derivative workflows that make CasADi useful for optimization.

Status: started.

Progress:

- Added initial graph-level reverse-mode `vjp(outputs, wrts, cotangents)` over `Expr` graphs, including adjoint accumulation for multiple outputs, broadcast unprojection, structural ops, matmul, and call-node inlining.
- Routed scalar `gradient(expr, wrt)` through reverse mode instead of constructing it as a full JVP-unrolled Jacobian.
- Added factory-level `Fwd` requests and a human `forward(fn, of, wrt)` wrapper for seeded `J(of, wrt) @ fwd:wrt` products.
- Added factory-level `Adj` requests and a human `adjoint(fn, of, wrt)` wrapper for seeded `J(of, wrt).T @ lam:of` reverse products.
- Added lower-level multi-seed `jvp_many` and `vjp_many` APIs using a leading seed axis.
- Added Scaly reverse-mode tests for scalar outputs, vector-seeded VJPs, broadcasted multi-output accumulation, structural/matmul paths, nested `CallOp` reverse AD, factory/API forward/adjoint directional derivatives, multi-seed AD, and clear errors for omitted directional seeds.
- Factory construction now reports unknown requested inputs, derivative-spec inputs/outputs, and aux outputs as explicit `ValueError`s instead of leaking raw mapping errors.
- Factory aux outputs now reject names that shadow real function outputs.

### Dense AD

- Replace naive JVP unrolling where necessary with graph-level forward and reverse passes.
- Expand and harden VJP/reverse mode beyond the initial `Expr` API.
- Treat multi-seed AD as a batched/looped derivative operation, not as Python cloning of one graph per seed. The current `jvp_many` / `vjp_many` leading-axis helpers are useful semantics, but their implementation must become loop-preserving before they are used for benchmark-sized sparse derivatives.
- Support nested `CallOp` AD through a policy: inline small callees, but call cached derivative functions for large or repeated callees. Repeated OCP stages and per-car dynamics should not be cloned blindly.
- Handle nonsmooth ops intentionally: reject, pick a convention, or mark nondifferentiable.
- A gradient and a Hessian of the same output currently share almost nothing. On the small `sumsqr(x * x * x)` example, their graphs share 3 nodes (15 for the gradient and 29 for the Hessian), and their generated code has 10 plus 41 assignments. Asking for both in one Function therefore costs the exact sum of asking separately. The cause is the AD pipeline: `gradient` returns a raw reverse-mode result, while `hessian` differentiates a simplified batched-forward form of that result. Sharing this work is independent of the derivative API and should be addressed as a separate AD change.

### Sparse AD

Sparse AD is the core of the next checkpoint. Do not keep `sparse_jacobian` as dense-Jacobian-then-gather except as a reference path for tiny tests.

- Add symbolic sparsity propagation.
- Add Jacobian sparsity estimation independent of numerical values.
- Add graph coloring for sparse Jacobian/Hessian evaluation.
- Use coloring to build compressed seed matrices and evaluate compressed JVPs/VJPs, then uncompress compact nonzero values.
- Preserve seed/color axes as loops or batch dimensions through lowering; do not scalarize them in the AD pass.
- Add compact vs full sparse derivative representations.
- Add symmetric/upper-triangular Hessian conventions needed by sparse SQP.
- Expand CSC/CSR sparsity metadata beyond the initial `SparsityType` conversion utilities and generated C header arrays where useful.
- Learn directly from the earlier sparse-Jacobian experience: unrolled CONST basis construction is a scalability trap; vmap/per-block approaches are the useful patterns.

### Factory language

Expand and harden CasADi-like factory requests:

- Expand `fwd:*` beyond the initial single-output `fwd:out:in` JVP request.
- Expand `adj:*` beyond the initial single-output `adj:out:in` VJP request.
- `Jac(of, wrt)`.
- `Grad(of, wrt)`.
- `Hess(of, wrt)`.
- `lam:out`.
- auxiliary outputs like `gamma`.
- possibly named aliases and grouped outputs.

Human-facing convenience APIs should stay as wrappers over this language, not a second derivative implementation:

- `gradient(fn, of, wrt)` -> `fn.factory(..., [Grad(of, wrt)])`
- `jacobian(fn, of, wrt)` -> `fn.factory(..., [Jac(of, wrt)])`
- `hessian(fn, of, wrt)` -> `fn.factory(..., [Hess(of, wrt)])`
- `lagrangian_hessian(fn, of, wrt)` -> `lam:*` inputs plus an aux output

Exit criteria:

- Dense and sparse Jacobians/Hessians match CasADi on a growing corpus.
- Lagrangian Hessian workflows needed by SQP are ergonomic and tested.
- AD through nested calls works for both inline and non-inline policies.

## Phase 3 — Mixed SX/MX-style lowering

Goal: make the central hypothesis real: one graph can contain scalar and block regions. Scalarization should happen here, not inside AD. AD should produce mathematical derivative graphs/linear maps with shape, sparsity, and seed metadata; lowering decides which pieces become scalar SX-like code, block loops, compact sparse assembly, or opaque calls.

**Status: loop-preserving lowering is complete; broader mixed lowering is on standby.** The
current compiler retains mapped structure through sparse Jacobians and exact sparse Hessians.
Block-region work remains deferred until a workload needs it.

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

### Loop-preserving lowering

Complete. The [automatic-differentiation design](../docs/how_it_works/autodiff.md) explains the
implementation. The [scalability results](../docs/results/scalability.md) own measured behavior.

### Policy

- Reassess whether manual per-expression hints are useful before choosing a region-formation interface.
  See [prototype status](#lowering-hint-prototype).
- Add heuristics later:
  - sparse scalar expressions -> scalar lowering;
  - dense matmul/reductions -> block lowering;
  - repeated integrator stages -> loop lowering;
  - solver calls -> opaque by default.

Exit criteria:

- A single `Function` can mix scalarized first-principles dynamics with dense block linear algebra.
- Expression/Program IR dumps clearly show region boundaries.
- Mixed lowering produces the same numerical results as all-scalar and all-block reference paths where both exist.

## Phase 4 — C ABI, codegen, JIT, and typed wrappers

Goal: make Scaly functions callable from generated C/C++ and compatible with CasADi-like consumers. The generated path should also become the default execution path: `Function.__call__` should eventually lazily JIT and cache native code, with the interpreter kept as a reference/debug mode.

Status: started.

Progress:

- C API header generation now emits `sz_arg`, `sz_res`, `sz_iw`, and `sz_w` helper declarations alongside the universal ABI.
- Added a standalone scalar C source renderer for the current expression op subset: constants, inputs, elementwise ops, reductions, structural reshape/transpose/slice/gather/scatter/stack/concat, and rank-1/rank-2 matmul.
- Added a compiled `ctypes` smoke test that builds generated C with `cc`, calls the universal ABI, checks the `sz_*` helpers, and verifies numerical outputs.
- `render_c_source(outer)` now emits nested callee raw bodies before callers and lowers `CallOp` instructions by invoking those internal raw bodies directly; only the root function is exported through the universal ABI for a rendered translation unit.
- C codegen now emits temporaries as local C arrays with contiguous slice/reshape aliases instead of monotonically growing caller workspace. This fixed the tracking benchmark symptom where Scaly's reported workspace grew with horizon; pure generated functions now usually have `SZ_W == 0`, while the ABI still reserves `w` for regions that need caller scratch.
- Added a compiled nested-call ABI test covering a callee with multiple outputs.
- Generated headers now expose compile-time ABI size macros and an inline C++ typed-buffer wrapper that calls the universal ABI internally.
- Added a C++ compile-and-run smoke test for the typed wrapper path.
- Generated headers/source now define named `SCALY_*` ABI status codes for success, null ABI arrays, missing workspace, null result slots, and null input slots.
- Expanded the compiled ABI test to verify those null-pointer error paths.
- Added `render_c_module`, which returns a paired generated header/source module with explicit output filenames and the source including the generated header.
- Updated the C++ wrapper smoke test to compile the generated C source separately and link it into a C++ caller.
- Generated ABI headers/source now include stateless `alloc_mem`, `init_mem`, and `free_mem` hooks; the compiled ABI test verifies the exported hook symbols.
- Added a compiled C-module smoke test for `sparse_jacobian(...)` factory output, verifying compact sparse derivative values through the universal ABI.
- Added a C++ typed-wrapper smoke test for `lagrangian_hessian(...)` factory output, including sanitized `lam:*` input names.
- The ABI spec now states caller ownership, row-major buffer layout, workspace/null-pointer rules, lack of default inputs, and alignment assumptions.
- Generated C++ typed wrappers now include `static_assert` checks that typed buffer structs have the expected flattened `double` size.

### Next concrete milestone: JIT as default execution path

**Status: done.** `Function.__call__` now lazily renders, compiles, caches, and dispatches through the universal ABI on first call. There is no interpreter fallback: missing compilers and lowering/codegen gaps fail loudly.

Landed:

- `src/scaly/codegen/jit.py` owns the JIT pipeline: SHA-256 cache key over (`_JIT_CACHE_VERSION`, `C_API_SIGNATURE`, function name, generated C source), per-user cache directory (`$SCALY_CACHE_DIR` overrides, otherwise `$XDG_CACHE_HOME/scaly/jit` or `~/.cache/scaly/jit`), `cc -O2 -fPIC -shared/-dynamiclib` invocation, and a `CompiledFunction` that wires `ctypes` against the universal ABI entry point and `_sz_w` helper.
- A process-local `_artifact_cache` lets multiple `Function` instances with identical generated source share the same `.so` after the first compile; the on-disk cache survives across processes.
- `Function._compiled` holds the per-instance handle. `Function.recompile()` drops the in-process handle and removes the cached source/library directory. `Function.__call__` and `Function.numerical_call` are the public execution entry points; `Function._flat_numerical_call` is the leaf-level seam beneath them.
- Internal `CALL`/`VMAP` nodes lower through Program IR, so nested Python execution and AOT share the same code path.
- Fixed a latent codegen bug uncovered by this work: `_skipped_instructions` was vacuously dropping output-only TRANSPOSE/ADD/SUB nodes because `all(...)` over an empty consumer list is `True`. Output instructions are now excluded from the skip set so the final copy loop always has a materialized buffer.
- Test coverage in `tests/scaly/test_scaly_jit.py` covers: JIT matches explicit NumPy references, cache keys are stable across `Function` instances of the same graph, `recompile()` invalidates both in-memory and on-disk caches, multi-output + keyword inputs, factory `sparse_jacobian` outputs (sparse compact buffer), nested `CALL` nodes, and shape-mismatch validation.

Resolved open questions:

- Multiple `Function` instances with identical generated source share an `.so` via `_artifact_cache` keyed by the SHA-256 hash, but each `Function` still holds its own `ctypes.CDLL` handle (cheap relative to the compile itself).
- The JIT cache lives under a separate Scaly directory (`~/.cache/scaly/jit/<hash>/`) so it is independent of tinygrad's cache.
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

## Phase 5 — QP and NLP solvers as Scaly Functions

Goal: make `sc.qp(...)` and `sc.nlp(...)` return real callable Scaly `Function`s whose oracles are driven by Scaly's derivative factory, with PIQP and IPOPT as the first concrete backends. Driving workload: the CBF safety filter described in [Current experiment pivot](#current-experiment-pivot).

### Repository split

Phase 5 starts by lifting Scaly out of the anvil repository into its own project. Reasons:

- the IR no longer imports from `anvil`; the dependency tree is meaningfully smaller (no `tinygrad`, `jax`, `scipy`, `networkx`);
- the build hook is about to grow significantly to vendor IPOPT + MUMPS + BLAS/LAPACK, and that machinery is scaly-specific;
- independent versioning lets Scaly release without dragging anvil's experimental state along;
- the PIQP shared library built for anvil and the one Scaly needs are the same artifact; cleaner if each project owns its own copy.

The new `scaly` repository owns all `src/scaly/`, `tests/scaly/`, `docs/scaly/`, `benchmarks/`, and `plugins/` artifacts from this worktree, plus its own `pyproject.toml`, per-plugin hatch hooks, `.github/workflows/`, worktrunk config, and `uv`/`ruff`/`ty` settings. The anvil repository keeps its current state; cross-references stay as documentation only.

### Vendored solver shared libraries

PIQP and IPOPT are built as shared libraries in their respective `plugins/scaly-{piqp,ipopt}/src/*/lib/` directories, with C headers in each plugin's `include/` directory. These artifacts are already used for local development and CI; making the final wheels fully redistributable still requires the static/runtime dependency cleanup tracked in `vendored_solvers.md`. The same artifacts are used:

- by the Python runtime, loaded via `ctypes` from the JITed wrapper Functions;
- by AOT C++ consumers that link against `-lpiqpc` / `-lipopt` using the include/lib/rpath flags reported by `scaly.solvers.graph.solver_compile_flags()`.

Build strategy:

- **PIQP**: reuse the existing anvil hatch-hook pattern in `plugins/scaly-piqp/hatch_build.py`. Clone PIQP v0.6.2, Eigen 3.4.1, blasfeo; build `piqp_c` as a shared library; copy headers. Cold build ~1-2 min.
- **IPOPT**: source build via a coinbrew-style hook. Clone coin-or/Ipopt 3.14+, `ThirdParty-Mumps`, upstream METIS 5 with GKlib, and either OpenBLAS (Linux) or rely on Apple Accelerate (macOS). Build MUMPS (sequential, no MPI) and IPOPT against them. The target is to **statically link `libgfortran`, `libgcc`, and `libstdc++` into `libipopt`** (`-static-libgfortran -static-libgcc -static-libstdc++`) so the resulting `.dylib`/`.so` has no runtime dependency on the host's Fortran toolchain. Current CI source builds pass but still dynamically depend on those runtime libraries; see `vendored_solvers.md`. Cold build ~5-8 min.
- **Caching**: identical to the anvil hook — skip each rebuild if its plugin lib + headers already exist.
- **Platform support**: macOS (arm64 + x86_64) and Linux (manylinux_2_28) for v1. Windows is deferred: MSVC has no Fortran, MinGW/intel Fortran would require its own build path. Document the limitation; revisit if a concrete Windows user appears.

Build requirements on the host: a C/C++ compiler, CMake, and `gfortran` (from `brew install gcc` on macOS, `apt install gfortran` on Linux). CI builds use these toolchains directly.

cyipopt was considered as a runtime dependency and rejected: although its wheel does bundle `libipopt`, the bundled library has a mangled SONAME, an `$ORIGIN`-relative RPATH pointing inside `cyipopt.libs/`, and no shipped C headers — none of which support the "external C++ links against our libipopt" use case. A source build with statically linked Fortran runtime is the only path that makes the same artifact usable from both the Python and the AOT entry points.

### Problem formulation: PIQP-style explicit constraints

Scaly deliberately departs from CasADi's bilateral `lba ≤ Ax ≤ uba` convention in favour of PIQP-style explicit constraint categories. Each category carries structural information the solver can exploit directly, without runtime detection of equalities from `lba == uba`.

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

**NLP oracle** for `sc.nlp(...)`:

- inputs: `x`, `p`
- outputs: `f`, `h_eq`, `g_ineq`
- generated helpers (factory-built, JIT-cached):
  - `nlp_eval`: `x, p -> f, h_eq, g_ineq`
  - `nlp_jac`: `x, p -> grad:f:x, jac:h_eq:x, jac:g_ineq:x` (sparse Jacobians)
  - `nlp_hess_l`: `x, p, lam_eq, lam_ineq -> hess:lagrangian:x:x` (sparse Lagrangian Hessian)

**QP oracle** for `sc.qp(...)`. The QP backend consumes the affine-in-`x` specialization of the same shape: at solver build time Scaly extracts `P = hess:f:x:x`, `c = grad:f:x` at `x=0`, `A_eq = jac:h_eq:x`, `b_eq = -h_eq(x=0)`, `G_ineq = jac:g_ineq:x`, and shifted bounds from `g_ineq(x=0)` via the factory. The resulting QP data are `Function`s of `p` only. This is the correct specialization for input-affine CBF filters, where `f(x)`, `g(x)`, and their Jacobians are evaluated once per step and become the QP's `p`.

Both schemas accept dict-form builders for ergonomics but normalize to the same internal `Oracle` object.

### Backends

A plugin-style registry similar to CasADi's `nlpsol` / `conic` plugins — but
external: a solver plugin is a separate pip-installable package registering an
`scaly.solvers` entry point, and since 2026-07-15 it also owns its C wrapper
template (contract in [`docs/dev/solver_plugins.md`](../docs/dev/solver_plugins.md); decision in
`notes/benchmark-buildout.md` §3.5):

- **`piqp`** (QP backend). Sparse interior-point QP. Suitable for the input-affine CBF safety filter, where the QP data is rebuilt every step but the sparsity is fixed. Bind through PIQP's C interface; Scaly supplies sparsity patterns from `SparsityType` so PIQP's sparse path is used directly without conversion overhead. PIQP's native `A_eq`/`G_ineq`/`x_lb`/`x_ub` fields map one-to-one to the scaly QP schema.
- **`ipopt`** (NLP backend). Standard NLP backend for the fully nonlinear CBF safety filter. Bind through IPOPT's C interface (`IpStdCInterface.h`), with Scaly supplying eval callbacks via the JITed oracle functions. Sparse Jacobian/Hessian patterns come from `SparsityType`; coloring stays internal to Scaly since IPOPT consumes structured sparse triplets, not colored seeds. The eq/ineq stacking conversion happens at solver construction.

Each backend is a `Function` wrapper around the solver. From the outside, calling a QP/NLP solver `Function` follows the universal ABI and the JIT-as-default path from Phase 4. The memory hooks (`alloc_mem`, `init_mem`, `free_mem`) are no longer no-ops here — they own the solver workspace, factorizations, and cached warm-start state.

### Solver API

```python
import scaly as sc

# QP — PIQP backend
qp = sc.qp(
    P=P_expr, c=c_expr,
    A_eq=A_eq_expr, b_eq=b_eq_expr,
    G_ineq=G_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr, x_ub=ub_expr,
    solver="piqp",
)

# NLP — IPOPT backend
nlp = sc.nlp(
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

Solvers are opaque `Function`s by default — their expression graph contains `SOLVER_CALL` outputs and outer graphs call them as named opaque callees. Fixed-iteration unrolling stays available as an explicit advanced option but is not the default; it conflicts with backend-internal warm starting.

### Safety-filter assembly

The concrete API target for the driving workload looks like:

```python
import scaly as sc

dynamics = sc.load_nn_dynamics("model.pth")  # input-affine: returns (f, g) Functions

@sc.function("safety_filter", {"x": (NX,), "u_ref": (NU,)})
def safety_filter(x, u_ref):
    f_x = dynamics.f(x)
    g_x = dynamics.g(x)
    h, h_grad = cbf(x)
    P = sc.eye(NU)
    c = -u_ref
    G_ineq = h_grad @ g_x
    l_ineq = -(h_grad @ f_x + alpha * h)
    u_ineq = sc.inf
    return {"u": sc.qp(P=P, c=c, G_ineq=G_ineq, l_ineq=l_ineq, u_ineq=u_ineq, solver="piqp")}
```

The same shape carries over to the NLP variant with `sc.nlp(...)` and `f(x, u)` evaluated inside the oracle.

Exit criteria:

- Scaly repository split out, with PIQP + IPOPT building and shipping in wheels for macOS (arm64 + x86_64) and Linux (manylinux_2_28).
- `libipopt.{dylib,so}` is either statically linked against libgfortran/libgcc/libstdc++ or repaired/vendored so a standalone C++ binary has no undeclared host Fortran-runtime dependency.
- Both safety-filter variants build, JIT, and execute through the universal ABI.
- PIQP and IPOPT bindings pass small NLP/QP test problems against reference solutions (CasADi+PIQP, CasADi+IPOPT).
- A C++ harness can call the generated safety filter through the universal ABI with realistic per-step runtime, linking against the vendored `libipopt` / `libpiqpc`.
- Two-sided general inequalities are exercised by tests on both backends.
- Lagrangian Hessian and sparse Jacobian factory paths are exercised end-to-end by the NLP variant.

## Benchmark-driven development

The workloads now serve as regression and measurement cases. Current behavior belongs in the
[results](../docs/results/index.md); new compiler work requires a failing test and measured workload,
then becomes an actionable item in [todo.md](todo.md).

## Test strategy

### Unit tests

- Expression construction, shape inference, evaluation.
- Expression graph topological ordering and Program IR verification.
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
4. **M3 Loop-preserving lowering**: VMAP-based per-stage callee reuse and colored sparse Jacobians constant in `N`/`C`, demonstrated on tracking NMPC `eq_constraints_jac` and VMAP-based unbumpercars `ineq_constraints_jac`. *Done.* Broader mixed scalar/block/opaque region formation is on standby until the safety-filter workload requires it.
5. **M4 JIT as default**: `Function.__call__` lazily renders, compiles, caches, and dispatches through the universal ABI; interpreter preserved as debug fallback. *Done.*
6. **M5 QP and NLP solvers**: `sc.qp(...)` with PIQP and `sc.nlp(...)` with IPOPT, driven by the CBF safety-filter workload. *Bindings + IR nesting + C codegen + AOT/C++ workflow landed*. Both builders ship, sparse Jacobian / sparse Lagrangian Hessian for IPOPT come through `Function.factory(...)`, and the new `ExprOp.SOLVER_CALL` op + `SolverFunction` subclassing `Function` lets a solver be embedded inside any other `@sc.function` graph. **Nested QP and NLP both compile to a single `.so` that links directly against the vendored `libpiqpc` / `libipopt` — no Python in the hot path.** An AOT Google Benchmark harness measured ~4 µs/solve (QP) and ~450 µs/solve (NLP) on a recent macOS arm64 dev box; that script has since been absorbed into `benchmarks/run.py`. See [`docs/guide/solvers.md`](../docs/guide/solvers.md). Direction (2026-07-14, `internal/notes/benchmark-buildout.md` §3.4 + L2), **landed 2026-07-15**: the generated single-`.so` path is the universal solver integration and the **only** solve path — sparse PIQP lives there (`sc.qp(..., sparse=True)`, CSC patterns baked at codegen, values-only updates), IPOPT callbacks point at generated kernels (workspace pass-through fixed the chain-scale crash) with full warm starts (`x0`, `lam_eq0`/`lam_ineq0`, sign-split `lam_box0`), and the scaly-owned stats struct resurfaces status/iterations/eval counts and the FE/solver/glue timing split. The Python-interleaved backends (nanobind `_piqp_ext`, ctypes IPOPT callbacks) are **deleted**; solver plugins ship vendored libs + headers + entry-point metadata + their C wrapper template (`render_wrapper`, moved out of core 2026-07-15 — see [`docs/dev/solver_plugins.md`](../docs/dev/solver_plugins.md) and `notes/benchmark-buildout.md` §3.5). Still open: PIQP warm-start handover (no C API for it upstream), implicit-function AD through `SOLVER_CALL`.

GPU codegen and broader CasADi/anvil interop are deferred until a workload's CPU runtime or migration need actually justifies them.
