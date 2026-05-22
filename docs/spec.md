# Alloy IR spec

Alloy is a small CasADi-like symbolic compiler living in pure Python under `src/alloy`. The name fits the anvil/metal theme and the core idea of mixing SX-like scalar lowering with MX-like block lowering in one graph.

See [`roadmap.md`](roadmap.md) for the milestones that take Alloy from the proven prototype to a Program-IR-based compiler with GPU support.

## Goals

- Provide a small CasADi-like symbolic core in pure Python.
- Treat `Function` as the unit of compilation and composition.
- Support local lowering choices: scalar/SX-like regions and block/MX-like regions in the same graph.
- Preserve a CasADi-style universal C ABI while allowing optional typed buffer wrappers for generated C++.

## Compiler pipeline (vocabulary)

Use **semantic IR** for the current `Expr` graph: it captures mathematical operations and domain structure (`matmul`, `call`, `map`, sparsity descriptors, solver boundaries). It is not a "tensor IR" — most values happen to be dense tensors today, but the layer is semantic, not tensor-shaped per se.

Use **Program IR** (planned, see roadmap Phase 4+) for the lower representation that looks like code: buffers, explicit loops, indices, loads/stores, calls, kernel launches, and memory spaces.

Phases vs artifacts:

```
User API (@al.function / Expr.sym / qp / nlp)
  │  tracing / construction phase
  ▼
Semantic IR (Function + Expr DAG + types/sparsity)
  │  semantic passes (AD factories, sparsity/coloring, CSE, MAP/SCAN analysis)
  ▼
[Lowering and placement phase: regions, device policy, layouts]
  ▼
[Program IR (planned): procedures, kernels, loops, buffers, loads/stores]
  │  backend schedule phase (CPU loops or GPU kernelization/tiling)
  ▼
Rendered artifacts (C today; CUDA/OpenCL/Metal/headers later)
  │  build / cache / dispatch phase
  ▼
Host ABI: int f(arg, res, iw, w, mem)
```

The verifier discipline (per IR level, see roadmap Phase 2) and PatternMatcher-backed specs follow the same separation: each artifact has an explicit spec table, and a verifier reports the first node that violates an invariant.

## Current public API

```python
import alloy as al

x = al.sym("x", 3)
y = (x.sin() + x * x).sum()
f = al.Function("f", [x], [y], ["x"], ["y"])

# Anvil-style scoped construction is the preferred user-facing direction.
@al.function("f_scoped", {"x": 3})
def f_scoped(x):
    return {"y": (x.sin() + x * x).sum()}

assert f([1.0, 2.0, 3.0]).shape == ()
assert f_scoped([1.0, 2.0, 3.0]).shape == ()

# Canonical factory API
g = f.factory("g", ["x"], ["grad:y:x"])
J = f.factory("J", ["x"], ["jac:y:x"])
fwd = f.factory("fwd", ["x", "fwd:x"], ["fwd:y:x"])
adj = f.factory("adj", ["x", "lam:y"], ["adj:y:x"])
spJ = f.factory("spJ", ["x"], ["spjac:y:x"])
spH = f.factory("spH", ["x"], ["sphess:y:x:x"])

# Human-friendly wrappers over the same factory language
g2 = al.gradient(f, "x", "y")
J2 = al.jacobian(f, "x", "y")
fwd2 = al.forward(f, "x", "y")
adj2 = al.adjoint(f, "x", "y")
spJ2 = al.spjacobian(f, "x", "y")
H2 = al.hessian(f, "x", "y")
spH2 = al.sphessian(f, "x", "y")

# Lower-level AD building blocks over Expr graphs
seed = al.sym("seed", y.shape)
(vjp_x,) = al.vjp((y,), (x,), (seed,))
seeds = al.sym("seeds", (4, *x.shape))
fwd_batch = al.jvp_many(y, x, seeds)  # shape (4, *y.shape)
```

## Core compiler concepts

Alloy's current names map to compiler/tinygrad concepts like this:

| Alloy concept | Compiler concept | tinygrad/anvil analogy |
| --- | --- | --- |
| `Ops` | semantic opcode enum | tinygrad `Ops` |
| `Expr` | semantic IR node (immutable, SSA-ish value) | tinygrad `UOp` |
| `Function` | named graph boundary and compilation unit | anvil `NumericalFunction`, CasADi `Function` |
| `Tape` / `Instruction` | transitional schedule/debug view | tinygrad linearized UOps, CasADi `algorithm_` |
| `Function.factory()` | derived graph/function builder | CasADi factory strings |
| `lowering` hint (`auto`/`scalar`/`block`/`opaque`) | region/codegen policy seed; will fold into Program IR placement policy | future scalar/block partitioning metadata |

`Ops` is a `StrEnum`, so opcodes are proper enum values while still being convenient to serialize and print as strings. The op set is intentionally semantic — `LOAD`/`STORE`/`THREAD_ID` and other Program IR concepts are deliberately kept out.

## MVP operation set

The initial supported operation set intentionally covers common modeling and control expressions before specialized numerics:

- Structural: `input`, `const`, `sum`, `reshape`, `vec`, `transpose`, `slice`/indexing, `split`, flat `gather`, `scatter`, `stack`, `concat`, `matmul`, `call`.
- Arithmetic: `add`, `sub`, `mul`, `div`, `pow`, `neg`.
- Reductions/helpers: `dot`, `sumsqr`, `norm_2` lower to existing elementwise, reshape, sum, and sqrt ops.
- Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`.
- Hyperbolic: `sinh`, `cosh`, `tanh`.
- Elementary nonlinearities: `exp`, `log`, `sqrt`, `abs`.
- Nonsmooth/common utilities: `floor`, `ceil`, `minimum`, `maximum` are evaluable but not all have AD rules yet.

`TensorType.diff` is metadata indicating whether an expression depends differentiably on symbolic inputs. Constants and `diff=False` symbols are non-differentiable, structural operations preserve the flag, differentiable arithmetic propagates it from differentiable inputs, and nonsmooth operations such as `floor`, `ceil`, `minimum`, and `maximum` mark outputs as non-differentiable.

Tensor and sparsity shapes must use nonnegative dimensions. If `TensorType.sparsity` is set, its rank-2 shape must match the tensor shape exactly.

Explicitly deferred:

- `expm1`, `log1p`: useful but not essential for the first modeling slice.
- Splines/interpolants: important, but their semantics need a separate design for knots, lookup, extrapolation, derivative behavior, table codegen, and sparsity.
- Matrix decompositions/exponentials and solver control flow.

## Tape (transitional)

`Tape` is a topologically sorted linear instruction stream of an expression graph — a debug/schedule view, not an AD tape. It is **transitional**: today it feeds the interpreter, the C renderer's lifetime/workspace planning, and debug printing. The long-term plan (see roadmap Phase 4+) is to move scheduling responsibilities into Program IR; `Tape` will then survive only as a small debug topological view, if at all.

Do not treat `Tape` as the future common compiler IR. New compiler work (verifiers, pattern rewrites, lowering) should sit on `Expr` (semantic) or, once it exists, Program IR — not on `Tape`.

Alloy currently exposes:

```python
tape = f.tape()
for instr in tape:
    print(instr.op, instr.inputs, instr.lowering)

print(tape.debug())
print(tape.lowering_regions())
```

The tape debug format annotates non-`auto` lowering hints inline, making early scalar/block/opaque boundaries visible before real region partitioning exists. `Tape.lowering_regions()` returns contiguous `TapeRegion` runs over the scheduled instructions; this is only a first metadata grouping, not full graph partitioning. The tape is meant to feed interpreters, C renderers, workspace planning, and later pattern rewriting.

## Call-node AD

`Function.call(...)` creates first-class `call` expressions and normalizes constant-like arguments to `Expr` nodes before shape checking. JVP currently differentiates through calls by inlining the callee derivative graph and substituting actual call arguments for formal inputs. That construction uses memoized JVP subgraphs and topological cached substitution so repeated named-call structure, such as an RK4 stage function inside a horizon-level equality constraint, does not repeatedly re-walk the same derivative graph. The lower-level `al.vjp(outputs, wrts, cotangents)` reverse pass follows the same initial policy for call nodes. This keeps nested named functions usable for `jac:*`, `grad:*`, and `hess:*` factory outputs while leaving room for future policies that call cached derivative functions instead of inlining.

## Sparsity

Alloy now has explicit `ScalarType` and `SparsityType` vocabulary alongside `TensorType`. `SparsityType` stores a rank-2 structural pattern as COO-style `(rows, cols)` plus shape, with mask conversion helpers.

```python
sp = al.jacobian_sparsity(y, x)
assert sp.shape == (y.size, x.size)
mask = sp.to_mask()
```

`jacobian_sparsity(expr, wrt)` is symbolic and value-independent. It currently covers the structural/arithmetic subset, matmul conservatively, and call nodes via boolean chain rule through the callee. `SparsityType` can convert its COO metadata to and from CSR/CSC pointer/index arrays for solver and codegen interfaces.

`al.sparse_jacobian(y, x)` returns a `SparseJacobian` with the structural pattern and a compact `values` expression containing the Jacobian entries at the nonzero coordinates. The default path first tries a structured decomposition: when `y` is (or is an axis-0 `concat` of) one or more `Ops.MAP` nodes whose outer tensors are exactly `x`, every MAP piece goes through `_sparse_jacobian_map`, which per formal input computes a local sparsity tile on the callee, colors that tile, materializes a constant local-color seed matrix, dispatches through `_call_jvp_many_const_function` (so the per-formal JVP body benefits from seed-constant DCE/CSE), wraps each per-formal JVP in `Ops.MAP` over the original slicing, and assembles the global compact nnz buffer through a constant scatter index table. The fallback (and the path for any non-MAP piece) is `al.sparse_jacobian_colored(y, x)`: it computes `jacobian_sparsity`, greedily colors structurally independent columns of the global pattern, builds one compressed seed vector per color, evaluates JVPs for those colors, simplifies/CSEs each compressed column, and gathers each nonzero from `(row, color[col])`. `SparseJacobian.to_dense()` scatters the compact values back into a dense matrix expression. `al.sparse_jacobian_reference(y, x)` remains available as the dense-Jacobian-then-gather correctness path for small tests. Greedy `column_coloring(...)` and `color_groups(...)` expose the graph-coloring vocabulary.

At the named-function layer, `spjac:out:in` and `sphess:out:in:in` factory requests return compact nonzero values and attach the corresponding pattern in `Function.output_sparsities`. `al.spjacobian(fn, input, output)`, `al.sphessian(fn, input, output)`, and `al.sparse_lagrangian_hessian(...)` are convenience wrappers over those requests.

Generated C headers include sparse output metadata for compact derivative buffers:

```c
#define f_spjac_y_x_NNZ 4
#define f_spjac_y_x_NROW 3
#define f_spjac_y_x_NCOL 4
static const int f_spjac_y_x_rows[4] = {0, 1, 1, 2};
static const int f_spjac_y_x_cols[4] = {0, 2, 3, 1};
static const int f_spjac_y_x_csr_row_ptr[4] = {0, 1, 3, 4};
static const int f_spjac_y_x_csr_col_ind[4] = {0, 2, 3, 1};
static const int f_spjac_y_x_csc_col_ptr[5] = {0, 1, 2, 3, 4};
static const int f_spjac_y_x_csc_row_ind[4] = {0, 2, 1, 1};
```

## Dtype model and device placement (Phase 1)

`DType` is a small interned descriptor with a name, bit-width, C-type spelling, and category flags. The canonical instances live in `dtypes`:

```python
al.dtypes.bool_      # uint8_t, 1 byte
al.dtypes.int32      # int32_t, 4 bytes
al.dtypes.int64      # int64_t, 8 bytes
al.dtypes.float32    # float,   4 bytes
al.dtypes.float64    # double,  8 bytes  (default)
```

For back-compat with prior `dtype="float64"` string usage, `DType.__eq__` accepts strings. Construction APIs (`Expr.sym`, `Expr.const`, `TensorType`, `BufferType`) accept either a `DType` instance or its string name and coerce in `__post_init__`.

Mixed-dtype binary ops are refused without an explicit cast — there is no implicit numeric promotion (`promote_dtype` only verifies that all operands share dtype). This will become more permissive when an explicit `Expr.cast(dtype)` op lands, but the conservative default keeps current `float64` workloads identical and makes future mixed-precision code explicit at the call site.

`DeviceSpec("host" | "cuda" | "opencl" | "metal", index)` is the public placement value:

```python
fn_gpu = fn.with_device("cuda:0")        # placement policy
assert fn_gpu.device == al.DeviceSpec("cuda", 0)
```

Each backend registers a `BackendSupport` capability table (`BACKEND_SUPPORT`). For example `metal` does not currently advertise `float64`, so `Function(..., device="metal:0")` over a `float64` graph fails at construction with a clear diagnostic — not at runtime, and not as a silent host fallback.

`host`, `metal:N`, and `cuda:N` all lower end-to-end today. `fn(np.array(...))` routes through `alloy.metal_runtime.MetalCompiledFunction` (`xcrun metal` + libobjc + Metal/Foundation via `ctypes`) for Metal, and through `alloy.cuda_runtime.CudaCompiledFunction` (nvcc + dlopen + `ctypes` on the universal ABI) for CUDA. CUDA auto-detects a compatible nvcc by smoke-launching a tiny kernel against the installed driver, walking `/usr/local/cuda-*` newest-first; `ALLOY_NVCC` overrides. `opencl:N` parses as a placement but has no runtime yet — `JitError("only host lowering is implemented")` triggers there.

Future phases will extend this:

- Phase 4+ — Program IR carries placement per region so a single host `Function` can contain CPU procedures, GPU kernels, and solver calls.
- Phase 8 follow-ups — kernel splitting at "thread-bound elementwise → REDUCE" boundaries (today the schedule pass falls back to grid=1/block=1 in that shape), multi-axis thread binding, OpenCL backend, libcuda-direct dispatch to skip the nvcc per-function compile.

## Mixed lowering

Every expression carries a lowering hint:

- `auto`: let the lowering pass choose.
- `scalar`: prefer SX-like scalar/unrolled code.
- `block`: prefer MX-like block/loop code.
- `opaque`: preserve a named call or future solver/integrator boundary instead of forcing it into symbolic scalar/block lowering.

Example:

```python
sparse_model = (x.sin() + x * x).scalar()
dense_layer = (A @ x).block()
opaque_boundary = dense_layer.opaque()
y = sparse_model + opaque_boundary
```

Today this is metadata preserved by the tape and shown by `Tape.debug()`. Later it should drive partitioning: scalar regions lower to explicit scalar instructions, dense regions lower to block kernels/loops, and the boundary inserts materialization/copy/project operations.

## Program IR (Phase 4)

`src/alloy/program.py` introduces the lower IR where explicit loops, buffers, loads/stores, calls, kernel launches, and memory spaces live. Phase 4 ships the vocabulary, the verifier, and the pretty-printer; the lowering pass that produces Program IR from semantic IR is Phase 5.

A single `PNode` class (frozen, hash-consed) carries an op tag from `POps`. Statement vs declaration vs scalar-expression is encoded by op tag, matching the tinygrad UOp style and reusing the spec-table verifier infrastructure from Phase 2:

```python
from alloy import program as p
from alloy.types import dtypes
from alloy.program import RangeKind, verify_program, format_program, spec_program_full

in_buf = p.buffer("in_", dtypes.float64, (16,))
out_buf = p.buffer("out_", dtypes.float64, (16,))
i = p.var("i", dtype=dtypes.int64)
k = p.kernel(
    "k_neg",
    [in_buf, out_buf],
    [
        p.for_(
            p.range_("i", 0, 16, kind=RangeKind.GLOBAL),
            [p.store(p.view(out_buf, [i]), p.neg(p.load(p.view(in_buf, [i]))))],
        ),
    ],
    grid_dims=1,
    device="cuda:0",
)
verify_program(k)
print(format_program(k))
```

`RangeKind` borrows tinygrad's `AxisType` vocabulary (`SERIAL`, `VECTOR`, `GLOBAL`, `THREAD`, `LOCAL`, `WARP`, `REDUCE`, `GROUP_REDUCE`, `UNROLL`). Backends use it to bind loops to launch axes, choose vectorized vs unrolled emission, or lower reductions. Buffer address spaces follow the OpenCL/CUDA naming (`global`/`local`/`private`/`constant`).

Verifier specs:

- `spec_program_shared` — invariants every Program IR node must satisfy (buffer attrs, view scalar args, load/store shapes, range kind enum, etc.).
- `spec_host_program` — host PROCs may not contain device-only ops (e.g. `BARRIER`).
- `spec_kernel_program` — KERNELs may not contain host-only ops (e.g. `LAUNCH`).
- `spec_program_full` — both, suitable for whole-program checks.

KERNEL nodes also carry a `bind_threads` attribute (default True) that the schedule pass sets to False when binding the first GLOBAL FOR to `(blockIdx, threadIdx)` would race — e.g. a top-level REDUCE that reads from a thread-private workspace produced by an earlier GLOBAL elementwise (the `sum(elementwise(x))` shape). The renderers honor it: when False, they emit a serial kernel and the host driver launches with `<<<1, 1>>>` (CUDA) or gates on `tid == 0` (Metal). Splitting into two kernels with a global scratch is the right long-term fix; the conservative fallback keeps results correct in the meantime.

Pretty-printing is backend-neutral and decoupled from C syntax — it shows host/device boundaries, range kinds, and launch specs so schedule decisions are debuggable before any renderer runs.

### Program-IR-backed C renderer (Phase 5, feature-flagged)

`alloy.lowering.lower_function(fun)` translates a small subset of semantic IR (today: elementwise unary/binary, `RESHAPE`, small `CONST`) into a host `PROC`. Setting ``ALLOY_USE_PROGRAM_IR_C=1`` routes ``Function`` instances in this subset through the new ``alloy.codegen.program_c.render_program_c_source`` renderer; anything outside the subset falls back to the legacy scalar renderer silently. The flag is the migration lever — extending Program IR coverage (``SUM``, ``MATMUL``, ``GATHER``, ``CALL``, ``MAP``, ``SOLVER_CALL``) is the upcoming work and lets the new path own more functions over time without disturbing benchmarks.

## Semantic loop ops: MAP and SCAN

`Ops.MAP` is the existing loop op for **independent repeated calls**: each iteration reads sliced inputs from outer tensors and produces an output, with no carry between iterations. Today this covers the tracking interstage residual loop, batched dynamics in unbumpercars, batched MLP rollouts, and similar workloads.

`Ops.SCAN` is reserved for **dependent recurrences** ``f(state_{i-1}, x_i) -> state_i`` — integrators, RNN cells, sequential filtering. It is declared in `Ops` so verifiers and printers can name it, but there is no constructor or codegen path yet: SCAN lands when a workload demands carry semantics. Do not extend MAP with recurrence semantics — keep the two ops separate.

For backward compatibility, ``al.scan`` is currently an alias for ``al.map_``. The alias is transitional and will move to the real SCAN constructor when one exists; new code wanting independent loops should prefer ``al.map_`` directly.

## Semantic IR verifier (Phase 2)

Each IR level should have an explicit spec table and verifier (see roadmap Phase 2). The semantic IR layer ships its spec in `src/alloy/spec.py`:

```python
import alloy as al

x = al.sym("x", 3)
y = (x.sin() + x * x).sum()
al.verify_expr(y)                       # silent on success
al.verify_expr(y, spec=al.spec_semantic)  # explicit spec choice
```

The verifier walks the DAG topologically and raises `VerifyError` at the first invalid node, naming the node, its op, and the failing rule. Two specs are exported today:

- `spec_semantic_shared` — rules every node must satisfy (non-negative shape, `DType`-typed dtype, arity matches `OP_INFO`, sparsity shape agrees with tensor shape).
- `spec_semantic` — `spec_semantic_shared` plus per-op rules (e.g. `RESHAPE` size, `TRANSPOSE` axes are a permutation, `MATMUL` contracting dims, `CALL` arg shapes match the callee, `MAP` outers are rank-1 with consistent `slice_size`, `CONST` value shape/dtype match the declared type).

Verification is opt-in: construction-time checks in `expr.py` keep the happy path fast. `verify_expr` is the explicit defensive check passes should run after non-trivial graph rewrites or AD transforms, and the harness for negative tests in `tests/alloy/test_verifier.py`.

## Structural equality and rewrites

`Expr.id` remains a construction-time identity, while `Expr.structural_key()`, `Expr.structural_hash()`, and `Expr.structurally_equal()` compare graph structure. This lets compiler passes identify equivalent subgraphs without changing user-visible node identity. `Expr.debug()` / `al.format_expr(...)` print stable topological `%0`, `%1`, ... names for small graph inspection.

Alloy also has a small graph rewrite layer:

```python
y_cse = al.cse(y)
y_clean = al.simplify(y_cse)
```

The first pass supports CSE, constant folding, and simple algebraic identities such as `x + 0`, `x * 1`, `x * 0`, and identity reshape/transpose. Rewrites use an op-indexed `PatternMatcher` and topological replacement cache so DAG sharing is preserved better than a recursive tree walk. This is still intentionally much smaller than tinygrad's `UPat`/`PatternMatcher`, but it establishes the compiler path for AD cleanup, canonicalization, and later tape-level peepholes.

## Execution: JIT by default, tape interpreter as reference

`Function.__call__` lazily renders the function to C, compiles it through `cc`, caches the resulting shared object on disk and in-process, and dispatches through the universal ABI via `ctypes` (see `src/alloy/jit.py`). Subsequent calls reuse the cached `.so`. Multiple `Function` instances with identical generated source share the same artifact via a SHA-256 key over (`_JIT_CACHE_VERSION`, ABI signature, function name, rendered C source).

The Python tape interpreter remains the reference path. `Function.eval_interpreter(*args, **kwargs)` always evaluates through the deterministic tape, and is what internal `CALL`/`MAP` nodes use when `Tape.evaluate` or `Expr.eval` walks the graph. The interpreter is automatically used as a fallback when no C compiler is available or codegen raises `NotImplementedError`, and explicitly when ``ALLOY_DISABLE_JIT=1``.

Environment variables that influence JIT behavior:

- ``ALLOY_DISABLE_JIT=1`` — skip JIT entirely and always use the interpreter.
- ``ALLOY_CACHE_DIR`` — override the on-disk cache root (default ``$XDG_CACHE_HOME/alloy/jit`` or ``~/.cache/alloy/jit``).
- ``ALLOY_CC`` — override the C compiler binary (default ``cc`` from ``$PATH``; works on Linux/macOS/BSD where ``cc`` is the POSIX symlink to the system's default C compiler).

`Function.recompile()` drops both the in-process compiled handle and the on-disk cache directory for the next call.

The tape also exposes a first workspace plan:

```python
plan = f.tape().plan_workspace()
assert plan.size <= sum(slot.size for slot in plan.slots)
```

The workspace plan is a deterministic lifetime-based scratch layout for non-leaf tape instructions. The scalar C renderer currently prefers local C temporaries, but the plan remains useful for future lowered regions that need caller-provided scratch buffers.

## Benchmark workload fixtures

The first benchmark-shaped Alloy fixture is the tracking NMPC equality constraint Jacobian in `tests/alloy/test_tracking_workload.py`. It defines scoped stage functions for initial-state pinning and RK4 interstage dynamics, composes them into a horizon-level named function through `Function.call(...)`, and checks small-horizon dense Jacobians against CasADi. It also checks that colored compact sparse-Jacobian metadata and values agree with the dense-gather reference path. An explicit opt-in sweep runs with `ALLOY_TRACKING_SWEEP=1 uv run pytest tests/alloy/test_tracking_workload.py -q -s`; it covers horizons `N=1,2,5,10`, records expression/tape node counts, sparsity nnz, coloring count, AD construction time, and generated C source size, and compares compact values against CasADi. For native timing without Python/`ctypes` overhead, `uv run python benchmarks/alloy_tracking_eq_jac_benchmark.py --horizons 1 2 5 10 -- --benchmark_min_time=0.01s` generates Alloy, CasADi SX, and CasADi MX C, compiles one Google Benchmark C++ binary per horizon, and reports runtime plus generated source sizes. The current tracking baseline preserves named stage derivative calls and batched color seeds, giving much smaller horizon growth than flat scalarization; after local temporaries, slice-aliasing, and constant-seed specialization, one `N=50` run produced Alloy at 364 KB / 13112 lines / 0 workspace / 5.1 us, CasADi SX at 466 KB / 25902 lines / 77 workspace doubles / 4.6 us, and CasADi MX at 2.0 MB / 63134 lines / 8046 workspace doubles / 8.4 us.

A second benchmark-shaped fixture is the official-size unbumpercars inequality Jacobian in `tests/alloy/test_unbumpercars_workload.py`. It uses the same MLP layer sizes as the example model (`256 -> 128 -> 3`) and loads the official `model_kinematic_mlp.pth` weights into the parameter vector, while keeping the normal unit test at `N=2`. Dense MLP matmuls are marked with `block` lowering metadata and sparse C3BF/wall/slack assembly with `scalar` metadata. Tests compare dense and colored sparse Alloy Jacobians against CasADi MX for `N=2`, including sparsity structure. `uv run python benchmarks/alloy_unbumpercars_ineq_jac_benchmark.py --stats-only --car-counts 2` prints source/workspace stats for Alloy, CasADi SX, and CasADi MX; `--backend alloy --backend casadi-mx` runs the native Google Benchmark without trying to compile the huge SX source. After stack-through-slice and slice-of-slice rewrites and scalar/vector inlining through elementwise/stack/concat consumers, a local `N=2` stats run produced Alloy at 38 KB / 1235 lines / 0 workspace, CasADi SX at 12.2 MB / 447759 lines / 36274 workspace doubles, and CasADi MX at 217 KB / 6689 lines / 141500 workspace doubles; the Alloy-vs-MX native run was ~94 us vs ~252 us. The SX source is correctly enormous for the dense MLP case and was not compiled in the quick benchmark loop.

Both benchmark harnesses also run a fail-fast correctness check before invoking Google Benchmark: each backend's compact Jacobian is scattered into a dense matrix using its own `(rows, cols)` sparsity, and compared to a Python alloy reference dense Jacobian written next to the generated sources. The check catches NaN-vs-finite mismatches and exits non-zero with a per-element diff. The compact-vs-compact diff that this replaces was wrong because Alloy uses COO row-major nnz order while CasADi uses CSC column-major.

The C codegen lifetime-packs storage-owning instructions into shared C locals `sN[max_size]`: instructions whose lifetimes don't overlap reuse the same slot. Aliases (RESHAPE, contiguous SLICE pointer aliases), inlined scalar/vector chains, and peephole-skipped intermediates all extend the underlying owner's last-use so that a reused slot is never overwritten while a downstream consumer can still observe it. This drops the per-JVP MLP stack frame on the official-size unbumpercars workload from ~25 KB to ~14 KB while keeping each slot a distinct `double` array — preserving the no-aliasing model the compiler reasons about for register promotion.

The C codegen previously had a correctness bug: `_contiguous_slice_offset` mis-detected column-style slices like `(slice(None), 1)` on a `(5, 7)` tensor as contiguous and emitted a pointer alias that read flat indices `[1, 2, 3, 4, 5]` instead of the column `[1, 8, 15, 22, 29]`. The Python tape interpreter used NumPy so unit tests passed, but compiled benchmarks could silently produce wrong derivative values. The fix now rejects int indices that appear after a slice unless the indexed dim has size 1, and there is a compile-and-verify regression test in `tests/alloy/test_core.py`.

## Scalability sweep

The full per-cell sweep across tracking horizons `N ∈ {1, 5, 10, 25, 50, 100, 200, 500, 1000}` and unbumpercars car counts `C ∈ {2, 4, 8, 16, 32}` is in `benchmarks/scalability_sweep.py`. It compiles one Google Benchmark binary per `(workload, size, backend)` cell with a configurable per-compile timeout (default 180 s) and a max-source-size cap (default 50 MB). Backends that hit a timeout or size cap at one cell are short-circuited to `skipped_after_failure` at all larger cells of the same workload. CSV output lives at `benchmarks/scalability_results.csv` and is summarized in [`scalability.md`](scalability.md). The short version: Alloy is within ~10 % of CasADi SX runtime and **smaller in C source** than SX past `N=10` on tracking, and beats CasADi MX by ~2.7× on the official-size unbumpercars workload; the remaining open work is constant-source loop preservation rather than per-stage performance.

## ABI

The baseline ABI is CasADi-compatible in spirit:

```c
int f(const double** arg, double** res, int* iw, double* w, void* mem);
```

This makes nested calls, generated solvers, and possible acados-style consumers easier. Alloy also supports optional typed buffer wrappers in generated headers, e.g. fixed-size structs for statically known shapes. The universal ABI stays the lowest common denominator; typed buffers are sugar for safer C++ integration.

The caller owns all ABI storage:

- `arg` is an array of `f_SZ_ARG` input pointers. Each non-null `arg[i]` points to a contiguous row-major `double` buffer with the statically known flattened input size.
- `res` is an array of `f_SZ_RES` output pointers. Each non-null `res[i]` points to caller-owned contiguous row-major `double` storage with the statically known flattened output size.
- `w` points to at least `f_SZ_W` doubles when `f_SZ_W > 0`; it may be null only when `f_SZ_W == 0`.
- `iw` is currently unused because `f_SZ_IW == 0`; callers may pass null.
- `mem` is currently unused by pure symbolic functions; callers may pass null after using the stateless memory hooks.
- Inputs do not have default values today: every required `arg[i]` must be non-null.
- Generated code assumes ordinary C `double`/`int` alignment for the buffers supplied by the caller.

Generated headers declare `f_sz_arg`, `f_sz_res`, `f_sz_iw`, and `f_sz_w` helpers, plus compile-time `f_SZ_*` constants for generated callers. The current source renderer emits standalone scalar C for the tape subset used by the core symbolic ops and returns nonzero error codes for missing ABI pointer slots. Tape temporaries are currently emitted as C locals rather than caller workspace, so pure generated functions usually report `f_SZ_W == 0`; the workspace ABI remains for future lowered regions or solver/integrator state that actually needs caller-provided scratch. When a rendered function contains `CallOp` nodes, `render_c_source` emits internal `static inline` raw callee bodies before the exported ABI wrapper and invokes those raw bodies directly from callers. Only the root rendered function is exported through the universal ABI in that translation unit; nested callees are implementation details unless rendered as roots themselves.

When typed buffers are enabled, generated headers also include a small C++ inline wrapper such as `f_call(const f_x_in& in_x, f_y_out& out_y)`. The wrapper builds the universal `arg`/`res` pointer arrays, allocates fixed-size stack workspace from `f_SZ_W`, and calls the canonical ABI. Generated C++ headers also emit `static_assert` checks that each typed buffer struct has the expected flattened `double` size. This is intentionally sugar; the pointer ABI remains the stable interface.

`render_c_module(fn)` packages the generated artifacts as a `CModule(header_name, source_name, header, source)`. The module source includes the generated header first, so it can be compiled as a C translation unit and linked into C++ callers that include the same header.

Generated modules expose stateless memory hooks today:

```c
void* f_alloc_mem(void);
int f_init_mem(void* mem);
void f_free_mem(void* mem);
```

For current symbolic functions `f_alloc_mem()` returns `NULL`, `f_init_mem(mem)` ignores `mem` and returns `ALLOY_SUCCESS`, and `f_free_mem(mem)` is a no-op. The hook shape leaves room for future solver/integrator memory without changing the exported ABI.

Generated functions return named ABI status codes:

| Code | Meaning |
| --- | --- |
| `ALLOY_SUCCESS = 0` | Evaluation succeeded. |
| `ALLOY_ERR_NULL_ABI = 1` | The top-level `arg` or `res` pointer array is null. |
| `ALLOY_ERR_NULL_WORK = 2` | The function needs floating workspace but `w` is null. |
| `ALLOY_ERR_NULL_RESULT = 3` | One required `res[i]` output buffer is null. |
| `ALLOY_ERR_NULL_INPUT = 4` | One required `arg[i]` input buffer is null. |

## QP and NLP solvers

`al.qp(...)` and `al.nlp(...)` build opaque `SolverFunction`s wrapping PIQP
(dense QP) and IPOPT (sparse-Jacobian, sparse-Lagrangian-Hessian NLP). The
data exprs may be Alloy `Expr`s of free parameters; oracle derivatives come
through `Function.factory(...)` (`grad`, `spjac`, `sphess` with a Lagrangian
`aux={"gamma": [...]}`). Each solver is a real `Function` whose body is
``Ops.SOLVER_CALL`` (one per output, all sharing a ``SolverDescriptor``
side-table). That means ``solver.call([...])`` returns ``Expr``s, so solvers
nest directly inside larger ``@al.function``-decorated graphs (the
safety-filter assembly pattern). ``SOLVER_CALL`` is marked
non-differentiable, and C codegen for it is not yet wired — parent Functions
containing a solver node fall back to the tape interpreter. The bound shared
libraries live under `src/alloy/lib/`. Full interface, sign conventions, and
current limitations are documented in [`solvers.md`](solvers.md).
