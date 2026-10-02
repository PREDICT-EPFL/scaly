# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why a number is or is not admissible** — [`internal/notes/benchmark_protocol.md`](notes/benchmark_protocol.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **What the core compiler work is and why it is ordered this way** —
  [`internal/notes/core_compiler_roadmap.md`](notes/core_compiler_roadmap.md): the decisions, the
  order of work, and one paragraph of design per item below.
- **What shape a refactoring should take** — [`internal/notes/refactorings.md`](notes/refactorings.md):
  one `#` section per refactoring, kept until that refactoring lands.

Reorganized 2026-09-29 around the core compiler roadmap, and again 2026-10-01 when the roadmap was
rewritten around three milestones; completed items were flushed and their
records live in git history and the frozen notes. Sections are themes. Inside each section, **Now**
holds what is actively worked on or next in line, and **Deferred** holds what is intentionally low
priority: the reasoning is still good, nothing depends on it yet. A finished item stays in place
with its box checked until it is flushed out by hand.

### Identifiers

Every item has an identifier `<PREFIX>-<n>`. The prefix names the section the item sits in; the
number comes from one counter shared by the whole file, which only ever grows.

**Next id: 157**

The frozen experimental `devrush` branch keeps its own todo list, whose ids from C-82 up name
different items than the same ids here. Cite devrush work by file and title, never by id.

| Prefix | Section |
|---|---|
| API | API |
| C | Compiler internals |
| S | Solvers |
| CAPI | C API |
| BH | Benchmark harness |
| BP | Benchmark problems |
| L | Licensing |
| D | Documentation |
| R | Release |

Rules:

- A new item takes the next id and bumps the counter. A deleted item never frees its number.
- Moving an item to another section changes its prefix and keeps its number. Grepping the number
  alone finds the item, or proves it is gone.
- Other documents cite the full id and the title, `S-16 Separate the IPOPT gap into version against
  build configuration`, so the reference survives both a move and a retitle.
- A new section adds a row with a prefix that is not in the table and never was.

Why identifiers at all: they give the short stable handle that Linear or GitHub issues give, while
the list stays git-tracked, lives next to the code, and changes per branch, so a worktree can add,
close and reorder its own items and the merge carries them. A global counter rather than one per
section because items move between sections more often than expected, because eight counters are
eight places to get wrong once completed items are deleted, and because the letters then carry
only the theme and nothing else has to stay stable.

## Priority order

From 2026-10-01. The roadmap's [order of work](notes/core_compiler_roadmap.md#order-of-work) has the
dependencies and the rules that let lanes run in parallel.

### 0.1.0

Settled 2026-10-02, for a release planned around the end of October. 0.1.0 takes what changes what
users write or link against, and the fixes for silent wrong answers; work that is internal or only
adds things waits for later 0.x releases, which may break interfaces anyway (`docs/dev/versioning.md`).
Nothing outside this list is started before 0.1.0 ships, so nothing reaches it half done.

- Fixes: C-86, then C-87 and C-93; C-88, C-89, C-91, C-92, S-155, S-142.
- Templates, the critical path: API-90, API-114, API-3, API-1 (shape holes only, keyed on the whole
  leaf type so dtype and pattern holes are additive), API-115, then API-2 on today's `VMAP`.
  API-1 ships the templates page of `docs/guide/functions.md`. API-156 any time.
- `sc.print`: C-124, float64 only.
- Toolchain: C-83, then R-71.
- Windows: R-38, in parallel with all of the above.
- Release: R-42, the status admonition in `docs/index.md`, Windows in the CI and release matrices,
  release notes.

Everything else follows 0.1.0 in the order below; C-94 is the first candidate after it.

### After 0.1.0

1. Milestone 1, differentiable loops in a debuggable core.
   - Foundations: C-86 first, then C-87 and C-93; C-88, C-89, API-90, C-91, C-92 and C-83 in any
     order; C-94 last. R-71 right after C-83, before C-139 and before milestone 3, so `CLibrary`
     speaks one flag dialect and milestone 3 is measured under one compiler.
   - Then C-124 (`sc.print`), and in parallel lanes: the AD engine C-95 to API-101; the templates
     API-114, API-3, API-1, API-115; C-103 and C-104; C-136; C-108 and C-109; the loops C-119 to
     C-123 and C-138.
2. Milestone 2, static sparse functions end to end: C-127, C-148, C-149, C-128, C-129, API-116,
   C-130 to C-132; C-154 and C-133 to C-135; C-125; C-139; C-105 to C-107.
3. Milestone 3, competitive kernels and generated solvers: C-110 to C-112; C-150, API-2; C-8, C-113,
   C-102, C-144; C-151, C-152; API-117, API-4; C-126; the milestone checks API-140 and API-141.
4. Then D-36, the GPU milestone definition.

Small fixes that fit between any two of these: C-82, S-17, C-143, C-145.

## API

### Now

- [ ] **API-90. An immutable Function.** One constructor, a public `Function.build` taking trees,
      `_replace` as the only copy path, the solver descriptor and a `role` as constructor fields,
      `_with_trees` no longer mutating `self`. Milestone 1. [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **API-101. `sc.custom_derivative` with residuals.** Rules set at construction: a JVP rule, and
      a `fwd`/`bwd` pair in JAX's `custom_vjp` form whose residuals (a factorization, a trajectory)
      are results of the same invocation, so a solve's derivatives reuse its factorization; honoured
      by `CALL`, `LOOP` and maps through one pair of body-derivative functions; unread tangents are
      never built; declared sparsity. After C-100 and API-90.
      [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **API-114. The playground's declaration surface.** `arg` and `group` replace `L` and `G`, one
      tree per parameter, keyword-only `outputs=` that may be left out. A scripted edit of every
      decorator in `src/`, `tests/`, `benchmarks/`, `plugins/`, `examples/`, `docs/` and the
      notebooks; bodies keep their single grouped parameter. Snapshots byte-identical.
      [Design](notes/core_compiler_roadmap.md#signatures-and-templates).
- [ ] **API-3. Parameter lists typed with `TypeVarTuple`, and the zero-input contract.** The
      playground's ladders and typed-`self` calls; a call with no arguments is numerical and
      `symbolic_call()` is symbolic, so parameterless solver oracles stay legal and `ad/forward.py`
      drops `_flat_symbolic_call`. The seeded wrappers and solver call convention change here; the
      playground's static tests move to `tests/typing/`. After API-114. Rationale: refactorings.md
      "Zero-input `Function`s, and the flat call seam that survives because of them".
      [Design](notes/core_compiler_roadmap.md#signatures-and-templates).
- [ ] **API-1. Function instances, holes and names.** The playground's `Function` in front of
      today's Function, which becomes the `ConcreteFunction` instance; holes for shapes, dtypes and
      patterns, keyed on whole leaf types; the bare mode; `instantiate`; mangled names without `__`
      that encode the nesting. After API-3 and C-87.
      [Design](notes/core_compiler_roadmap.md#signatures-and-templates).
- [ ] **API-115. Lifted derivatives.** The nine derivative wrappers through `lift`, cached per
      source instance, with seeds and multipliers copying the whole leaf type, not only the shape.
      After API-1.
- [ ] **API-116. Pattern holes and the numerical sparse fast path.** A pattern hole binds from a
      sparse expression or a SciPy argument; a concrete sparse leaf also takes its values vector.
      With C-128. [Design](notes/core_compiler_roadmap.md#signatures-and-templates).
- [ ] **API-2. Typed vmap.** `sc.vmap(f, N)` returns a mapped callable, lifted with `lift`, that
      builds a map at the call site and returns the callee's output tree with a leading axis;
      `sc.window` and `sc.broadcast` markers; views read in place when affine; template callees
      instantiated from slice shapes. Built on today's `VMAP`, so 0.1.0 does not wait for `LOOP`:
      C-150 later moves it onto `LOOP`, and until C-119 each used output is its own loop, as with
      today's `vmap`. After API-115. Rationale: refactorings.md
      "`vmap` and the AD entry points erase the callee's declared trees".
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **API-156. Remove `vjp_many` from the public API** before 0.1.0, rather than as a break
      after it: `sc`, `scaly.ad.__all__`, `docs/api/ad.md` and `docs/dev/codebase.md`; its only
      caller is a test. Taken out of C-99.
- [ ] **API-117. Function-level `sc.jvp` and `sc.vjp` that keep declared trees.** AD-built callees
      stop being `Any`-typed at run time. After C-98 and API-3.
- [ ] **API-4. Finish the npmpc `FunctionTemplate` example**, and write the template page of
      `docs/guide/functions.md`. The public typed decorators, exact `Function` annotations, and
      shared Scaly/CasADi runtime parameters landed first. Replace the remaining
      decoder-architecture builders with templates. This benchmark may use the packed parameter
      length as its specialization key because it does not add more MLP layouts; a general template
      must distinguish individual layer shapes because equal parameter counts do not prove equal
      architectures. After API-115.
- [ ] **API-140. TinyMPC on main.** Port devrush's `examples/tinympc` onto the finished core as a
      test of it: loops, constant folding, loop-invariant parameters, the library dense solves.
      Within 10% of devrush's reported timings. After C-123, C-133 and C-113.
      [Checks](notes/core_compiler_roadmap.md#milestone-checks).
- [ ] **API-141. A generated sparse QP interior-point solver.** The devrush IPM's structure rebuilt
      on the library sparse LDL^T, loops, predicates and reductions, passing devrush's
      Maros-Meszaros subset, with devrush's split into a setup (the scaling, run once while the
      matrices stay fixed) and a solve, which is what PIQP's solve timer measures. An example, not
      a library. After C-135 and C-8.
      [Checks](notes/core_compiler_roadmap.md#milestone-checks).

### Deferred

- **API-118. Static Python-valued template arguments.** A horizon or an ordering as a
  specialization key. Closures and factory functions cover them today, and refusing body defaults in
  API-3 keeps the syntax free.
- **API-6. A QP-subproblem contract so scaly-sqp can use other QP plugins.** Today `scaly-sqp`
  imports only `include_dir`/`lib_dir` from `scaly_piqp` and its C template calls
  `piqp_setup/update/solve` and reads `qp->result` directly, so a future OSQP, ProxQP or HPIPM
  plugin would be a standalone solver but not an SQP backend. The contract is narrower than
  `render_wrapper`: set up a QP with fixed sparsity, refill values, solve, read the step and
  multipliers in one sign convention, report status and iteration count, clean up. The hard part is
  form reconciliation (two-sided rows and box bounds versus OSQP's single `l <= Ax <= u`, and
  stage-structured solvers) the way CasADi's `conic` layer does it. Documented as a limitation in
  `docs/guide/solver_backends.md`.
- **API-7. Specialized OCP problem/solver tier** in scaly (structured staged OCP lowering to general
  form), then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as
  claims demand.

## Compiler internals

Start each compiler item by reading how the tools that shaped Scaly solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Scaly's are modelled on,
expands small tensor ops into scalar UOps and simplifies them symbolically, and has a scheduler
that fuses elementwise producers into their consumers and a symbolic index arithmetic that turns
strided views into closed-form index expressions. MLIR's affine dialect and its loop-fusion,
affine-map and memref-normalization passes are the standard treatment of exactly the loops we emit,
and their design notes state the legality conditions we would otherwise rediscover. JAX's `vmap`
batching rules are the reference for what a mapped derivative rule should produce without
materializing per-trip index tables. The goal is to port the smallest idea that fits Scaly's two
dialects, not to adopt a framework; write down what was read and what was rejected in the item's
section of `internal/notes/core_compiler_roadmap.md` before the implementation.

### Foundations

- [ ] **C-86. Split `passes/lowering.py` into a package**, refuse a second `@lowers` for one op,
      and delete the unused multi-index `VIEW` door. First in milestone 1. Snapshots byte-identical.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-87. Key callees by identity, and give generated C one name authority.** Two Functions
      with the same name share one procedure and one invocation today, silently computing the wrong
      result (reproduced 2026-09-29). One `NameScope` for C and C++ keywords, libm names, loop
      variables, temporaries and header identifiers; closes the `_h{n}` collision guard C-53 left
      open. After C-86. [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-88. Interning sets fields once, floats are keyed by bits, constants are frozen.** Fixes
      intern hits that replace `value` and `attrs` in both dialects, `sc.const` aliasing the
      caller's array, `const_float(0.0) is const_float(-0.0)`, and `1`, `True` and `1.0` sharing a
      node. [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-89. Refuse wrong answers at the boundary.** Non-float64 input and output leaves raise
      until C-107; an active derivative through `SOLVER_CALL` raises instead of returning zero;
      `MINIMUM`/`MAXIMUM` fold with `fmin`/`fmax` like the C.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-91. Remove the GPU placeholders**: non-host `DeviceSpec`s, `BACKEND_SUPPORT`,
      `KERNEL`/`LAUNCH`/`BARRIER`, and the range kinds `THREAD`, `LOCAL`, `WARP`, `GROUP_REDUCE`.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-92. Run the verifier, test op coverage, remove recursion.** `verify_expr` at Function
      construction and before lowering, rules for `SLICE` and `SOLVER_CALL`, one test that every
      `ExprOp` is classified everywhere and that no pass names a member of the elementwise table,
      iterative `_depends_on`, `_jac_mask` and `structural_key`.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-93. `SCATTER` accumulates repeated indices**, as its docstring says; `segment_sum` as a
      builder; linear-time gather and unbroadcast adjoints. The first step of C-136. After C-86.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-83. Fingerprint the host and the compiler in the JIT cache key.** Absorbs C-85. The CPU
      features and compiler version macros from the probe `toolchain.native_recipe` already runs,
      the compiler's `--version`, real path, full command and effective flags; `Compiler` becomes a
      command tuple for R-71, which follows it. Gate: changing any of them misses the cache.
      [Design](notes/core_compiler_roadmap.md#foundations).
- [ ] **C-94. A JIT index key that skips lowering on a hit.** A graph digest of the immutable
      Function plus versions, render options and C-83's fingerprint; atomic index writes through
      `os.replace` and cache-relative paths, which hold on Windows (R-38); `invalidate_cache`
      without rendering. Gate: a fresh process with another `PYTHONHASHSEED`
      loads from the cache without calling `lower_function`; one test per hole devrush's reviews
      found; a `SCALY_JIT_KEY=verify` mode that renders on every hit, which CI runs. After C-88 and
      API-90.
      [Design](notes/core_compiler_roadmap.md#foundations).

### AD engine

- [ ] **C-95. A differential AD harness before any AD change.** Every op against the unrolled
      path, the structural path, finite differences and forward/reverse duality, with unsupported
      ops listed and their refusal tested; extended call, map and constant-seed fixtures. Proven to
      fail on a perturbed partial. [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-96. One derivative-helper cache and naming rule** in `ad/helpers.py`; a procedure mark
      set by lowering from the Function's `role` replaces `_force_noinline_raw`'s `"_fwd"` substring
      test. After C-87 and API-90. [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-97. Move the call and map derivative rules into `ad/calls.py`**, unchanged. Snapshots
      byte-identical. [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-98. One forward traversal with a leading seed axis.** Steps 1 to 3 of the plan: `jvp` is
      `jvp_many` with one seed; `_seed_axis`'s stack becomes broadcasting; callee bodies use the
      same traversal; zero tangents are structural and typed; the rank-4 transpose cap goes;
      `_jvp_many_structural`, `_jvp_many_unrolled` and `SCALY_STRICT_JVP_MANY` are deleted.
      Regenerates snapshots. [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-99. One elementwise table: partials, folding and C spelling**, lazily built, with
      stable contraction forms and the masked-cotangent helper. [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-100. Derivatives with respect to intermediate expressions**, and `sc.stop_gradient`.
      Without it, forward mode returns silent zeros for a slice of a loop carry.
      [Design](notes/core_compiler_roadmap.md#one-ad-engine).
- [ ] **C-102. Reverse mode keeps the call boundary**: one adjoint helper call per invocation, map
      adjoints through the accumulating scatter, inlining left to lowering. Milestone 3, after
      C-113; measured on every benchmark problem. [Design](notes/core_compiler_roadmap.md#one-ad-engine).

### Dtypes and the scalar vocabulary

- [ ] **C-103. One scalar coercion rule.** Weak Python scalars, strong NumPy values, explicit casts
      between expression dtypes, `sc.const` keeping ndarray dtypes, same-kind numerical casts,
      `Expr.__bool__` raising, dtype checks at calls, NumPy integer indices.
      [Design](notes/core_compiler_roadmap.md#dtypes-and-the-scalar-vocabulary).
- [ ] **C-104. Predicates, `select`, `cast`, integer `//` and `%`** as rows of the elementwise
      table in both dialects; integer and boolean formats for `sc.print`. The IR and lowering half
      after C-108; the AD half, with masked cotangents and a nonlinear treatment in the QP affinity
      proof, after C-99. [Design](notes/core_compiler_roadmap.md#dtypes-and-the-scalar-vocabulary).
- [ ] **C-105. Nonsmooth derivatives and extremum reductions.** Ties split equally, `abs` slope 0 at
      0, zero slope for `floor` and `ceil`; `max` and `min` as `REDUCE` kinds propagating NaN;
      `argmax`, `argmin` and the two norms as builders. After C-104 and C-109.
      [Design](notes/core_compiler_roadmap.md#dtypes-and-the-scalar-vocabulary).
- [ ] **C-106. float32 through lowering and the program passes.** Dtype-faithful folding and
      literals, libm names by dtype, float32 in `scalarize`, `widen_ranges`, `coalesce_stores` and
      `pack_workspace`, no float64 default on `ProgramNode`, dtype checks in `verify_program`,
      integer accumulators. [Design](notes/core_compiler_roadmap.md#dtypes-and-the-scalar-vocabulary).
- [ ] **C-107. A typed entry signature and workspace.** Double for all-float64 functions
      (unchanged), float for all-float32, `void` pointers with a byte-addressed workspace otherwise;
      typed headers, C++ buffers and JIT arrays; CasADi and solver wrappers stay float64. The
      float32 parity gate. After C-106.
      [Design](notes/core_compiler_roadmap.md#dtypes-and-the-scalar-vocabulary).

### Tensor core and the loop compiler

- [ ] **C-108. The loop-nest emitter and the iteration-plan interface.** `nest`, `reduce` and `read`
      on `LowerCtx`, with access maps that are affine or table-driven from the start so sparse
      kernels share the interface; affine view indices, quotient and remainder only where a
      composition requires them; empty extents emit no store; elementwise, movement, `STACK` and
      `CONCAT` migrated; the div/mod coordinate helpers deleted. Regenerates snapshots. After C-86.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-109. Accumulators and the contraction schedule.** Reductions as `ASSIGN` accumulators;
      `SUM` and rank-2 `MATMUL` on them; C-43's layout branches deleted and replaced by one
      scheduling decision. Gate: npmpc N=12 and unbumpercars C=8 within 5%. Regenerates snapshots.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-110. Axis reductions.** `REDUCE` with `add`/`max`/`min` over axes replaces `SUM`;
      `sum(axes, keepdims)`, `mean`, `sc.diag`; `A @ ones` folds to a reduction (C-10's residue).
      After C-98 and C-109. [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-111. Batched matmul** with NumPy semantics, its AD and sparsity rules. After C-110.
- [ ] **C-112. `einsum`** as a builder: pairwise contractions through batched matmul, ellipses,
      diagonals for repeated labels. After C-111.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-150. `VMAP` folds into `LOOP`** as a loop without carries: one predicate replaces the
      `ExprOp.VMAP` tests in 14 files, one invocation model, one adjoint route. Snapshots
      byte-identical, `vmap.c` included. After C-121 and C-119.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-8. Fusion decided at lowering time**, in the shape of tinygrad's rangeify: movement ops
      never materialized unless required, single-consumer elementwise producers inlined, a
      reduce-under-broadcast cap, the consumer-agreement merge, and sparse producers fused only into
      consumers that share their traversal; `fuse_elementwise.py` deleted. Gates: race-car `SZ_W`
      zero at W = 1, chain M=5 workspace under 100k doubles (109,944 today), npmpc within 5%, the
      IPM's KKT assembly in one loop. Regenerates snapshots. After C-109, C-112 and C-130. The
      2026-09-08 study (`notes/perf_2026_09_07/tinygrad_rangeify.md`) is the reference.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-113. Lanes over nests, and block callees inlined into their maps.** Per-lane accumulators
      on any reduction, with a remainder loop on reduce ranges with loaded bounds, lanes over nests
      and the seed axis, float32 lanes, loop order by unit-stride access. Gates: a `vmap` of a
      12x512 matvec within 2x of `X @ W.T`, and the regressions C-79 left open (unbumpercars C=2,
      C=16, C=32, closed-loop function evaluation) resolved under the C-79 protocol; its design and
      gates are in [`c77_c79_implementation.md`](notes/c77_c79_implementation.md#the-c-79-todo-entry-at-closure).
      After C-8. [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-151. Predictable lowering of hand-written kernels.** A register-blocked product written
      in Python keeps its structure: blocks as straight-line code over C locals, small loop carries
      in locals, slices read in place, packing hoisted, Function boundaries and lowering hints kept;
      no `restrict`, which devrush measured as neutral. Each property tested on the generated C; a 4×4 product and a
      panel Cholesky measured against BLASFEO. After C-121 and C-8.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-152. Demanded entries.** Which entries of a node its consumers read, so a node is computed
      over its support intersected with them; generalizes C-130's sampled adjoint (devrush: 984
      entries computed for 527 gathered; its CS-12). After C-130 and C-8.
      [Design](notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler).
- [ ] **C-82. A frame budget in `pack_workspace` instead of the per-buffer spill threshold.**
      Fusion and lane staging move memory from full-length intermediates into per-stage locals, and
      inlined callees add their locals to the caller's frame; the hand-written kernel uses no `w[]`
      and about 40 kB of stack. Today a slot spills to `w[]` only when it alone reaches 1024
      doubles, so nothing bounds the frame. Replace it with a per-procedure estimate (local buffers
      plus inlined callees' locals plus a fixed allowance for scalar spills) against a budget,
      default about 64 kB on hosts and a compile option for embedded builds, where zero sends
      everything to a caller-provided `w[]`; spill the largest buffers until the estimate fits;
      report the estimate beside `SZ_W` in the header. Verify the estimate with the compiler: a test
      builds a generated module with `-Wframe-larger-than=<budget> -Werror` on gcc and clang and
      fails if the estimate was optimistic. Any time after C-86: the budget is computed over the
      final program, so it stays valid as schedules change.

### Sparse tensors

- [ ] **C-127. The stored pattern in `TensorType`**: canonical, interned, compared by identity, over
      the last two axes with shared-pattern batch axes; storage shape `(*batch, nnz)`; fixed pattern
      rules shared with C-148; ops without one refuse sparse operands; the storage-size audit. After
      API-1 and C-92. [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-148. Support on every node.** `Expr.support`, lazy, cached, outside the intern key, one
      rule per op with `None` as the sound default; used by simplification, `ad/sparsity.py` (which
      keeps relations over nonzero entries only) and scalar lowering; a randomized soundness
      harness. After C-127. [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-149. No work proportional to a matrix's full size.** Gate: build, analyse, differentiate
      and lower a 10⁵×10⁵ matrix with 10⁶ entries under fixed time and memory limits; fixes
      `zeros_like`, scatter folding, `expr.size` row pointers, `arange(size)` rules, `_matmul_mask`
      and constant hashing. After C-148. [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-128. The sparse boundary.** Sparse leaves with a pattern and a layout (CSC, CSR or a given
      coordinate order), sparse `sc.const` from SciPy or from values and a pattern, `PACK` and
      `VALUES`, `to_dense`, `from_dense` and `sparsify` as builders, storage-aware calls and maps,
      numerical sparse arguments and results. After C-149.
      [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-129. Sparse inputs in the generated C, C++ and CasADi interfaces**, values in each leaf's
      layout, batched leaves publishing their local pattern once, no zero-length arrays for empty
      patterns. After C-128. [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-130. Sparse kernels and the sampled adjoint.** Iteration plans for sparse operands: one
      compact map with build-time union and intersection regions, `MATMUL` with a sparse operand,
      `REDUCE` over stored entries; four kernel forms per region; outputs written in the declared
      layout; derivatives on stored coordinates and the sampled `MATMUL` adjoint. Gate: devrush's
      KKT assembly and products on its Maros-Meszaros subset with no dense `(m, n)` buffer. After
      C-128, C-109 and C-98. Regenerates snapshots. [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-131. Sparse-sparse products**: the Boolean product pattern at build time, a column loop
      over a dense accumulator, bounded table and workspace growth. After C-130.
      [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- [ ] **C-132. Sparse derivative outputs become sparse leaves** with a declared layout and triangle,
      and `Function.output_sparsities` goes away; values switch to the declared layout under the
      versioning policy. After C-129 and C-130. [Design](notes/core_compiler_roadmap.md#sparse-tensors).

### Loops, conditionals and printing

- [ ] **C-124. `sc.print`.** An identity op carrying a format, printing whenever its value is
      computed, a trace-time error for unreachable prints, `SCALY_PRINTF` to compile prints out.
      float64 first; integer and boolean formats with C-104, float32 with C-106. The JIT flushes C
      `stdout` after a call that printed, since a Windows process (R-38) may not share it with
      Python's. Right after the foundations, after C-86 and C-87.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-119. One invocation per callee-op call.** `callees_of` replaces the hard-coded
      `{CALL, VMAP}` sets; `invocation_key` groups sibling outputs and custom-rule residuals; each
      invocation is emitted once with all its used results, so a `VMAP` with several used outputs
      becomes one loop. Regenerates snapshots. After C-87 and C-108.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-120. Carried loops in the program dialect.** A `carried` range attribute, `BREAK_IF`,
      `exit_var`, and every pass guard pinned by a test before the op exists. After C-119.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-121. `LOOP`, with `scan` and `while_loop`.** One callee op with an optional condition, a
      user-facing run-time trip count under a static bound, several carries, an `in_place`
      contract, documented results including an int64 trip count and per-carry trajectories;
      derivatives raise until C-122. Gate: RK4 rollout at N=200 builds in constant C size. After
      C-120 and C-104. [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-122. Forward AD through `LOOP`**, single and multi-seed, taking the primal's steps.
      After C-121 and C-99.
- [ ] **C-123. Reverse AD through `LOOP`.** One backward loop over the stored trajectory, visiting
      exactly the steps taken; differentiable trajectories; `max_trajectory=`. After C-122.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-154. Unrolling static loops by a straight-line budget** that counts every right-hand
      side, a constant in lowering until a measurement asks for a `BuildRecipe` field, honouring
      lowering hints, so small library factorizations become straight-line code without an option; the
      reference schedule's summation order unchanged. After C-121.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-125. `COND`** with branch Functions, a boolean or clamped integer selector, program `IF`,
      and AD over branches; no separate `SWITCH`. After C-121.
      [Design](notes/core_compiler_roadmap.md#loops-conditionals-and-printing).
- [ ] **C-126. Checked builds**: bounds and non-finite checks naming the function and op, behind a
      render option. After C-136 and C-124.

### Linear algebra as library Functions

- [ ] **C-133. Triangular solves, Cholesky, and `solve(assume="pos")`** as library Functions over
      `LOOP` under one solve contract: stated matrix classes, one stored triangle, pivot status (a
      pivot at most epsilon times its diagonal entry fails), an
      implicit derivative reusing the factorization through API-101's residuals, a reference
      schedule. After API-101, C-121, C-138 and C-154.
      [Design](notes/core_compiler_roadmap.md#linear-algebra-as-library-functions).
- [ ] **C-134. Dense LDL^T and LU, and `solve(assume="sym"|"gen")`**, as library Functions. After
      C-133.
- [ ] **C-135. Sparse LDL^T as a library Function** over a digest-named symbolic analysis that also
      replaces `scaly-sqp`'s `_ldl_symbolic`: an outer loop over columns, inner loops with run-time
      trip counts, `in_place` carries, column chunks as a `COND` over widths. Gate, and the test of
      the closed language: devrush's direct kernel's loop structure and timings. After C-130,
      C-134 and C-125. [Design](notes/core_compiler_roadmap.md#linear-algebra-as-library-functions).

### Runtime indexing and external code

- [ ] **C-136. `GATHER` and `SCATTER` with index operands**, absorbing C-137: a constant index is
      the static case, `combine` in `set`, `add`, `max`, `min`, `mode` `fill` or
      `promise_in_bounds`, defined duplicate and out-of-range behaviour, AD and sparsity; segment
      reductions, `take` and `dynamic_slice` as builders. Snapshots byte-identical for constant
      indices. After C-93 and C-104.
      [Design](notes/core_compiler_roadmap.md#runtime-indexing-in-place-updates-and-external-code).
- [ ] **C-138. In-place loop carries** by buffer reuse in lowering: last use, same index, disjoint
      indices; read rules stated per op, never taken from derivative patterns; the `in_place`
      contract raises instead of copying; `inout` procedure parameters.
      After C-136 and C-121.
      [Design](notes/core_compiler_roadmap.md#runtime-indexing-in-place-updates-and-external-code).
- [ ] **C-139. `sc.extern` and `sc.CLibrary`.** A concrete C function from a frozen build
      description, called through `EXTERN` nodes; pure and reentrant; differentiable only through
      `custom_derivative`; the solver plugins move onto it and `SOLVER_CALL` goes away. Flags in
      clang's dialect and libraries named rather than spelled as linker flags, so Windows (R-38)
      adds a platform without changing the API. After C-119, API-1 and R-71; typed pointers after
      C-107.
      [Design](notes/core_compiler_roadmap.md#runtime-indexing-in-place-updates-and-external-code).

### Deferred

- **C-143. One copy of each constant table per generated module.** Devrush emitted a spline table
  once per function that used it (941 kB of C; a 128x128 bicubic went from 34.9 to 5.9 MB once
  deduplicated). Main lowers every `CONST` to a buffer per procedure, so it likely has the same
  duplication; measure before changing.
- **C-144. Share a call's primal with its derivative.** A `CALL` node's value and its derivative each
  run the callee's forward pass; devrush's neural MPC case study hit it, and its DiffMPC episode ran
  in 200 ms against 161 ms without the duplicate. C-96's cache knows which helper is the derivative of which
  callee, and API-101's residuals are the mechanism. After C-102.
- **C-153. Structure inside a sparse pattern.** Detect dense blocks, bands and supernodes inside a
  static pattern and give them dense kernels; also different patterns per batch entry and
  automatic compaction of dense-typed values. After C-131, when a workload asks.
  [Design](notes/core_compiler_roadmap.md#sparse-tensors).
- **C-145. Fold identity products in QP extraction.** Devrush found `_qp_data` in
  `solvers/qp.py` building `I @ X` products and searching full patterns for dense colouring, which
  costs cubic time on dense QPs. Main has the same function; confirm and fold.
- **C-146. Generation and compile time of straight-line code.** Devrush measured about 1.2 ms of
  Python per scalar op of unrolled code, and about 1,000 one-element buffers live until the final
  stores doubling gcc's time. Measure both on main after C-8, since fusion changes them.
- **C-57. The static metadata that still grows with N after C-9.** With every affine index
  table gone, race_cars metadata is 39,578 bytes at N=50 and 420,681 at N=500, so it still
  grows roughly linearly. Three things are left, none of them index arithmetic. The generated
  header's sparsity tables are `O(nnz)` by construction (23,677 bytes at N=50: rows, cols, the
  CSR and CSC pointers and both value permutations) and the question is whether a banded or
  per-stage-block encoding can describe them in closed form for a multistage problem instead of
  listing them. The sparse-assembly gather is a genuine `nnz`-length permutation with no affine
  structure (`k44`, 657 entries at N=50), and would need the assembly itself restructured, not
  its index compressed. And three `double` constant tables that C-45's periodic-tile bake did
  not reach (`k0` 1200, `k22` and `k26` 1224 entries at N=50) grow with N; find out which
  tangent or weight each one is and whether the bake's period test is simply too narrow.
  Deferred 2026-09-09: growing metadata is reported separately from executable code and
  artifacts must stay within the compile cap. Revisit if measured artifact size becomes a
  deployment limit. C-132 removes the value permutations. Re-measure the constant tables, since
  C-78's periodic-tile folding landed after this was written.
- **C-11. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
  42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
  computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
  body, then a scatter. The
  [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#what-is-inherent-to-forward-over-reverse-here)
  finds composition accounts for most excess operations, with symmetry the remaining
  second-order opportunity. Reassess only if a current workload justifies the work. Rediscovered
  empirically on 2026-09-22 while hand-optimizing the race-car Hessian kernel
  ([note](notes/perf_2026_09_22/README.md)): each of the 4 seed directions yields all 4 rows of
  the block, 16 entries where the lower triangle needs 10. There the 6 are dead rows that C-77's
  fusion deletes for free; only where the coloring width itself grows does a second-order
  reverse sweep (edge pushing) that touches each nonzero once pay for its complexity. Devrush's
  AC-OPF case study hit the same wall at scale: 74 colours at case9241 gave 277 MB of C.
- **C-14. Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Scaly already wins.
- **C-58. Inline small pure callees before differentiation.** Revisit a bounded expansion policy
  if measured workloads justify it. Excluded from the C-49 closeout to preserve mapped structure
  without introducing a new expansion policy. Diagnosis: [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#ranked-rule-and-composition-edits-for-an-implementer).
- **C-54. Add memory-aware program common-subexpression elimination and dead-code cleanup.**
  Deferred unless a measured workload requires it; C-8's lowering-time fusion removes most of the
  intermediates it targeted. Build on the shared matcher and arithmetic rules with definition/use
  tracking and conservative read/write/alias handling, including C-138's `inout` parameters; retain
  required calls and output stores, and test repeated loads across writes. Broader loop motion
  follows demonstrated workload need; load-node interning alone is not a current stale-value bug.
  [Design](notes/algebraic_simplification_2026_09_08.md#separate-value-cleanup-from-memory-optimization).

## Solvers

### Now

- [ ] **S-155. Vendor PIQP 0.6.4.** 0.6.2's dual recovery (`KKTSystem::solve`) reads one entry
      past its index of lower-bounded rows, so in about one process in ten `ocp/linear_mpc.ipynb`'s
      condensed QP hits `MAX_ITER` with a NaN solution. Upstream fixed it in 0.6.4 (PIQP issue 42).
      Devrush has the bump, a `plugins/scaly-piqp/hatch_build.py` that rebuilds when a pin differs
      from the versions the notices record, and `test_piqp_dual_recovery.py`, which makes the bad
      read certain with a preloaded `malloc`. A cold rebuild is 5 to 8 minutes.
- [ ] **S-142. Validate PIQP option names before code generation.** Unknown settings reach the
      generated C and fail at compile time with a C error. Before C-139 moves the plugins onto
      externs, so it is not written twice. Rationale:
      [`documentation_api_review.md`](notes/documentation_api_review.md#piqp-option-errors-reach-c-compilation).

### Deferred

- **S-80. Parameter-only oracle prologue.** Every oracle subgraph that depends on `p` alone
  gives the same result at every solver iteration but is recomputed on every call: the
  race-car cost block, the 402 `cos`/`sin` of the reference heading that the cost tangent and
  adjoint callees each recompute per stage, and in the bumper-car safety filter everything
  derived from the cars' current states. Compute it once per solve, as `bounds` already is.
  After `_lowered` in `solvers/nlp.py` has built `base`, `grad`, `jac` and the Hessians, one
  pass over all their outputs marks every node that depends on `x` or the multipliers. The
  parameter-only nodes that a marked node reads become the outputs of one shared `prologue`
  Function, and each oracle takes them as extra parameter inputs in place of the subgraph.
  Doing this after differentiation also catches derivative blocks that are constant in `x`,
  such as the Hessian of a quadratic tracking cost. Inside a `CALL` or `VMAP`, split the
  callee into a mapped parameter-only part and a body that takes its outputs as extra mapped
  inputs. This is the expression-level counterpart of `hoist_invariant`, which hoists across
  loop trips rather than across solver iterations. Leave in place any value cheaper to
  recompute than to load (a view, a reshape, a parameter itself). Both plugins call the
  prologue beside `bounds`, and the descriptor carries it. The marking asks the same structural
  question that `_prove_quadratic` and `_prove_variable_independent_bounds` in `solvers/qp.py`
  answer with `_jac_mask`, so share one implementation. A QP is the limiting case where the
  whole oracle is prologue. Gate: on race cars and bumper cars, the factored oracles match
  the unfactored ones at random `x`, `p` and multipliers, the prologue runs once per solve,
  and closed-loop function evaluation time drops. Function or solver level, not a program
  pass.
- **S-16. Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with
  our hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: [protocol](notes/benchmark_protocol.md).
- **S-17. Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library
  search paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".
- **S-18. CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its
  globalization, regularization and QP path differ from `scaly-sqp`. Add only if review asks for it.

## C API

The generated C, C++ and CasADi-compatible interface. [Design](notes/generated_interface_2026_09_18.md).

### Deferred

- **CAPI-147. Dense matrices in the CasADi layer.** `casadi=True` refuses a dense argument or
  result with both dimensions above one, because CasADi stores column-major and Scaly row-major.
  Transpose at the boundary, or document the refusal next to the option, when a user needs it.
  Row-major against column-major is the dense case of the value layout a sparse leaf declares in
  C-128 and C-129; extend that field to dense leaves rather than adding a CasADi-only transpose.

## Benchmark harness

### Deferred

- **BH-22. Embedded hardware benchmarks** (Raspberry Pi / Jetson).

## Benchmark problems

### Deferred

- **BP-24. Replace the race-car tracking NMPC with the MPFC distillation**, in the same `race_cars`
  package, and **laopt as an external baseline** for it once laopt is published.
- **BP-25. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- **BP-26. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
  authors so the width study becomes a measured closed-loop column instead of an extrapolation.
  Also worth telling them their released episode's reported cost metric cannot be reproduced from
  the trajectory it ships with.
- **BP-27. Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.**

## Licensing

Scaly and the three plugins are BSD-2-Clause. The plugin wheels also ship other people's binaries,
so each wheel carries its dependencies' license texts the way CasADi does
(`casadi/include/licenses/<dep>/LICENSE`), except that CasADi's `mumps-external` and
`metis-external` entries are the COIN-OR wrapper's EPL text rather than the real MUMPS and METIS
licenses, which we do not copy. Surveyed 2026-09-07. What we ship and what it asks of us:

| Package | Component | License | Obligation |
|---|---|---|---|
| scaly, scaly-sqp | our code | BSD-2 | none |
| scaly-piqp | PIQP, BLASFEO | BSD-2 | notice |
| | Eigen | MPL-2.0 | notice. PIQP does not define `EIGEN_MPL2_ONLY` itself, so `hatch_build.py` passes it through `CMAKE_CXX_FLAGS`; PIQP 0.6.2 compiles under it, which proves no LGPL Eigen file reaches the library |
| | LDL inside PIQP (`piqp/sparse/LDL_License.txt`) | LGPL-2.1-or-later | notice plus the LGPL-2.1 text. PIQP's `sparse/ldlt` is a modified LDL, instantiated in `ldlt.cpp` and compiled into `libpiqpc`, so the shared library we ship contains LGPL code. That is allowed: the LGPL text travels with it, the modified source is PIQP's public tag, and `libpiqpc` is a separately loaded shared library the user can replace |
| scaly-ipopt | IPOPT | EPL-2.0 | notice, upstream source of the pinned version; a separate dynamically loaded module, so our BSD-2 is unaffected |
| | MUMPS 5.8.2, via COIN-OR `ThirdParty-Mumps` 3.0.12 | CeCILL-C, EPL-2.0 for the wrapper | notice for each |
| | OpenBLAS (static, Linux) | BSD-3 | notice |
| | libgfortran, libquadmath | GPL-3 + GCC runtime exception | notice; the exception covers this use |
| | METIS 5.2.1 | Apache-2.0 | notice |
| | GKlib | Apache-2.0, plus two glibc-derived headers under LGPL-2.1-or-later and one BSD-3-Clause file, per its `LICENSES.md` | notice for each |

The build hooks generate each wheel's notices from this table's pins, and
`plugins/*/tests/test_*_notices.py` fails when a dependency in `build_config.json` has no license
directory. A new vendored dependency needs a `build_config.json` entry and a license the hook copies.

## Documentation

### Deferred

- **D-36. GPU backend milestone definition**: what a first accelerator target must demonstrate before
  any backend work starts. After the core compiler roadmap; C-91 leaves no placeholders to inherit.

## Release

The four packages were released at `0.1.0a1` on 2026-09-29 through `release.yml`, tagged
`<package>-v0.1.0a1`. GitHub Pages serves the docs of the latest release and Cloudflare Pages every
branch; the [release workflow note](notes/release_workflow_design.md) has the design.

### Now

- [ ] **R-42. Freeze measurements on `0.1.0a1`** and publish a durable archive. Absorbs BH-21: the
      archive holds the release's source, lockfile, inputs, generated code, logs, statistics and
      manifest, with every raw run retained. The status admonition in `docs/index.md` stays until
      0.1.0.
- [ ] **R-71. `zig cc` as the JIT's preferred compiler.** Order: `SCALY_CC`, then
      `python -m ziglang cc` when `ziglang` is importable, then `CC`, then `cc` on `PATH`. It wraps
      clang, so the JIT always speaks one flag dialect and gcc-only behaviour (the sincos merge
      that blocks vectorization, the 40% gap to clang measured in C-81) leaves the JIT path; on
      the x86 reference machine zig's clang 21 was within 10% of clang 20 on the baseline and
      linked `-lmvec` natively (11.6 µs on the 8-lane libmvec variant). The compiler becomes a
      command list rather than a binary path (C-83 prepares this), which `scaly_toolchain` must
      print. Milestone 1, right after C-83 and before C-139 and milestone 3 (roadmap
      [order of work](notes/core_compiler_roadmap.md#order-of-work)). Before flipping the default,
      check that the solver plugins' JIT paths (the `scaly-sqp` wrapper, the PIQP and IPOPT hooks)
      link against the vendored libraries with zig's driver on all three operating systems, and
      measure cold compile latency, since zig builds its own libc on first use. Expose it as the `scaly[toolchain]` extra, required on Windows (R-38).
- [ ] **R-38. Windows support** for 0.1.0: the core JIT, `scaly-sqp` and `scaly-piqp`; `scaly-ipopt`
      later, since MUMPS needs Fortran, which zig lacks. The candidate toolchain is `zig cc`
      throughout, `ziglang` required on Windows through a `sys_platform == 'win32'` marker. The
      investigation so far, including how CasADi does it, was all done from macOS and Linux; the
      first step is to try it on an actual Windows machine: a zig-built DLL loaded by ctypes, the
      cache and DLL search path, a zig build of PIQP and BLASFEO. Runs in parallel with the rest of
      0.1.0. [Investigation](notes/windows_support_2026_10_02.md).
