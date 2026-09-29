> Planned work, maintained like `refactorings.md`. It records the design of the core compiler
> features that come before GPU support and before application libraries (splines, integrators,
> model predictive control). `internal/todo.md` holds one item per pull request and links here for
> the reasoning. Delete a section once all of its items have landed.

# Core compiler roadmap

Drafted 2026-09-29 from a survey of main at `e4f3a64` and of the experimental `origin/devrush`
branch, then revised after two independent reviews. Devrush implemented most of this list in a few
weeks, with many examples and benchmarks, including a TinyMPC solver and a PIQP-style
interior-point method (IPM) generated from Scaly and specialized to one problem structure. Its
reports say both run faster than the libraries they reproduce. That is the evidence that every
feature below is within reach, not a promise about our timings. Devrush's code is not the plan:
its 2026-09-27 review reproduced 13 bugs, found each derivative formula written three times, a
2,300-line `lowering.py` and a Function object that collected attributes after construction. We
reimplement each feature on main one reviewable pull request at a time, and use devrush's tests and
measurements as targets.

Devrush stays a frozen experimental branch. Its code is a source of ideas and test cases; nothing
merges from it. It keeps its own todo list, and its ids from C-82 up name different items than the
same ids on main: devrush's C-86 is its `while_loop`, main's C-86 is the lowering package split
below. Do not confuse the two. This document and `internal/todo.md` cite devrush work by file and
title only. This document supersedes the "Function templates" section that `refactorings.md` held
until now.

Abbreviations: AD is automatic differentiation, ABI the generated C calling convention, JIT the
just-in-time compile path in `codegen/jit.py`, CSC and CSR compressed sparse column and row storage.

## Contents

1. [Decisions](#decisions)
2. [Order of work](#order-of-work)
3. [Rules for every item](#rules-for-every-item)
4. [Foundations](#foundations)
5. [One AD engine](#one-ad-engine)
6. [Dtypes and the scalar vocabulary](#dtypes-and-the-scalar-vocabulary)
7. [Tensor core and the loop compiler](#tensor-core-and-the-loop-compiler)
8. [Signatures and templates](#signatures-and-templates)
9. [Sparse tensors](#sparse-tensors)
10. [Loops, conditionals and printing](#loops-conditionals-and-printing)
11. [Linear solves](#linear-solves)
12. [Runtime indexing, in-place updates and external code](#runtime-indexing-in-place-updates-and-external-code)
13. [Milestone checks](#milestone-checks)
14. [What devrush gives us](#what-devrush-gives-us)
15. [Open questions](#open-questions)
16. [Evidence from devrush](#evidence-from-devrush)
17. [How this roadmap was made](#how-this-roadmap-was-made)

## Decisions

The rest of the roadmap is built from these. Each settles a choice where the design memos or
devrush disagreed. Decisions 4, 6, 8 and 11 were confirmed on 2026-09-29, together with removing
the GPU placeholders (C-91). Decision 12 is open: the template API gets its own design session.

1. **One forward traversal, one table of elementwise partials.** A forward rule is written once,
   with the tangent carrying a leading seed axis, and reverse mode reads the same partials. No new
   differentiable op gets its AD rules before C-98 and C-99 land; otherwise each op is written
   three times, as on devrush. The IR and lowering halves of new ops may land earlier.
2. **N-d lowering and the loop compiler are one project.** The expression dialect is already mostly
   rank-generic; the rank limits live in lowering, which emits one flat loop per op with div/mod
   coordinates. We write a loop-nest emitter and put N-d ops on it, rather than extending the flat
   style and rewriting it for C-8 later.
3. **Keep `MATMUL`, make it batched, make `einsum` frontend sugar.** No `dot_general` op and no
   einsum op in the IR. `einsum` contracts pairwise through batched matmuls, transposes, reshapes,
   reductions and diagonal gathers.
4. **Sparsity is part of `TensorType`.** A tensor type gains an optional canonical,
   interned CSC pattern over its last two axes. Its shape stays mathematical; its storage is
   `(*batch, nnz)`. Devrush kept sparsity in a Python wrapper over a dense values vector, which made
   AD free but left the IR blind to matrix structure, gave the pattern a second home, and made
   template keys and later kernel choices (a BLAS call, a CSR traversal) impossible to express in
   the IR. Ops without a sparse rule refuse sparse operands. Nothing densifies silently.
5. **Sparse support does not wait for the loop compiler.** A CSC product stores at table-indexed
   positions, which is a scatter and outside the loop compiler's output-index model anyway. Sparse
   lowering needs only the loop-nest emitter (C-108). The loop compiler fuses sparse kernels better
   later.
6. **One loop op for scan and while.** `LOOP` has an optional condition, a static
   bound, and documented numbered results: final carries, stacked outputs, the trip count as int64,
   and per-carry trajectories. Devrush had separate `SCAN` and `WHILE` ops, a hidden `output=-1`
   trajectory node and duplicated AD builders.
7. **One node per output, grouped by an invocation key.** `CALL` already emits one node per output.
   Every callee op (`CALL`, `VMAP`, `LOOP`, `COND`, external calls) shares one `invocation_key`, and
   lowering emits each invocation once with all its used results. No tuple type.
8. **No global options.** Devrush's `sc.options` context variable keyed every
   derivative cache on every option and still missed cases. A choice that shapes the graph is an
   explicit builder argument. Nonsmooth derivatives follow one convention: ties split equally, as in
   JAX and PyTorch; `floor` and `ceil` have zero derivative; `abs` has zero slope at zero.
9. **Two indexing ops, after one accumulating scatter.** `SCATTER` accumulates repeated indices now,
   as its docstring already says, instead of adding a separate segment-sum op. Once runtime indexing
   lands, `TAKE` and `PUT(combine=set|add|max|min)` with the index as an operand absorb `GATHER` and
   `SCATTER`; a constant index is the static case.
10. **Custom behaviour at Function boundaries.** `sc.custom_derivative` sets rules when a Function
    is built. An external C function is a graph node whose signature comes from a Function
    declaration. There is no expression-level customization: a small Function gives names, a cache
    boundary and reuse for free.
11. **The entry signature follows the leaf dtypes.** All-float64 functions keep
    today's `const double**` entry and `double* w` byte for byte. All-float32 functions get
    `const float**` and `float* w`. Anything else gets `const void**`, one cast per leaf in the
    prologue, and a byte-addressed workspace. Integers and booleans cross as their own C types,
    never as doubles.
12. **The template API is open.** The draft below follows devrush: one decorator with one slot per
    positional parameter, which would be a scripted break of 432 call sites in 63 Python files and
    55 in the published docs. The maintainer likes one decorator and more default names, but not
    the asymmetry between inputs and outputs or a declaration syntax loose enough to accept almost
    anything with or without `sc.L`, so the surface is redesigned in a separate session. What does
    not depend on the syntax stays: an instance's signature is the tree skeleton plus one
    `TensorType` per leaf, instances are keyed and named from it, and the compiler consumes only
    concrete Functions. API-114, API-3 and API-1 onwards wait for that session; nothing in waves 0
    and 1 outside that lane does. `sc.S` (C-128) is built as a leaf beside `sc.L` in today's form,
    so the redesign applies to it unchanged.

## Order of work

The work runs in lanes. Inside a lane items are sequential; arrows are hard dependencies. Two rules
hold across lanes: a pull request that regenerates C snapshots is never in flight at the same time
as another one (C-98, C-108, C-109, C-8 and C-119 regenerate them), and edits to the shared
vocabulary (`ir/expr.py`, `ir/program.py`, the verifiers) merge one at a time.

```
Wave 0, foundations
  C-86 lowering package, first; then C-87 names and C-93 scatter (both edit lowering)
  independent: C-88 interning, C-89 wrong answers, API-90 immutable Function,
               C-91 GPU placeholders, C-92 verification, C-83 JIT fingerprint
  C-94 JIT index key (after C-88, API-90)

Wave 1
  AD        C-95 harness -> C-96 helpers and names -> C-97 ad/calls.py
            -> C-98 one forward traversal -> C-99 partials and reverse -> C-100 intermediates
            -> API-101 custom_derivative
  lowering  C-108 loop-nest emitter -> C-109 accumulators and contraction schedule
  API       API-114 output= codemod -> API-3 one slot per parameter
  dtypes    C-103 coercion -> C-104 predicates, select, cast: IR and lowering (after C-108)
  sparse    C-127 pattern in TensorType -> C-128 sparse boundary -> C-129 sparse interfaces
            (after C-88, C-92)
  any time  C-82 frame budget (after C-86)

Wave 2
  dtypes    C-104 AD half (after C-99) -> C-105 nonsmooth, max/min (after C-109)
            C-106 float32 lowering -> C-107 typed entry signature
  tensor    C-110 axis reductions -> C-111 batched matmul -> C-112 einsum (after C-98, C-109)
  API       API-1 templates (after API-3, C-87) -> API-115 lifted derivatives
            -> API-116 pattern holes (after C-128)
  sparse    C-130 sparse algebra (after C-98, C-108, C-93) -> C-131 assembly, sparse products
  loops     C-119 invocation model (after C-87, C-108); C-124 sc.print (after C-104)

Wave 3
  loops     C-120 carried loops in the program dialect -> C-121 LOOP -> C-122 forward AD
            -> C-123 reverse AD
  tensor    API-2 typed vmap (after API-1, C-110, C-119)
            C-8 lowering-time fusion (after C-112) -> C-113 lanes and block callees
            -> C-102 reverse keeps the call boundary
  API       API-117 Function-level jvp/vjp (after C-98) -> API-4 npmpc example and guide
  sparse    C-132 sparse derivative outputs as sparse leaves (after C-129, C-130)

Wave 4
  solves    C-133 triangular and Cholesky (after API-101, C-109) -> C-134 LDL and LU
            -> C-135 sparse LDL op pair (after C-130, C-104)
  loops     C-125 COND/SWITCH ; C-126 checked builds (after C-136, C-124)
  indexing  C-136 TAKE/PUT -> C-137 GATHER/SCATTER folded in -> C-138 in-place carries (after C-121)
  external  C-139 EXTERNAL_CALL (after C-107, API-1, C-119)
  checks    API-140 TinyMPC on main ; API-141 generated sparse QP interior-point solver

Then D-36, the GPU milestone definition.
```

Why this order:

- Wave 0 fixes silent wrong answers found on main during the survey; the worst is two callees with
  one name sharing a procedure. Templates multiply generated names, so the name fixes come first.
  The lowering package split goes before every other lowering edit, so nothing rebases across a
  file move.
- The AD engine is the barrier for every track that adds differentiable ops. Its first step is a
  test harness, then helper identity, then a byte-identical extraction of the call rules, and only
  then the rewrite, so each step can be bisected.
- The sparse boundary (pattern type, sparse leaves and constants, headers) depends on neither the AD
  engine nor the emitter, so it starts in wave 1. Wave 1 then ends with sparse matrices passing
  through Function signatures, and templates in wave 2 key on patterns from their first commit.
  Sparse algebra waits for the AD engine and the emitter.
- Typed vmap needs templates (a mapped template is instantiated from its slice shapes) and the
  invocation model (one loop for all used outputs).
- Reverse mode keeping the call boundary (C-102) waits until fusion can inline block callees
  (C-113). Before that, keeping adjoint calls would cost run time that nothing could recover.
- Linear solves wait for `custom_derivative` and for program-level `select`, which the LDL width
  dispatch uses.

## Rules for every item

- The whole suite, `uv run ruff format`, `uv run ruff check`, `uv run ty check`, the import-layer
  and public-name tests. A new module gets an `IMPORT_LAYERS` entry, a one-line docstring and a line
  in the package map of `docs/dev/codebase.md`.
- Before an item changes an IR, AD or codegen path that only a benchmark exercises, copy a small
  reproduction into `tests/` (AGENTS.md). Every new gate is shown to fail by perturbing what it
  checks.
- C snapshots stay byte-identical unless the item says it regenerates them, and a regeneration comes
  with the reviewed diff and measurements of source size, generation time and run time on the
  benchmark problems.
- The pull request that finishes a user-visible feature ships its guide page and API entries.
  Anything unfinished stays out of `docs/`, filtered from the API page, and free of roadmap wording
  in rendered docstrings.

## Foundations

Wave 0. Each item is small and removes a class of bugs the later tracks would multiply.

**C-86. Lowering becomes a package.** Split `passes/lowering.py` into
`passes/lowering/{ctx,elementwise,movement,reduce,contraction,calls,gather}.py`, registration
unchanged, and refuse a second `@lowers` for one op. Delete the unused multi-index `VIEW` door
(`program.py:320`'s `rank` attr, the per-component branch at `program_spec.py:76`, the renderer
error at `c.py:336`): the emitter keeps a view a single index expression. Snapshots byte-identical.

**C-87. Callees keyed by identity, and one name authority.** Verified on main:
`_ensure_callee` stores procedures in `ctx.callees[callee.name]` and `_lower_call` deduplicates
invocations by `(callee.name, arg_names)`. Two different Functions named `f` (or two decorated
lambdas, both `<lambda>`) therefore share one procedure, and the second silently returns the first's
result. Key procedures, invocations, `solver_fns` and solver-oracle references by Function identity,
and map each to its emitted symbol in one place. Replace `utils/names.py::c_ident`'s five reserved
words and `passes/program/_common.py::allocated_name` with one `NameScope`:

- a program scope seeded with C23 and C++ keywords, reserved patterns (leading `_[A-Z]`, any `__`),
  libm and libc names including the `f` variants, the ABI parameter names and generated globals
  (`k{n}` tables, vector typedefs, `SCALY_*` macros);
- the exported entry symbol kept exactly as given, with a clash raising (a Function called `log`
  collides with `<math.h>` today), and deterministic renames for internal `_raw` procedures;
- a child scope per procedure for parameters, buffers, loop variables and temporaries, used by
  lowering's `i_{name}` loop variables (they collide with an input called `i_y`), `hoist_invariant`,
  `widen_ranges`, `hoist_reciprocals` and the renderer's `_h{n}` temporaries, which closes the
  collision guard C-53 left open;
- a header scope seeded with C++ keywords for `abi.buffer_idents` and `cpp._sparse_namespace`.

Tests in `tests/codegen/test_name_clash.py`: two same-named callees called on the same arguments, a
Function named `log`, input `i_y` beside output `y`, inputs named `new` and `default`, a self-named
sparse output in the `.hpp`.

**C-88. Interning sets fields once, floats are keyed by bits, and constants are frozen.** On an
intern hit, `Expr` and `ProgramNode` return the cached node and the dataclass `__init__` then
reassigns every field, so a later construction replaces `value` and `attrs` and resets `_key_cache`.
`Expr.const` keeps the caller's array, so mutating that array changes an interned node.
`ir/program.py::_attrs_key` keys floats by value, so `const_float(0.0) is const_float(-0.0)` and a
constant's sign can flip. Use `@dataclass(init=False)` and set fields only on a miss; encode floats
by their bits in both `_attrs_key` functions (signed zeros stay distinct, NaN interns); copy and
freeze constants, index tables and patterns (`writeable=False`) and freeze `attrs` mappings. A
determinism test forbids module-level counters in `src/`, since generated names must stay
content-derived.

**C-89. Refuse what silently computes wrong results at the boundary.**

- `CompiledFunction` converts every input to float64, and `L.flatten_numerical` casts a float to an
  int without complaint (`tree.py:165`). Until C-107, lowering and the JIT raise for any *input or
  output leaf* that is not float64. Boolean and integer values inside a function stay allowed, so
  C-104 is not blocked.
- `SOLVER_CALL` has a hard-wired zero derivative in `ad/forward.py`, `ad/reverse.py` and
  `ad/sparsity.py`. An active derivative through a solver raises instead. Solver derivatives through
  the implicit function theorem come later with their own design.
- `OP_INFO` folds `MINIMUM`/`MAXIMUM` with `np.minimum`/`np.maximum` while `c.py` renders
  `fmin`/`fmax`, so constant folding and generated code disagree on NaN. Fold with `np.fmin`/`np.fmax`.

**API-90. An immutable Function.** One private constructor with one `_validate`, and a public
`Function.build(name, input_tree, inputs, output_tree, outputs, *, output_sparsities=None,
descriptor=None, role=None, ...)` used by the decorator, `factory`, the AD callees and `solvers/`.
`_replace` is the only copy path: `with_device`, `_with_outputs` and `_with_trees` become one-liners,
and `_with_trees` stops mutating `self`. The solver descriptor becomes a constructor field instead
of an attribute set in `solvers/model.py`. `role` records what built a Function (a forward helper,
an adjoint helper), so later passes can read it instead of guessing from names. The compiled handle
and the digest memo live outside the immutable definition. `_from_exprs` becomes a two-line
`flat_tree` adapter.

**C-91. Remove the GPU placeholders.** The `DeviceSpec` entries for cuda, opencl and metal,
`BACKEND_SUPPORT`, the `KERNEL`/`LAUNCH`/`BARRIER` program ops, and the range kinds `THREAD`,
`LOCAL`, `WARP` and `GROUP_REDUCE` exist only to raise "deferred". `GLOBAL`, `REDUCE`, `VECTOR`,
`SERIAL` (the `range_` default, used by `widen_ranges` for its chunk loop) and `UNROLL` stay. D-36
defines what a GPU target needs, and it should not inherit guesses.

**C-92. Verification, op coverage, and no recursion.** `verify_expr` is never called in the
pipeline, and `SLICE` and `SOLVER_CALL` have no verifier rule. Run it when a Function is built and
before lowering. Add one test that classifies every `ExprOp` in the verifier, forward and reverse AD,
`ad/sparsity.py`, lowering and (after C-94) the graph digest, so a new op cannot be half-wired. Make
`ad/sparsity.py::_depends_on`, `_jac_mask` and `Expr.structural_key` iterative: devrush found 800
unrolled Euler steps overflow the Python stack in walks of this kind.

**C-93. Accumulating scatter.** The `SCATTER` docstring says repeated indices accumulate, but the
builder rejects them (`expr.py:688`) and lowering overwrites. Accept duplicates. Lowering keeps
today's plain store loop when the indices are unique (so results stay bit-identical, including
signed zeros) and emits an accumulate loop, summing in index order, when they repeat.
`segment_sum(values, ids, n)` is a builder over it, not a new op. `ad/reverse.py::_gather_vjp`
becomes one scatter (devrush: 1.3 s to 1.6 ms at n = 2000), `_unbroadcast` stops building
per-element `gather().sum()` stacks, and `combine_scatter_sums` accepts accumulate loops.

**C-83. The host and the compiler in the JIT cache key.** Absorbs C-85. `_compute_cache_key` hashes
the flag string `-march=native`, not the CPU it resolves to, and not the compiler. Hash the full
output of the `cc -march=native -dM -E` probe that `toolchain.native_recipe` already runs (the CPU
features and the compiler's version macros), the compiler's `--version` output and real path, and the
full command and effective flags, so a wrapper or a patched compiler at the same path still misses.
`Compiler` becomes a command tuple so R-71's `zig cc` drops in. Bump the cache version.

**C-94. A JIT index key that skips lowering.** Today every warm process lowers and renders the C
before it can find its library (devrush: 1.4 s of a 1.8 s warm start). Add an index keyed on:

- `graph_digest(fun)` from a new `function/digest.py`, iterative and memoized per immutable
  Function, hashing each node's op, type, name, lowering hint, value bytes and attrs through an
  explicit encoder (floats by bits, callees by digest, custom rules, solver descriptors and external
  code by content) that raises on an unknown attr type;
- the scaly version and a digest of the installed scaly and plugin sources, computed once per
  process, since a codegen change no longer shows up in the key by itself;
- the render options and `BuildRecipe`, C-83's fingerprint, compile and link flags, the
  generation-affecting environment variables, and the libc version when `vector_libm="glibc"`.

The index maps to the rendered-C hash, which still names the artifact directory. Writes are atomic;
a missing artifact behind a valid index entry is rebuilt; `invalidate_cache` uses the index instead
of rendering. Test: a subprocess with a different `PYTHONHASHSEED` loads from the cache and never
calls `lower_function`.

## One AD engine

Wave 1. The maintainer's four-step plan, with the details filled in.

**C-95. The differential harness, before any change.** A parameterized test under `tests/ad/`
comparing, for every `ExprOp`: the current single-seed `jvp` stacked per seed, the current
structural `jvp_many` with the fallback disabled where it is supported, central finite differences
on smooth domains, and the reverse product through `<Jv, l> = <v, J^T l>`. Nonsmooth points are
tested separately against the stated convention, and every unsupported op is listed and its refusal
tested, rather than skipped. Cover scalar operands, unequal broadcast ranks, empty dimensions, zero,
one and several seeds, repeated operands, the four `MATMUL` cases and every structural op. Extend
`test_joint_jvp.py`, `test_vmap.py` and `test_const_seed_bake.py` with nested calls, shared formal
symbols, several active formals, mixed constant and runtime seeds, zero-length maps, overlapping
windows, periodic and non-periodic tiles, local coloring and packed contributions, and compare a
baked constant seed with the same seed passed at run time. Prove the harness fails on a perturbed
partial and on a wrong seed permutation. The unrolled path is the oracle during the migration only.

**C-96. One derivative-helper cache and one naming rule.** A new `ad/helpers.py` owns the key of
every derived callee: callee identity, requested outputs, active inputs, direction, seed count and
layout, constant seed bytes and dtype, custom rules, inherited lowering hint. Names are a readable
stem plus a digest of the key, checked for collisions and reserved through C-87's scope. Helpers
are built with API-90's `role`. Lowering reads the role to mark procedures that must not be
inlined, and `codegen/c.py` prints the mark; this replaces `_force_noinline_raw`'s
`"_fwd" in proc_name`, which also matches a user function called `car_fwd`. Byte-identical where
names do not change.

**C-97. Call rules move to `ad/calls.py`.** `body_tangents`, `body_cotangents`, helper construction,
mapped seed layout, constant-seed baking, `_periodic_seed_tiles`, `_local_seed_colors` and
`_pack_jvp_maps` move out of `forward.py` and `reverse.py` into one module shared by both modes. No
behaviour change; snapshots byte-identical. This makes C-98 a rewrite of rules, not of call
plumbing.

**C-98. One forward traversal with a leading seed axis.** Replace `_jvp` and
`_jvp_many_structural` with one iterative routine taking several outputs and a map from independent
expressions to seed tensors, so several active Function inputs need no joint vector. `jvp` adds a
seed axis of length one and removes it; `jvp_many` checks `(nseed, *wrt.shape)`.

- Structural ops shift their axes by one: `SLICE` prefixes `slice(None)`, `GATHER` offsets each
  seed by `input.size`, `TRANSPOSE` maps `axes` to `(0, *(a + 1))`, `STACK`/`CONCAT` add one to the
  axis, `SUM` reduces primal axes only. Remove the rank-4 cap in `_lower_transpose`, which already
  emits one range per axis, since the seed axis adds a rank.
- `MATMUL` keeps the seed axis as a batch axis. Until C-111 adds batched matmul, fold it into the
  matrix dimension: the left term as `dA.reshape(s*m, k) @ B`, the right term by arranging `dB` as
  `(k, s*n)`, multiplying by `A` and restoring seed-major order. This deletes
  `_jvp_many_matmul_left/right`'s per-seed Python loops.
- Step 2 of the plan, answered from the code: `_seed_axis` stacks `nseed` references to the primal
  and inserts missing broadcast dimensions. Nothing needs the copies for correctness; broadcasting
  already works, and scalars already skip the stack. The copies were free on the scalarized tape
  and cost `nseed` copy loops under block lowering. Replace it with an alignment helper returning
  shape `(1, *missing_ones, *shape)`.
- Callee bodies are differentiated with the same traversal through `ad/calls.py`, instead of the
  per-seed unrolling inside callees today. Constant-seed baking, periodic tiles, local coloring and
  packing stay, as schedules around the one engine.
- Zero tangents are structural: a known zero never builds `0 * x`, which the QP affinity proof would
  read as a dependence. Integer and boolean values have no tangent at all, which is different from a
  zero tangent. Zeros, ones and seeds take the primal's dtype.
- Delete `_JVPManyUnsupported`, `_jvp_many_structural`, `_jvp_many_unrolled` and
  `SCALY_STRICT_JVP_MANY` (step 3). A missing rule raises with the op and the callee named; nothing
  catches it and unrolls.

Regenerates snapshots where seed broadcasting changes the C.

**C-99. A shared table of partials, and reverse-mode cleanup.** A new `ad/rules.py` maps each
elementwise op to a function returning its local partials for the requested operand positions,
built lazily (an inactive exponent must not build `log(x)`). Forward contracts partials with
tangents; reverse multiplies by the cotangent and reduces broadcast axes. An entry may prescribe a
stable contraction form, as division does today (`(dx - y * db) / b`), so the finite-scale cases in
`test_joint_jvp.py` keep passing. Structural ops keep explicit pushforward and transpose rules in
their mode modules. The helper that applies a partial to a cotangent is also where masked cotangents
live once `SELECT` lands (C-104). Remove `vjp_many` from `sc`, `scaly.ad.__all__`, `docs/api/ad.md`
and `docs/dev/codebase.md` (step 4); its only caller is a test.

**C-100. Derivatives with respect to intermediate expressions.** Devrush's `independent(exprs,
wrts)` substitutes stand-in inputs for the chosen intermediates and maps the result back. Without
it, forward mode returns silent zeros for a `wrt` that is a slice of a loop carry, which breaks a
Newton step inside a while body. Apply it in both modes and in sparsity analysis, and test
overlapping selections. `sc.stop_gradient` comes with it.

**API-101. `sc.custom_derivative`.** `sc.custom_derivative(fn, jvp=, vjp=, sparsity=)` returns a new
Function through API-90's constructor; the rules are fields, never set afterwards. A JVP rule takes
primal inputs and tangents; a VJP rule takes primal inputs, primal outputs and output cotangents.
Every derivative of a Function body goes through `body_tangents` and `body_cotangents`, so `CALL`,
`VMAP` and later `LOOP` all honour the rule. Tangents the rule never reads are not built, so an
implicit solve does not differentiate a factorization. Several seeds call a single-seed JVP rule
through one `VMAP` over the seeds. Nested differentiation differentiates the rule graph, and primal
outputs passed to a VJP keep their dependence, since freezing them gives wrong Hessians. Declared
sparsity becomes pattern blocks at construction, which `ad/sparsity.py` reads without importing
Function; without a declaration the pattern is dense. Tests: randomized linearity and duality of the
rules, forward over reverse and forward over forward. Devrush measured an implicit-rule gradient of
a solve 5.6 times faster than differentiating through the steps, and exact.

**C-102. Reverse mode keeps the call boundary.** Wave 3, after C-113. Reverse mode through `CALL`
inlines the callee's adjoint by substitution today, so no call survives. Emit one adjoint helper call
per invocation instead, sharing the cotangents of sibling outputs, and route `VMAP` adjoints through
the accumulating scatter, which also handles stride-0 and overlapping windows. Lowering decides
whether to inline a small helper. Measured on every benchmark problem. A call's value and its
derivative compute the callee's forward pass twice today (devrush's DiffMPC episode ran in 200 ms
against 161 ms without the duplicate); C-144 shares it later.

## Dtypes and the scalar vocabulary

Waves 1 and 2. On main float32 is a type nothing honours: `x_f32 * 2.0` raises, AD builds float64
zeros and seeds, a float32 function reads `float` values through `double*` and calls double `sin`,
and neither dialect has a cast, comparison or select.

**C-103. One coercion rule.** A helper `_coerce(x, like)` in `ir/expr.py`, used by every builder and
operator. Python scalars are weak and take `like`'s dtype: a Python int with a range check, and a
Python float beside an integer expression raises. NumPy scalars and arrays are strong and must match.
Mixed expression dtypes raise and ask for `sc.cast`. `sc.const(ndarray)` keeps the array's dtype,
which changes `sc.const(np.arange(3))`; lists and Python scalars stay float64. Numerical calls accept
`np.can_cast(src, leaf, "same_kind")` and raise otherwise. `Expr.__bool__` raises and points at
`sc.where`: comparisons raise today because `Expr` has no comparison operators, but a bare `if x:`
is silently true. `==` stays identity because interning depends on it, and `sc.equal` compares
values. Symbolic calls check dtypes, not only shapes. Integer indexing accepts NumPy integers
(`x[np.int64(2)]` is refused today).

**C-104. Predicates, select, cast.** Expression and program ops `LT LE EQ NE AND OR NOT ISFINITE
SELECT CAST COPYSIGN`, plus integer `//` and `%` with NumPy floor semantics. Predicates are boolean
and not differentiable; `CAST` keeps its derivative only between float dtypes. The C renders
operators, `?:`, `isfinite` and casts; `isfinite` makes `-ffast-math` unsupported, which the toolchain
asserts. `scalarize` accepts int64 and bool along with float64: its gate exists because narrower
stores round or truncate, which cannot happen for these. The AD half waits for C-99: `SELECT`
forward selects tangents, reverse sends `where(c, t, 0)` and `where(c, 0, t)`, and C-99's
masked-partial helper keeps a NaN partial of the unchosen branch (`sqrt` at 0) out of the result.
Sparsity takes the union of branches, and the QP affinity proof treats a select condition as
nonlinear; devrush's proof accepted `|x|` as quadratic because its pattern ignored predicates.

**C-105. Nonsmooth derivatives and extremum reductions.** Ties split equally, `floor` and `ceil`
have zero derivative, and `abs` has `sign(x)` with `sign(0) = 0` (main's `x/|x|` is NaN at 0).
`reduce_max`, `reduce_min`, `argmax`, `argmin`, `norm_inf` and `norm_1`, full reductions now and over
axes once C-110 lands. `max` and `min` propagate NaN, so a convergence test on a NaN residual fails.
They lower to C-109's accumulator form with a select; lanes come from `widen_ranges` and later
C-113, not from devrush's hand-written four-lane code.

**C-106. float32 through lowering and the program passes.** Fold constants in the node's dtype
(`passes/arith.py`, `fold_arith`) and render float32 literals as `%.9g` with an `f` suffix; never
pass int64 constants through `float(v)` (`lowering.py:492`). libm names by dtype (`sinf`, `fabsf`,
`fminf`) and integer `abs`/`min`/`max`. The scalarize gate admits float32 once folding keeps each
store's rounding. `widen_ranges` handles float32 lanes; `coalesce_stores` generalizes `STORE_PAIR`
or a test pins float32 as untouched; `pack_workspace` spills by C-107's workspace rule.
`ProgramNode` loses its float64 default so every implicit double shows up, and `verify_program`
checks that loads and stores match their buffer's dtype. Integer sums use integer accumulators.
Tests: a float32 twin of every op test against NumPy float32 with ulp tolerances.

**C-107. A typed entry signature and workspace.** `codegen/abi.py::c_api_signature(fun)` replaces
the constant `C_API_SIGNATURE`. All float64 (leaves and every buffer the workspace holds): today's
signature and `double* w`, unchanged. All float32: `const float** arg, float** res, float* w`.
Anything else: `const void** arg, void** res, void* w`, with one cast per leaf in the prologue
(defined, because each pointer came from an object of that type) and a byte-addressed workspace in
which every spilled buffer sits at an offset aligned to its type and `SZ_W` counts bytes. Settle the
C spelling of that workspace in the header (a caller `malloc` or a union) with a test compiled under
`-fstrict-aliasing`. `int32_t`, `int64_t` and `bool` leaves cross as themselves (`<stdint.h>`,
`<stdbool.h>`). Headers and the C++ `Buffer<T, ...>` follow; the JIT passes `c_void_p` pointers and
allocates outputs in the leaf dtype. The CasADi layer and the solver wrappers (`codegen/solver.py`
and each plugin's `render_wrapper`) stay float64 and raise otherwise. `docs/dev/versioning.md`
states the rule as the stable signature. This removes C-89's guard and is the float32 parity gate:
every op test passes in float32, plus one float32 C snapshot.

## Tensor core and the loop compiler

Waves 1 to 3. This track absorbs C-8, closes C-79's follow-up and deletes C-43's layout rules.

What main supports: `TensorType` has no rank limit, broadcasting is N-d, `transpose` takes any axes
at build time and stepped slices lower already. The limits are `matmul` (rank 2), lowering's
`TRANSPOSE` (rank 4), `SUM` (full reduction only), `vmap` (rank-1 outers, flat output) and
`jvp_many`'s rank-3 transpose. Downstream, every op is one flat loop with div/mod coordinates,
reductions accumulate into the output element, and matmul has hand-written layouts.

**C-108. The loop-nest emitter.** `LowerCtx` gains `nest(shape) -> (ranges, coords)`, `reduce(extents)`
with bounds that may be expressions (sparse rows need `rowptr[i]` to `rowptr[i+1]`), and
`read(expr, coords)` returning a scalar program node. A singleton axis gets coordinate zero; a zero
extent suppresses the nest, so an empty tensor emits no store. A view index is the affine sum of
strides times coordinates. A reshape after a transpose can leave an index that is not affine in the
consumer's coordinates; then the emitter uses quotient and remainder on that axis only, or
materializes the transpose, and a test pins which. This is devrush's `delinearize_loops` idea done at
emission, where the structure is known, instead of recovered afterwards. Migrate elementwise ops,
movement ops (`RESHAPE`, `TRANSPOSE`, `SLICE`, `GATHER` through `index_at`), `STACK` and `CONCAT`;
delete `_coord_p`, `_broadcast_index_p` and `_flat_index_p`. Tests: NumPy differential tests per op
and shape class, transpose-reshape-slice chains, empty dimensions. Gates: race-car N=200 within the
C-79 limits, `test_fuse_ranges` and `test_widen_ranges` still asserting the same properties, lowering
time within 10%. Regenerates snapshots.

**C-109. Accumulators and the contraction schedule.** A reduction becomes `acc = identity; for k {
acc = acc op x }; out = acc`: `scalarize` learns `ASSIGN`, and `widen_ranges._ordered_reduction`
matches the new form. Empty reductions produce the identity. `SUM` and `MATMUL` (rank 2 as today)
move to it, and `_lower_matmul`'s layout branches and `_mm_accumulate` are deleted. The two C-43
layouts come back as one scheduling decision in `passes/lowering/schedule.py`: a unit-stride reduce
axis splits the neighbouring output axis into four `UNROLL` lanes with one accumulator each, and a
strided reduce axis goes outermost. Each output keeps its summation order. Gates: npmpc N=12 and
unbumpercars C=8 within 5% of the C-43 kernels. Regenerates snapshots.

**C-110. Axis reductions.** `ExprOp.REDUCE` with `kind` (`add`, `max`, `min`) and `axes` replaces
`SUM`, a mechanical rename across `ad/`, `passes/expr.py` and lowering. `sum(axes, keepdims)`, `mean`
and `sc.diag` are builder sugar. AD: the reverse rule of an axis reduction is a broadcast, which
replaces C-93's interim `_unbroadcast` scatter. `A @ ones` and `ones @ A` fold to reductions, the
part of C-10 that waited for an axis reduction. A product reduction waits for a user: its derivative
at zeros needs the exclusive-product rule.

**C-111. Batched matmul.** `MATMUL` with NumPy semantics: leading axes broadcast, the last two
contract, rank-1 operands promote and drop as today. AD, sparsity (`_matmul_mask`) and
`_matmul_transpose_fold` generalize; C-98's folded seed-axis matmul becomes one batched node.

**C-112. `einsum`.** A new `ir/einsum.py`: parse the spec with ellipses; extract diagonals for labels
repeated inside one operand (a gather, so `ii->i` and trace work); reduce labels that occur in one
operand only; contract operands pairwise, left to right, each pair as a batched `MATMUL` after
permuting to (batch, free, contracted) and reshaping; permute to the output. No contraction-order
search. Until C-8 lands, a transpose is a copy loop, so einsum is correct but copy-heavy; after C-8
the transposes become index remaps.

**API-2. Typed vmap.** The tensor track's leading-axis design merged with devrush's views plan.
`sc.vmap(f, N)` returns a mapped callable whose call builds `VMAP` nodes at the call site (a `CALL`
would hide the map from `fuse_ranges`, `widen_ranges` and structured sparse AD) and returns the
callee's whole output tree, each leaf shaped `(N, *shape)`. Argument resolution, first match wins:
`sc.window(x, start, stride)` for overlapping rank-1 windows (race-car's `(z, 0, 4)`);
`sc.broadcast(p)`; an array whose leading axis is `N`; the API-70 size rules. A view is read in
place by replaying its `SLICE`/`RESHAPE`/`TRANSPOSE` chain on `np.arange` to recover
`(base, start, stride)`, so the node equals the hand-written one and the C is byte-identical; a
non-affine view (`A.T`) is copied, which is semantics, not a fallback. A template callee is
instantiated from the resolved slice shapes. Lowering emits one loop for all used outputs through
C-119. `vmap.c` stays byte-identical.

**C-8. Fusion decided at lowering time.** The three cases of tinygrad's scheduler, applied in
`LowerCtx.read`, where consumer counts are known: movement ops are never materialized unless an
output or a non-affine composition requires it; an elementwise node with one consumer is inlined at
the consumer's coordinates; a node with several consumers, a call or map output, a scatter, a
reduction or an output is materialized. A producer read under a range it does not depend on is
materialized unless it is cheap, by the caps `fuse_elementwise` applies today. `fuse_elementwise.py`
and `inline_producer` are deleted. The consumer-agreement merge (share a producer when all consumers
index it the same way) stays out until a workload needs it. Memoize `read` on `(expr, coords)`,
since re-walking deferred subgraphs is what made `fuse_ranges._rewrite` slow before. Gates from the
old C-8: race-car `SZ_W` zero at W = 1, chain M=5 workspace under 100k doubles (109,944 today),
npmpc within 5%, `_assert_fused` still holds. Regenerates snapshots. `fuse_ranges`, `widen_ranges`
and `hoist_invariant` stay: inlining a scalarized callee into its mapped range is not something
lowering-time deferral can do.

**C-113. Lanes over nests and block callees.** The `UNROLL` split for per-lane accumulators on any
reduction; `widen_ranges` over nests and over the seed axis; float32 lanes (after C-106);
`fuse_ranges` inlining non-scalarized callees, with loop order chosen by unit-stride accesses. The
last part makes a `vmap` of a matrix-vector product with one constant matrix run like a matrix
product: the batch axis goes innermost so the lanes run over it and the matrix entry is invariant
(devrush's neural MPC case study ran at 0.05 times PyTorch there). Gates: a `vmap` of a 12x512
matvec within 2x of the explicit `X @ W.T`; the regressions C-79 left open (unbumpercars C=2, C=16,
C=32 and the closed-loop function evaluation), re-measured under the C-79 protocol. C-77 and C-79
close into this track; the C-79 design and gates are in `c77_c79_implementation.md`.

**C-82. Frame budget.** As the todo describes. It can land any time after C-86: a budget computed
over the final program stays valid as the schedules change.

Out of scope: `dot_general`, an einsum op, multi-index views, a product reduction, a device scheduler.

## Signatures and templates

Waves 1 to 3. Devrush's final template design gives the semantics (its plan is
`internal/notes/function_templates_plan_2026_09_27.md` on that branch, tested with 56 mutants and
byte-identical snapshots). Its object model does not: templates forwarded 21 graph attributes
through `__getattr__`, and rules and descriptors were set after construction. The sketch in
`typing_playground/templates.py` (a separate `FunctionTemplate`, one input tree, two traces in bare
mode) is superseded.

**The signature.** An instance's signature is its bound parameter list: the tree skeleton plus one
`TensorType` per leaf, `(shape, dtype, diff, pattern | None)`. Leaf kind is not a separate field:
`sc.G` is structure, and `sc.S` is a leaf whose type has a pattern. Because patterns are interned,
the specialization key `(skeleton, types)` hashes in time proportional to the number of leaves, not
of nonzeros. A symbolic and a numerical call that bind the same way get the same instance object,
since call interning and AD caches key on identity.

**The classes.** `Function` is the template, which may have holes. `ConcreteFunction(Function)` is
what the compiler consumes: lowering, the renderers, the CLI and viz accept only a
`ConcreteFunction` and raise `NotConcrete` otherwise. A fully declared function is a
`ConcreteFunction` from the decorator on, with its bare name, which keeps snapshots byte-identical.

**What a user writes.** Main today declares one input tree and one output tree, positionally:

```python
@sc.function(sc.G(sc.L("x", 3), sc.L("p", ())), sc.L("f", ...))
def cost(inputs):
  x, p = inputs
  return (x * x).sum() * p
```

The playground sketch keeps that two-tree form and adds a separate `@template` decorator in which a
leaf without a shape is a hole:

```python
@template(G(L("x"), L("p")), L("f"), name="cost")
def cost(inputs):
  x, p = inputs
  return (x * x).sum() * p
```

Devrush, and this roadmap, use one decorator with one slot per positional parameter and the output
as a keyword. A slot is a shape, a `TensorType`, a name, or an `sc.L`/`sc.G`/`sc.S` tree; leaf names
default to the parameter names. A slot without a shape is a hole, and no slots at all is bare mode:

```python
@sc.function(3, (), output="f")        # fully declared: traced at the decorator, symbol cost
def cost(x, p):
  return (x * x).sum() * p

@sc.function(sc.L(), output="y")       # a hole: one instance per argument shape
def double(x):
  return 2.0 * x

double(np.ones(3))                     # traces double_3
double(np.ones((2, 2)))                # traces double_2x2

@sc.function                           # bare: structure, shapes and outputs from the first call
def square(x):
  return x * x
```

The break is the second positional argument. `@sc.function(inputs, outputs)` becomes
`@sc.function(inputs, output=outputs)`, which API-114 does by script at every call site; the body
keeps its single `inputs` parameter until someone chooses to spread it into several.

**API-114. `output=` codemod.** `sc.function(*slots, output=None, name=None)`. The only forced edit at
the call sites in `src/`, `tests/`, `benchmarks/`, `plugins/`, `examples/`, `docs/` and the notebooks
is moving the output declaration, which a script does. A single `sc.G(...)` slot with `def f(inputs)`
stays valid, so bodies do not change. The decorator checks its arity against the body's positional
parameters and suggests `output=` when they disagree. Snapshots byte-identical.

**API-3. One slot per parameter, and zero-input Functions.** `Function[**PS, **PN, SO, NO]` with a
decorator overload ladder of 0 to 8 slots and `TypeVar` defaults; variadic calls; leaf names default
to parameter names. Calls dispatch on argument kind as today; a call with no arguments is numerical,
and `f.symbolic_call()` is the symbolic spelling. That settles API-3: parameterless solver oracles
stay legal and `ad/forward.py` drops `_flat_symbolic_call`. The seeded wrappers and the solver call
convention change in this PR (about 130 call sites), separately from the codemod. Inlining constant
zero-input derivative callees is an AD change for after C-98. Tests: runtime structure tests, and
static arity 0, 1, 2 and 8 in `tests/typing/test_arity.py` under `ty check --error-on-warning`;
the ladder's check time is measured and kept.

**API-1. Templates.** Holes `Hole(dims, dtype, diff)` with partial dims and open dtypes, on inputs
only; output shapes come from the trace. `instantiate(*specs, name=)` for ahead-of-time export. A
dtype hole binds from an expression's dtype or from a floating ndarray's dtype, which is how
templates reach float32. Instance names are `{template}_{tokens}`: `3x4`, `s` for a scalar, only the
open dims of a partial shape, `f32` or `i64` for a bound dtype, `p` plus 8 hex digits of a pattern
digest, `t` plus 6 of a nesting. No `__`, which C++ reserves. A collision raises, and the mangling
joins the exported-symbol rules in `docs/dev/versioning.md`. Bare mode (`@sc.function` with no slots)
infers the output tree in the same single trace. A fast argument cache keyed on the tree skeleton,
shapes, dtypes and pattern identities skips binding. Tests: one trace per instance (counted), the
same object for symbolic and numerical binding, two instances give two procedures, a cross-process
JIT hit, the CLI refusing a holed template, typing assertions.

**API-115. Lifted derivatives.** The nine derivative wrappers go through one `_lift`: a derived
template whose instance is the transform of the source instance, cached per source instance, with
seed and multiplier slots typed from the source instance. `of` and `wrt` default when unique.

**API-116. Pattern holes.** `sc.S(name)` without a pattern binds from a sparse expression or a SciPy
argument; bare mode binds a sparse argument's own pattern. The key and the `p` token already cover
it. Numerical sparse arguments pay one canonicalization and an O(nnz) comparison per call; the fast
path is a concrete `S` leaf fed its values vector.

**API-117. Function-level `sc.jvp` and `sc.vjp`.** Arity ladders, built through `factory` and
`Function.build`, so AD-built callees keep their trees and stop being `Any`-typed at run time. After
C-98. Closes API-2's second half, as recorded in `refactorings.md`.

**API-4. npmpc template example and the functions guide.** As the todo says, plus the template page
of `docs/guide/functions.md` that D-32 left out. Retire the playground's template sketch.

Deferred: static Python-valued arguments (API-118). Closures and factory functions cover them today,
and refusing body defaults keeps the syntax free.

## Sparse tensors

Waves 1 to 3. The phases follow the independent sparse design memo, checked against devrush's
reports (KKT assembly of 42k nonzeros in 37 µs, 16 times faster than SciPy's `bmat`; a generated
sparse LDL^T within 1.04 to 1.5 times QDLDL-class C with loops, 4 to 5 times faster unrolled on small
systems).

**C-127. The pattern in `TensorType`.** `TensorType(shape, dtype, diff, sparsity=P)`. `P` is a
canonical CSC pattern over the last two axes (sorted, unique rows per column), immutable, interned by
content with a cached hash. It lists the stored coordinates, explicit zeros included; everything else
is identically zero. Leading axes are batch axes sharing the pattern, which is how seed axes and
mapped outputs carry sparse values: `shape = (*batch, m, n)`, `storage_shape = (*batch, nnz)`.
Different patterns per batch entry stay out. `shape` and `size` keep their mathematical meaning. The
existing `SparsityPattern` keeps its arbitrary order for derivative outputs and gains a
canonicalization that returns the CSC pattern and the value permutation. `structural_key`,
`substitute` and the verifier include the pattern. Builders refuse a sparse operand to any op without
a sparse rule, so `exp` of a sparse matrix raises at construction, and the verifier (running since
C-92) enforces the same; this bounds the audit of the 89 places that assume `size` equals buffer
length. Main removed an inert `TensorType.sparsity` as C-84; this one has a meaning from its first
commit.

Storage sparsity is not derivative sparsity: a diagonal input can have a dense solve derivative, and
a dense output can have a sparse Jacobian.

**C-128. The sparse boundary.** `sc.S(name, P)` declares a sparse leaf, a `Tree[Expr, csc_array]`, and
`sc.S(name, ...)` infers an output pattern. `sc.const` accepts SciPy sparse arrays (copied, duplicates
summed, indices sorted, explicit zeros kept) and `sc.const(values, pattern=P)` in the pattern's own
order. New ops: `SPARSE_PACK` and `SPARSE_VALUES` between a pattern and its values vector, `TO_DENSE`,
and `FROM_DENSE`, a projection that drops entries outside the pattern. Lowering, calls and `VMAP`
allocate and offset by storage size. A symbolic call requires the exact pattern; a numerical call
canonicalizes a copy and checks it; a concrete `S` leaf also takes a 1-D values array in canonical
order. Outputs come back as `csc_array`s sharing the pattern's read-only index arrays.

**C-129. Sparse inputs in the generated interfaces.** The ABI passes only values. The C and C++
headers publish dimensions, `nnz`, CSC tables and, on request, CSR permutations for inputs as well as
outputs; `casadi.py` stops forcing input patterns to `None`. An empty pattern emits no zero-length C
array.

**C-130. Sparse algebra with full AD.** Transpose; add and subtract (pattern union); elementwise
multiply (intersection); scaling by scalars, rows and columns; full sum; sparse times a dense vector
or matrix; dense times sparse. Absent entries are structural zeros, so a sparse product skips them
even where `0 * inf` would be NaN; documented and tested. Zero-preserving unary ops keep the pattern;
other unary ops need an explicit `to_dense()`. AD works in stored coordinates: tangents and cotangents
carry the primal pattern, several seeds store `(nseed, nnz)`, and a Jacobian is output storage size
by input storage size. `ad/sparsity.py` propagates dependencies over stored coordinates, and its
matmul rule walks the structural product instead of the dense loop. Lowering uses the emitter: CSR
traversal is a row range with a reduce range whose bounds are loaded, CSC products are accumulating
scatters. Three tests pin the contract with the loop compiler: an accumulating scatter loop is never
widened, a carried range is never split, a reduce range with loaded bounds is never fused. This is
the first usable sparse milestone.

**C-131. Assembly and sparse products.** `blockdiag`, `concat`, `kron`, repatterning, `diagonal` and
axis sums, all with static coordinate maps. Sparse times sparse with the Boolean product pattern
computed at build time, lowered as a column loop over a dense accumulator column, instead of
devrush's enumeration of every scalar product pair. Tests pin bounded table and workspace growth.

**C-132. Sparse derivative outputs become sparse leaves.** `sparse_jacobian` and `sparse_hessian`
return `S`-typed outputs, and `Function.output_sparsities` goes away, so a sparse output has one
spelling. The header tables, the CasADi gather and `casadi_output_sparsities` read the type. Values
switch from arbitrary order to CSC order, which removes the value-permutation tables: an interface
change under `docs/dev/versioning.md`.

Deferred, beside this track: different patterns per batch entry, block and banded execution layouts,
dedicated BLAS or sparse-BLAS lowerings. C-57 (header tables that grow with N), C-11 and C-14
(coloring width, row coloring) stay deferred.

## Loops, conditionals and printing

Waves 2 to 4. Devrush's reports give the stakes: an RK4 rollout plus its gradient at N=200 built in
80 ms instead of 22 s unrolled, with 126 lines of C at any N. TinyMPC, the generated IPM, adaptive
integrators and Newton solvers all need scan and while.

**C-119. The invocation model.** `callees_of(node)` in `ir/expr.py` replaces the hard-coded
`{CALL, VMAP}` sets in `solvers/graph.py`, `solvers/qp.py`, `codegen/aot.py`, `codegen/solver.py`,
`ir/text.py` and `function/model.py`. `invocation_key(node)` is the op, callees by identity, argument
ids and attrs except `output`. `LowerCtx` computes the used results of each invocation once from the
topological order it already walks, and the first sibling emits the invocation with all of them. This
replaces the name-keyed `call_invocations` and fixes `VMAP` lowering one full loop per selected
output, which is why AD packs derivative outputs into one today. Regenerates snapshots for the merged
loops. Derived callees come from C-96's cache so that siblings stay grouped.

**C-120. Carried loops in the program dialect.** A `carried` attribute on a range marks a loop whose
trips depend on earlier trips. It is not `SERIAL`, which `widen_ranges` already emits for its chunk
loops. `BREAK_IF` is allowed only inside a carried range with `exit_var`, and `FOR(exit_var=)`
declares the loop variable before the loop so it holds the trip count. Pass guards, each pinned by a
test before `LOOP` exists: `scalarize.work` refuses carried ranges (it would unroll any constant-bound
loop, bounded under `auto` and unbounded under the scalar hint); `unroll_unit_loops` keeps exit
loops; `widen_ranges` and `fuse_ranges` refuse carried ranges; `hoist_invariant` may hoist a pure
prologue; `pack_workspace` and C-82 count trajectories.

**C-121. `LOOP`.** A callee op with `args = (*inits, *outers[, trip])` and attrs `callee`, `cond`
(a Function with the body's inputs that returns one bool, or `None`), `n_carry`, `length`, `starts`,
`strides` (non-negative; stride 0 is a loop-invariant parameter), `index`, `reverse` and `output`. The
body is `(*carries, [k], *xs) -> (*carries', *ys)`; carries may have mixed dtypes, and each carry's
shape, dtype and pattern are the same at entry and exit. Semantics: `T = min(trip, length)`; the
condition is evaluated before each step on the entering carries, and the loop stops at the first
false; `k` counts `0..T-1`, or `T-1..0` with `reverse`. Results: the final carries, the stacked ys,
the int64 trip count, and one trajectory per carry, where `trajectory[k]` is the carry entering step
`k`. Stacked ys and trajectory slots at or past the trip count are zero-filled, which costs a
conditional loop an O(length) fill and a plain scan nothing. The condition sees only carries and
parameters, so any residual it needs must be a carry. Builders `scan(body, init, xs, *, length,
index=False, reverse=False)` and `while_loop(cond, body, init, *, max_iter, params=(), index=False)`
share `vmap`'s slice resolver. The `trip` operand is never user-facing; only the reverse rule of a
conditional loop creates it.

Lowering: one carried range, `[call cond; BREAK_IF !flag;] call body`, with carries in two
alternating slots, in `length + 1` slots when a trajectory is used, or in one slot when proven in
place (C-138). Sparsity is a boolean fixed point over the carries plus the dependencies of stacked
outputs. Any derivative raises until C-122. Tests against NumPy loops and Python-unrolled graphs:
lengths 0 and 1, exit at step 0, `max_iter` reached, index, reverse, parameters, mixed-dtype carries,
nested loops, a loop inside a `VMAP` body.

A scan is a `LOOP` with no condition and no `trip` operand, so its trip count is the static
`length` and every pass can read that from the attributes. That is all a dedicated scan op would
give: the carried range has constant bounds, stacked outputs and trajectories have exactly `length`
slots and need no zero-fill, no `BREAK_IF` is emitted, the backward loop of C-123 has the same
static bound, and slice offsets are constants. Unrolling a short static scan is possible for the
same reason, and stays out until a workload shows it pays, since loops stay loops. What a
condition costs (a flag per step, a runtime trip count, the zero-fill) is paid only by loops that
have one.

**C-122. Forward AD through `LOOP`.** The tangent loop is the same loop with tangent carries added,
seed axis leading. The condition reads primal carries only, so the tangent loop takes the primal's
steps. Integer and boolean carries get no tangent. Devrush's per-step Jacobian compression (when the
seeded local dimension is smaller than the seed count, form the body's Jacobian once per step and
multiply) is a later optimization with measured thresholds.

**C-123. Reverse AD through `LOOP`.** The rule requests the trajectory result of the same
invocation, so the primal loop stores it once. One backward loop (`reverse=True`, same starts and
strides) carries the carry cotangent plus one accumulator per differentiated parameter, and emits the
cotangents of the sliced inputs, scattered into their outers. A conditional loop passes its primal
trip count as the backward loop's `trip`, so the backward pass visits exactly the steps taken, in
reverse (devrush masked all `max_iter` steps, and value plus gradient cost 5.8 times the value).
Trajectory results are differentiable: a cotangent on `trajectory[k]` adds to the carry cotangent at
step `k`, which is what reverse over reverse needs. `max_trajectory=` is a per-loop argument counting
stored elements; nested loops each count their own, and the product is not bounded. Exceeding it
raises and names `sc.custom_derivative`. Tests: finite differences, duality, zero-trip and early-exit
loops, reverse scans, a trajectory output, and a Hessian of a while loop by forward over reverse.

**C-124. `sc.print`.** `y = sc.print("x={} k={}", x, k)` returns `x`. `PRINT` is an identity op that
carries the format, so prints travel with data through substitution, inlining, `VMAP`, `LOOP` and AD.
The rule, documented with its limits: a print runs whenever generated code computes its value. Once
per step in a loop and once per trip in a map; identical prints intern to one; independent prints
have no guaranteed order; a callee whose outputs nothing uses is never called, so its prints do not
run. A derivative Function prints when it computes the primal value, which forward and reverse rules
usually do, since partials read the primal; the AD rule of `PRINT` keeps the primal node rather than
bypassing it. The tracer records the prints it creates and raises if one is unreachable from the
outputs. Lowering emits a `PRINT` statement; `hoist_invariant`, `scalarize` and `widen_ranges` leave
procedures containing one alone, and `prune_procedures` keeps them. Formatting uses `{}`
placeholders: `%.17g` for float64, `%.9g` for float32, `%lld` with a cast to `long long` for int64,
`%d` for int32 and bool, tensors flat as `[a, b, ...]`. The source defines `SCALY_PRINTF` as `printf`
only when it contains a print, so `-DSCALY_PRINTF(...)=` compiles prints out and an embedded target
routes them to a UART. JIT prints go to the process's C `stdout`, which some notebook front ends do
not capture. JAX's effect tokens were rejected: they need an effect system threaded through every
transform.

**C-125. `COND` and `SWITCH`.** Once masked cotangents exist, `where` is correct in both AD modes, so
`COND` is about cost and about effects inside branches. `COND(selector, *operands)` with a tuple of
branch Functions; a boolean selector for two branches, an integer clamped like `lax.switch` for more.
Lowered to a program `IF` chain calling each branch procedure. Forward AD is a `COND` over the branch
tangent Functions; reverse is a `COND` over branch adjoints that recompute their primal. Sparsity is
the union of branches. `widen_ranges` refuses a map body containing one.

**C-126. Checked builds.** A render option `checks=True` emits `CHECK(cond, message)` statements:
bounds on `TAKE`/`PUT` with `in_range=True`, non-finite values in loop carries and procedure outputs,
each message naming the function and the op as `format_expr` numbers it. `SCALY_CHECK_FAILED`
defaults to `fprintf(stderr, ...); abort()` and can be overridden.

## Linear solves

Wave 4. Devrush's generated PIQP rests on these, and its review asked for one implementation per
factorization.

**Contract for every solve.** The matrix classes are stated per solve. A solve reads one stored
triangle of a symmetric matrix (the lower), and an off-diagonal stored entry stands for both
mirrored entries, so its cotangent is the sum of both mirrored contributions. A factorization that
meets an unusable pivot reports it through a status output, never through a NaN alone. `solve(K, b)`
is a semantic boundary built with `custom_derivative`: the primal and all derivative solves share
one factorization, forward is `dx = K^-1 (db - dK x)`, reverse is `bbar = K^-T xbar` with
`Kbar = -bbar x^T` sampled on the read triangle. Second-order tests have both the matrix and the
right-hand side active. The summation order of a factorization is fixed by its definition, not by
its schedule, so unrolling or chunking never changes its bits.

**C-133. Triangular solves and Cholesky.** `TRISOLVE` (lower or upper, transposed, unit diagonal,
one or many right-hand sides) and `CHOLESKY` as structural ops with closed-form tangent rules, and
`solve(A, b, assume="pos")`. Unrolling is decided in lowering by size, not stored as a build-time
attribute that duplicates an option. Lowering in `passes/lowering/linalg.py`. Re-measure plain loops
under the emitter and lanes before porting devrush's 4x4 register-tiled Cholesky, and never hard-code
a partial-sum count that a tile constant claims to control (devrush's `CHOLESKY_TILE` dropped terms
for any value but 4).

**C-134. Dense LDL^T and LU.** `LDL` without pivoting for symmetric quasi-definite matrices, `LU`
with partial pivoting for general ones, and `solve(assume="sym"|"gen")`. The factor outputs refuse
differentiation; the solves are differentiable through the contract above.

**C-135. Sparse LDL^T as one op pair.** A symbolic analysis module below `Function` (natural, RCM and
minimum-degree orderings, the elimination tree, the pattern of L, column chunks), immutable and named
by a digest of its tables; it replaces `scaly-sqp`'s private `_ldl_symbolic`. `SPARSE_LDL` and
`SPARSE_LDL_SOLVE` ops with typed table attributes instead of devrush's string-keyed dicts, and one
implementation in which unrolling is a lowering decision. Devrush had three schedules that rounded
differently, so the factor's bits changed when the work crossed a threshold. The factor lowers to one
left-looking loop nest with chunks of up to eight columns summed in registers (devrush: 1.5 to 2.2
times faster than its scan schedule), for quasi-definite matrices, with the pivot status of the
contract. `inertia()` and `health()` come with it; iterative refinement comes once `LOOP` exists.

## Runtime indexing, in-place updates and external code

Wave 4. Needed by interpolation search, the IPM's loops and in-place loop carries. The sparse LDL op
pair above does not need it.

**C-136. `TAKE` and `PUT` with runtime indices.** `TAKE(x, idx; fill, in_range)` reads through a
clamped address and returns `fill` out of range; an empty source returns `fill` without reading.
`PUT(base, idx, values; combine, in_range)` with `combine` in `set`, `add`, `max`, `min` writes
out-of-range lanes to a scratch slot, so writes stay unconditional. Duplicates resolve in index
order: the last write wins for `set`, `add` sums in order, `max` and `min` include the base value and
propagate NaN like the reductions. `in_range=True` is a promise; breaking it is undefined, and
checked builds catch it. `idx` is an int64 operand, rank-1 or the last axis first, an `axis`
attribute after C-110. AD: `take` and `put(add)` are adjoint to each other, `put(set)` masks the
cotangent to the winning lanes, `max` and `min` split ties. Sparsity is exact for a constant index and
a conservative row otherwise. `scalarize` reports a data-dependent address as not eligible instead of
failing in `_Frame.pointer`.

**C-137. `GATHER` and `SCATTER` become constant-index `TAKE` and `PUT`.** `LowerCtx.index_at`'s affine
recovery (C-9) keeps working on constant indices; `ad/sparse.py`, the identity-gather rule,
`fuse_ranges._index_values` and the constant-seed bake read the index operand. Snapshots
byte-identical. Devrush's `INDEX_ADD` and `INDEX_SET` are never introduced.

**C-138. In-place loop carries.** One prover in `passes/lowering/in_place.py` replacing devrush's
two: it finds the next carry as a chain of `PUT`s rooted at the carry input, evaluates every index at
every step with one batched NumPy interpreter over `OP_INFO`, and checks that the entries read are
disjoint from the entries later links write. The in-place procedure takes one `inout` buffer
parameter instead of two aliased pointers; `scalarize` refuses it, `hoist_invariant` never treats it
as invariant, and `pack_workspace` counts it as read and written. Devrush measured an update of four
entries in a 100k carry 354 times faster than with two alternating slots.

**C-139. `EXTERNAL_CALL`.** A frozen descriptor below the frontend (import layer 1 or 2, since it must
not import `Function`) holding a signature taken from a Function declaration, C source text or a
linked symbol, declarations and link dependencies, and a workspace size. The first contract allows
only deterministic, reentrant calls: inputs read-only, outputs fully written, no retained pointers,
no hidden state, no aliasing, no failure status. Typed pointers follow C-107; sibling outputs share
one invocation (C-119). AD only through `custom_derivative`, otherwise an active request raises;
sparsity is declared or dense. Source content and library identity enter C-94's digest.
`ExternalOracle` can later adapt to this descriptor for solver slots. `Expr.opaque()` stays a
lowering hint.

## Milestone checks

Each wave ends with something a user can rely on. The last row ports two devrush examples onto main
as tests of the core, not as libraries.

| After | A user can | Check |
|---|---|---|
| Wave 0 | trust that generated code matches the graph; start a warm process without lowering | the name-clash tests; the cross-process cache test |
| Wave 1 | differentiate every op main has today through one engine; write an implicit derivative; pass sparse matrices through a Function and export its header | the AD harness with the fallback deleted; a sparse identity function round-trips through C, C++ and CasADi |
| Wave 2 | write float32, N-d and einsum code; declare templates; add and multiply sparse matrices with derivatives | float32 op twins; einsum against NumPy; sparse algebra against SciPy with finite differences |
| Wave 3 | write loops with prints, map templates, get fused N-d kernels | RK4 rollout gradient at N=200: build time and C size constant in N; the C-8 workspace gates |
| Wave 4 | factor and solve, index at run time, call external C | API-140 TinyMPC on main within 10% of devrush's reported timings; API-141 a generated sparse QP interior-point solver passing devrush's Maros-Meszaros subset |

## What devrush gives us

| Devrush work | Take | Leave |
|---|---|---|
| Predicates, select, cast | op set and semantics, masked cotangents | the separate masked traversal, two coercion helpers |
| `sc.options` | nothing | the context variable and cache over-keying |
| Accumulating scatter, segment ops | all of it | nothing |
| `scan`, `while_loop` | callee design, alternating slots, trajectory as storage, one backward loop | two ops, the hidden `output=-1` node, the sibling scan over the whole graph, float64 counters |
| Multi-seed through loops | `[c, Dc]` carries, the per-step Jacobian idea | global counters, exceptions as probes, id-keyed memos |
| `custom_derivative` | the `body_tangents`/`body_cotangents` funnel | attributes set after construction |
| Runtime indexing, in-place | `take`/`put` semantics, scratch-slot writes | five ops, two provers, `_inplace` names outside the name authority |
| `delinearize_loops` | the idea: no division in nests | the pass (recursive, exponential on shared graphs) |
| `SparseMatrix`, `sc.S` | build-time patterns, exact-pattern checks, values-only ABI | the wrapper type, the pattern outside the IR, sorting caller arrays in place |
| Sparse LDL^T | analysis, the one-loop-nest factor op, the implicit solve rule | three schedules, string-keyed tables, process-global names |
| Dense linalg | ops and closed-form rules | the hard-coded partial sums behind `CHOLESKY_TILE` |
| Templates | semantics and naming tokens | `__` in names, attribute forwarding |
| Integrators, interpolation, MPC, IPM | test problems and timings for the milestone checks | the libraries themselves, for now |

## Open questions

Choices this roadmap leaves open on purpose. Each names the item or session that settles it.

- **The template API** (decision 12). One decorator and more default names are wanted; the
  asymmetry between input slots and `output=`, and a declaration syntax loose enough to accept
  almost anything with or without `sc.L`, are not. A separate design session settles the surface
  before API-114, API-3 and API-1 start. Inputs for it: the "Signatures and templates" section above,
  `typing_playground/`, `refactorings.md` "Open problems", and devrush's
  `internal/notes/function_templates_plan_2026_09_27.md` and `vmap_views_plan_2026_09_28.md`.
- **The C spelling of a mixed-dtype workspace** (C-107): a caller `malloc`, a union, or a byte array
  accessed through `memcpy`, decided under `-fstrict-aliasing` tests.
- **Whether reverse mode should keep call boundaries** (C-102). The design keeps them and lets
  lowering inline; the item lands only if the benchmark problems do not slow down after C-113.
- **Unrolling short static scans** (C-121): possible because the length is an attribute; left out
  until a workload shows it pays.
- **Per-step Jacobian compression in loop derivatives** (after C-122): devrush's idea, with
  thresholds to be measured rather than guessed.
- **The consumer-agreement merge in lowering-time fusion** (C-8): left out until a workload needs a
  producer shared by consumers that index it the same way.
- **A product reduction** (C-110): waits for a user, since its derivative at zeros needs the
  exclusive-product rule.
- **Derivatives through solver calls by the implicit function theorem**: C-89 makes them raise; a
  later design defines residuals, regularity assumptions and solve status.
- **Batched sparse tensors with a different pattern per batch entry, block and banded layouts, and
  BLAS or sparse-BLAS lowerings**: after C-131, when a workload asks.
- **The GPU milestone** (D-36): after this roadmap; C-91 leaves nothing to inherit.

## Evidence from devrush

The measurements the roadmap's gates and priorities lean on, as devrush's reports state them. They
are targets and arguments, not results on main; each was measured on devrush's hardware and
protocol, not the reference machine of `benchmark_protocol.md`.

| Claim | Devrush source |
|---|---|
| RK4 rollout plus gradient at N=200 builds in 80 ms with `scan` against 22 s unrolled, 126 lines of C at any N | `internal/notes/tier1_summary_report.html`, `tests/integration/test_scan.py` |
| Straight-line code costs about 1.2 ms of generation per scalar op, which caps unrolling | `internal/todo.md` "Generation time of straight-line code" |
| A warm process spends 1.4 s of a 1.8 s start rendering C before the cache lookup | `internal/notes/case_study_e8_report.html`, devrush `internal/todo.md` "A JIT cache hit still renders the C" |
| An in-place four-entry update of a 100k carry is 354 times faster than two alternating slots | `internal/notes/tier1_pr6_report.html` |
| An implicit-rule gradient of a solve is 5.6 times faster than differentiating through the steps | `internal/notes/tier1_pr7_report.html` |
| Multi-seed forward through loops: MPC Hessian 109 to 58 µs, RK4 Hessian 1079 to 93 µs at N = 100 | `internal/notes/tier1_implementation_status.md` |
| Linear-time gather adjoint: 1.3 s to 1.6 ms at n = 2000 | devrush `internal/todo.md` "Accumulating `scatter`, segment reductions, linear-time `gather` VJP" |
| Sparse KKT assembly of 42k nonzeros in 37 µs, 16 times SciPy's `bmat` | `internal/notes/tier2_pr4_report.html` |
| Generated sparse LDL^T within 1.04 to 1.5 times QDLDL-class C with loops, 4 to 5 times faster unrolled on small systems | `internal/notes/tier2_implementation_status.md`, `tier2_pr9_report.html` |
| The one-loop-nest factor op is 1.5 to 2.2 times faster than the scan schedule | `internal/notes/ipm_speed_report.html` |
| The generated sparse PIQP runs in 0.78 times PIQP's warmed solve time on the 38 problems above 50 µs; its decision traces match PIQP's on 48 of 48 problems | `internal/notes/ipm_speed_report.html`, `tier3_implementation_status.md` |
| TinyMPC written in Scaly: geometric-mean speed-ups of 1.38 to 2.79 over the library, losing at nu = 16 to 32 where Eigen's SIMD kernels win | `internal/notes/tinympc_benchmark_report.html` |
| A `vmap` of matrix-vector products with one constant matrix runs at 0.05 times PyTorch | `internal/notes/case_study_e2_report.html` |
| 74 colours on AC-OPF case9241 gave 277 MB of C | `internal/notes/case_study_e6_report.html` |
| A call's value and its derivative run the callee's forward pass twice; on DiffMPC's episode that is 200 against 161 ms | `internal/notes/case_study_e2_report.html`, `case_study_e5_report.html` |

## How this roadmap was made

Written 2026-09-29 so a later session can check or extend the reasoning. Five read-only surveys
covered main's IR, AD, lowering and codegen; devrush's control flow, indexing and AD; devrush's
sparse linear algebra, templates and float32 work; devrush's reviews and case studies; and every
item of the old `internal/todo.md`. Four design memos followed, one each for the tensor core and
loop compiler, AD and customization, templates with the float32 ABI and the foundations, and
control flow with printing and runtime indexing, plus one independent memo on the sparse IR. Two
independent reviews of the draft led to the current text; their main corrections were the
same-name invocation key, the float64 guard that would have blocked predicates, batched sparse
storage, the `SERIAL` collision, and serializing snapshot-regenerating work.

The devrush files worth rereading, all on `origin/devrush`:

- reviews and maps: `internal/notes/code_review_2026_09_27.html` (13 reproduced bugs, about 250
  findings), `scaly_capabilities_2026_09_27.html`, `pipeline_analysis_2026_09_22.html`,
  `claude_code_handoff.md` ("traps learned the hard way");
- plans: `function_templates_plan_2026_09_27.md`, `vmap_views_plan_2026_09_28.md`,
  `tier1_primitives_plan_2026_09_25.html`, `piqp_plan_2026_09_26.html`;
- reports: `tier1_*`, `tier2_*`, `tier3_*`, `ipm_speed_report.html`, `tinympc_benchmark_report.html`,
  `case_study_e1_report.html` to `case_study_e8_report.html`;
- code: `src/scaly/ir/expr.py`, `passes/lowering.py`, `ad/forward.py`, `ad/reverse.py`,
  `function/{model,tree,api,sugar}.py`, `linalg/{sparse,sparse_factor,symbolic,dense}.py`,
  `utils/options.py`, and the tests under `tests/integration/` and `tests/linalg/` that the items
  above cite as targets.

On main, the notes the design builds on are `perf_2026_09_07/tinygrad_rangeify.md` (the rangeify
model behind C-108 and C-8), `c77_c79_implementation.md` (range fusion and lanes), `perf_2026_09_22/`
(the race-car kernel study), `refactorings_landed_2026_09.md` (what was read and rejected for the
affine index maps and mapped ranges) and `documentation_api_review.md` (API gaps found while
writing the guide).

