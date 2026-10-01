> Planned work, maintained like `refactorings.md`. It records the design of the core compiler
> features that come before GPU support and before application libraries (splines, integrators,
> model predictive control). `internal/todo.md` holds one item per pull request and links here for
> the reasoning. Delete a section once all of its items have landed.

# Core compiler roadmap

Drafted 2026-09-29 from a survey of main at `e4f3a64` and of the experimental `origin/devrush`
branch, then revised after two independent reviews. Rewritten 2026-10-01 after devrush's
restructure: the core opset is cut to about 50 ops in a closed language, every linear-algebra
algorithm becomes a library `Function`, sparsity is carried as a stored pattern in the type plus a
derived support on every node, sparse kernels are planned in lowering in TACO's manner, and the work
is ordered as three milestones. The reasoning, the alternatives and the investigations behind the
rewrite are in [`small_core_static_sparsity_2026_09_30.md`](small_core_static_sparsity_2026_09_30.md);
this document states the result and does not repeat the argument.

Devrush implemented most of this list in a few weeks, with many examples and benchmarks, including
a TinyMPC solver and a PIQP-style interior-point method (IPM) generated from Scaly and specialized
to one problem structure. Its reports say both run faster than the libraries they reproduce. That
is the evidence that every feature below is within reach, not a promise about our timings.
Devrush's code is not the plan: its 2026-09-27 review reproduced 13 bugs, found each derivative
formula written three times, a 2,300-line `lowering.py` and a Function object that collected
attributes after construction, and its restructure moved the linear algebra out of the core as
eight registered ops rather than removing them. We reimplement each feature on main one reviewable
pull request at a time, and use devrush's tests and measurements as targets.

Devrush stays a frozen experimental branch. Its code is a source of ideas and test cases; nothing
merges from it. It keeps its own todo list, and its ids from C-82 up name different items than the
same ids on main: devrush's C-86 is its `while_loop`, main's C-86 is the lowering package split
below. Do not confuse the two. This document and `internal/todo.md` cite devrush work by file and
title only.

Abbreviations: AD is automatic differentiation, ABI the generated C calling convention, JIT the
just-in-time compile path in `codegen/jit.py`, CSC and CSR compressed sparse column and row storage,
KKT the Karush–Kuhn–Tucker system of an IPM, TACO the Tensor Algebra Compiler.

## Contents

1. [Decisions](#decisions)
2. [The core language](#the-core-language)
3. [Order of work](#order-of-work)
4. [Rules for every item](#rules-for-every-item)
5. [Foundations](#foundations)
6. [One AD engine](#one-ad-engine)
7. [Dtypes and the scalar vocabulary](#dtypes-and-the-scalar-vocabulary)
8. [Tensor core and the loop compiler](#tensor-core-and-the-loop-compiler)
9. [Signatures and templates](#signatures-and-templates)
10. [Sparse tensors](#sparse-tensors)
11. [Loops, conditionals and printing](#loops-conditionals-and-printing)
12. [Linear algebra as library Functions](#linear-algebra-as-library-functions)
13. [Runtime indexing, in-place updates and external code](#runtime-indexing-in-place-updates-and-external-code)
14. [Milestone checks](#milestone-checks)
15. [What devrush gives us](#what-devrush-gives-us)
16. [Open questions](#open-questions)
17. [Evidence from devrush](#evidence-from-devrush)
18. [How this roadmap was made](#how-this-roadmap-was-made)

## Decisions

The rest of the roadmap is built from these. All were confirmed by the maintainer on 2026-09-29 or
2026-10-01.

1. **One forward traversal, one table of elementwise partials.** A forward rule is written once,
   with the tangent carrying a leading seed axis, and reverse mode reads the same partials. No new
   differentiable op gets its AD rules before C-98 and C-99 land; otherwise each op is written
   three times, as on devrush. The IR and lowering halves of new ops may land earlier.
2. **A closed language.** The expression dialect has about 50 ops, most of them rows of one
   elementwise table that no pass names individually (listed in [The core language](#the-core-language)).
   There is no op registry, no traits, no public pass slots. An op belongs in the core only if a
   pass needs a rule for it that composing other core ops cannot supply without losing derivative
   or structural sparsity, a loop staying a loop, work proportional to the real data, or a lowering
   decision that cannot be recovered afterwards. Everything else is a builder (Python producing
   core ops) or a library `Function`.
3. **Keep `MATMUL`, make it batched, make `einsum` a builder.** Every analysis in Scaly walks the
   graph node by node, and a matrix product written as a broadcast multiply and a sum would show
   each of them an `(m, k, n)` intermediate larger than its operands and its result. One node per
   pairwise contraction avoids that. A contraction of several operands is a sequence of pairwise
   ones whose order is a cost decision, made once by the `einsum` builder. No `dot_general` and no
   einsum op; if the transposes around `MATMUL` ever stop being free, `MATMUL` becomes a pairwise
   contraction with labelled axes.
4. **Sparsity is a stored pattern in the type plus a support on every node.** A tensor type gains
   an optional canonical, interned pattern over its last two axes, with leading batch axes sharing
   it; its shape stays mathematical and its storage is `(*batch, nnz)`. The pattern is set by fixed
   rules when the graph is built, like shape inference, so storage is a contract: an op whose rule
   gives no pattern refuses a sparse operand instead of densifying, and generated C does not change
   when an analysis gets sharper. Separately, every node has a lazily computed, cached support, the
   entries that may be nonzero whatever the storage, used by simplification, derivative sparsity
   and lowering. Derivative sparsity stays a third analysis, which consults the support. The order
   of stored values inside the graph belongs to lowering; at the ABI a sparse leaf declares its
   layout (CSC, CSR or a given coordinate order).
5. **Sparse kernels are planned in lowering, in TACO's manner.** Every pattern is known at build
   time, so the index merging and output-pattern assembly TACO does at run time happen in Python.
   Lowering builds a private iteration plan per kernel (domains, access maps, the scalar body,
   reduction order, workspace, writes) and chooses per region among straight-line code, loops over
   constant tables, affine loops and dense blocks. The dense loop nest of C-108 is the all-dense
   case of these plans, so there is one loop compiler. Sparse kernels do not wait for fusion, but
   the plan's interface is designed for both before C-108 fixes it.
6. **One loop op for maps, scans and while loops.** `LOOP` has an optional condition, a static
   bound, a trip count that may be given at run time under that bound, and documented numbered
   results: final carries, stacked outputs, the trip count as int64, and per-carry trajectories. A
   map is a `LOOP` with no carry and no condition; `VMAP` folds into it once `LOOP` exists, as a
   rename that keeps every snapshot byte for byte. Devrush had separate `SCAN` and `WHILE` ops, a
   hidden `output=-1` trajectory node and duplicated AD builders.
7. **One node per output, grouped by an invocation key.** `CALL` already emits one node per output.
   Every callee op (`CALL`, `LOOP`, `COND`, `EXTERN`) shares one `invocation_key`, and lowering emits
   each invocation once with all its used results, including the residuals a custom forward rule
   returns for its backward rule. No tuple type.
8. **No global options.** Devrush's `sc.options` context variable keyed every
   derivative cache on every option and still missed cases. A choice that shapes the graph is an
   explicit builder argument. Nonsmooth derivatives follow one convention: ties split equally, as in
   JAX and PyTorch; `floor` and `ceil` have zero derivative; `abs` has zero slope at zero.
9. **Two indexing ops, `GATHER` and `SCATTER`, with the index as an operand.** A constant index is
   the static case, a computed int64 index the runtime case; `SCATTER` takes a base, a `combine`
   (`set`, `add`, `max`, `min`) and a `mode`. `TAKE`, `PUT`, `PUT_ADD`, `SEGMENT_REDUCE`,
   `INDEX_ADD` and `INDEX_SET` never exist on main; segment reductions are builders.
10. **Algorithms are library Functions; custom behaviour lives at Function boundaries.**
    Factorizations, solves, KKT assembly and interpolation search are `Function`s over core ops.
    `sc.custom_derivative` sets rules when a Function is built, in JAX's `custom_vjp` form: a forward
    rule returns outputs and residuals, a backward rule consumes them, so a solve's derivatives
    reuse its factorization. External C is its own construct, `sc.extern`, always concrete, with the
    build description in `sc.CLibrary`. There is no expression-level customization.
11. **The entry signature follows the leaf dtypes.** All-float64 functions keep
    today's `const double**` entry and `double* w` byte for byte. All-float32 functions get
    `const float**` and `float* w`. Anything else gets `const void**`, one cast per leaf in the
    prologue, and a byte-addressed workspace. Integers and booleans cross as their own C types,
    never as doubles.
12. **The template API is the typing playground's.** `typing_playground/` and its README fix the
    surface and the instance model: one tree per parameter (`arg`, `group`), keyword `outputs=` that
    may be left out, holes wherever a leaf has no shape, a bare mode, `lift` for derived functions,
    and `ConcreteFunction` instances held by the `Function`. The bound key grows from shapes to
    whole leaf types (dtype and pattern). Templates land before the sparse boundary, so sparse
    leaves key on patterns from their first commit.
13. **A reference schedule, not bits fixed by definition.** Each kernel has one reproducible
    reference schedule. Target schedules (lanes, blocking, unrolling) may round differently and are
    checked by numerical and solver-robustness tests against it. This replaces the earlier rule
    that a factorization's summation order is part of its definition.
14. **Predictable lowering.** Users and libraries write performance-critical kernels, such as
    register-blocked dense products with tile sizes chosen in Python, as ordinary expressions.
    Lowering keeps the structure they wrote (Function boundaries, loops, lowering hints), keeps
    small loop carries in C locals, reads affine slices in place and hoists packing out of loops.
    Scaly does not try to rediscover BLASFEO's kernels by itself.

## The core language

| Family | Ops |
|---|---|
| leaves | `INPUT`, `CONST` (with a uniform form, so `zeros_like` of a 10⁵×10⁵ matrix is O(1)) |
| elementwise table | the 17 unary math ops on main; `ADD SUB MUL DIV POW ATAN2 MINIMUM MAXIMUM`; `FLOORDIV MOD`; `LT LE EQ NE AND OR NOT ISFINITE`; `SELECT`; `CAST` |
| shape | `RESHAPE`, `TRANSPOSE`, `SLICE` (with steps), `CONCAT` |
| reduction and contraction | `REDUCE(x; kind=add\|max\|min, axes)`, `MATMUL(a, b)` batched |
| indexing | `GATHER(x, idx; axis, mode)`, `SCATTER(base, idx, values; axis, combine, mode)` |
| sparse boundary | `PACK(values; pattern)`, `VALUES(x)` |
| callees | `CALL`, `LOOP`, `COND` |
| effects and external code | `PRINT`, `EXTERN` |

`stack`, `pad`, `split`, `broadcast_to`, `diag`, `einsum`, `argmax`, `argmin`, the norms,
`segment_sum`, `take`, `dynamic_slice`, `to_dense`, `from_dense`, `copysign` and `where` over
several branches are builders. `COPYSIGN` leaves the vocabulary: `sc.copysign` over `where`, `abs`
and a comparison differs only in the sign of zeros and NaNs. Main has 38 ops today, devrush 65
(57 built in and 8 registered by `scaly.linalg`), and the previous version of this roadmap ended
near 67.

What devrush's registered ops needed from the compiler, and the core capability that replaces each:

| Devrush op (`linalg/ops/`) | What it needed | Core capability, item |
|---|---|---|
| `ragged_add`, `ragged_dot` | inner runs `lo[g] <= p < hi[g]` of run-time length | a run-time trip count on `LOOP` under a static bound, C-121 |
| `ragged_add` on the work column, row swaps in `lu`, the sweeps' updates | updating a large carry without copying it | buffer reuse in lowering with an in-place contract, C-138 |
| `cholesky`, `ldl`, `trisolve` with `Unroll` | straight-line code on small sizes | unrolling static loops by the target's budget, C-154 |
| `trisolve`'s four partial sums, `blocked_sum` | lanes on a reduction whose bounds are loaded | lanes with a remainder loop, C-113 |

## Order of work

The work is three milestones. Each ends with something a user can rely on; inside a milestone,
items run in lanes, sequential inside a lane, with arrows as hard dependencies. Two rules hold
across lanes: a pull request that regenerates C snapshots is never in flight at the same time as
another one (C-98, C-108, C-109, C-119, C-8 and C-130 regenerate them), and edits to the shared
vocabulary (`ir/expr.py`, `ir/program.py`, the verifiers) merge one at a time.

```
Milestone 1: differentiable loops in a debuggable core
  foundations  C-86 lowering package, first; then C-87 names and C-93 scatter (both edit lowering)
               independent: C-88 interning, C-89 wrong answers, API-90 immutable Function,
                            C-91 GPU placeholders, C-92 verification
               C-83 JIT fingerprint -> R-71 zig cc as the JIT's compiler
               C-94 JIT index key (after C-88, API-90)
  print        C-124 sc.print, float64 (after C-86, C-87)
  AD           C-95 harness -> C-96 helpers and names -> C-97 ad/calls.py
               -> C-98 one forward traversal -> C-99 partials and reverse -> C-100 intermediates
               -> API-101 custom_derivative with residuals
  templates    API-114 declaration surface -> API-3 parameter lists -> API-1 instances and holes
               -> API-115 lifted derivatives
  dtypes       C-103 coercion -> C-104 predicates, select, cast (IR and lowering after C-108,
               AD half after C-99)
  indexing     C-136 GATHER and SCATTER with index operands (after C-93, C-104)
  emitter      C-108 loop nests and the iteration-plan interface -> C-109 accumulators
  loops        C-119 invocation model (after C-87, C-108) -> C-120 carried ranges -> C-121 LOOP
               -> C-122 forward AD -> C-123 reverse AD; C-138 in-place carries (after C-121, C-136)
  any time     C-82 frame budget (after C-86)

Milestone 2: static sparse functions end to end
  sparse       C-127 pattern in TensorType (after API-1, C-92) -> C-148 support on every node
               -> C-149 no full-size work -> C-128 sparse boundary -> C-129 interfaces
               -> API-116 pattern holes
               C-130 sparse kernels and the sampled adjoint (after C-128, C-109, C-98)
               -> C-131 sparse-sparse products -> C-132 sparse derivative outputs
  linalg       C-133 dense factorizations and solves (after API-101, C-121, C-138, C-154)
               -> C-134 dense LDL^T and LU -> C-135 sparse LDL^T (after C-130, C-125)
  loops        C-154 unrolling static loops (after C-121); C-125 COND (after C-121)
  external     C-139 extern and CLibrary (after C-119, API-1, R-71)
  dtypes       C-105 nonsmooth, extremum reductions (after C-104, C-109)
               C-106 float32 lowering -> C-107 typed entry signature

Milestone 3: competitive kernels and generated solvers
  tensor       C-110 axis reductions -> C-111 batched matmul -> C-112 einsum
               C-150 VMAP folded into LOOP (after C-121, C-119)
               API-2 typed vmap (after API-1, C-110, C-150)
               C-8 lowering-time fusion, sparse-aware (after C-109, C-130)
               -> C-113 lanes and block callees -> C-102 reverse keeps the call boundary
               -> C-144 the primal shared with its derivative
  kernels      C-151 predictable lowering of hand-written kernels; C-152 demanded entries
  API          API-117 Function-level jvp/vjp (after C-98, API-3) -> API-4 npmpc example and guide
  checks       C-126 checked builds (after C-136, C-124); API-140 TinyMPC; API-141 generated IPM

Then D-36, the GPU milestone definition.
```

Why this order:

- Milestone 0's old wave stays first: it fixes silent wrong answers found on main during the survey;
  the worst is two callees with one name sharing a procedure. The lowering package split goes before
  every other lowering edit, so nothing rebases across a file move.
- `sc.print` comes right after the foundations because it is the first tool that makes generated
  code debuggable, before the graph viewer is made useful. It needs only the lowering package and
  the name authority; integer and boolean formats join with C-104.
- The AD engine is the barrier for every track that adds differentiable ops. Its first step is a
  test harness, then helper identity, then a byte-identical extraction of the call rules, and only
  then the rewrite, so each step can be bisected.
- Templates land in milestone 1 because the sparse boundary keys instances on patterns from its
  first commit (decision 12).
- Milestone 1 ends with loops and their derivatives, because every algorithm that becomes a library
  Function (decision 10) is built from `LOOP`, `GATHER` and `SCATTER` and needs in-place carries.
- Milestone 2 is where the closed-language decision is tested: the library sparse LDL^T must match
  devrush's direct kernel (C-135). If it needs an op to get there, decision 2 is reopened.
- Fusion and lanes come last because they change the emitted code of everything before them, and
  they must know about sparse traversals (C-8).
- R-71 (`zig cc` as the JIT's compiler) lands in milestone 1, right after C-83 prepares it. Before
  C-139, because `CLibrary`'s flags are then written in one clang dialect on every operating system
  and the plugins' link paths are checked with zig's driver once, before they move onto externs.
  Before milestone 3, because changing the JIT's compiler shifts every timing, and that
  milestone's gates must be measured under one compiler. R-38 (Windows) stays deferred, but C-139,
  C-94 and C-124 are designed so it adds a platform without changing them.

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
- A new op is added to the elementwise table or to one of the families above, never as a one-off;
  C-92's coverage test lists every op in every pass.
- The pull request that finishes a user-visible feature ships its guide page and API entries.
  Anything unfinished stays out of `docs/`, filtered from the API page, and free of roadmap wording
  in rendered docstrings.

## Foundations

Milestone 1. Each item is small and removes a class of bugs the later tracks would multiply.

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
`flat_tree` adapter. This is today's Function, which becomes the playground's `ConcreteFunction`
in API-1.

**C-91. Remove the GPU placeholders.** The `DeviceSpec` entries for cuda, opencl and metal,
`BACKEND_SUPPORT`, the `KERNEL`/`LAUNCH`/`BARRIER` program ops, and the range kinds `THREAD`,
`LOCAL`, `WARP` and `GROUP_REDUCE` exist only to raise "deferred". `GLOBAL`, `REDUCE`, `VECTOR`,
`SERIAL` (the `range_` default, used by `widen_ranges` for its chunk loop) and `UNROLL` stay. D-36
defines what a GPU target needs, and it should not inherit guesses.

**C-92. Verification, op coverage, and no recursion.** `verify_expr` is never called in the
pipeline, and `SLICE` and `SOLVER_CALL` have no verifier rule. Run it when a Function is built and
before lowering. Add one test that classifies every `ExprOp` in the verifier, forward and reverse AD,
`ad/sparsity.py`, the support rules (after C-148), lowering and (after C-94) the graph digest, so a
new op cannot be half-wired, and that no pass names an individual member of the elementwise table.
Make `ad/sparsity.py::_depends_on`, `_jac_mask` and `Expr.structural_key` iterative: devrush found
800 unrolled Euler steps overflow the Python stack in walks of this kind.

**C-93. Accumulating scatter.** The `SCATTER` docstring says repeated indices accumulate, but the
builder rejects them (`expr.py:688`) and lowering overwrites. Accept duplicates. Lowering keeps
today's plain store loop when the indices are unique (so results stay bit-identical, including
signed zeros) and emits an accumulate loop, summing in index order, when they repeat.
`segment_sum(values, ids, n)` is a builder over it, not a new op. `ad/reverse.py::_gather_vjp`
becomes one scatter (devrush: 1.3 s to 1.6 ms at n = 2000), `_unbroadcast` stops building
per-element `gather().sum()` stacks, and `combine_scatter_sums` accepts accumulate loops. This is
the first step of C-136's `SCATTER`: today's op is `SCATTER(zeros, idx, values, add)`.

**C-83. The host and the compiler in the JIT cache key.** Absorbs C-85. `_compute_cache_key` hashes
the flag string `-march=native`, not the CPU it resolves to, and not the compiler. Hash the full
output of the `cc -march=native -dM -E` probe that `toolchain.native_recipe` already runs (the CPU
features and the compiler's version macros), the compiler's `--version` output and real path, and the
full command and effective flags, so a wrapper or a patched compiler at the same path still misses.
`Compiler` becomes a command tuple so R-71's `zig cc` drops in. Bump the cache version.

**C-94. A JIT index key that skips lowering.** Today every warm process lowers and renders the C
before it can find its library (devrush: 1.4 s of a 1.8 s warm start). Add an index keyed on:

- `graph_digest(fun)` from a new `function/digest.py`, iterative and memoized per immutable
  Function, hashing each node's op, type (patterns by digest), name, lowering hint, value bytes and
  attrs through an explicit encoder (floats by bits, callees by digest, custom rules, solver
  descriptors, externs and their libraries by content) that raises on an unknown attr type;
- the scaly version and a digest of the installed scaly and plugin sources, computed once per
  process, since a codegen change no longer shows up in the key by itself;
- the render options and `BuildRecipe`, C-83's fingerprint, compile and link flags, the
  generation-affecting environment variables, and the libc version when `vector_libm="glibc"`.

The index maps to the rendered-C hash, which still names the artifact directory. Writes are atomic,
through `os.replace`, whose semantics also hold on Windows (R-38), and paths in the index are stored
relative to the cache root;
a missing artifact behind a valid index entry is rebuilt; `invalidate_cache` uses the index instead
of rendering. Test: a subprocess with a different `PYTHONHASHSEED` loads from the cache and never
calls `lower_function`.

## One AD engine

Milestone 1. The maintainer's four-step plan, with the details filled in.

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
- `_seed_axis` stacks `nseed` references to the primal and inserts missing broadcast dimensions.
  Nothing needs the copies for correctness; broadcasting already works, and scalars already skip the
  stack. The copies were free on the scalarized tape and cost `nseed` copy loops under block
  lowering. Replace it with an alignment helper returning shape `(1, *missing_ones, *shape)`.
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
built lazily (an inactive exponent must not build `log(x)`). The table is the elementwise table of
decision 2: one row per op also gives its NumPy fold and its C spelling by dtype. Forward contracts
partials with tangents; reverse multiplies by the cotangent and reduces broadcast axes. An entry may
prescribe a stable contraction form, as division does today (`(dx - y * db) / b`), so the
finite-scale cases in `test_joint_jvp.py` keep passing. Structural ops keep explicit pushforward
and transpose rules in their mode modules. The helper that applies a partial to a cotangent is also
where masked cotangents live once `SELECT` lands (C-104). Remove `vjp_many` from `sc`,
`scaly.ad.__all__`, `docs/api/ad.md` and `docs/dev/codebase.md` (step 4); its only caller is a test.

**C-100. Derivatives with respect to intermediate expressions.** Devrush's `independent(exprs,
wrts)` substitutes stand-in inputs for the chosen intermediates and maps the result back. Without
it, forward mode returns silent zeros for a `wrt` that is a slice of a loop carry, which breaks a
Newton step inside a while body. Apply it in both modes and in sparsity analysis, and test
overlapping selections. `sc.stop_gradient` comes with it.

**API-101. `sc.custom_derivative` with residuals.** `sc.custom_derivative(fn, jvp=, fwd=, bwd=,
sparsity=)` returns a new Function through API-90's constructor; the rules are fields, never set
afterwards. A JVP rule takes primal inputs and tangents. The reverse rule is JAX's `custom_vjp`
pair: `fwd` takes the primal inputs and returns the outputs plus residuals (a factorization, a loop
trajectory), and `bwd` takes the residuals and the output cotangents. The residuals are results of
the same invocation (decision 7), so the primal computes them once; they keep their dependence on
the inputs, since freezing them gives wrong Hessians. Every derivative of a Function body goes
through `body_tangents` and `body_cotangents`, so `CALL`, `LOOP` and the map case all honour the
rule, library Functions included. Tangents the rule never reads are not built, so an implicit solve
does not differentiate a factorization. Several seeds call a single-seed JVP rule through one map
over the seeds. Nested differentiation differentiates the rule graph. Declared sparsity becomes
pattern blocks at construction, which `ad/sparsity.py` reads without importing Function; without a
declaration the pattern is dense. A sparse output's stored pattern needs no rule: it is part of the
declared output type. Tests: randomized linearity and duality of the rules, forward over reverse,
forward over forward, and a solve whose derivative reuses its factorization. Devrush measured an
implicit-rule gradient of a solve 5.6 times faster than differentiating through the steps, and
exact.

**C-102. Reverse mode keeps the call boundary.** Milestone 3, after C-113. Reverse mode through
`CALL` inlines the callee's adjoint by substitution today, so no call survives. Emit one adjoint
helper call per invocation instead, sharing the cotangents of sibling outputs, and route map
adjoints through the accumulating scatter, which also handles stride-0 and overlapping windows.
Lowering decides whether to inline a small helper. Measured on every benchmark problem. A call's
value and its derivative compute the callee's forward pass twice today (devrush's DiffMPC episode
ran in 200 ms against 161 ms without the duplicate); C-144 shares it, through the residual
machinery of API-101.

## Dtypes and the scalar vocabulary

Milestones 1 and 2. On main float32 is a type nothing honours: `x_f32 * 2.0` raises, AD builds
float64 zeros and seeds, a float32 function reads `float` values through `double*` and calls double
`sin`, and neither dialect has a cast, comparison or select.

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

**C-104. Predicates, select, cast.** Rows of the elementwise table in both dialects: `LT LE EQ NE
AND OR NOT ISFINITE SELECT CAST`, plus integer `FLOORDIV` and `MOD` with NumPy floor semantics.
Predicates are boolean and not differentiable; `CAST` keeps its derivative only between float
dtypes. The C renders operators, `?:`, `isfinite` and casts; `isfinite` makes `-ffast-math`
unsupported, which the toolchain asserts. `scalarize` accepts int64 and bool along with float64: its
gate exists because narrower stores round or truncate, which cannot happen for these. `sc.print`
gains its integer and boolean formats here. The AD half waits for C-99: `SELECT` forward selects
tangents, reverse sends `where(c, t, 0)` and `where(c, 0, t)`, and C-99's masked-partial helper
keeps a NaN partial of the unchosen branch (`sqrt` at 0) out of the result. Sparsity and support
take the union of branches, and the QP affinity proof treats a select condition as nonlinear;
devrush's proof accepted `|x|` as quadratic because its pattern ignored predicates.

**C-105. Nonsmooth derivatives and extremum reductions.** Ties split equally, `floor` and `ceil`
have zero derivative, and `abs` has `sign(x)` with `sign(0) = 0` (main's `x/|x|` is NaN at 0).
`reduce_max` and `reduce_min` as `REDUCE` kinds; `argmax`, `argmin`, `norm_inf` and `norm_1` as
builders; full reductions now and over axes once C-110 lands. `max` and `min` propagate NaN, so a
convergence test on a NaN residual fails. They lower to C-109's accumulator form with a select;
lanes come from `widen_ranges` and later C-113, not from devrush's hand-written four-lane code.

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

Milestones 1 and 3. This track absorbs C-8, closes C-79's follow-up and deletes C-43's layout rules.

What main supports: `TensorType` has no rank limit, broadcasting is N-d, `transpose` takes any axes
at build time and stepped slices lower already. The limits are `matmul` (rank 2), lowering's
`TRANSPOSE` (rank 4), `SUM` (full reduction only), `vmap` (rank-1 outers, flat output) and
`jvp_many`'s rank-3 transpose. Downstream, every op is one flat loop with div/mod coordinates,
reductions accumulate into the output element, and matmul has hand-written layouts.

**C-108. The loop-nest emitter and the iteration-plan interface.** `LowerCtx` gains
`nest(shape) -> (ranges, coords)`, `reduce(extents)` with bounds that may be expressions (sparse
rows need `rowptr[i]` to `rowptr[i+1]`), and `read(expr, coords)` returning a scalar program node.
These are the all-dense case of decision 5's iteration plans, so the interface takes access maps
that are affine or table-driven from the start, even though only dense ops use it here. A singleton
axis gets coordinate zero; a zero extent suppresses the nest, so an empty tensor emits no store. A
view index is the affine sum of strides times coordinates. A reshape after a transpose can leave an
index that is not affine in the consumer's coordinates; then the emitter uses quotient and
remainder on that axis only, or materializes the transpose, and a test pins which. This is
devrush's `delinearize_loops` idea done at emission, where the structure is known, instead of
recovered afterwards. Migrate elementwise ops, movement ops (`RESHAPE`, `TRANSPOSE`, `SLICE`,
`GATHER` through `index_at`), `STACK` and `CONCAT`; delete `_coord_p`, `_broadcast_index_p` and
`_flat_index_p`. Tests: NumPy differential tests per op and shape class, transpose-reshape-slice
chains, empty dimensions. Gates: race-car N=200 within the C-79 limits, `test_fuse_ranges` and
`test_widen_ranges` still asserting the same properties, lowering time within 10%. Regenerates
snapshots.

**C-109. Accumulators and the contraction schedule.** A reduction becomes `acc = identity; for k {
acc = acc op x }; out = acc`: `scalarize` learns `ASSIGN`, and `widen_ranges._ordered_reduction`
matches the new form. Empty reductions produce the identity. `SUM` and `MATMUL` (rank 2 as today)
move to it, and `_lower_matmul`'s layout branches and `_mm_accumulate` are deleted. The two C-43
layouts come back as one scheduling decision in `passes/lowering/schedule.py`: a unit-stride reduce
axis splits the neighbouring output axis into four `UNROLL` lanes with one accumulator each, and a
strided reduce axis goes outermost. Each output keeps the summation order of the reference schedule
(decision 13). Gates: npmpc N=12 and unbumpercars C=8 within 5% of the C-43 kernels. Regenerates
snapshots.

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

**C-150. `VMAP` folds into `LOOP`.** After C-121 and C-119. `VMAP`'s attributes (`callee`, `length`,
`slice_size`, `starts`, `strides`, `output`) are a subset of `LOOP`'s, and a map is a `LOOP` with no
carry and no condition. Rename the op, and make `fuse_ranges`, `widen_ranges`, the periodic seed
tiles, structured sparse AD and the 14 files that name `ExprOp.VMAP` test "a loop without carries"
through one predicate beside `callees_of`. The forward and reverse rules become `LOOP`'s with no
carry, which removes a second invocation model and a second adjoint route. Snapshots byte-identical,
`vmap.c` included.

**API-2. Typed vmap.** The tensor track's leading-axis design merged with devrush's views plan and
the playground's `vmap` candidate (`typing_playground/function.py`), lifted with `lift`.
`sc.vmap(f, N)` returns a mapped callable whose call builds a map at the call site (a `CALL` would
hide the map from `fuse_ranges`, `widen_ranges` and structured sparse AD) and returns the callee's
whole output tree, each leaf shaped `(N, *shape)`. Argument resolution, first match wins:
`sc.window(x, start, stride)` for overlapping rank-1 windows (race-car's `(z, 0, 4)`);
`sc.broadcast(p)`; an array whose leading axis is `N`; the API-70 size rules. A view is read in
place by replaying its `SLICE`/`RESHAPE`/`TRANSPOSE` chain on `np.arange` to recover
`(base, start, stride)`, so the node equals the hand-written one and the C is byte-identical; a
non-affine view (`A.T`) is copied, which is semantics, not a fallback. A template callee is
instantiated from the resolved slice shapes. A sparse leaf mapped over `N` stages shares one local
pattern (decision 4's batch axes). Lowering emits one loop for all used outputs through C-119.
`vmap.c` stays byte-identical.

**C-8. Fusion decided at lowering time.** The three cases of tinygrad's scheduler, applied in
`LowerCtx.read`, where consumer counts are known: movement ops are never materialized unless an
output or a non-affine composition requires it; an elementwise node with one consumer is inlined at
the consumer's coordinates; a node with several consumers, a call or map output, a scatter, a
reduction or an output is materialized. A producer read under a range it does not depend on is
materialized unless it is cheap, by the caps `fuse_elementwise` applies today. Fusion reads
iteration plans, not only consumer counts: a sparse producer is inlined only into a consumer that
shares its traversal, and otherwise kept in a workspace or materialized. The consumer-agreement
merge of tinygrad's `run_rangeify` (share a producer when all consumers index it the same way) is in
scope: devrush's many short IPM loops justify it. `fuse_elementwise.py` and `inline_producer` are
deleted. Memoize `read` on `(expr, coords)`, since re-walking deferred subgraphs is what made
`fuse_ranges._rewrite` slow before. Gates from the old C-8: race-car `SZ_W` zero at W = 1, chain M=5
workspace under 100k doubles (109,944 today), npmpc within 5%, `_assert_fused` still holds; plus the
IPM's KKT assembly in one fused loop. Regenerates snapshots. `fuse_ranges`, `widen_ranges` and
`hoist_invariant` stay: inlining a scalarized callee into its mapped range is not something
lowering-time deferral can do.

**C-113. Lanes over nests and block callees.** The `UNROLL` split for per-lane accumulators on any
reduction, including reduce ranges with loaded bounds, which get a remainder loop; `widen_ranges`
over nests and over the seed axis; float32 lanes (after C-106); `fuse_ranges` inlining
non-scalarized callees, with loop order chosen by unit-stride accesses. The last part makes a `vmap`
of a matrix-vector product with one constant matrix run like a matrix product: the batch axis goes
innermost so the lanes run over it and the matrix entry is invariant (devrush's neural MPC case study
ran at 0.05 times PyTorch there). Gates: a `vmap` of a 12x512 matvec within 2x of the explicit
`X @ W.T`; the regressions C-79 left open (unbumpercars C=2, C=16, C=32 and the closed-loop function
evaluation), re-measured under the C-79 protocol. C-77 and C-79 close into this track; the C-79
design and gates are in `c77_c79_implementation.md`.

**C-151. Predictable lowering of hand-written kernels.** Decision 14 as tests and the changes they
need. A register-blocked dense product written in Python (4×4 blocks of slices, a `LOOP` over the
`k` panels carrying the block accumulator) must lower to: the block as straight-line code over C
locals; the accumulator carry in C locals rather than workspace slots; slices read in place; a
packed layout the user materializes hoisted out of the loops; Function boundaries, loops and
lowering hints (`.scalar()`, `.block()`) kept; generated buffer parameters marked `restrict`, which
they are not today. Each property is a test on the generated C, and a small kernel library in
`tests/` (a 4×4 product, a panel Cholesky) is measured against the same kernels in BLASFEO.

**C-152. Demanded entries.** The backward counterpart of support: which entries of a node any
consumer reads, computed in lowering, so a node is computed over its support intersected with its
demanded entries. Devrush's evidence: 984 entries computed where 527 are gathered, and a
parameter-dependent QP Hessian built dense every solve and then gathered (its todo item CS-12). It
generalizes C-130's sampled `MATMUL` adjoint to any producer of a gather.

**C-82. Frame budget.** As the todo describes. It can land any time after C-86: a budget computed
over the final program stays valid as the schedules change.

Out of scope: `dot_general`, an einsum op, multi-index views, a product reduction, a device scheduler.

## Signatures and templates

Milestone 1. Decision 12: `typing_playground/` is the design. Its README lists the guarantees,
the mapping to `src/scaly` and the decisions of 2026-09-30; this section turns them into items and
adds what the playground left to the roadmap. Devrush's template work gives test cases and naming
tokens (`internal/notes/function_templates_plan_2026_09_27.md` on that branch); its object model,
with a `ConcreteFunction` subclass and 21 graph attributes forwarded through `__getattr__`, does
not apply.

**The model, from the playground.** A `Function` is a body plus a declaration whose leaves may be
holes. It owns one `ConcreteFunction` per binding, which is everything that reaches C; a fully
declared Function has one instance, traced at the decorator and named after the function. The
compiler consumes only instances. Derived functions (the derivative wrappers, `vmap`, solver
descriptors) are made with `lift`, whose transform runs per source instance. The numerical leaf
type is `np.ndarray | scipy.sparse.sparray`, so a leaf's sparseness need not be declared
statically, and a numerical call on a sparse output returns the union.

**What a binding carries.** The playground keys instances on shapes only (`LeafKey(shape)` in
`trees.py`). The key becomes the bound leaf type, `(shape, dtype, diff, pattern)` with a layout for
sparse leaves, and a leaf declaration gains optional dtype and pattern fields, each of which may be
a hole. A dtype hole binds from an expression's dtype or a NumPy argument's (C-103's rules); a
pattern hole binds from a sparse expression's type or from a SciPy argument, and a dense argument
binds no pattern. Because patterns are interned, the key hashes in time proportional to the number
of leaves. Differentiability is declared, never a hole.

**API-114. The declaration surface.** `arg` declares a leaf and `group` a group, replacing `L` and
`G`; the decorator takes one tree per parameter and a keyword-only `outputs=`, which may be left
out to read the output tree off the trace (one leaf named after the function, several `out0`,
`out1`...). A scripted edit of every decorator in `src/`, `tests/`, `benchmarks/`, `plugins/`,
`examples/`, `docs/` and the notebooks: today's `@sc.function(inputs, outputs)` with
`def f(inputs)` becomes one parameter holding the group and `outputs=`, so bodies do not change. The
decorator checks its arity against the body's parameters. Snapshots byte-identical.

**API-3. Parameter lists typed with `TypeVarTuple`, and zero-input Functions.** The playground's
typing: `Function[SI, NI, SO, NO]` whose input parameters are tuples of parameter types, a decorator
ladder of 0 to 8 parameters with and without `outputs=`, `group`'s ladder, calls unpacked through a
typed `self`, and seeded modes appending one parameter. Calls dispatch on argument kind as today; a
call with no arguments is numerical, and `f.symbolic_call()` is the symbolic spelling, so
parameterless solver oracles stay legal and `ad/forward.py` drops `_flat_symbolic_call`. The seeded
wrappers and the solver call convention change in this PR (about 130 call sites), separately from
the codemod. The playground's static tests move to `tests/typing/` under `ty check
--error-on-warning`, and the ladder's check time is measured and kept.

**API-1. Instances, holes and names.** The registry in front of today's Function, which becomes the
instance: holes on input leaves for shapes, dtypes and patterns; output shapes and patterns from the
trace; `instantiate` for ahead-of-time export; the bare mode `@sc.function()` reading structure and
shapes off each call, instantiated only by calling; a fast argument cache keyed on the call
skeleton and leaf types. Names follow the playground's rule (the declaration decides whether an
instance name is mangled, never the call history) with tokens: `3x4`, `s` for a scalar, only the
open parts of a partial declaration, `f32` or `i64` for a bound dtype, `p` plus 8 hex digits of a
pattern digest, `t` plus 6 of a nesting. Mangling encodes the nesting and never uses `__`, which C++
reserves; the playground's `_mangle` still does and is fixed with it. A collision raises, and the
mangling joins the exported-symbol rules in `docs/dev/versioning.md`. Tests: one trace per instance
(counted), the same object for symbolic and numerical binding, two instances give two procedures, a
cross-process JIT hit, the CLI refusing a holed template, the playground's typing assertions.

**API-115. Lifted derivatives.** The nine derivative wrappers through `lift`, cached per source
instance. Seeds and multipliers copy the source's whole leaf type, not only its shape (the
playground's `_decl` copies the shape): a forward seed of a float32 input is float32, the seed of a
sparse input carries its pattern, and the multipliers of a sparse output know theirs. `of` and
`wrt` default when unique; inferred names are checked at binding.

**API-116. Pattern holes and the numerical fast path.** A numerical call with a SciPy argument
canonicalizes it and digests its pattern to find the instance, which is O(nnz) per call; a concrete
sparse leaf also accepts its values vector directly, which is the fast path. With C-128.

**API-117. Function-level `sc.jvp` and `sc.vjp`.** Arity ladders, built through `lift` and
`Function.build`, so AD-built callees keep their trees and stop being `Any`-typed at run time. After
C-98. Closes API-2's second half, as recorded in `refactorings.md`.

**API-4. npmpc template example and the functions guide.** As the todo says, plus the template page
of `docs/guide/functions.md` that D-32 left out.

Deferred, as the playground's README records: overload sets under one name, constraints on holes,
and static Python-valued arguments (API-118). Closures and factory functions cover the last today.

## Sparse tensors

Milestone 2. Decision 4 and decision 5; the alternatives and the review that chose this design are
in the design note's "Sparsity" and "Sparse lowering" sections.

**C-127. The stored pattern in `TensorType`.** `TensorType(shape, dtype, diff, pattern=P)`. `P` is a
canonical pattern over the last two axes, immutable, interned by content with a digest computed
once and compared by identity; `ir/expr.py::_attrs_key` keys it by identity instead of hashing index
arrays per construction. It lists the stored coordinates, explicit zeros included; everything else
is identically zero. Leading axes are batch axes sharing the pattern, which is how seed axes and
mapped outputs carry sparse values: `shape = (*batch, m, n)`, `storage_shape = (*batch, nnz)`.
`shape` and `size` keep their mathematical meaning. The existing `SparsityPattern` keeps its
arbitrary order for derivative outputs until C-132 and gains a canonicalization that returns the
canonical pattern and the value permutation. Each op's pattern rule is fixed, like its shape rule,
and shared with C-148's support rules restricted to sparse operands: union for addition and
subtraction, intersection for multiplication, the Boolean product for `MATMUL`, unchanged for
zero-preserving unary ops. An op whose rule gives no pattern raises on a sparse operand at
construction (`exp` of a sparse matrix asks for `to_dense()`), and the verifier enforces the same.
`structural_key`, `substitute` and the verifier include the pattern. The 89 places on main that
assume `size` equals buffer length get a storage size; a missed one is a wrong address, so the audit
is a list in the pull request. Main removed an inert `TensorType.sparsity` as C-84; this one has a
meaning from its first commit.

**C-148. Support on every node.** `Expr.support`, computed lazily, iteratively and cached per node,
never part of the intern key; `None` means any entry may be nonzero. One rule per op, defaulting to
`None`, so a missing rule is sound and only imprecise: the image under the static coordinate map for
movement ops, `GATHER`, `SCATTER` and `CONCAT`; union for `ADD`, `SUB`, `MAXIMUM`, `MINIMUM` and
`SELECT` (per entry when the condition is constant); intersection for `MUL`; the numerator for
`DIV`; unchanged for zero-preserving unary ops; the projection for `REDUCE`; the Boolean product for
`MATMUL`, computed with SciPy; the callee's support for `CALL`, tiled over stages for maps; a fixed
point over carries for `LOOP`. Constants give their nonzero entries and a sparse type bounds the
support by its pattern. Uses: `passes/expr.py::_is_zero` becomes "support is empty" and disjoint
products fold; `ad/sparsity.py` consults support in `MUL`, `DIV`, `SELECT` and `MATMUL`, so a dense
constant with zero blocks no longer gets a dense Jacobian, and keeps its relations only over each
node's nonzero entries, never `expr.size` rows; scalar lowering skips known zeros. A randomized
harness checks soundness: random values on random patterns, and every entry outside the support is
zero. Simplification over the reals already treats constant zeros as strong (`x * 0` folds to `0`
in `passes/arith.py`); the guide states that a sharper analysis may remove a NaN from a product with
a structural zero.

**C-149. No work proportional to a matrix's full size.** A gate test builds, analyses,
differentiates and lowers a 10⁵×10⁵ matrix with 10⁶ entries under fixed time and memory limits.
What it catches on main today: `zeros_like` builds a full array, `passes/expr.py::_evaluate` folds a
scatter of constants by allocating the full shape, `ad/sparsity.py` allocates CSR row pointers of
length `expr.size`, transpose and slice rules build `arange(size)`, `_matmul_mask` enumerates the
dense contraction domain, and constant hashing serializes full arrays. Uniform constants
(`CONST`'s uniform form) fix the first; each of the others gets its own fix in this item.

**C-128. The sparse boundary.** A sparse leaf declares its pattern (or a pattern hole, API-116) and
its layout: CSC by default, CSR, or a coordinate list in an order the caller gives, for solvers that
read triplets in their own order. The layout is a field of the leaf, not of the pattern, and enters
the instance key. `sc.const` accepts SciPy sparse arrays (copied, duplicates summed, indices sorted,
explicit zeros kept) and `sc.const(values, pattern=P)` in the pattern's own order. Two ops convert:
`PACK(values; pattern)` makes a sparse tensor from a values vector, and `VALUES(x)` exposes the
stored values; `to_dense` is a builder over `SCATTER` and `from_dense(x, P)`, a projection that
drops entries outside the pattern, a builder over `GATHER` and `PACK`. An explicit `sc.sparsify(x)`
compacts a dense-typed value over its support. Lowering, calls and maps allocate and offset by
storage size. A symbolic call requires the exact pattern; a numerical call canonicalizes a copy and
checks it. Outputs come back as SciPy sparse arrays sharing the pattern's read-only index arrays.

**C-129. Sparse inputs in the generated interfaces.** The ABI passes only values, in each leaf's
declared layout. The C and C++ headers publish dimensions, `nnz` and the pointer and index tables of
that layout for inputs as well as outputs; `casadi.py` stops forcing input patterns to `None`. A
batched sparse leaf publishes its local pattern once, which closes the header half of C-57. An empty
pattern emits no zero-length C array.

**C-130. Sparse kernels and the sampled adjoint.** Decision 5's iteration plans for sparse
operands, in `passes/lowering/plans.py`. Absent entries are structural zeros, so a sparse product
skips them even where `0 * inf` would be NaN; documented and tested. The first kernels:

- one generic compact map, a loop over the output's stored entries that reads each operand
  through a static position table with a zero slot for absent entries. It covers movement ops,
  `CONCAT`, `GATHER`, `SCATTER`, elementwise ops and `SELECT`, with the union and intersection
  regions of operands with different patterns computed in NumPy when the graph is built, so no
  index comparison runs in the generated code. KKT assembly is this kernel;
- `MATMUL` with a sparse operand: CSR row loops with loaded bounds for sparse times dense, the
  transposed traversal for dense times sparse, with the CSR order a lowering choice and the
  permutation folded into the tables;
- `REDUCE` over stored entries, with absent entries included for `max` and `min`.

Each region takes one of four forms: straight-line code when small, a loop over constant tables when
irregular, an affine loop when `LowerCtx.index_at` recovers the index (regular runs, repeated stage
blocks), or a dense block. Output values are written straight into a boundary's declared layout.
AD: the tangent of a sparse value carries its pattern, several seeds store `(nseed, nnz)` against
one shared index table, and the elementwise partials of C-99 are evaluated on stored coordinates.
The reverse rule of `MATMUL` with a sparse operand samples the cotangent on the pattern
(`Abar[p] = ybar[row[p]] * x[col[p]]`) instead of building the outer product
`ybar[:, None] * x[None, :]` that `ad/reverse.py::_matmul_vjp` builds today. `ad/sparsity.py`'s
matmul rule walks the structural product. Three tests pin the contract with the loop compiler: an
accumulating scatter loop is never widened, a carried range is never split, a reduce range with
loaded bounds is never fused before C-8 says so. Gate: devrush's IPM KKT assembly and matrix-vector
products on its Maros–Meszaros subset with no dense `(m, n)` buffer and Jacobian patterns equal to
devrush's. This is the first usable sparse milestone.

**C-131. Sparse-sparse products.** The Boolean product pattern computed at build time, lowered as a
column loop over a dense accumulator column with the entries to clear known statically, instead of
devrush's enumeration of every scalar product pair (`linalg/sparse.py::_spgemm`). Work and fill grow
as the sum of squared row degrees, which no table can remove; tests pin bounded table and workspace
growth on `G.T @ diag(w) @ G`. Block assembly, `diagonal`, Kronecker products and repatterning are
compositions of existing ops and need no item.

**C-132. Sparse derivative outputs become sparse leaves.** `sparse_jacobian` and `sparse_hessian`
return sparse-typed outputs with a declared layout and triangle
(`sc.sparse_hessian(f, of, wrt, layout="csr", triangle="upper")`), and `Function.output_sparsities`
goes away, so a sparse output has one spelling. The header tables, the CasADi gather and
`casadi_output_sparsities` read the type. Values switch from arbitrary order to the declared layout,
which removes the two value-permutation tables: an interface change under `docs/dev/versioning.md`.
The mapped path (`ad/sparse.py::_sparse_jacobian_vmap`) keeps its local coloring and returns a
batched pattern for a map of independent stages.

Deferred, beside this track: different patterns per batch entry, dense-block and supernode
detection inside a pattern (C-153), automatic compaction of dense-typed values. C-11 and C-14
(coloring width, row coloring) stay deferred.

## Loops, conditionals and printing

Milestones 1 and 2. Devrush's reports give the stakes: an RK4 rollout plus its gradient at N=200
built in 80 ms instead of 22 s unrolled, with 126 lines of C at any N. TinyMPC, the generated IPM,
adaptive integrators, Newton solvers and every factorization of decision 10 need loops.

**C-124. `sc.print`.** Right after the foundations. `y = sc.print("x={} k={}", x, k)` returns `x`.
`PRINT` is an identity op that carries the format, so prints travel with data through substitution,
inlining, maps, `LOOP` and AD. The rule, documented with its limits: a print runs whenever generated
code computes its value. Once per step in a loop and once per trip in a map; identical prints
intern to one; independent prints have no guaranteed order; a callee whose outputs nothing uses is
never called, so its prints do not run. A derivative Function prints when it computes the primal
value, which forward and reverse rules usually do, since partials read the primal; the AD rule of
`PRINT` keeps the primal node rather than bypassing it. The tracer records the prints it creates
and raises if one is unreachable from the outputs. Lowering emits a `PRINT` statement;
`hoist_invariant`, `scalarize` and `widen_ranges` leave procedures containing one alone, and
`prune_procedures` keeps them. Formatting uses `{}` placeholders: `%.17g` for float64 here, then
`%.9g` for float32 (C-106), `%lld` with a cast to `long long` for int64 and `%d` for int32 and bool
(C-104); tensors print flat as `[a, b, ...]`. The source defines `SCALY_PRINTF` as `printf` only
when it contains a print, so `-DSCALY_PRINTF(...)=` compiles prints out and an embedded target
routes them to a UART. JIT prints go to the process's C `stdout`, which some notebook front ends do
not capture, and which a Windows process (R-38) may not share with Python's `sys.stdout`; the guide
says so, and the JIT flushes C `stdout` after each call that printed. JAX's effect tokens were rejected: they need an effect system
threaded through every transform.

**C-119. The invocation model.** `callees_of(node)` in `ir/expr.py` replaces the hard-coded
`{CALL, VMAP}` sets in `solvers/graph.py`, `solvers/qp.py`, `codegen/aot.py`, `codegen/solver.py`,
`ir/text.py` and `function/model.py`. `invocation_key(node)` is the op, callees by identity, argument
ids and attrs except `output`. `LowerCtx` computes the used results of each invocation once from the
topological order it already walks, and the first sibling emits the invocation with all of them,
including the residuals of a custom rule (API-101). This replaces the name-keyed
`call_invocations` and fixes `VMAP` lowering one full loop per selected output, which is why AD
packs derivative outputs into one today. Regenerates snapshots for the merged loops. Derived
callees come from C-96's cache so that siblings stay grouped.

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
`strides` (non-negative; stride 0 is a loop-invariant parameter), `index`, `reverse`, `in_place` and
`output`. The body is `(*carries, [k], *xs) -> (*carries', *ys)`; carries may have mixed dtypes, and
each carry's shape, dtype and pattern are the same at entry and exit. Semantics: `T = min(trip,
length)`; the condition is evaluated before each step on the entering carries, and the loop stops
at the first false; `k` counts `0..T-1`, or `T-1..0` with `reverse`. Results: the final carries, the
stacked ys, the int64 trip count, and one trajectory per carry, where `trajectory[k]` is the carry
entering step `k`. Stacked ys and trajectory slots at or past the trip count are zero-filled, which
costs a conditional loop an O(length) fill and a plain scan nothing. The condition sees only
carries and parameters, so any residual it needs must be a carry. Builders `scan(body, init, xs, *,
length, trip=None, index=False, reverse=False, in_place=())` and `while_loop(cond, body, init, *,
max_iter, params=(), index=False)` share `vmap`'s slice resolver. The `trip` operand is user-facing:
it gives a run of run-time length under the static bound `length`, which is what devrush's
`ragged_add` and `ragged_dot` were for, and the reverse rule of a conditional loop also creates it.

Lowering: one carried range, `[call cond; BREAK_IF !flag;] call body`, with carries in two
alternating slots, in `length + 1` slots when a trajectory is used, or in one slot when C-138 proves
the update in place. Small carries live in C locals (C-151). Sparsity is a boolean fixed point over
the carries plus the dependencies of stacked outputs. Any derivative raises until C-122. Tests
against NumPy loops and Python-unrolled graphs: lengths 0 and 1, exit at step 0, `max_iter`
reached, a run-time trip count, index, reverse, parameters, mixed-dtype carries, nested loops, a
loop inside a map body.

A scan is a `LOOP` with no condition and no `trip` operand, so its trip count is the static
`length` and every pass can read that from the attributes. The carried range has constant bounds,
stacked outputs and trajectories have exactly `length` slots and need no zero-fill, no `BREAK_IF` is
emitted, the backward loop of C-123 has the same static bound, and slice offsets are constants. What
a condition or a run-time trip costs (a flag per step, a runtime trip count, the zero-fill) is paid
only by loops that have one.

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

**C-154. Unrolling static loops by the target's budget.** `LOOP` lowering unrolls a loop whose
length is static and whose per-step slices are constants when the body, times the length, fits the
target's straight-line budget, honouring the node's lowering hint. Constant folding then deletes
the masked terms of a triangle written with `where(iota < j, ...)`, so a library Cholesky of a small
dense block gives the straight-line code devrush's `Unroll` option produced, without an option.
Unrolling never changes the reference schedule's summation order (decision 13).

**C-125. `COND`.** Once masked cotangents exist, `where` is correct in both AD modes, so `COND` is
about cost and about effects inside branches. `COND(selector, *operands)` with a tuple of branch
Functions; a boolean selector for two branches, an integer clamped like `lax.switch` for more, and
no separate `SWITCH`. Lowered to a program `IF` chain calling each branch procedure. Forward AD is a
`COND` over the branch tangent Functions; reverse is a `COND` over branch adjoints that recompute
their primal. Sparsity and support are the union of branches. `widen_ranges` refuses a map body
containing one.

**C-126. Checked builds.** A render option `checks=True` emits `CHECK(cond, message)` statements:
bounds on `GATHER`/`SCATTER` with `mode="promise_in_bounds"`, non-finite values in loop carries and
procedure outputs, each message naming the function and the op as `format_expr` numbers it.
`SCALY_CHECK_FAILED` defaults to `fprintf(stderr, ...); abort()` and can be overridden.

## Linear algebra as library Functions

Milestone 2. Decision 10: every factorization and solve is a `Function` in `scaly.linalg` over
`LOOP`, `GATHER`, `SCATTER` and the elementwise table, with `sc.custom_derivative` on the solves.
None of them adds an op. Devrush's `linalg/ops/{dense,trisolve,sparse_ldl,ragged}.py` are the
reference for the algorithms and the measurements, not for the structure.

**Contract for every solve.** The matrix classes are stated per solve. A solve reads one stored
triangle of a symmetric matrix (the lower), and an off-diagonal stored entry stands for both
mirrored entries, so its cotangent is the sum of both mirrored contributions. A factorization that
meets an unusable pivot reports it through a status output, never through a NaN alone.
`solve(K, b)` is a semantic boundary built with `custom_derivative`: the forward rule returns the
solution and the factorization as residuals, so the primal and all derivative solves share one
factorization; forward is `dx = K^-1 (db - dK x)`, reverse is `bbar = K^-T xbar` with
`Kbar = -bbar x^T` sampled on the read triangle. Second-order tests have both the matrix and the
right-hand side active. Each factorization has one reference schedule (decision 13); lanes,
unrolling and chunking are checked against it numerically.

**C-133. Triangular solves, Cholesky, and `solve(assume="pos")`.** Library Functions over `LOOP`:
lower or upper, transposed, unit diagonal, one or many right-hand sides. Small sizes become
straight-line code through C-154 rather than an `Unroll` option, and the partial sums of a long
reduction come from C-113's lanes rather than hand-written four-lane code (devrush's `CHOLESKY_TILE`
dropped terms for any value but 4). Gates: npmpc's and TinyMPC's dense blocks against devrush's
`linalg/ops/dense.py` and `trisolve.py` timings.

**C-134. Dense LDL^T and LU, and `solve(assume="sym"|"gen")`.** `LDL` without pivoting for
symmetric quasi-definite matrices, `LU` with partial pivoting for general ones; row swaps are
same-index updates that C-138 keeps in place. The factor outputs refuse differentiation; the solves
are differentiable through the contract above.

**C-135. Sparse LDL^T.** A symbolic analysis module below `Function` (natural, RCM and
minimum-degree orderings, the elimination tree, the pattern of L, column chunks), immutable and named
by a digest of its tables; it replaces `scaly-sqp`'s private `_ldl_symbolic` and runs on a sparse
type's pattern. The factorization is a library Function: an outer `LOOP` over columns carrying `L`,
`D` and a dense work column, inner loops with run-time trip counts over each column's runs (C-121),
gathers and accumulating scatters through the analysis tables, and `in_place=("L", "D", "w")`.
Chunks of up to eight columns sharing rows are a `COND` over chunk widths, each branch summing in
registers (devrush: 1.5 to 2.2 times faster than its scan schedule); measure the code size of eight
widths on the Maros–Meszaros subset before keeping them. The pivot status follows the contract;
`inertia()` and `health()` come with it; iterative refinement is a `LOOP` around the solve. Gate,
and the test of decision 2: the generated C has the loop structure of devrush's direct lowering
(`linalg/ops/sparse_ldl.py::_lower_sparse_ldl`) and runs within its measured times (within 1.04 to
1.5 times QDLDL-class C with loops). If it needs an op to get there, decision 2 is reopened.

## Runtime indexing, in-place updates and external code

Milestones 1 and 2. Needed by interpolation search, the IPM's loops, every library factorization and
the solver plugins.

**C-136. `GATHER` and `SCATTER` with index operands.** `GATHER(x, idx; axis, mode)` and
`SCATTER(base, idx, values; axis, combine, mode)`, with `idx` an int64 operand of any shape: a
constant index is the static case, and every rule that reads `attrs["indices"]` today reads the
constant operand (`LowerCtx.index_at`'s affine recovery from C-9, `ad/sparse.py`, the identity-gather
rule, `fuse_ranges._index_values`, the constant-seed bake). `combine` is `set`, `add`, `max` or
`min`; duplicates resolve in index order: the last write wins for `set`, `add` sums in order, `max`
and `min` include the base value and propagate NaN like the reductions. `mode` uses JAX's names:
`fill` reads through a clamped address and returns a fill value out of range, an empty source
returns the fill without reading, and an out-of-range write goes to a scratch slot so writes stay
unconditional; `promise_in_bounds` is a promise, breaking it is undefined, and checked builds catch
it. Lowering emits a memset instead of a copy when the base is a zero constant. AD: `GATHER` and
`SCATTER(add)` are adjoint to each other, `SCATTER(set)` masks the base cotangent at the winning
lanes, `max` and `min` split ties. Sparsity and support are exact for a constant index and a
conservative row otherwise. `scalarize` reports a data-dependent address as not eligible instead of
failing in `_Frame.pointer`. `segment_sum`, `segment_max`, `take`, `dynamic_slice`, `index_add` and
`index_set` are builders. Snapshots byte-identical for constant indices.

**C-138. In-place loop carries.** Buffer reuse in `passes/lowering/in_place.py`. Semantics stay
functional; writing into the old buffer is a lowering decision, by three rules, cheapest first:

1. *Last use.* A `SCATTER` writes into its base's buffer when nothing reads the base after it in the
   schedule; inside a loop step, the next carry reuses the old carry's slot when every read of the
   old carry is scheduled before the update. This covers a left-looking LDL^T: step `j` reads the
   columns of `L` before `j`, then writes column `j`.
2. *Same index.* `SCATTER(base, idx, f(GATHER(base, idx)))` reads only what it writes, so its read
   and write loops may be fused, for any index, including a pivot chosen at run time.
3. *Disjoint indices.* When the old value is read after an earlier write in the same step, devrush's
   proof applies: every index is evaluated at every step with one batched NumPy interpreter over the
   elementwise table, and the entries read must be disjoint from those later links write.

`LOOP(in_place=...)` names carries that must be updated in place, and lowering raises when it
cannot prove it, instead of silently copying O(size) per step. The in-place procedure takes one
`inout` buffer parameter instead of two aliased pointers; `scalarize` refuses it, `hoist_invariant`
never treats it as invariant, and `pack_workspace` counts it as read and written. Devrush measured
an update of four entries in a 100k carry 354 times faster than with two alternating slots.

**C-139. `sc.extern` and `sc.CLibrary`.** External C code is its own construct, not a Function body:

- `sc.CLibrary(name, sources=, headers=, include_dirs=, defines=, compile_flags=, link=, versions=)`
  is a frozen description of how to compile and link some C. It is the unit of compilation: the C
  sources are compiled once however many symbols are taken from them, and their flags apply to that
  translation unit only. Flags are written in clang's dialect, which the JIT always speaks after
  R-71, and libraries to link are named, not spelled as `-l` or `.lib` flags: the toolchain
  resolves names to paths and flags per platform, so Windows (R-38) adds a platform without
  changing the API. Shared-library loading, including a separate linker namespace where the
  platform has one, belongs to the JIT, not to the description. Its content (source text, flags, the identities and versions of linked
  libraries) enters C-94's digest. The JIT compiles or links the libraries a graph reaches into the
  same shared object; ahead-of-time export writes their flags into a build manifest beside the
  generated C, since Scaly does not own the user's build.
- `sc.extern(library, symbol, inputs=..., outputs=..., workspace=0)` declares one C function with a
  concrete signature: every leaf has a shape, dtype and pattern, so an extern is concrete by
  construction. Its calling convention is the generated one, one typed pointer per leaf and a
  workspace (C-107). Calling it builds `EXTERN` nodes, one per output, grouped by invocation
  (C-119).

A shape-generic C kernel is a Python function that builds one extern per shape, generating the
source if needed; the template machinery is not involved. The first contract allows only
deterministic, reentrant calls: inputs read only, outputs fully written, no retained pointers, no
hidden state, no aliasing; a failure status is an ordinary output. An extern has no derivative
unless a Function wrapping it gets `sc.custom_derivative`, and an active request raises otherwise;
its derivative sparsity is declared or dense. The solver plugins become externs over their plugin's
library, which removes `SOLVER_CALL` and C-89's zero-derivative guard, and replaces devrush's
`function.extern = callee` set after construction (`function/extern.py` on that branch).
`Expr.opaque()` stays a lowering hint.

## Milestone checks

| After | A user can | Check |
|---|---|---|
| Milestone 1 | trust that generated code matches the graph; start a warm process without lowering; print from generated code; differentiate every op through one engine; write an implicit derivative; declare templates; write loops with run-time trip counts and differentiate them | the name-clash tests; the cross-process cache test; prints inside a loop and a map; the AD harness with the fallback deleted; an RK4 rollout with its gradient at N=200 in build time and C size constant in N; a Newton iteration inside a while loop, with early exit, zero-trip loops and a second derivative |
| Milestone 2 | pass sparse matrices through Functions in a chosen layout; assemble and multiply them with derivatives; factor and solve dense and sparse systems with library Functions; call external C | the full-size-work gate; KKT assembly and matrix-vector products on devrush's Maros–Meszaros subset with no dense `(m, n)` buffer; sparse algebra against SciPy with finite differences; the library sparse LDL^T against devrush's direct kernel; PIQP and IPOPT through `sc.extern` |
| Milestone 3 | write float32, N-d and einsum code; get fused kernels and lanes; write register-blocked kernels in Python | float32 op twins; einsum against NumPy; the C-8 workspace gates; C-151's kernel tests against BLASFEO; API-140 TinyMPC within 10% of devrush's reported timings; API-141 a generated sparse QP IPM passing devrush's Maros–Meszaros subset |

## What devrush gives us

| Devrush work | Take | Leave |
|---|---|---|
| Predicates, select, cast | op set and semantics, masked cotangents | the separate masked traversal, two coercion helpers, `COPYSIGN` |
| `sc.options` | nothing | the context variable and cache over-keying |
| Accumulating scatter, segment ops | all of it, as `SCATTER` and builders | `SEGMENT_REDUCE` as an op |
| `scan`, `while_loop` | callee design, alternating slots, trajectory as storage, one backward loop | two ops, the hidden `output=-1` node, the sibling scan over the whole graph, float64 counters |
| Multi-seed through loops | `[c, Dc]` carries, the per-step Jacobian idea | global counters, exceptions as probes, id-keyed memos |
| `custom_derivative` | the `body_tangents`/`body_cotangents` funnel | attributes set after construction |
| Runtime indexing, in-place | `take`/`put` semantics, scratch-slot writes, the disjointness proof | five ops, two provers, `_inplace` names outside the name authority |
| `delinearize_loops` | the idea: no division in nests | the pass (recursive, exponential on shared graphs) |
| `SparseMatrix`, `sc.S` | build-time patterns, exact-pattern checks, values-only ABI | the wrapper type, the pattern outside the IR, gathers and segment sums standing for products, pair enumeration in `_spgemm`, sorting caller arrays in place |
| Op registry, `scaly.ext` | the idea that linear algebra is not core vocabulary | `register_op`, traits, pass slots, option namespaces |
| Linear algebra (`linalg/ops/`, `symbolic.py`) | algorithms, the symbolic analysis, the chunk schedule, the implicit rules, test problems and timings | eight registered ops, three schedules that round differently, string-keyed tables, process-global names, `CHOLESKY_TILE` |
| Extern callees (`function/extern.py`) | `BuildRequirements`' contents as `CLibrary` fields | `function.extern` set after construction, the render protocol |
| Templates | test cases and naming tokens | `__` in names, attribute forwarding, the `ConcreteFunction` subclass |
| Integrators, interpolation, MPC, IPM | test problems and timings for the milestone checks | the libraries themselves, for now |

## Open questions

Choices this roadmap leaves open on purpose. Each names the item that settles it.

- **The C spelling of a mixed-dtype workspace** (C-107): a caller `malloc`, a union, or a byte array
  accessed through `memcpy`, decided under `-fstrict-aliasing` tests.
- **Whether reverse mode should keep call boundaries** (C-102). The design keeps them and lets
  lowering inline; the item lands only if the benchmark problems do not slow down after C-113.
- **Per-step Jacobian compression in loop derivatives** (after C-122): devrush's idea, with
  thresholds to be measured rather than guessed.
- **Column chunk widths in the sparse LDL^T** (C-135): one `COND` branch per width multiplies
  procedures; measure code size before keeping eight.
- **A product reduction** (C-110): waits for a user, since its derivative at zeros needs the
  exclusive-product rule.
- **Derivatives through solver calls by the implicit function theorem**: C-89 makes them raise; a
  later design defines residuals, regularity assumptions and solve status, on `sc.extern`.
- **Batched sparse tensors with a different pattern per batch entry, automatic compaction of
  dense-typed values, and dense-block or supernode detection inside a pattern** (C-153): after
  C-131, when a workload asks.
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
| The dense backend cannot use matrix-vector kernels while every sparse product goes through index tables; a parameter-dependent QP Hessian is built dense every solve | `internal/notes/perf_gaps_proposal_2026_09_30.html` (A7), devrush `internal/todo.md` CS-12 |
| 984 entries computed where 527 are gathered | `internal/notes/codegen_speed_o7_report.html` |

## How this roadmap was made

Written 2026-09-29 so a later session can check or extend the reasoning. Five read-only surveys
covered main's IR, AD, lowering and codegen; devrush's control flow, indexing and AD; devrush's
sparse linear algebra, templates and float32 work; devrush's reviews and case studies; and every
item of the old `internal/todo.md`. Four design memos followed, one each for the tensor core and
loop compiler, AD and customization, templates with the float32 ABI and the foundations, and
control flow with printing and runtime indexing, plus one independent memo on the sparse IR. Two
independent reviews of the draft led to that text; their main corrections were the same-name
invocation key, the float64 guard that would have blocked predicates, batched sparse storage, the
`SERIAL` collision, and serializing snapshot-regenerating work.

Rewritten 2026-10-01 after devrush's restructure. Four independent investigations (the opset, the
sparse representation, TACO and its successors, and the whole system as an adversary of the
roadmap) and one adversarial review of the sparse design produced
[`small_core_static_sparsity_2026_09_30.md`](small_core_static_sparsity_2026_09_30.md), and the
maintainer settled its decisions. The templates section follows `typing_playground/`, which the
maintainer designed before that session.

The devrush files worth rereading, all on `origin/devrush`:

- reviews and maps: `internal/notes/code_review_2026_09_27.html` (13 reproduced bugs, about 250
  findings), `scaly_capabilities_2026_09_27.html`, `pipeline_analysis_2026_09_22.html`,
  `claude_code_handoff.md` ("traps learned the hard way"), `restructure_plan_2026_09_28.md`;
- plans: `function_templates_plan_2026_09_27.md`, `vmap_views_plan_2026_09_28.md`,
  `tier1_primitives_plan_2026_09_25.html`, `piqp_plan_2026_09_26.html`,
  `perf_gaps_proposal_2026_09_30.html`, `codegen_speed_plan_2026_09_30.html`,
  `blasfeo_kernels_analysis_2026_09_30.html`;
- reports: `tier1_*`, `tier2_*`, `tier3_*`, `ipm_speed_report.html`, `tinympc_benchmark_report.html`,
  `case_study_e1_report.html` to `case_study_e9_report.html`, `codegen_speed_o1_report.html` to
  `codegen_speed_o7_report.html`;
- code: `src/scaly/ir/expr.py`, `passes/lowering.py`, `ad/forward.py`, `ad/reverse.py`,
  `function/{model,tree,api,sugar,extern}.py`, `linalg/{sparse,sparse_factor,symbolic,dense}.py`,
  `linalg/ops/`, `utils/options.py`, and the tests under `tests/integration/` and `tests/linalg/`
  that the items above cite as targets.

On main, the notes the design builds on are `perf_2026_09_07/tinygrad_rangeify.md` (the rangeify
model behind C-108 and C-8), `c77_c79_implementation.md` (range fusion and lanes), `perf_2026_09_22/`
(the race-car kernel study), `refactorings_landed_2026_09.md` (what was read and rejected for the
affine index maps and mapped ranges) and `documentation_api_review.md` (API gaps found while
writing the guide).
