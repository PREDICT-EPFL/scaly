# Function templates and multi-parameter functions: implementation plan (2026-09-27)

Status: **plan, not started**. Implements todo API-1 (and unblocks API-4, part of API-3). Supersedes the
"Function templates" section of `refactorings.md` and the "Deferred: multi-parameter bodies" and
"fold the outer group into the decorator" items of `typing_playground/README.md`; those two places keep
the older reasoning, and this note says where it changed.

Read first: `typing_playground/README.md` (section *Templates*), `typing_playground/templates.py`,
`src/scaly/function/{model,tree,api}.py`, `docs/dev/codebase.md` (import layers, *Where to add things*).

## 1. What we are building

Today:

```python
@sc.function(sc.G(sc.L("A", (NX, NX)), sc.L("B", (NX, NU)), sc.L("x", NX)), sc.L("y", NX))
def f(inputs):
  A, B, x = inputs
  ...
f((A_np, B_np, x_np))
```

Target:

```python
@sc.function                               # nothing declared: every property bound at the call
def f(A, B, x):
  ...

@sc.function(sc.L((NX, NX)), sc.L((NX, None)), sc.L(NX), output=sc.L("y", NX))   # partial
def f(A, B, x):
  ...

@sc.function(sc.L((NX, NX)), sc.L((NX, NU)), sc.L(NX), output=sc.L("y", NX))      # fully declared
def f(A, B, x):
  ...

f(A_np, B_np, x_np)                        # numerical: instantiate (cached), compile, run
f(A_sym, B_sym, x_sym)                     # symbolic, inside another trace: instantiate, CALL node
```

Decisions (settled with Colin on 2026-09-27):

1. **One decorator, `sc.function`.** Every decorated function is a *template*. A declaration without
   holes is instantiated eagerly at the decorator and reports `f.is_concrete == True`; `f.concrete` is
   then its single concrete instance. There is no `sc.concrete_function` and no `sc.template`.
2. **One body, many instances.** "Several functions under one name" means one Python body instantiated
   per argument signature, cached on the template object (`f.instances`), each instance given a
   deterministic unique name. Overload sets (several bodies dispatched on argument type) stay deferred,
   for the reasons in the playground README.
3. **One declaration slot per Python parameter**, `sc.function(slot_1, …, slot_n, output=…)`, and the
   body takes `n` parameters. Leaf names default to the parameter names. The old single-tree
   convention is exactly the one-slot case (`sc.function(sc.G(...), output=...)` with `def f(inputs)`),
   so every existing function can be migrated without touching its body or its call sites.
4. Hard break, no deprecation shim: the whole tree is migrated in the PR that changes the signature.

Defaults chosen here, overridable (see §9): holes cover whole shapes, per-dimension sizes and sparse
patterns; dtype is not a hole in v1; an undeclared single output is named after the function.

## 2. Answer to "does the multi-input format cause complexities?"

It works, and it is cheaper than it looks, because **grouping was never part of the C signature**:
the leaves are flattened in order, so `f(a, b)` and the old `f((a, b))` lower to byte-identical C.
Lowering, AD, codegen and the JIT only ever see flat leaves (`Function.inputs`, `_flat_*_call`) and do
not change. The complexities are all in the frontend:

| Issue | Resolution |
| --- | --- |
| **Static typing.** `Function[SI, NI, SO, NO]` types one tree argument. A variadic call needs the *symbolic* and *numerical* parameter lists as separate type variables; `TypeVarTuple` cannot be mapped, and a class may hold only one. | Make `Function` generic over **two ParamSpecs**: `Function[**PS, **PN, SO, NO]`, `__call__` overloaded on `PN` then `PS`. The decorator's width ladder (0..8 slots) fills them: slot `k` of type `Tree[Sk, Nk]` contributes `Sk` to `PS` and `Nk` to `PN`. **Probed with ty 0.0.84**: typed symbolic and numerical calls, wrong-arity and mixed-kind calls rejected, decorator/body arity mismatch rejected, `gradient` passing `PS, PN` through. Probe files are reproduced in §8.1. |
| **Seeded derivatives append a parameter.** `Concatenate` only prepends. | `forward`, `adjoint`, `lagrangian_hessian`, `sparse_lagrangian_hessian` get an arity ladder (`Function[[SA],[NA],…] -> Function[[SA, Expr],[NA, ndarray],…]`, …). Probed: works in ty. |
| **Derived call conventions change.** `fwd((inputs, seed))` becomes `fwd(*inputs, seed)`; the solver `solver((v0, lam_box0, lam_eq0, lam_ineq0, p))` becomes `solver(v0, lam_box0, lam_eq0, lam_ineq0, p)`. | Part of the migration PR. ~125 call sites of `sc.forward/adjoint/lagrangian_hessian/sparse_lagrangian_hessian/solver` plus the plugin tests' `symbolic_call((...))`. |
| **Zero-argument functions.** `f()` has no leaf to dispatch on. | Symbolic if called while a body is being traced, numerical otherwise (a context variable set around the trace in `ConcreteFunction.__init__`). This also resolves the dispatch half of API-3 (§6.3). |
| **The old two-positional form is silently reinterpreted** (`sc.function(IN, OUT)` would read `OUT` as a second input). | The decorator checks slot count against the body's positional parameters and raises with the migration hint: "`sc.function` takes one declaration per parameter and `output=`; did you mean `sc.function(<IN>, output=<OUT>)`?" |
| **Keyword calls** `f(A=…, B=…)`. | Bound at run time through the body's `inspect.Signature`. The typed ladder gives positional-only parameter lists, so keywords are typed only in the bare mode (where `PS` is the body's own signature). Acceptable. |
| **Bodies with defaults, `*args`, `**kwargs`, keyword-only parameters.** | Refused at decoration in v1. Defaults are reserved for *static arguments* (§9). |
| **Grouped slots still exist.** A parameter may itself be a tuple (`def f(state, params)` with `state = (x, v)`). | Unchanged: a slot is any `Tree`; `sc.G(...)` in a slot means that parameter is a tuple. |

## 3. Object model

Two classes, one of them user-facing:

- **`ConcreteFunction`** — today's `Function`, renamed, with its input tree generalized to a
  parameter list (§4.1). It is what the compiler consumes: CALL nodes, `factory`, AD, lowering,
  codegen, the JIT, solvers. Never has holes. Keeps `_from_exprs`, `_flat_symbolic_call`,
  `_flat_numerical_call`, `_with_trees`, `_with_outputs`, `custom_jvp/vjp/sparsity`.
- **`Function`** — the new user-facing template (the playground's `FunctionTemplate`, renamed). Owns
  the body, the per-slot declarations (possibly with holes), the output declaration (possibly
  absent), `instances: dict[SpecKey, ConcreteFunction]`, the instance-name map, and for derived
  templates a `(source, transform)` pair. `sc.Function` names this class.

Common protocol on `Function`:

| Member | Meaning |
| --- | --- |
| `name` | template name (the bare function name, or `name=`) |
| `is_concrete` | declaration has no holes (then exactly one instance exists, built at the decorator) |
| `concrete` | that instance; raises `NotConcrete` listing the holes otherwise |
| `instantiate(*args, **kwargs)` | bind holes from example arguments — `Expr`, `SparseMatrix`, arrays, **or** `TensorType`/shape declarations — trace (cached), return the `ConcreteFunction`. The explicit AOT entry. |
| `instances` | the cache; everything in it reaches C when used |
| `__call__`, `symbolic_call`, `numerical_call` | as today, but variadic (§4.3) |
| `input_names`, `output_names`, `inputs`, `outputs`, `input_shapes`, `output_shapes`, `factory`, `with_device`, `device`, `output_sparsities`, `solver_stats`, `recompile` | explicit delegating properties/methods to `concrete`, so a fully declared `Function` reads exactly like today's. `recompile` on a template recompiles every instance. |

One helper, **`as_concrete(fn) -> ConcreteFunction`** in `function/template.py`, accepts either class
and raises `NotConcrete` with the fix (`f.instantiate(...)`) for a template with holes. Every consumer
that *inspects* a callee calls it: `sugar.{vmap,scan,while_loop,custom_derivative}`, the derivative
wrappers for a concrete source, `codegen.aot.{render_c_module,write_module}` and its CLI target,
`solvers` (problem oracles are built from `ConcreteFunction` already), `linalg/sparse_factor.py`,
`solvers/ipm/kkt.py`. Consumers that only *call* a function need nothing: a symbolic call inside a
traced body instantiates at trace time.

`Function._from_exprs(...)` stays available on the public class as a thin wrapper
(`Function._wrap(ConcreteFunction._from_exprs(...))`), because 658 test/example lines use
`sc.Function._from_exprs` and should not churn. `src/` uses `ConcreteFunction._from_exprs` directly.
`_from_exprs` keeps today's tree shape: 0 inputs → no parameters, 1 → one `L` slot, n → one private
group slot; so existing `fn((a, b))` calls on such functions are unchanged.

Import layers: `function/template.py` (the `Function` class, `SpecKey`, mangling, `as_concrete`,
`NotConcrete`, the `DerivedFunction` subclass taking a transform callable) at **layer 3** beside
`model.py`, so `sugar` (4) and `api`/`solvers`/`linalg` (5) can import it. The lifted wrappers live in
`api.py` (5) and pass transforms *down* to `DerivedFunction`, so no upward import is needed. Add the
`IMPORT_LAYERS` entry and module docstring; `tests/test_import_boundaries.py` pins `sc.Function`,
`sc.ConcreteFunction`, `sc.NotConcrete`.

## 4. Semantics in detail

### 4.1 Parameter-list trees

A new private tree `_Params(parts, names)` in `function/tree.py`: structurally `_G(public=False)`
(flat names/decls in order, `with_types`, `infer`, `sparsities`, `relabel`) with two differences:
`symbols()` returns the tuple the body is *splatted* with, and it records the Python parameter names
for keyword binding. `ConcreteFunction.input_tree` is always a `_Params` (possibly of width 0 or 1).
`ConcreteFunction.__init__` calls `fn(*symbols)` instead of `fn(symbols)`. The flattened
`input_names`/`inputs` and hence the C signature are unchanged.

### 4.2 Declarations and holes

`L` gets optional arguments; dispatch on the type of the first positional:

| Spelling | Meaning |
| --- | --- |
| `sc.L()` | name from the parameter, shape a hole (any rank) |
| `sc.L(3)`, `sc.L((NX, NX))` | name from the parameter, fixed shape |
| `sc.L((NX, None))` | fixed rank 2, first dimension fixed, second a hole |
| `sc.L("y")`, `sc.L("y", ...)` | explicit name, shape a hole (today's output spelling keeps working) |
| `sc.L("y", (NX, NX))`, `sc.L("y", TensorType(...))` | today's forms |
| `sc.L(..., dtype=sc.dtypes.int64)` | new keyword; convenience over passing a `TensorType` |

`ShapeDecl` gains `None` entries; `Tree.decls` stores either a `TensorType` or a `ShapeHole(dims:
tuple[int | None, ...] | None, dtype, diff)` in place of today's bare `Ellipsis` (keep accepting
`...` as the whole-shape hole). `Tree.has_holes`, `Tree.bind(values) -> Tree` (resolve holes from
actual leaves, checking fixed dims) are new; `resolved()`/`with_types()` keep their output-side role.

**Leaf naming.** An unnamed `L` in slot `k` takes the parameter name. Unnamed leaves inside a group
slot for parameter `p` are named `p_0`, `p_1`, nested `p_0_1`. An explicit name always wins. Names
must remain unique across all slots (checked by the existing `_check_unique`).

`S` gets the same treatment: `sc.S()` / `sc.S("P")` with no pattern is a **pattern hole** bound from a
`SparseMatrix` (symbolic) or SciPy sparse array (numerical) argument. Phase 5.

**dtype** is not a hole in v1: a leaf's dtype is its declared one, default `float64`. Numerical
arguments are coerced exactly as `L.flatten_numerical` does today. A symbolic argument whose dtype
differs from the declared one is an error; add that check to `_flat_symbolic_call` too, which today
compares shapes only (a latent bug independent of this work — give it its own test).

**Differentiability** is taken from the declaration (default `diff=True`), never from the argument,
and is not part of the key; `_flat_symbolic_call` already derives the call node's `diff` from the
actuals.

### 4.3 Binding arguments, calling

`Function.__call__(*args, **kwargs)`:

1. Bind to the body signature (`inspect.Signature.bind`), giving one value per slot.
2. Dispatch on leaf kind across all slots, as `Tree.is_symbolic`/`is_numerical` do now; mixed is the
   existing error. Zero leaves: symbolic iff `_TRACING.get() > 0`.
3. Resolve the instance: if `is_concrete`, the single instance directly (**no key computation on the
   hot path**; C-100's 2.9 µs trivial call must not regress). Otherwise compute the `SpecKey` from the
   hole leaves only and look it up; on a miss, bind the declaration, build a `ConcreteFunction`, store.
4. Call the instance's `symbolic_call(*values)` / `numerical_call(*values)`.

Structure is checked before shapes (the playground's order). Error vocabulary: a fixed dimension that
does not match is a `ValueError` from the call site (today's `flatten_*` wording), not the
`TypeError` of `resolved()`.

**Bare mode** (`@sc.function` with no parentheses, or `@sc.function()`): every slot is `sc.L()`, except
that the *structure* of each argument is read from the call: a `tuple` is structure, every other
value is a leaf (`Expr`, `SymbolicValue`, `ndarray`, Python scalar, list → `np.asarray`). This
deliberately relaxes the playground's refusal of array-likes: with one slot per parameter, the old
`(a, b)`-is-two-leaves-or-a-vector ambiguity now only arises for tuples, and tuples are always
structure. The output is inferred from the traced value by the same rule. Bare templates key on the
nested skeleton plus leaf shapes, so different structures give different instances.

**Trace once.** `ConcreteFunction.__init__` accepts `outputs=None`, meaning: build the output tree
from the traced value (port `inferred_tree` and `skeleton` from `typing_playground/trees.py` into
`function/tree.py`), with default names (§4.5). This removes the playground's
double trace.

### 4.4 Specialization key and instance names

`SpecKey = tuple` over the *hole* leaves, in flat leaf order, of `(shape, dtype, pattern_digest|None)`,
plus, in bare mode only, the nested skeleton. Anything that changes the concrete graph must be in the
key; anything fixed by the declaration need not be.

Instance name, deterministic and a valid C identifier (`utils/names.c_ident` must accept it unchanged):

- **No holes → the bare template name.** Every fully declared function keeps exactly today's C symbol,
  so `tests/test_c_snapshot.py` and all benchmark/CasADi symbols are untouched.
- Otherwise `"{name}__" + "_".join(tokens)` with one token per hole leaf: dims joined by `x` with
  only the *hole* dims printed for a partial shape (`(NX, None)` bound to `(4, 7)` → `7`), `s` for a
  scalar, `p{8 hex}` for a pattern hole (sha256 of shape, indptr, indices), and later a dtype suffix
  if dtype holes arrive. Bare mode with any tuple argument appends `t{6 hex}` of the skeleton.
- If the result exceeds 64 characters: `"{name}__h{12 hex}"` of the full key.
- The template keeps `name -> key`; two keys producing one name raise (a bug in the scheme, not
  something to paper over with counters). Creation-order suffixes (`f_1`, `f_2`) are forbidden: they
  make C symbols and JIT cache entries depend on call order.

Cross-template name clashes (two templates both called `f`) behave as today: lowering refuses two
different Functions with one name in a graph; `name=` fixes it.

### 4.5 Outputs

`output=` is optional. Omitted, the output is inferred: a single leaf is named **after the function**
(so `sc.gradient(cost, wrt="x")` yields `grad_cost_x`), a tuple result names its leaves `out0, out1, …`
(nested: `out0_1`). A declared output may have holes (`sc.L("y")`) as today; fixed output shapes are
checked against the trace at instantiation (eagerly when the inputs are concrete). Verify with a test
in `tests/codegen/test_name_clash.py` that an output named like its function is fine in both the C and
the C++ header (`namespace f { … f … }`).

### 4.6 Transforms over templates (lifting)

As in the playground: consumers that call need nothing; consumers that inspect need an instance.

- **Derivative wrappers** (`jacobian`, `gradient`, `hessian`, `sparse_jacobian`, `sparse_hessian`,
  `forward`, `adjoint`, `lagrangian_hessian`, `sparse_lagrangian_hessian`): given a concrete source,
  build as today and return `Function._wrap(result)`; given a template with holes, return a
  `DerivedFunction` whose instances are `transform(source.instantiate(<source part of the key>))`.
  Name checks (`of`/`wrt`) stay eager; shape-dependent checks (scalar output for `gradient`) move to
  instantiation. Seed/multiplier slots are holes checked against the generated inputs (the
  playground's `_Derived._build` check). Make `of` optional when the source has one output.
- **`custom_derivative`**: lift the same way; `jvp`/`vjp` rules may themselves be templates and are
  instantiated at the matching shapes.
- **`vmap`, `scan`, `while_loop`**: `as_concrete(callee)`. A template with holes must be instantiated
  explicitly (`f.instantiate(TensorType(...), ...)` or with example arguments). Inferring the callee's
  shapes from `inputs`/`init`/`xs` is possible for `scan` and `while_loop` (carry shape = `init`
  shape; params given) and is a follow-up, not v1. Typed `vmap` stays API-2.
- **`sc.solver`** returns `Function._wrap(concrete)`, typed
  `Function[[SV, SV, Expr, Expr, SP], [NV, NV, ndarray, ndarray, NP], tuple[...], tuple[...]]`.
- **`sc.problem`** is out of scope; its body already takes `(vars, params)` as two parameters, so it
  is consistent with the new convention. Templated problems are a later item.

## 5. Phases (one PR each, in order)

Each PR: `uv run pytest -n=auto`, `uv run ruff format`, `uv run ruff check`, `uv run ty check`,
`uv run ty check --error-on-warning typing_playground`, a report in `internal/notes/` in the style of
the tier reports, and the matching `todo.md` update. Never cite a branch commit hash.

### P0 — Prove the typing in the playground

- Port `typing_playground/{function,templates,trees}.py` to: `Function[**PS, **PN, SO, NO]`,
  `function(*slots, output=)` with the 0..8 width ladder plus the bare overload
  `function(fn: Callable[P, R]) -> Function[P, ..., R, Any]`, arity ladders for seeded transforms,
  one template class with `is_concrete`/`concrete`.
- `tests/test_typing.py`: `assert_type` for symbolic/numerical calls at arities 0, 1, 2, 8; expected
  errors (`# ty: ignore[...]`) for wrong arity, mixed kinds, decorator/body arity mismatch, seed of the
  wrong kind; `gradient`/`forward`/`lagrangian_hessian`/`solver` result types.
- Run under the **pinned** ty (`uv run ty`, floor `ty>=0.0.75` in `pyproject.toml`); the probe in §8.1
  passed on 0.0.84. If the floor fails, bump it in this PR.
- Update the playground README: multi-parameter bodies are no longer deferred; say why (§2).
- **Gate:** both playground checks green. Stop and report if ParamSpec pairs do not hold up; the
  fallback is arity-specific classes `Function0..Function8`, which is uglier and must be discussed
  before proceeding.

### P1a — Rename `Function` → `ConcreteFunction` (no behavior change)

- Rename the class in `function/model.py` and every `src/` and `plugins/*/src` reference (≈400 refs in
  47 files; word-boundary rename, then read the diff: prose that means "a function" stays). Keep
  `Function = ConcreteFunction` as a temporary alias in `function/__init__.py` and `scaly/__init__.py`
  so tests are untouched in this PR.
- **Gate:** suite green, C snapshots unchanged, import-layer test green.

### P1b — Parameter lists and the new decorator signature (still no holes on inputs)

- `_Params` tree; `ConcreteFunction.__init__` splats; `__call__`/`symbolic_call`/`numerical_call`
  variadic with keyword binding; `_TRACING` context variable and zero-argument dispatch.
- `ConcreteFunction` generic over `[**PS, **PN, SO, NO]`; retype `api.py` wrappers (ladders for the
  seeded ones), `solvers/{solver,qp,nlp,model}.py`.
- `function(*slots, output=, name=)` in `api.py` with the arity check and migration hint (§2). In this
  PR it still requires fully shaped input slots and returns the concrete object (the template arrives
  in P2); unnamed `L` in a slot takes the parameter name.
- Derived conventions: `forward`/`adjoint` → `[*source slots, seed]`; `lagrangian_hessian`/
  `sparse_lagrangian_hessian` → `[*source slots, lam]` with `lam` shaped as the output tree; solver →
  five slots. Update internal callers: `linalg/sparse_factor.py:70`, `solvers/ipm/kkt.py:{111,272}`,
  `solvers/qp.py:222`, `solvers/nlp.py:{107,155,192}`, `solvers/model.py:127`.
- **Codemod** (a throwaway script, kept in the PR report, not committed): with `ast` positions, rewrite
  every `sc.function(IN, OUT, ...)`/`@function(IN, OUT, ...)` to `sc.function(IN, output=OUT, ...)`.
  Bodies and call sites stay as they are (one group slot == old convention). Then fix the ≈125 derived
  call sites by hand (`fwd(((x, p), seed))` → `fwd((x, p), seed)`, `solver((…))` → `solver(…)`),
  including the plugin tests (`plugins/scaly-{ipopt,sqp,piqp}/tests`). Docs code blocks too (20 in
  `docs/`, plus `README.md`, `examples/README.md`).
- **Gates:** `tests/test_c_snapshot.py` byte-identical (the flat signature did not change); the full
  suite; `tests/typing/test_arity.py` under `ty check --error-on-warning` extended with the arity-0/2/8
  cases; the trivial-call timing within noise of C-100's.

### P2 — Templates: holes, instances, `is_concrete`

- `function/template.py`: `Function` (template), `SpecKey`, mangling, `as_concrete`, `NotConcrete`,
  `DerivedFunction`. `sc.Function` now names the template; drop the P1a alias. `Function._from_exprs`
  and `Function._wrap` as in §3.
- `tree.py`: optional `L` arguments, `ShapeHole`, `None` dims, `has_holes`, `bind`.
- Decorator returns a `Function` always; eager instantiation when no holes.
- `as_concrete` in every inspecting consumer (§3 list; grep `isinstance(.*, Function)` — 13 sites in
  `src/` — and review each: CALL-node callees are always `ConcreteFunction`).
- `codegen/aot.py`: `render_c_module`/`write_module`/CLI accept a concrete `Function`; a template with
  holes is refused with its instance list and the `instantiate` hint.
- Tests (new `tests/function/test_template.py`, ported from `typing_playground/tests/test_templates.py`
  but *compiling and running*): instantiate-and-cache; distinct shapes give distinct instances and
  distinct C symbols; partial dims checked; fully declared is eager and keeps the bare name; symbolic
  instantiation inside another trace (the playground's `caller`); cache reuse across calls does not
  retrace (count body invocations); JIT cache hit on a second process for the same instance name;
  `vmap`/`scan` refuse a holed template with the hint and accept `instantiate(...)`; `NotConcrete`
  messages; name-collision guard (force two keys to one name with a monkeypatched mangler); dtype
  mismatch on a symbolic call is refused (the `_flat_symbolic_call` fix); lowering of a graph calling
  two instances of one template (two procedures, distinct names).
- **Gates:** C snapshots unchanged; suite; ty.

### P3 — Bare mode and inferred outputs

- `@sc.function` / `@sc.function()`; skeleton-keyed instances; the leaf rules of §4.3; output
  inference with default names (§4.5); single trace via `ConcreteFunction(outputs=None)`.
- Tests: the user's example (`def f(A, B)` called with `(4,4)`/`(4,2)` and with `(3,3)`/`(3,1)`),
  scalars and lists as leaves, tuple arguments as structure, tuple returns, body traced exactly once
  per instance, differentiation of a bare template by parameter name (`sc.gradient(f, wrt="x")`), an
  output named after its function in C and C++ (§4.5).

### P4 — Lifted transforms

- All wrappers in `api.py` accept templates via `DerivedFunction`; `of` optional for one output;
  `custom_derivative` over templates.
- Tests: the playground's `test_gradient_of_a_template_is_a_template`,
  `test_seeded_transforms_extend_the_tree_with_holes`, `test_scalar_checks_move_to_instantiation`,
  compiled and compared against the concrete-source result and a finite difference; instance names
  (`cost__3_s_grad_cost_x`); deriving builds nothing until called.

### P5 — Sparse pattern holes

- `sc.S()` without a pattern on inputs; pattern binding from `SparseMatrix`/SciPy arguments; `p{hex}`
  token; the existing exact-pattern refusal applies after binding.
- Tests: two patterns → two instances; same pattern with different explicit-zero layout is a
  different pattern (as `S` already treats it); a structural-sparsity derivative over a pattern-hole
  template.

### P6 — Documentation, examples, cleanup

- Rewrite `docs/guide/functions.md` (*Declare a function*, *Symbolic and numerical calls*, *A single
  leaf is unpacked*, *Grouping and the C signature*, *Compose functions*) around the new decorator,
  with a new *Templates and instances* section; update `getting_started.md`, `sparsity.md`,
  `solvers.md`, `index.md`, `how_it_works/{architecture,ir}.md`, `README.md`, `docs/dev/codebase.md`
  (package map, import-layer table, *Where to add things*).
- Convert `examples/` to idiomatic multi-parameter bodies (bare mode where nothing needs declaring);
  API-4 (npmpc `FunctionTemplate` example) can then be closed.
- Remove the "Function templates" section of `refactorings.md`; tick API-1 in `todo.md`; update the
  API-3 entry (dispatch solved; the AD zero-input inlining question remains); point the playground
  README's open items here.
- Optional follow-up: with `_from_exprs` results callable as `fn.symbolic_call(*args)` including zero
  arguments, `ad/forward.py` no longer needs the sanctioned `_flat_symbolic_call` seam; retiring it
  (and its entry in `test_flat_call_seams_stay_inside_their_sanctioned_modules`) is its own PR.

## 6. Risks and how each is caught

1. **Call overhead.** Template dispatch in front of every numerical call. Concrete fast path skips key
   computation; measure the trivial call as C-100 did and record the number in the P1b/P2 reports.
2. **C symbol drift.** Any change to fully declared names moves symbols in users' builds. Gate:
   `tests/test_c_snapshot.py` byte-identical through P1a–P2.
3. **Instance explosion.** Bare templates called with many shapes compile many libraries. Expose
   `f.instances`; log (debug level) each new instantiation with its name; no cap in v1.
4. **Zero-argument dispatch** now depends on a context variable. Test a zero-argument function called
   inside a trace, outside, and inside a nested trace; test that an exception in a body resets the
   depth (use a `ContextVar` token and `try/finally`).
5. **ty regressions.** ParamSpec support in ty is recent; the P0 gate uses `--error-on-warning` so an
   expected error that disappears fails the check.
6. **Codemod misses.** After P1b, `rg "sc\.function\([^)]*\)\s*$"`-style checks are insufficient; the
   decorator's arity check is the real net, since every miss fails at import.

## 7. Files touched (checklist)

`src/scaly/function/{model,tree,api,sugar,factory,__init__}.py`, new `function/template.py`;
`src/scaly/__init__.py`; `src/scaly/solvers/{solver,qp,nlp,model,_oracle}.py`;
`src/scaly/linalg/sparse_factor.py`; `src/scaly/solvers/ipm/kkt.py`;
`src/scaly/codegen/aot.py`; `tests/test_import_layering.py` (`IMPORT_LAYERS`),
`tests/test_import_boundaries.py`, `tests/typing/test_arity.py`, `tests/function/*`,
`tests/codegen/test_name_clash.py`; every `@sc.function` site in `tests/`, `examples/`, `benchmarks/`,
`plugins/*/tests`, `typing_playground/tests/definitions.py`, `docs/`; `typing_playground/*`;
`internal/{todo.md,notes/refactorings.md}`.

## 8. Evidence

### 8.1 ty probe (ty 0.0.84, `--python-version 3.12`)

```python
class Function[**PS, **PN, SO, NO]:
  @overload
  def __call__(self, *args: PN.args, **kwargs: PN.kwargs) -> NO: ...
  @overload
  def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> SO: ...

@overload
def function[SA, NA, SB, NB, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], /, *, output: Tree[SO, NO]
) -> Callable[[Callable[[SA, SB], SO]], Function[[SA, SB], [NA, NB], SO, NO]]: ...

@function(L(), L(), output=L())
def f(x: Expr, y: Expr) -> Expr: ...
assert_type(f(Buf(), Buf()), Buf)      # ok
assert_type(f(Expr(), Expr()), Expr)   # ok
f(Buf())                               # error: no matching overload   (expected)
f(Expr(), Buf())                       # error: no matching overload   (expected)

@function(L(), L(), output=L())
def g(x: Expr) -> Expr: ...            # error: argument is incorrect  (expected: arity)

# append a seed: arity ladder works, Concatenate[Expr, PS] (prepend) works too
@overload
def fwd[SA, NA, SB, NB, SO, NO](fn: Function[[SA, SB], [NA, NB], SO, NO]
) -> Function[[SA, SB, Expr], [NA, NB, Buf], Expr, Buf]: ...

def gradient[**PS, **PN](fn: Function[PS, PN, Any, Any], wrt: str) -> Function[PS, PN, Expr, Buf]: ...
# gradient(one)(Buf()) is Buf, (Expr()) is Expr, (Buf(), Buf()) rejected

# zero slots: function(*, output=L()) -> Function[[], [], SO, NO]; zero() is Buf
# bare: function(fn: Callable[P, R]) -> Function[P, ..., R, Any]; calls resolve to Any (numerical
# overload first, PN = ...), i.e. untyped, as the playground's bare mode already accepts.
```

### 8.2 Why this is compatible with the existing design

The playground established that grouping is not signature, that `Ellipsis` decls and `with_types`
already carry output holes, and that templates are purely additive over concrete Functions. Of its
deferred items this plan takes up exactly one, multi-parameter bodies, on the evidence of §8.1: its
objection was that a second calling convention forks every call surface, and with the single-tree
convention subsumed as the one-slot case there is only one convention.

## 9. Open decisions (defaults above; change before P2 if wanted)

- **dtype holes**: bind dtype from the argument (float32 → separate instance) instead of coercing.
  Default: no. Adds a name token and a coercion-versus-specialization rule for Python scalars.
- **Static arguments**: parameters with defaults (or a `static=` marker) taking Python values (a
  horizon `N`, a callable, a flag) that select an instance and are part of the key, as JAX's
  `static_argnums`. Natural next step; v1 refuses defaults so the syntax stays free.
- **Keyword declaration form** `@sc.function(A=(NX, NX))`: declare only what is constrained, by
  parameter name. Cannot be typed on the numerical side; could be added as untyped sugar later.
- **Default output names** for tuple returns (`out0…` versus `{name}_0…`).
- **Shape inference for `scan`/`while_loop` callees** from `init`/`xs`/`params` instead of explicit
  `instantiate`.
- **Templated `sc.problem`** (holes in `vars`/`params`).
