# Function templates and multi-parameter functions: implementation plan (2026-09-27, v2)

Status: **done** on `claude/function-templates`: P1a to P4 and P6, P5 deferred as todo API-77.
Summary: `templates_summary_report.html`; one report per phase beside it. Implements todo API-1 (and unblocks API-4, part of
API-3). Supersedes the "Function templates" section of `refactorings.md` and the "Deferred:
multi-parameter bodies" and "fold the outer group into the decorator" items of
`typing_playground/README.md`. v1 was reviewed by five agents (typing, frontend semantics, compiler
consumers, migration, user-facing API); §10 lists what changed and why. The step-by-step list is §5.

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
@sc.function                                   # nothing declared: every property bound at the call
def f(A, B, x):
  ...

@sc.function((NX, NX), (NX, None), NX, output="y")   # partial: B's second dimension is a hole
def f(A, B, x):
  ...

@sc.function((NX, NX), (NX, NU), NX, output="y")     # fully declared: a ConcreteFunction, traced now
def f(A, B, x):
  ...

f(A_np, B_np, x_np)                            # numerical: instantiate (cached), compile, run
f(A_sym, B_sym, x_sym)                         # symbolic: instantiate, CALL node
```

A slot is a *tree spec*: a `Tree` (`sc.L`, `sc.G`, `sc.S`), or shorthand for one leaf: a shape
(`3`, `(NX, NU)`, `(NX, None)`, `...`, a `TensorType`) or a name (`"y"`, a leaf with a shape hole).
`sc.L((NX, NX))` and `(NX, NX)` are the same slot.

Decisions (settled with Colin on 2026-09-27, unchanged by the review):

1. **One decorator, `sc.function`.** Every decorated function is a *template*. A declaration without
   input holes is instantiated eagerly at the decorator and reports `f.is_concrete == True`;
   `f.concrete` is then its single concrete instance. There is no `sc.concrete_function` and no
   `sc.template`.
2. **One body, many instances.** One Python body instantiated per argument signature, cached on the
   template, each instance given a deterministic unique name. Overload sets stay deferred.
3. **One declaration slot per Python parameter**, `sc.function(slot_1, …, slot_n, output=…)`. Leaf
   names default to the parameter names. The old single-tree convention is the one-slot case.
4. Hard break, no deprecation shim: the whole tree is migrated in the PR that changes the signature.

## 2. The multi-input format

Grouping was never part of the C signature: leaves are flattened in order, so `f(a, b)` and the old
`f((a, b))` lower to byte-identical C. Lowering, AD, codegen and the JIT see flat leaves and do not
change. The complexities are in the frontend:

| Issue | Resolution |
| --- | --- |
| **Static typing.** A variadic call needs the symbolic and numerical parameter lists as separate type variables. | `Function[**PS, **PN, SO, NO]`, `__call__` overloaded numerical (`PN`) first, then symbolic (`PS`). A decorator width ladder (0..8 slots) fills them; more than 8 slots hits an untyped catch-all. **Verified on the pinned ty 0.0.77** (and identical on 0.0.75 and 0.0.84; no bump). |
| **Shorthand slots** (`@sc.function(3, (NX, NU))`) have no static tree type. | Slot parameters are `Tree[SA, NA] \| Spec` with `SA`/`NA` `TypeVar`s defaulting to `Expr`/`ndarray` (`typing_extensions.TypeVar(..., default=...)` under `TYPE_CHECKING`; the project floor is 3.12). A shape or name slot is then typed as a leaf. Verified on ty 0.0.77. |
| **Seeded derivatives append a parameter.** | `forward`, `adjoint`, `lagrangian_hessian`, `sparse_lagrangian_hessian` get an arity ladder. Verified. |
| **Derived call conventions change.** `fwd((inputs, seed))` → `fwd(*inputs, seed)`; `solver((v0, lam_box0, lam_eq0, lam_ineq0, p))` → `solver(v0, lam_box0, lam_eq0, lam_ineq0, p)`. | Part of P1b-ii: 118 call sites (72 run by the suite, 46 only in docs, examples/casadi, benchmarks and `tests/typing`), one rule: strip one level of parentheses from a tuple literal, otherwise splat with `*`. |
| **Calls built from a function's own tree** (`f(f.input_tree.unflatten(flat))`). | The input tree is now a parameter list and `unflatten` returns the argument tuple: `f(*f.input_tree.unflatten(flat))`. 10 sites plus 2 zero-input `fn(())` → `fn()`. |
| **Zero-argument functions.** `f()` has no leaf to dispatch on. | `f()` is always numerical; `f.symbolic_call()` is the symbolic spelling. This matches the static types (a zero-argument call types as numerical) and needs no trace-depth context variable. |
| **The old two-positional form is silently reinterpreted.** | The decorator checks slot count against the body's positional parameters and raises: "`sc.function` takes one declaration per parameter and `output=`; did you mean `sc.function(<IN>, output=<OUT>)`?" |
| **Keyword calls** `f(A=…, B=…)`. | Bound at run time through the body's `inspect.Signature`, only when keywords are present (binding costs ~0.6-1 µs, a third of a trivial call). Statically, keywords are errors on declared functions (the ladder gives positional-only lists) and typed on a bare function's `symbolic_call`. Functions without a body (`_from_exprs`, derived, solvers) are positional-only. |
| **Bodies with defaults, `*args`, `**kwargs`, keyword-only parameters.** | Refused at decoration in v1. Defaults are reserved for static arguments. |

## 3. Object model

**`ConcreteFunction` is a subclass of `Function`.** Both live in `function/model.py` (a
`template.py` beside it would import `model.py` for `ConcreteFunction` while `model.py` imports it for
the base class: a cycle).

- **`Function[**PS, **PN, SO, NO]`**: the template. Owns the body (or, for a derived template, a
  source and a transform), the per-slot declaration with holes (or none, in bare mode), the output
  declaration (or none), the body signature, the device, a private `key -> ConcreteFunction` cache
  and the public `instances: dict[str, ConcreteFunction]`. `sc.Function` names it.
- **`ConcreteFunction(Function)`**: today's class. What the compiler consumes. `is_concrete` is
  `True`, `concrete` is `self`, `instances` is `{name: self}`, `instantiate(...)` checks and returns
  `self`.

A fully declared `@sc.function(...)` returns the `ConcreteFunction` itself, as do `_from_exprs`,
`sc.solver`, and every derivative wrapper given a concrete source. So `isinstance(x, sc.Function)`,
`.descriptor`, `._flat_*`, `.factory` and the 618 `sc.Function._from_exprs` lines outside `src/` keep
working with no wrapper, no delegation table and no extra frame on the call path. `src/` never holds a
template: everything it builds comes from concrete sources.

| Member | On a template | On a `ConcreteFunction` |
| --- | --- | --- |
| `name` | template name | instance name |
| `is_concrete` | `False` | `True` |
| `concrete` | raises `NotConcrete` listing the holes and the `instantiate` fix | `self` |
| `instantiate(*specs)` | bind holes, trace (cached), return the instance | check the specs, return `self` |
| `instances` | `{name: instance}` built so far | `{name: self}` |
| `__call__`, `symbolic_call`, `numerical_call` | resolve the instance from the arguments, forward | today's, variadic |

`NotConcrete(TypeError)`: not an `AttributeError`, which `hasattr` duck typing would swallow.

**`instantiate` takes a declaration per slot**, as the decorator does: a tree spec (a shape, a
`TensorType`, a hole-free `sc.L`/`sc.G`/`sc.S`), or an example value (`ndarray`, `Expr`,
`SparseMatrix`). A tuple of ints is a shape; any other tuple is structure.

Public entry points that *inspect* a callee call `.concrete` after their type check: `vmap`, `scan`,
`while_loop`, `custom_derivative` (the function and its rules), `lower_function`, the five
`scaly.codegen.aot` renderers and the CLI, `render_program_c_source`, `render_expr_assembly` (layer
1: duck-typed), and `viz`'s `expr_graph` and recording. A holed template there raises `NotConcrete`
with the fix. `while_loop` instantiates holed `cond`/`body` itself: carry from `init`, the step index,
then `params`, all determined. `scan` and `vmap` do not infer: a slice's size is not its stride.

Import layers: unchanged. `function/model.py` stays at layer 3.

## 4. Semantics in detail

### 4.1 Parameter lists

`ConcreteFunction.input_tree` is always a parameter list: `params(*slots) = _G(slots, public=False)`
(no new class; `_G` already has width 0 and 1 with `public=False`). `__init__` calls
`fn(*params.symbols())`. `unflatten` returns the argument tuple. The flat `input_names`, `inputs` and
C signature are unchanged.

- `_from_exprs` keeps today's convention: 0 inputs → no parameters, 1 → one leaf, n → one private
  group slot. Existing `fn(x)` and `fn((a, b))` calls on such functions are unchanged; `fn(())` →
  `fn()`.
- NLP oracles rebuilt with `_with_trees` (`solvers/nlp.py`) keep one private group slot.
- The solver has five slots, built once in `descriptor_function`.
- `forward`/`adjoint`: `(*source slots, seed)`; `lagrangian_hessian` and the sparse one:
  `(*source slots, lam)` with `lam` shaped as the output tree.

**Call path.** `ConcreteFunction.__call__(*args, **kwargs)`: bind keywords only if present; dispatch
on leaf kinds across all arguments (none → numerical; mixed → the existing error, which names
`sc.const`); then one loop over `input_tree.parts` (not `_G.flatten_*`, which measured +0.4 µs). An
arity mismatch raises `TypeError("f() takes 2 arguments (x, p), got 3")`. `_jit()` caches the module
after its first import (0.32 µs per call today).

### 4.2 Declarations and holes

`L(*args, dtype=None, diff=True)`, dispatching on the type of the first positional:

| Spelling | Meaning |
| --- | --- |
| `sc.L()`, `sc.L(...)` | unnamed, shape a hole (any rank) |
| `sc.L(3)`, `sc.L((NX, NX))`, `sc.L(())` | unnamed, fixed shape (`()` is a scalar) |
| `sc.L((NX, None))` | fixed rank 2, first dimension fixed, second a hole |
| `sc.L("y")`, `sc.L("y", ...)` | named, shape a hole |
| `sc.L("y", (NX, NX))`, `sc.L("y", TensorType(...))` | today's forms |
| `sc.L(..., dtype="int64", diff=False)` | dtype and differentiability; `dtype=` with a `TensorType` is refused |

`sc.L(None)` is refused (`as_shape(None)` would silently mean a scalar). A leaf declaration is a
`TensorType` or a `Hole(dims: tuple[int | None, ...] | None, dtype: DType | None, diff: bool)`;
`...` is `Hole(None, None, True)`. `TensorType` never gets `None` dims: it is the IR type.

**Input holes only.** `has_holes` and the name tokens look at the input declaration. Output holes
(`L("y")`, 184 sites) are resolved by the trace as today; they do not make a function a template.
On outputs a `Hole` with `dtype=None` takes the traced dtype. On inputs it takes the argument's
dtype when the argument is an `Expr` and `float64` when it is numerical (numerical values are always
coerced; a symbolic argument keeps its dtype, so an `int64` index reaches a bare helper).

**Leaf naming.** Leaves may be unnamed until the decorator names them: an unnamed leaf in slot `p`
takes the parameter name, unnamed leaves in a group slot take `p_0`, `p_1`, nested `p_0_1`. An
explicit name wins. Unnamed output leaves take the template name the same way (`f`, `f_0`, …).
Default names come from the *template* name, never the instance name, so `of`/`wrt` are the same for
every instance. `_check_unique` runs after naming.

**Binding** (`Tree.bind(value, what) -> Tree`): walks the declaration and the argument together,
checking structure first (a group needs a tuple of its width), then fixed dimensions and declared
dtypes (a `ValueError` in the call-site vocabulary), and returns the tree with every hole resolved to
a `TensorType`. Differentiability comes from the declaration, never the argument.
`_flat_symbolic_call` gains the dtype check it lacks today (its own test).

### 4.3 Bare mode

`@sc.function`, `@sc.function()`, `@sc.function(name=...)` and `@sc.function(output=...)` on a body
with parameters: every slot is a whole hole and each argument's *structure* comes from the call.

- A tuple is structure. A tuple of Python numbers is refused ("pass a list or an array for a vector").
- `Expr` is a leaf with its shape and dtype. A `SymbolicValue` supplies its own leaf declaration
  through a hook (`SparseMatrix` gives `S(name, pattern)`), so `tree.py` does not import `linalg`.
- `ndarray`, NumPy scalars, Python numbers and lists are numerical leaves coerced to `float64`
  arrays; a float, an int, `np.float64` and a 0-d array share one instance. A list containing an
  `Expr` is refused ("use sc.stack"). A SciPy sparse argument is refused in v1 ("declare the slot
  with sc.S(pattern)"): pattern holes are P5.
- A bare body with no parameters has no holes: it is traced at the decorator like any fully declared
  function.

**Outputs.** `output=` is optional everywhere. Omitted, the output tree is built from the traced value
in the same trace (`ConcreteFunction(outputs=None)`): a tuple is structure, a leaf is `L`, a
`SymbolicValue` uses its hook. A single leaf is named after the template, tuple leaves `{name}_0`,
`{name}_1`, nested `{name}_0_1`. `output="y"` is shorthand for `sc.L("y")`.

### 4.4 Specialization key and instance names

The key is the bound input declaration: its flat resolved types (shape, dtype, diff) and sparsity
patterns, plus, in bare mode, the nested structure. Keying on resolved types rather than on the raw
arguments means a symbolic and a numerical call that bind the same way share one instance.

Instance names are deterministic valid C identifiers:

- **No input holes → the template name.** Fully declared functions keep today's C symbols; the C
  snapshots are the gate.
- Otherwise `"{name}__" + "_".join(tokens)`, one token per hole leaf: the hole dimensions joined by
  `x` (only the `None` dimensions of a partial shape: `(NX, None)` bound to `(4, 7)` gives `7`), `s`
  for a scalar, then the dtype name when a dtype hole bound to something other than `float64`
  (`sint64`). Bare mode appends `t{6 hex}` of the structure when any argument is a tuple. Tokens have
  no `_`, so they cannot form `_grad_` and friends. No length fallback.
- The template keeps `name -> key`; two keys giving one name raise. Creation-order suffixes are
  forbidden.
- Derived instances are named by the transform from the source instance (`cost__3_s_grad_f_x`);
  with `name=` given, `{name}__{source tokens}`.
- Digests use `hashlib` over canonical bytes (int64 `indptr`/`indices`), never `hash()`.

`__` is reserved in C++; clang warns only under `-Wreserved-identifier`. Kept, documented.
Lowering's name-clash refusal (`_check_function_names`) keys on `c_ident(name)`, since `f:_3` and
`f__3` spell one C symbol today and die in the C compiler instead of with a clear error.

### 4.5 Derivatives

- **Name arguments.** `sc.gradient(f, of, wrt)` stays. One positional string is `wrt`
  (`sc.gradient(f, "x")`, mirroring the `Expr` form); `of` defaults when the function has one output,
  `wrt` when it has one input leaf. Naming an output as `wrt` raises with both spellings.
- **Over templates.** Given a concrete source, the wrappers build as today and return a
  `ConcreteFunction`. Given a template, they return a derived template whose instance for a call is
  `transform(source instance)`, keyed by the source instance: the seed and multiplier slots are
  determined by the source instance and need no holes of their own. Name checks stay eager where the
  names are declared; shape checks (scalar output for `gradient`) happen at instantiation. Nothing is
  built until the derived template is called or instantiated.
- **`custom_derivative`** lifts the same way; `jvp`/`vjp` rules may be templates and are instantiated
  at the matching shapes.

### 4.6 C and C++ names

Parameter names become C names by default. `c_ident` reserves the C and C++ keywords (`new`,
`default`, `this`, …) as it reserves the ABI's parameter names. A sparse output named like its
function breaks the `.hpp` today (`_sparse_namespace` uses the raw name inside `namespace f`); it
uses the output's buffer identifier instead. `tests/codegen/test_name_clash.py` covers a dense and a
sparse self-named output in C and C++.

## 5. Phases and steps

One commit per phase on `claude/function-templates`. Each phase:

- `uv run pytest -n=auto`, `uv run ruff format`, `uv run ruff check`, `uv run ty check --error-on-warning`:
  no new diagnostics against the 58 already there (49 in tracked notebooks and examples);
- `uv run pytest typing_playground`;
- the node-ID baseline regenerated whenever tests are added or renamed;
- mutation checks on each new behavior (break it, see a test fail, restore);
- the trivial-call timing A/B against the previous phase (`internal/notes/perf_2026_09_27_templates/`);
- a short HTML report `internal/notes/templates_pN_report.html`, the API-1 entry in `todo.md` updated.

Never cite a branch commit hash. Colin's local `tiny_qp` example and its gallery test are left alone
and deselected; after P1b-ii its solver call needs the five-argument form.

### P1a: rename `Function` → `ConcreteFunction` (no behavior change)

1. Word-boundary rename over tracked `src/` and `plugins/*/src` files; read the diff so prose meaning
   "a function" and CasADi's `ca.Function` stay. Change the string at `ir/text.py:54`.
2. `Function = ConcreteFunction` alias in `function/model.py`, `function/__init__.py`,
   `scaly/__init__.py` (`test_import_boundaries.py` imports it from `model`).
3. Gates: suite, C snapshots unchanged, import layering.

### P1b-i: the decorator signature, one slot

1. `function(*slots, output, name=None)`: exactly one slot for now; the arity check against the body's
   positional parameters with the migration hint; refuse defaults, `*args`, `**kwargs`, keyword-only.
2. Codemod (ast positions, UTF-8 byte offsets): `sc.function(IN, OUT, ...)` → `sc.function(IN,
   output=OUT, ...)` over tracked `.py`, `.ipynb` code cells and `.md` python fences. 306 sites in 85
   files. Kept in the report, not committed.
3. Hand edits: the bash heredoc in `docs/guide/installation.md`, `def solve_with_ordering(K, m=m)` in
   `sparse_fem_topology.ipynb`, the two direct `sc.Function(...)` constructions in
   `examples/qp_solvers/generated_piqp.py`, the stale constructor in `docs/how_it_works/ir.md`.
4. Tests: the arity refusal and its hint, refused parameter kinds. Typing: `output=` keyword.

### P1b-ii: parameter lists

1. `tree.py`: `params()`; unnamed leaves and `named(base)`; `as_tree(spec)` for shapes, names and
   `TensorType`s in slots, `output=` and `G`.
2. `model.py`: parameter-list input trees; splatting trace; `_from_exprs` convention (§4.1); variadic
   `__call__`/`symbolic_call`/`numerical_call` with the fast path; keyword binding; zero-argument
   calls numerical; `_jit` cached; `_with_trees` takes a parameter list.
3. Typing: `ConcreteFunction[**PS, **PN, SO, NO]`; the 0..8 decorator ladder with spec slots and
   `TypeVar` defaults plus the catch-all; `G` with spec parts; seeded-derivative ladders; solver
   type with five parameters.
4. `api.py`: seeded conventions (§4.1). `solvers/model.py`: five-slot `descriptor_function`, used by
   `qp.py`, `nlp.py`, `plugins/scaly-sqp/src/scaly_sqp/external.py`. `solvers/qp.py`'s zero-parameter
   probe calls `numerical_call` explicitly.
5. `utils/names.py`: C and C++ keywords reserved.
6. Migration: the 118 derived and solver call sites, the 12 tree-built and zero-input calls, the
   four-argument `Function[...]` annotations (`tests/typing`, `benchmarks/problems/npmpc`), the message
   test at `tests/solvers/test_codegen.py:34`, and the docs prose for the changed conventions
   (`guide/derivatives.md`, `guide/solvers.md`, `guide/getting_started.md`, `how_it_works/solvers.md`,
   `README.md`, `docs/index.md`).
7. Tests: multi-slot bodies, width 0 and 8, keywords, arity errors, unnamed-leaf naming, spec slots,
   seeded and solver conventions, reserved keywords. Typing: arities 0, 1, 2, 8; wrong arity; mixed
   kinds; decorator/body mismatch; keywords refused statically; zero-argument `symbolic_call`;
   `forward`/`lagrangian_hessian`/`solver` result types.
8. Extra gates: `uv run benchmarks/run.py smoke`; the example scripts that pytest does not run.

### P2: templates

1. `model.py`: `Function` base (template) and `ConcreteFunction(Function)`; `NotConcrete`;
   `is_concrete`, `concrete`, `instances`, `instantiate`; the decorator returns a `ConcreteFunction`
   when the inputs have no holes; drop the alias; export `sc.ConcreteFunction`, `sc.NotConcrete`.
2. `tree.py`: `Hole`, `None` dimensions, `sc.L()`, `dtype=`/`diff=`, `has_holes`, `bind`.
3. Keys, mangling and the collision guard (§4.4).
4. Consumers (§3): `.concrete` at every inspecting entry; `while_loop` instantiation; the CLI treats
   any `Function` attribute as a function (a `ConcreteFunction` is not a factory) and refuses a holed
   template with the "export `f.instantiate(...)`" hint; `_check_function_names` on `c_ident`;
   `_flat_symbolic_call` dtype check.
5. `docs/dev/codebase.md` (package map line), `docs/api/core.md`, `test_import_boundaries.py` pins.
6. Tests (`tests/function/test_template.py`): instantiate and cache; distinct shapes give distinct
   instances and C symbols; partial dims checked; fully declared is concrete and keeps the bare name;
   symbolic instantiation inside another trace; one trace per instance (count body calls);
   `f(a) is f(a)` and identity hashing; JIT cache hit for the same instance in a fresh process;
   `vmap`/`scan` refuse a holed template with the hint and accept `instantiate(...)`; `while_loop`
   instantiates; `NotConcrete` messages; the collision guard (monkeypatched mangler); the dtype check;
   two instances lower to two procedures; the CLI cases.

### P3: bare mode and inferred outputs

1. The bare forms (§4.3), including `output=` with zero slots.
2. Structure inference from arguments, the `SymbolicValue` hook, the refusals, dtype from `Expr`.
3. Output inference in one trace with default names; `output="y"`.
4. The sparse namespace fix and `test_name_clash` cases (§4.6).
5. Typing: bare overloads (`_Bare` protocol), output-omitted overloads per width (`_Inferred`
   protocol, `NO = Any`).
6. Tests: `def f(A, B)` called with `(4,4)/(4,2)` and `(3,3)/(3,1)`; scalar forms share an instance;
   tuples as structure; tuple returns; traced once; the refusals; a `SparseMatrix` argument.

### P4: lifted transforms

1. Derived templates (§4.5) for all nine wrappers; `of`/`wrt` defaults and the one-string form.
2. `custom_derivative` over templates.
3. Tests: gradient of a template is a template; seeded transforms append the right slot; scalar
   checks at instantiation; compiled results against the concrete source and finite differences;
   instance names with and without `name=`; nothing built until called; `sc.gradient(f, "x")` on a
   bare template.

### P5: sparse pattern holes (deferred)

No test, example or benchmark needs a pattern-polymorphic input. `sc.S()` without a pattern, pattern
binding from `SparseMatrix` and SciPy arguments and the `p{8 hex}` token become a todo item.

### P6: documentation, examples, cleanup

1. Rewrite `docs/guide/functions.md` from simplest to most declared: a first function (bare), symbolic
   and numerical calls, one body many shapes, declaring shapes, partial declarations, names, structured
   arguments and results, dtype and differentiability, sparse matrices, differentiating, composing,
   loop callees, ahead of time, a declaration reference table.
2. `getting_started.md`, `sparsity.md`, `solvers.md`, `index.md`, `installation.md`,
   `how_it_works/{architecture,ir}.md`, `README.md`, `docs/dev/codebase.md`, `docs/api/*`,
   `examples/README.md`.
3. Examples to idiomatic multi-parameter bodies (shape slots; bare where nothing needs declaring),
   keeping every name a CasADi comparison or benchmark depends on.
4. Playground: the README records that multi-parameter bodies and templates landed and why (§2); its
   sketches stay the record of the single-tree design.
5. `refactorings.md` loses its "Function templates" section; `todo.md`: API-1 ticked, API-3 updated
   (dispatch settled), API-4 unblocked, new items for P5, static arguments and `scan` inference.

## 6. Risks and how each is caught

1. **Call overhead.** Fully declared functions are `ConcreteFunction`s: no template frame. Keywords
   are bound only when present. A/B timing per phase.
2. **C symbol drift.** `tests/test_c_snapshot.py` byte-identical through every phase.
3. **Instance explosion.** `f.instances` is public; each instantiation logs its name at debug level.
4. **A template leaking into the compiler.** Every inspecting entry calls `.concrete`; `src/` builds
   only from concrete sources.
5. **ty regressions.** `--error-on-warning` fails on an expected error that disappears.
6. **Codemod misses.** The decorator's arity check fails every miss at import; the suite, the smoke
   benchmark run and the unexercised example scripts cover the rest.

## 7. Files touched

`src/scaly/function/{model,tree,api,sugar,factory,__init__}.py`; `src/scaly/__init__.py`;
`src/scaly/solvers/{solver,qp,nlp,model,problem}.py`; `src/scaly/linalg/{sparse,sparse_factor}.py`;
`src/scaly/solvers/ipm/kkt.py`; `src/scaly/passes/lowering.py`; `src/scaly/codegen/{aot,c,cpp}.py`;
`src/scaly/ir/text.py`; `src/scaly/viz/{graph,recording}.py`; `src/scaly/utils/names.py`;
`plugins/*/src` (rename, `scaly_sqp/external.py`); `tests/test_import_boundaries.py`,
`tests/typing/test_arity.py`, `tests/function/*`, `tests/codegen/test_name_clash.py`; every
`sc.function` site in `tests/`, `examples/` (with notebooks), `benchmarks/`, `plugins/*/tests`,
`docs/`; `typing_playground/README.md`; `internal/{todo.md,notes/refactorings.md}`.

## 8. Evidence

- **Typing** (probes under the session scratchpad, pinned ty 0.0.77, identical on 0.0.75 and 0.0.84):
  typed and rejected calls at arities 0, 1, 2, 8; the decorator ladder with an arity mismatch
  rejected on the decorator's first line; `output: Tree | None = None` does *not* work (SO/NO solve to
  Unknown) and a separate output-omitted overload does; bare `@function`/`@function()` beside the
  ladder; seeded ladders appending `Expr` or the source's output type; `gradient` passing `PS, PN`
  through; `ConcreteFunction[**PS, **PN, SO, NO](Function[PS, PN, SO, NO])`; `TypeVar` defaults for
  spec slots. Two `TypeVarTuple`s on one class are refused, as v1 claimed.
- **Hot path** (M3 Max, Python 3.14): `f(x)` 2.95-3.07 µs (C-100: 2.9). `Signature.bind` on every
  call +0.95 µs; bound only for keywords +0.03 µs. `_jit()`'s deferred import 0.32 µs per call.
- **JIT key**: sha256 over the name, the rendered C and the flags. The rename touches neither;
  deterministic instance names give cross-process hits.
- **Lowering**: procedures keyed by name, invocations by `(name, args)`; two instances lower to two
  procedures, one instance called twice to one. The template cache must return the same object:
  CALL-node interning and AD's caches are identity-keyed.
- **Migration**: 306 decorator sites, all two-positional, all bodies one parameter but one; the
  codemod dry run made 306 insertions, re-parsed everything, and round-tripped every notebook.

## 9. Deferred

- dtype holes on numerical arguments (float32 instances); static arguments (defaults as `static=`,
  JAX's `static_argnums`); the keyword declaration form `@sc.function(A=(NX, NX))`; shape inference for
  `scan`/`vmap` callees; templated `sc.problem`; sparse pattern holes (P5); a solver call defaulting
  the multipliers to zero.

## 10. Review round (2026-09-27): what changed from v1

- **`ConcreteFunction` subclasses `Function`** (frontend, consumers, migration). v1's unrelated
  classes needed `_wrap`, a delegation table missing a dozen attributes read outside `src/`
  (`descriptor` ×55, `_flat_numerical_call` ×111, `_compiled` including a write), `as_concrete` at
  sites v1 did not list, and an extra frame on every call. `template.py` is gone (cycle).
- **No `_Params` class, no `SpecKey` type, no `DerivedFunction` subclass, no 64-character hash
  fallback**: `_G(public=False)`, a resolved-type key, a transform on the template, readable names.
- **Zero-argument calls are numerical**; no `_TRACING` context variable. It disagreed with the static
  type and missed `sc.problem`'s traces.
- **`is_concrete` means no *input* holes.** Output holes are ordinary (184 sites, both C snapshot
  functions).
- **Keyword binding only when keywords are present** (measured +0.95 µs otherwise).
- **Tree specs** (shapes and names as slots, `output="y"`, `G("a", "b")`), **`wrt` as the one
  positional string**, **`wrt`/`of` defaults**, **bare inputs with a declared `output=`**, **default
  tuple output names `{name}_i`** (one naming rule with inputs), **dtype from `Expr` arguments** for
  undeclared leaves, **`while_loop` instantiates its callees** (API review).
- **P1b split in two** (P1b-i mechanical, P1b-ii the convention change); **P5 deferred**; **no P0
  port**: the typing gate was run by the review on the pinned ty. Docs prose for the changed
  conventions moves to P1b-ii.
- **Missed sites added**: tree-built calls, zero-input `fn(())`, `forward`/`adjoint`/Lagrangian
  wrappers (no `isinstance`, so v1's grep missed them), three more codegen entry points, viz, the CLI
  factory rule, `nlp.py`'s derivative results, `sparse_factor.py`'s `custom_derivative`, the sqp
  plugin's `external_nlp`, the sparse C++ namespace, C++ keywords in `c_ident`.
