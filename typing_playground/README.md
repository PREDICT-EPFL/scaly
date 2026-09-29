# Typing playground

Interface sketches for Scaly's typed API: functions over declared pytrees, each realized once per
binding of its shape holes, derivative wrappers, problems and solvers. Nothing here lowers or runs.
`Expr` and `Buffer` are stand-ins and every body fabricates values of the right shape. The package
exists to prove a design at type-check time and at run time before it is implemented in
`src/scaly`, and to keep the reasoning after it lands.

Two checks, and both must stay green:

```sh
uv run pytest typing_playground                              # behavior
uv run ty check --error-on-warning typing_playground         # static: every `ty: ignore` is an expected error
```

The static tests live in `tests/test_typing.py` as an `if TYPE_CHECKING:` block of `assert_type`
calls for the positive cases and `# ty: ignore[<code>]` on the expected errors. With
`--error-on-warning`, an unused ignore fails the check, so an error that stops being one is caught.
That is the "prove the gate can fail" rule applied to types. CI also runs strict type checking over the
package; ruff and pytest reach it through `pyproject.toml`.

## Layout

| Module | Owns |
| --- | --- |
| `expr.py` | `Expr` and `Buffer` stand-ins. `Expr.degree` tracks polynomial degree for the QP gate. |
| `trees.py` | `Tree`, `L`, `G`; parameter lists; flatten, unflatten and structure checks; the inferred tree of the bare mode. |
| `concrete.py` | `ConcreteFunction`, one instance with resolved shapes, and the transforms over one. |
| `function.py` | `Function`, `@function`, `lift`, the derivative wrappers, a candidate `vmap`. |
| `opti.py` | `ProblemSpec`, `Problem`, `@problem`, `solver`, the QP gate, `qp_problem`. |
| `tests/definitions.py` | The functions, problems and solvers every test uses. |

## What the design guarantees

- **One public function type.** Everything a user holds, a declared function, a derivative, a
  solver, is a `Function`: a body plus a declaration that may leave leaf shapes as holes, realized
  as one `ConcreteFunction` per binding. See "Functions and their instances".
- **The declared structure is the source of truth.** Nothing is inferred from the body's signature.
  ty checks that decorator and body agree, and rejects calls with the wrong structure, count or
  leaf kind on both the symbolic and the numerical side (`test_typing.py`, "functions").
- **One declaration per parameter.** `@function(L("x"), L("p"), outputs=L("f"))` declares
  `def cost(x, p)`, called `cost(x, p)`, so the decorator reads like the signature. A group is one
  parameter whose value is a tuple; zero to eight parameters are typed. The input type parameters
  are always tuples of parameter types (see "Parameter lists").
- **Outputs are keyword-only**, so the declaration reads as inputs, then `outputs=`.
- **`f(*args)` dispatches on the leaves**, as `__call__` does in `src/scaly/function/model.py`:
  all-`Expr` arguments build a call node, all-numerical ones evaluate, and a mix is a `TypeError`
  at run time and matches no overload statically. `Tree.is_symbolic` and `is_numerical` are the
  `TypeGuard`s the dispatch narrows with. `symbolic_call` and `numerical_call` stay for when the
  distinction is the point.
- **One structure, two leaf types.** `Tree[Symbolic, Numerical]` is the same nesting over `Expr`
  and over `Buffer`; mixing them in a call is a type error. Typed numerical calls are the seam
  between Scaly and the rest of a user's program and are not negotiable.
- **Grouping is for readability, not signature.** Grouped and flat declarations of the same leaves
  are the same C signature (`test_grouping_is_not_a_signature`).
- **Names are metadata, one per leaf, unique within a tree, never how a value is addressed.** They
  spell the generated header, the sparse tables and the `of`/`wrt` strings of the derivative
  wrappers. The name a body binds is independent of the declared one. Unknown `of`/`wrt` fails
  when the derivative is built, with the declared choices in the message.
- **Output shapes may be holes** (`L("f")` or `L("f", ...)`) and are then traced; a written shape
  is checked against the trace. Wrong output count or shape fails when the instance is traced,
  which for a fully shaped declaration is at the decorator.
- **Derivatives keep the source's parameters and are therefore typed.** `gradient` is
  `Function[SI, NI, Expr, Buffer]`. Seeded modes append one parameter: `forward` and `adjoint` a
  seed, `lagrangian_hessian` the multipliers typed as the source's output tree.
- **A problem is a `ProblemSpec` of expressions** (cost, equality groups, bounded inequality groups,
  box bounds shaped like the variables) traced over declared `vars` and `params`. Multi-block
  variables are a grouped tree, and the solution and warm start have that structure.
- **A solver is a plain `Function`** with parameters `vars_init, lam_box0, lam_eq0, lam_ineq0, params`
  returning `(vars, lam_box, lam_eq, lam_ineq)`. Multipliers are always present, size 0 when the
  category is absent, so the signature never depends on the problem.
- **A QP backend is gated by a structural proof** that the cost is quadratic and the constraints
  affine in the variables. An NLP backend takes any spec.
- **Shape mismatches at call time are runtime errors** (`ValueError`). The type system checks
  structure, not shape. That was accepted from the start and is what makes holes possible.

## Why a wrapper class exists

Every user would rather write `("x", 3)` than `L("x", 3)`. The wrapper is not storing metadata; it
is the type-level map from the `Expr` structure to the `Buffer` structure, done by brute force:
`G`'s eight overloads spell out, width by width, that grouping `Tree[SA, NA]` and `Tree[SB, NB]`
gives `Tree[tuple[SA, SB], tuple[NA, NB]]`, and the decorator's do the same for its parameters.
Python's type system has no other way to express it:

- A bare `("x", 3)` literal is `tuple[str, int]` whatever classes exist.
- `typing.Annotated[T, meta]` is erased to `T` by every checker; the metadata survives only at run
  time, which is what FastAPI and jaxtyping use. It could replace `L` for the runtime half and
  contributes nothing to the static half. The stubs nanobind emits for PIQP,
  `Annotated[NDArray[float64], "[m, 1]"]`, are documentation, and they are documentation because the
  type system has nowhere to hold a shape.
- `TypeVarTuple` captures a heterogeneous tuple but has no `Map` operator to transform it; one was
  discussed for PEP 646 and never landed. `ParamSpec` is equally opaque. What `TypeVarTuple` can
  do is unpack a tuple into parameters and append to it, which is all the parameter lists need
  once a ladder has built both sides ("Parameter lists").
- Checker plugins could do the map. ty and pyright have none by design.

So the design space is three points: type only the symbolic side (numerical calls become `Any`),
make the user write both structures, or carry both in a wrapper with a width ladder. This is the
third. The lever for the notation is therefore ceremony, not existence: better names than `L` and
`G`, a defaulted output declaration. Those are open (see below); the wrapper is not.

## Parameter lists

The decorator takes one tree per parameter, and a ladder of overloads, one per width from 0 to 8,
builds both parameter lists from them:

```python
@overload
def function[SA, NA, SB, NB, SO, NO](a: Tree[SA, NA], b: Tree[SB, NB], /, *, outputs: Tree[SO, NO]) -> Callable[[Callable[[SA, SB], SO]], Function[tuple[SA, SB], tuple[NA, NB], SO, NO]]: ...
```

`Function`'s input type parameters are therefore always tuples, and from there a `TypeVarTuple`
does the rest without any map. The calls unpack them through a typed `self`,
`def numerical_call[*Ns](self: Function[SI, tuple[*Ns], SO, NO], *args: *Ns) -> NO`, and a seeded
mode appends in one overload: `forward` takes `Function[tuple[*Ss], tuple[*Ns], ...]` to
`Function[tuple[*Ss, Expr], tuple[*Ns, Buffer], ...]`. ty, pyright and mypy agree on every
expected error in `test_typing.py`. Two `TypeVarTuple`s in one function signature are fine; PEP 646
refuses two on one class, which is why `Function` keeps plain type variables.

A group is one parameter: `@function(G(L("x"), L("p")), outputs=...)` declares `def f(xp)`. Groups
are never normalized, so `G(L("x"))` is a one-element tuple, not a leaf. With no parameters,
`@function(outputs=...)` is a constant: `f()` evaluates it and `f.symbolic_call()` embeds it,
since an empty call has no leaves to dispatch on. A ninth parameter is a static error; group some.

The `devrush` branch spells the declaration the same way and types it differently
(`src/scaly/function/model.py` and `api.py`): `Function[**PS, **PN, SO, NO]`. A class may take two
`ParamSpec`s, so its calls need no typed `self`. But a `ParamSpec` can only be extended at the front
(`Concatenate`), so each seeded wrapper there repeats a ladder of ten overloads. Here the decorator's
ladder and `G`'s are the only ones. Mixing the two, a `ParamSpec` class matched by `TypeVarTuple`s
(`Function[[*Ss], [*Ns], ...]`), is rejected by pyright and mypy and misread by ty. `ParamSpec`
would add keyword binding by the body's parameter names, but the ladders spell the lists
positionally, so typed keyword calls are lost there too.

## Functions and their instances

A `Function` is a body plus a declaration whose leaf shapes may be holes. It owns one
`ConcreteFunction` per binding of the holes, cached in `instances`, which is everything that reaches
C. Binding happens wherever shapes first appear: a numerical call, a symbolic call inside another
traced body (`caller` in `tests/definitions.py` binds `cost__2_2` while it is itself being
decorated), or `instantiate(shapes)` ahead of time. Which shape combinations are valid needs no
machinery: a binding whose body raises during the trace fails at that first call.

A declaration without holes is the fully formed case, and it needs nothing else: no second
decorator and no flag. Its one instance is traced at the decorator, so it fails early, and it keeps
the function's own name, so its C symbol does not change. `instantiate()` with no arguments returns
it. Instances of a declaration with holes are named with the bound shapes mangled in
(`cost__30x40_5`; `s` is a scalar).

`Function` holds nothing shape-dependent: `inputs` and `outputs` are the declared trees, holes
included. `input_shapes`, `output_shapes` and `c_signature` belong to the instance, and its calls
are typed exactly as `Function`'s. `ConcreteFunction` is not exported and users never build one;
they meet it only through `instances` and `instantiate`, for ahead-of-time export. The instance is
held, not inherited from. Devrush makes `ConcreteFunction` a subclass and a fully declared function
one of them; here every value a user holds has one type, and there is no "is this one concrete"
question.

Shapes were never part of the static types, so the holes cost the typing nothing. A call whose
shapes contradict the declaration is a `ValueError` at the call; `instantiate` with the wrong
shapes is a `TypeError` from the declaration.

**Transforms.** Two kinds of consumer, two costs:

- Anything that *calls* a function symbolically needs no support at all. Composition inside a
  traced body binds the callee at trace time, as `caller` shows.
- Anything that needs an *instance* (derivative wrappers, `vmap`, the solver descriptor) is lifted.
  `lift(source, inputs, outputs, transform)` is the one mechanism: the declared trees give the
  static types, with holes where the source has them, and `transform` runs per binding on the
  source's instance. `source_shapes` maps the derived function's shapes to the source's: a prefix
  for the seeded modes, one axis less for `vmap`. Each wrapper is one signature over `Function`.
  Seeded modes append a parameter declared from the source (`fwd:x` shaped as `x`, `lam:f` as
  `f`), which is a hole only where the source has one, and `lift` checks the bound shapes against
  what the transform produced. Name checks happen when the derivative is built, because names need
  no shapes. So does everything else when the source has no holes, since the derived declaration
  then has none either; otherwise the checks that need shapes, such as "gradient needs a scalar
  output", run at binding.

**The bare mode.** `@function()` with no declaration reads structure, leaf shapes and auto names
(`in0`, `out0`) from the first call. It types as `Function[tuple[*Ss], Any, SO, Any]` with the
symbolic side read off the body's annotations: the arity is checked, leaf kinds only where
annotated (`Unknown` elsewhere), and the numerical side is given up explicitly. Its rules:

- Leaves must be exactly `Expr` or `np.ndarray`, and every tuple is structure. Array-likes are
  refused with a `TypeError` that says how to fix it, not a warning: `(a, b)` could be two leaves
  or one vector, and `[[1, 2], [3, 4]]` a matrix or two rows, so there is no correct guess. The
  declared form has no such ambiguity, since the declaration says which tuples are structure, and
  may keep coercing array-likes as `flatten_numerical` does today.
- Each distinct call structure is its own instance (the cache key is the nested shape skeleton),
  so a bare helper may be called with different structures. It has no declared names, so it cannot
  be transformed or instantiated by shape; declare the trees for that.
- In practice bare functions are helper libraries called symbolically inside traced bodies, where
  every leaf is an `Expr` and the ambiguity above never arises.

**Deferred from the original proposal**, and why:

- *Overload sets under one name.* The proposal collects several bodies registered under the same
  string (`def cost(inputs)` and `def cost(x, y)`) and, at a call, picks "the most defined
  template that matches". That is what C++ calls an overload set with overload resolution, or what
  Julia calls multiple dispatch. It needs a registry keyed by name across modules, which is the
  global-cache trap, and a partial order over declarations: if one declaration fixes `x`'s shape
  and another fixes `p`'s, a call matching both is ambiguous and needs tie-break rules. One
  `Function` per Python binding covers the same body at any shape, which is most of the value, and
  the module namespace is already the registry. Dispatch can be layered on top later.
- *Parameterized constraints on holes*, such as "a matrix with 5 rows" or a predicate over shapes.
  They belong in the leaf declaration, where `ShapeDecl` would grow a constraint form; nothing
  here blocks them.

## Mapping to `src/scaly`

Every name here is a stand-in for machinery that exists and must be kept, with its interface
swapped:

- `Buffer` is `np.ndarray`; there is no new runtime type. `Expr` is `scaly.Expr`, and `degree`
  stands in for the structural dependency analysis `ad/sparsity.py`'s `_jac_mask` already does.
  The QP proof must be written on `_jac_mask` over the real gradient and Hessian expressions.
- `ConcreteFunction` is today's `sc.Function`: its `__init__` traces the body as `@sc.function`
  does, `symbolic_call` is today's `CALL` node and `numerical_call` today's JIT call. Lowering,
  code generation and the CLI keep consuming it. `Function` is the new registry in front of it.
  `function/tree.py` already stores `Ellipsis` decls and resolves them with `with_types`; `L` needs
  its shape argument made optional and nothing else. The bare mode's inferred trees are
  `flat_tree`'s job.
- The per-instance transforms in `concrete.py` are the existing wrappers in `function/api.py` with
  the specs in `function/factory.py`, with `extra_inputs` gone because the source's whole input
  tree is kept. The public wrappers lift them.
- `solver` builds the descriptor and `SOLVER_CALL` nodes exactly as `solvers/nlp.py` and
  `solvers/qp.py` do. `ProblemSpec.eq`/`ineq` groups concatenate into today's `h_eq`/`g_ineq` with
  `l_ineq`/`u_ineq`; `lb`/`ub` become `x_lb`/`x_ub` over the concatenated variable. Multi-block
  `vars` need one new IR utility: substituting each block symbol with a slice of one internal
  decision symbol. `BACKENDS` is the entry-point registry in `solvers/registry.py`; `NotQuadratic`
  is the one new error.
- The mangled instance name becomes the C symbol and the JIT cache key, so the scheme must be
  deterministic and documented. `c_signature` is illustrative; the real header comes from
  `codegen/aot.py` unchanged, with leaf names in tree order.

## Open items

The Function and solver API over declared trees is implemented in `src/scaly` in its single-tree
form. The registry, parameter lists and the typed `vmap` candidate remain sketches. Follow-up work
is tracked as API-1, API-2, API-3 and API-4 in [`internal/todo.md`](../internal/todo.md), with the
production design in
[`internal/notes/core_compiler_roadmap.md`](../internal/notes/core_compiler_roadmap.md#signatures-and-templates).

- **Roadmap API-3** has this declaration, one argument per parameter, but types it as
  `Function[**PS, **PN, SO, NO]` with a ladder on every seeded wrapper; "Parameter lists" is the
  case for tuple parameters and `TypeVarTuple` instead. Decide before API-3 starts.
- **Names for `L` and `G`.** They carry no meaning to a reader who did not design them; `leaf` and
  `group` are the honest pytree words. Verbosity is a separate complaint and is not fixed by
  renaming: default the output declaration to one traced leaf named after the function.
- **Mangling flattens structure.** Two nestings with the same flat shapes would collide in C symbol
  names; the real scheme must encode the nesting.
- **Tensor metadata is absent from the binding.** Define how dtype, differentiability and sparsity
  patterns bind when a shape is inferred. The instance key and C names must distinguish metadata
  that changes the concrete graph, and lifted seed trees must match the generated inputs.
- **The bare mode traces twice**, once to learn the output structure and once in
  `ConcreteFunction`. Trace once.
- **`vmap` typed by its callee's trees** is one of the two open problems in
  `internal/notes/refactorings.md`. `function.py`'s `vmap` is a candidate answer, every leaf
  gaining a leading axis with the structure preserved, not the real slicing semantics.
- **Constraints on holes** and **overload sets**, above, when a need shows.
- **ty's messages on typed-`self` calls** report an arity error as a mismatch of the parameter tuple
  (`Expected tuple[Buffer, Buffer], found tuple[Buffer]`); pyright names the missing argument.
  mypy rejects the implementation signature of the self-typed `__call__` overloads (`misc`); ty
  and pyright accept it.
