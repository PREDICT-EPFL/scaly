# Typing playground

Interface sketches for Scaly's typed API: functions over declared pytrees, templates that
instantiate them per shape, derivative wrappers, problems and solvers. Nothing here lowers or runs.
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
| `trees.py` | `Tree`, `L`, `G`; flatten, unflatten and structure checks; the inferred tree of the bare mode. |
| `function.py` | `Function`, `@function`, the derivative wrappers, a candidate `vmap`. |
| `templates.py` | `FunctionTemplate`, `@template`, `lift`, and the wrappers' template overloads. |
| `opti.py` | `ProblemSpec`, `Problem`, `@problem`, `solver`, the QP gate, `qp_problem`. |
| `tests/definitions.py` | The concrete functions, templates, problems and solvers every test uses. |

## What the design guarantees

- **The declared structure is the source of truth.** Nothing is inferred from the body's signature.
  ty checks that decorator and body agree, and rejects calls with the wrong structure, count or
  leaf kind on both the symbolic and the numerical side (`test_typing.py`, "functions").
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
  is checked against the trace. Wrong output count or shape fails at the decorator.
- **Derivatives keep the source's inputs and are therefore typed.** `gradient` is
  `Function[SI, NI, Expr, Buffer]`. Seeded modes pair the inputs with the new group: `forward` and
  `adjoint` with a seed, `lagrangian_hessian` with multipliers typed as the source's output tree.
- **A problem is a `ProblemSpec` of expressions** (cost, equality groups, bounded inequality groups,
  box bounds shaped like the variables) traced over declared `vars` and `params`. Multi-block
  variables are a grouped tree, and the solution and warm start have that structure.
- **A solver is a plain `Function`** over `(vars_init, lam_box0, lam_eq0, lam_ineq0, params)`
  returning `(vars, lam_box, lam_eq, lam_ineq)`. Multipliers are always present, size 0 when the
  category is absent, so the signature never depends on the problem.
- **A QP backend is gated by a structural proof** that the cost is quadratic and the constraints
  affine in the variables. An NLP backend takes any spec.
- **Shape mismatches at call time are runtime errors.** The type system checks structure, not
  shape. That was accepted from the start and is what makes templates possible.

## Why a wrapper class exists

Every user would rather write `("x", 3)` than `L("x", 3)`. The wrapper is not storing metadata; it
is the type-level map from the `Expr` structure to the `Buffer` structure, done by brute force:
`G`'s eight overloads spell out, width by width, that grouping `Tree[SA, NA]` and `Tree[SB, NB]`
gives `Tree[tuple[SA, SB], tuple[NA, NB]]`. Python's type system has no other way to express it:

- A bare `("x", 3)` literal is `tuple[str, int]` whatever classes exist.
- `typing.Annotated[T, meta]` is erased to `T` by every checker; the metadata survives only at run
  time, which is what FastAPI and jaxtyping use. It could replace `L` for the runtime half and
  contributes nothing to the static half. The stubs nanobind emits for PIQP,
  `Annotated[NDArray[float64], "[m, 1]"]`, are documentation, and they are documentation because the
  type system has nowhere to hold a shape.
- `TypeVarTuple` captures a heterogeneous tuple but has no `Map` operator to transform it; one was
  discussed for PEP 646 and never landed. `ParamSpec` is equally opaque.
- Checker plugins could do the map. ty and pyright have none by design.

So the design space is three points: type only the symbolic side (numerical calls become `Any`),
make the user write both structures, or carry both in a wrapper with a width ladder. This is the
third. The lever for the notation is therefore ceremony, not existence: better names than `L` and
`G`, the outer group folded into the decorator's positional arguments, a defaulted output
declaration. Those are open (see below); the wrapper is not.

## Templates

A **template** is a body plus a declaration with shape holes. It instantiates one concrete
`Function` per combination of leaf shapes, cached on the template, with the signature mangled into
the instance name (`cost__30x40_5`; `s` is a scalar). `template.instances` is the list of
concrete functions that reach C, and `instantiate(shapes)` is the explicit ahead-of-time entry.

Instantiation happens wherever shapes first appear: a numerical call, a symbolic call inside
another traced body (`caller` in `tests/definitions.py` instantiates `cost__2_2` while it is
itself being decorated), or eagerly at the decorator when the declaration has no holes, so a fully
shaped template keeps today's fail-early behavior. Which shape combinations are valid needs no
machinery: an instantiation whose body raises during the trace fails at that first call.

The static guarantees are unchanged because shapes were never part of the static types.
`FunctionTemplate[SI, NI, SO, NO]` has `Function`'s four variables and its calls typecheck
identically (`test_typing.py`, "templates").

**Purely additive.** `Function` is exactly today's class, untouched. Partiality lives in two places:
`Ellipsis` inside `Tree.decls`, which output trees have always used and `with_shapes` has always
resolved, and the bare mode's absent declaration on the template. No `| None` reaches `Tree`,
`Function`, tracing or code generation. In the real library the only signature change is `L`'s
shape argument becoming optional.

**Transforms over templates.** Two kinds of consumer, two costs:

- Anything that *calls* a function symbolically needs no support at all. Composition inside a
  traced body resolves the template at trace time, as `caller` shows.
- Anything that *inspects* the `Function` object (derivative wrappers, `vmap`, the solver
  descriptor) needs to be lifted. `lift(source, inputs, outputs, transform)` is the one mechanism:
  the declared trees give the static types, with holes where the source has them, and `transform`
  runs per instantiation on the source's instance. With it, every wrapper gains a six-line
  overload: `gradient`, `hessian`, `forward` and `lagrangian_hessian` are in `templates.py`, and
  `jacobian`, `adjoint` and the sparse forms follow the same pattern. Seeded modes extend the
  input tree with a hole for the seed (`G(inputs, L("fwd:x"))`) and `lift` checks the bound shapes
  against what the transform produced. Name checks stay eager because names need no shapes;
  checks that need shapes, such as "gradient needs a scalar output", move to instantiation.
- `vmap` is the exception: the real one builds a node from the callee's leaf sizes and never calls
  it, and the per-iteration shape is the very thing it maps over. A template callee is
  instantiated explicitly, `vmap(t.instantiate(shapes), n)`, and that is the right spelling.

**The bare mode.** `@template()` with no declaration reads structure, leaf shapes and auto names
(`in0`, `out0`) from the first call, and types as `FunctionTemplate[Any, Any, Any, Any]`: the
notation-free spelling, with every static guarantee given up explicitly rather than by bending the
typing rules. Its rules:

- Leaves must be exactly `Expr` or `np.ndarray`, and every tuple is structure. Array-likes are
  refused with a `TypeError` that says how to fix it, not a warning: `(a, b)` could be two leaves
  or one vector, and `[[1, 2], [3, 4]]` a matrix or two rows, so there is no correct guess. The
  declared form has no such ambiguity, since the declaration says which tuples are structure, and
  may keep coercing array-likes as `flatten_numerical` does today.
- Each distinct call structure is its own instance (the cache key is the nested shape skeleton),
  so a bare helper may be called with different structures. It has no declared names, so it cannot
  be differentiated or instantiated by shape; declare the trees for that.
- In practice bare templates are helper libraries called symbolically inside traced bodies, where
  every leaf is an `Expr` and the ambiguity above never arises.

**Deferred from the original proposal**, and why:

- *Overload sets under one name.* The proposal collects several bodies registered under the same
  string (`def cost(inputs)` and `def cost(x, y)`) and, at a call, picks "the most defined
  template that matches". That is what C++ calls an overload set with overload resolution, or what
  Julia calls multiple dispatch. It needs a registry keyed by name across modules, which is the
  global-cache trap, and a partial order over declarations: if one template fixes `x`'s shape and
  another fixes `p`'s, a call matching both is ambiguous and needs tie-break rules. One template
  per Python binding covers the same body at any shape, which is most of the value, and the module
  namespace is already the registry. Dispatch can be layered on top of templates later without
  changing them.
- *Parameterized constraints on holes*, such as "a matrix with 5 rows" or a predicate over shapes.
  They belong in the leaf declaration, where `ShapeDecl` would grow a constraint form; nothing
  here blocks them.
- *Multi-parameter bodies* (`def cost(x, y)`). The single-tree calling convention is what makes the
  twin symbolic/numerical typing work; a second convention forks every call surface.

## Mapping to `src/scaly`

Every name here is a stand-in for machinery that exists and must be kept, with its interface
swapped:

- `Buffer` is `np.ndarray`; there is no new runtime type. `Expr` is `scaly.Expr`, and `degree`
  stands in for the structural dependency analysis `ad/sparsity.py`'s `_jac_mask` already does.
  The QP proof must be written on `_jac_mask` over the real gradient and Hessian expressions.
- `Function.__init__` traces the body as `@sc.function` does; `symbolic_call` is today's `CALL`
  node, `numerical_call` today's JIT call. `function/tree.py` already stores `Ellipsis` decls and
  resolves them with `with_types`; `L` needs its shape argument made optional and templates need
  nothing else from it. The bare mode's inferred trees are `flat_tree`'s job.
- The derivative wrappers are the existing ones in `function/api.py` with the specs in
  `function/factory.py`, typed as here, with `extra_inputs` gone because the source's whole input
  tree is kept. Template overloads wrap them through `lift`.
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

The Function and solver API over declared trees is implemented in `src/scaly`. Templates and the typed
`vmap` candidate remain sketches. Follow-up work is tracked as API-1, API-2 and API-4 in
[`internal/todo.md`](../internal/todo.md), with the production design in
[`internal/notes/core_compiler_roadmap.md`](../internal/notes/core_compiler_roadmap.md#signatures-and-templates).

- **Names for `L` and `G`.** They carry no meaning to a reader who did not design them; `leaf` and
  `group` are the honest pytree words. Verbosity is a separate complaint and is not fixed by
  renaming: fold the outer group into `function(*inputs, outputs=...)` by moving the width ladder
  onto the decorator, and default the output declaration to one traced leaf named after the
  function.
- **Mangling flattens structure.** Two nestings with the same flat shapes would collide in C symbol
  names; the real scheme must encode the nesting.
- **Tensor metadata is absent from specialization.** Define how dtype and differentiability bind
  when a shape is inferred. Cache keys and C names must distinguish metadata that changes the
  concrete graph, and lifted seed trees must match the generated Function inputs.
- **The bare mode traces twice**, once to learn the output structure and once in `Function`. Trace
  once.
- **Error vocabulary.** Instantiation-time shape mismatches surface as `TypeError` from `resolved`;
  the real one should use the call-site `ValueError` vocabulary.
- **`vmap` typed by its callee's trees** is one of the two open problems in
  `internal/notes/refactorings.md`. `function.py`'s `vmap` is a candidate answer, every leaf
  gaining a leading axis with the structure preserved, not the real slicing semantics.
- **Constraints on holes** and **overload sets**, above, when a need shows.
