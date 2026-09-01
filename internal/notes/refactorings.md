> Planned work, not a frozen record. One `#` section per refactoring; add new ones alongside
> rather than editing settled ones, and remove a section once it has landed.

# Solver problem construction: typed arity, one `Problem`, one `solver`

Decided 2026-08-26, superseding the 2026-08-19 draft that two sessions left unsettled. Not started.
Tracked as `internal/todo.md` D3. D2, D1, D1.5 and A11 have landed; this is next. The prototype that
settled the design is `arity.py` at the repository root. It is an **interface sketch, not a
reference implementation**: `Buffer` is `np.ndarray`, `Expr.degree` stands in for `_jac_mask`, and
every body (`Function`, the derivative wrappers, `solver`) is a stand-in for machinery that already
exists in `src/alloy` and must be kept, with its interface swapped. Its docstring lists, per name,
what is real and what is stubbed; read that before touching `src/`. Its two test sections are the
acceptance tests of the design and move to `tests/typing/` and `tests/` when it lands; the file goes
then.

## The problem

Unchanged from the draft, in order of how much it hurts:

- **Parameter order is discovered, not declared.** `collect_free_inputs` sorts by `(name, id)`, so
  the call signature of `al.nlp`/`al.qp` depends on alphabetical accident, and `docs/guide/solvers.md`
  has to warn that two same-shaped parameters swapped give a wrong answer instead of an error.
- **No problem/solver split.** The safety filter in `benchmarks/problems/unbumpercars/filters.py`
  and the backend sweeps in `benchmarks/run.py` differentiate the same expressions once per backend.
- **Multi-block variables are index arithmetic.** `[u; s]` is concatenated by hand and the result
  sliced by hand (`out["x"][:n_u]`), with nothing checking the two agree.
- **Nothing is typed.** `Function.__call__` returns "an array or a tuple", the number and shapes of
  inputs are checked at run time only, and a nested `solver.call([...])` takes a positional list.

## What we are changing

### 1. Inputs and outputs are declared pytrees, built from two names

```python
class Tree[Symbolic, Numerical]: ...     # names, shapes in C order; symbols(); relabel()
class L(Tree[Expr, Buffer]): ...         # L("x", 3), L("P", (n, n)), L("f", ...) for a traced shape
def G(a, b, ...) -> Tree[tuple[SA, SB, ...], tuple[NA, NB, ...]]   # width 2..8, overloads; nest beyond
```

The two type parameters are the symbolic and the numeric spelling of the same structure. `L` is
one named tensor (the real one also takes a `TensorType` for dtype and `diff`); `G` groups trees
side by side at any width up to eight and any depth. Grouping is a choice for readability, never a
requirement: `G(G(state, u), G(pw, physics, dt))` and `G(state, u, pw, physics, dt)` are the same
C signature. Both are exported at top level, `al.L` and `al.G`: two short names is a smaller
vocabulary than a family of classes.

`G`'s width overloads are the one ladder in the design, in one place, seven stub lines. It is the
same ladder `tuple[A, B, C]` is built from inside the type system, which we cannot borrow: deriving
the `Buffer` structure from the `Expr` structure needs a type-level map, and Python has none
(PEP 646 left `Map` out). That is also why a bare `("u", 2)` literal cannot be the declaration,
however much lighter it reads: its type is `tuple[str, int]` whatever classes exist, subclassing
`tuple` does not change a literal's type, and typing it into `Expr`/`Buffer` would need one overload
per structure. The `L` wrapper is the smallest thing that carries the map.

This is one corner of a trilemma, chosen 2026-08-27 after the other two were drafted and rejected.
Of {no class ladder, flat positional inputs, distinct symbolic/numeric leaf types} only two are
available at once:

- A generated ladder `Arity1..Arity8` keeps flat inputs and distinct leaves. Rejected: a class per
  width plus one overload per rung for every seeded derivative.
- `TypeVarTuple` (`Function[Out, *Ins]`) keeps flat inputs with no ladder, and ty handles the
  concatenation `Function[Expr, *Ins, Expr]` correctly. Rejected: both calls must take the same
  `*Ins`, so numeric calls are typed as `Expr` or not at all. Typed numeric calls are the
  connection to the rest of a user's code and are not negotiable.
- Pytrees keep distinct leaves with no class ladder. Chosen; `G`'s overloads are the residue.

### 2. `Function` is generic in its two trees

```python
@al.function(G(L("x", 3), L("y", 3)), L("prod", ...))
def multiply(inputs: tuple[Expr, Expr]) -> Expr:
  return inputs[0] * inputs[1]

multiply.symbolic_call((a, b))            # tuple[Expr, Expr] -> Expr, checked by ty
multiply.numerical_call((a_buf, b_buf))   # tuple[Buffer, Buffer] -> Buffer
```

`Function[SymIn, NumIn, SymOut, NumOut]`. An output leaf's shape may be `...`, inferred by
tracing; a written one is checked against the trace, and the stored output tree carries the traced
shapes so derived trees (multipliers) have them. `symbolic_call` is today's `Function.call`;
`numerical_call` is today's `__call__`. A wrong structure, a wrong count, or a numeric value passed
to the symbolic side is a type error at the call site, and checked in `tests/typing/` (see item 9).
Per-output metadata that AD produces, `output_sparsities` and `output_coloring_widths`, stays on
`Function`; the trees declare names and shapes only.

### 3. One `ProblemSpec`, everything an `Expr`

```python
@dataclass(frozen=True)
class ProblemSpec:
  minimize: Expr
  eq: tuple[Expr, ...] = ()
  ineq: tuple[Bounded, ...] = ()        # al.bounded(expr, lo, hi, name=...)
  lb: SymbolicVars | None = None        # same pytree structure as vars
  ub: SymbolicVars | None = None

@al.problem(vars=G(L("u", NU), L("s", NS)), params=G(L("x", NX), L("u_ref", NU)))
def filter(vars: tuple[Expr, Expr], params: tuple[Expr, Expr]) -> ProblemSpec:
  u, s = vars
  x, u_ref = params
  return ProblemSpec(minimize=..., ineq=(al.bounded(barriers(x, u) + s, lo=0.0, name="cbf"),), lb=(NO_LB, const(0.0)))
```

`NO_LB` and `NO_UB` are scalar `Expr` constants containing IEEE negative and positive infinity.
They leave one variable leaf open while preserving the exact symbolic variable tree; scalar bound
expressions broadcast over their leaf. Solver plugins translate the infinities to their native bound
convention.

`al.problem` traces the body once with the declared symbols and returns
`Problem[SymVars, NumVars, SymParams, NumParams]`, which holds the spec and both trees. Roles are
dataclass fields, so they are statically checkable; `eq` and `ineq` are tuples of groups
concatenated by alloy, which removes the manual `al.concat(parts)` in the chain and race-car
problems; `diff` follows from the role. Constraint groups take an optional `name=` for
"which block is violated" diagnostics. `params=None` keeps inference for a problem built inside an
`@al.function` body over the enclosing symbols, the case `benchmarks/run.py` has; with `params`
declared, a reachable undeclared input is a build error.

The body is written against `u` and `s`, but the Lagrangian Hessian wants one `wrt`. `al.solver`
builds one internal `x` of the total size and substitutes `u -> x[:n_u]`, `s -> x[n_u:]`. There is
no substitution utility in `src/alloy` today (`Function.call` is a `CALL` node, not inlining); it is
a hash-consed rebuild over `topo`, and the one new IR piece in this work. Item 5 uses it too.

### 4. One `solver`, returning a plain `Function`

```python
filt = al.solver(filter, "sqp", name="filter", options={...})
(u, s), lam = filt.numerical_call(((u0, s0), lam_box0, lam_eq0, lam_ineq0, (x_val, u_ref_val)))
```

A solver is `Function[tuple[SV, SV, Expr, Expr, SP], ...]`: inputs are the tree
`(vars_init, lam_box0, lam_eq0, lam_ineq0, params)`, outputs `(vars, lam_box, lam_eq, lam_ineq)`.
Once nesting is the norm there is nothing for a separate `SolverFunction` type to add; the
descriptor stays as the attr on the `SOLVER_CALL` nodes, as today. Box multipliers have the vars
tree's structure (`relabel("lam:")`); `lam_eq0`/`lam_ineq0` are leaves that are always present with
size 0 when the category is absent, which `nlp.py` and `qp.py` already do, so the signature never
depends on the problem. `eq`/`ineq` are discovered from the body, so their multipliers are leaves
of unknown length; that is accepted rather than declared.

The backend is positional and `name` belongs to the builder call: the main and fallback solvers in
the filter come from one problem and need two artifact names. `al.nlp` and `al.qp` are deleted; a
`ProblemSpec` expresses everything they did.

**The cache boundary is the symmetric `SparseJacobian`, not the `Function`.** After A11 the
descriptor Hessian differs per backend (IPOPT lower, SQP upper), so "nothing branches on the
solver" stops being true at the descriptor. Everything expensive is still backend-free: the
dependency mask, the colouring, the compressed JVPs and the recovery table are symmetric. `Problem`
caches the gradient, Jacobian and the full symmetric Lagrangian Hessian at the expression level;
`al.solver` applies the backend's triangle and wraps the `Function`s. Consequence for `sparse_hessian` as landed: the triangle must stay an operation on a `SparseJacobian` after
construction, with the gather indices fixed at construction time, never resolved inside the kernel.
Filtering the recovery table (so `SparseJacobian` keeps the compressed products and the table) does
that unconditionally; the `gather(gather(x, a), b)` rewrite does it only if it is guaranteed to
fire. Take the recovery-table option and keep the "no full-nnz buffer in the C" gate.

### 5. QP backends are gated by a structural proof, not a type

`al.solver(problem, "piqp")` looks up `backend.kind`. For `"qp"` it proves the spec quadratic before
building the descriptor: `_jac_mask(hess_values, x)` (`ad/sparsity.py`) with every row empty proves
the Hessian is structurally independent of `x`, hence `f` quadratic in `x` for all `p`; the same
mask on `jac(h, x)` and `jac(g, x)` proves the constraints affine. Then

```
P = hess(f, x)         c = grad(f, x)|_{x=0}
A = jac(h, x)          b = -h|_{x=0}
G = jac(g, x)          bounds shifted by g|_{x=0}
```

with `|_{x=0}` through the substitution utility of item 3, and the sparse PIQP path unchanged:
`_qp_matrix_sparsity` already takes arbitrary expressions and probes them. The proof is
conservative: `x*x/x` is quadratic and is rejected, and the error names the oracle that failed. For
`f = 0.5 x'Px` with parametric `P`, AD yields `0.5 (P + Pᵀ)`: `n²` adds per call and a free
symmetrisation, which PIQP's `triu` wants anyway; for constant `P` it folds at build time. NLP
backends never run the extraction; a quadratic spec handed to IPOPT is solved as written.

This closes the draft's open question 2 (what `al.qp` returns): nothing, because it does not exist.
QP-ness is a property proven of the spec. It also answers the "updatable QP" use case: a data block
that should change per call is a parameter, forwarded into `piqp_update` by an identity oracle
(dense: the column-major transpose the wrapper already does; sparse: a constant-index gather). A
data block that is genuinely constant is a constant. No second API.

### 6. `qp_problem(n, n_eq, n_ineq)`: the typed data form

```python
def qp_problem(n, n_eq, n_ineq) -> Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]]:  # QPData[T] = tuple[tuple[T, T], tuple[T, T], tuple[T, T, T]]
  @problem(vars=L("x", n), params=G(G(L("P", (n, n)), L("c", n)), G(L("A", (n_eq, n)), L("b", n_eq)), G(L("G", (n_ineq, n)), L("g_lb", n_ineq), L("g_ub", n_ineq))))
  def qp(x, params):
    (P, c), (A, b), (G, g_lb, g_ub) = params
    return ProblemSpec(minimize=0.5 * x @ P @ x + c @ x, eq=(A @ x - b,), ineq=(al.bounded(G @ x, g_lb, g_ub),))
  return qp
```

A function, not a decorator, because the body is fully determined by the sizes. The data tree is
fixed, grouped as objective, equalities, inequalities: an absent block is a `(0, n)` leaf, the same
size-0 convention as the multipliers, so every QP has one checked signature. Box bounds go through
`ProblemSpec.lb`/`ub` like any other problem. It goes through item 5's extraction like any other spec,
which makes it the natural first differential test: `qp_problem` through `"piqp"` against today's
`al.qp(P=..., c=...)` on the same data. Lives in `solvers/` next to `solver`.

### 7. Names are metadata, one per leaf

Names survive in the places where they earn their place and nowhere as an addressing mechanism. The
generated header (`codegen/aot.py`) spells the typed buffer structs, the C++ wrapper parameters and
the sparse metadata tables after them (`filter_u_out`, `spjac_g_x`), and that is the AOT consumer's
surface; the text asm and viz print `%zprev: f64[3]` rather than `%in0`. The derivative wrappers
take them as `of`/`wrt`, and every `{kind}_{of}_{wrt}` output name is built from them. The
generated C body never uses them (`arg[i]`/`res[i]`).

Every `L` carries its name, so a tree is fully named by construction; duplicate names within one
tree are a runtime error, and there are no default names. An output leaf's shape may be `...`,
meaning inferred by tracing; a written shape is checked. Derived trees are made by `relabel`
(`lam:`, `fwd:`), so the header spells `lam:u` for the multiplier of `u`. The declared name and the
name the body binds are independent: `G(L("u", ...), L("s", ...))` may be received as
`def body(vars, params): u, s = vars`; the body's names are local, the declared ones are external,
and keeping them equal is a habit for the reader that nothing enforces. Document this in the guide
and the `L` docstring.

Unknown `of`/`wrt` is a `ValueError` naming the declared choices, raised when the derivative is
built, never at evaluation time; `arity.py` tests it. Static names are possible, a `LiteralString`
bound keeps `Function[..., Literal["x", "p"], Literal["f"]]` and `fn.gradient("f", "z")` then fails
at check time, but only through methods on `Function`, since a free function solves the name
variable from both arguments and widens to `str`. Not adopted; it would reopen D1's free-function
form and reject runtime-built names such as `nlp.py`'s.

### 8. Derivatives keep the source's inputs, so they are typed

Leaning, not settled (2026-08-26). Drafted in `arity.py`.

`gradient(fn, of, wrt)` returns `Function[SI, NI, Expr, Buffer]`: the same input tree as `fn`,
one output leaf; `jacobian`, `hessian`, `sparse_jacobian`, `sparse_hessian` likewise. Seeded modes
pair the source's inputs with the new group and stay generic, with no overloads:
`forward` is `Function[tuple[SI, Expr], tuple[NI, Buffer], Expr, Buffer]`, and `lagrangian_hessian`
is `Function[tuple[SI, SO], tuple[NI, NO], Expr, Buffer]`, whose multipliers have the source's
*output* tree type, one multiplier per output (`fn.outputs.relabel("lam:")`), so multipliers in the
wrong structure are a type error. `of` and `wrt` stay strings and are checked at build time.

What this replaces is `extra_inputs`, landed in D1: it selects a subset of the source's inputs at
runtime, which is exactly what makes a derived function's arity unknowable. Its only consumer in
`src/` is `nlp.py`, which uses it to say "all of them" (`(x, *params)`), so keeping every input is
what the solver path already does. A derived function then carries inputs it does not read, which
in C is an unused pointer argument. `fn.factory([...])` over a runtime list stays dynamic; it is
the "several derivatives at once" shorthand and its typed equivalent is calling the wrappers
separately. This is the change that lets the typed layer be used inside AD, `vmap` and the solver
builders rather than only at the user surface, which is the point of the typing work.

### 9. Typing tests

`tests/typing/` holds files of `if TYPE_CHECKING:` blocks with `assert_type(...)` for the positive
cases and `# ty: ignore[<code>]` on the expected errors. It runs as
`uv run ty check --error-on-warning tests/typing`: ty reports an unused `ty: ignore` as
`unused-ignore-comment`, so an expected error that stops being one fails the check. That is the
"prove the gate can fail" rule applied to types. `arity.py` is the shape of the first file.

Requires `ty>=0.0.75` (done 2026-08-26, tree clean under it): the pinned 0.0.37 inferred `Unknown`
for a class-scoped `type` alias used as an annotation and the design did not check under it.

## Sequencing

1. `ty` bump. Independent; done.
2. Reorder `sparse_hessian` so the triangle is applied after the JVP batch: as landed, it filters
   `sparsity` and `recovery` before `_sparse_jacobian_colored`, so the triangle is baked into
   construction and a cached symmetric Hessian cannot be re-cut per backend. Build `compressed`
   once and gather with `recovery[keep]`; same values, same constant indices, JVPs shared. Tiny, and
   a prerequisite for the `Problem` cache in item 4.
3. `substitute` in `ir/`, with tests. Independent except for the `MAP`/`VMAP` spelling; do after D2.
4. `Tree`, `L`, `G` with its width overloads, generic `Function`, typed derivative wrappers replacing
   `extra_inputs` (item 8), `tests/typing/` harness. Touches `function/`; after D1.
5. `ProblemSpec`, `Problem`, `al.problem`, `al.solver` for NLP backends, deleting `al.nlp` and
   `SolverFunction`; move the
   three benchmark problems and the filter. After A11, since it owns the descriptor Hessian fields.
6. Quadratic proof and extraction, `qp_problem`, deleting `al.qp`; differential against the old
   builder before it goes.
7. Docs: `docs/guide/solvers.md` rewritten around `problem`/`solver`; `docs/guide/functions.md` for
   the arity form of `@al.function`.

## Considered and rejected

- **Two problem types, `Problem` and a data-form `QP`, with conversions.** Proposed in this round
  and dropped for item 5: two representations of one thing, with the presence of one deciding which
  backends work. A proof over one type is the same information, checkable.
- **`al.qp(P=...)` returning a `Problem` with the matrices stashed in an optional field.** Same
  objection.
- **A helper that wraps constants as parameters so every QP block is updatable.** Changes the call
  signature for blocks that are genuinely constant; passing a symbol already makes a block a
  parameter. No new concept needed.
- **`al.var("u", n, lb=, ub=)` symbols with attached bounds, declared where used, and a plain
  dataclass with `vars=[u, s]`.** Proposed to fix declaration locality without a decorator. The
  arity form gets locality from the decorator line plus the checked annotation, and binds the type
  variables, which the symbol list cannot. Bounds live on `ProblemSpec` instead.
- **An imperative `ca.Opti`-style builder (`m.subject_to(...)`).** Still rejected: accumulating
  state buys nothing over a tuple, loses static role checking, and the one thing it is for
  (declaring variables inline) the arity form gives through the annotation.
- **Operator-overloaded constraints**, **a returned dict of roles**, **stage structure in the
  API**, **`Function`-valued fields**, **a variable-layout system**: rejected in the draft for
  reasons that still hold. Cost and constraints share one graph, so they are expressions, and call
  nodes let the user pick the compile boundary independently.
- **Typing the `eq`/`ineq` multipliers by declaring constraint arity in the decorator.** Constraints
  are discovered from the body; declaring them twice to type two outputs is not worth it.

# One matcher: op-indexed tables and one walk-rebuild

Decided 2026-08-14 as phase 9 of the restructure, deferred out of it and left standing on
2026-08-19. Not started. Conditional: it lands only if the result is smaller.

## The problem

Three places index entries by op and scan for the first that fires, and two places walk a
hash-consed graph rebuilding it bottom up. They were written separately and never reconciled.

**Two op-indexed tables.** `ir/match.py`'s `PatternMatcher` and `ir/spec.py`'s `Spec` are the same
data structure spelled twice: an `any` list for the `op is None` entries, a `by_op` defaultdict, a
`candidates(op)` generator yielding the first then the second, and a scan that stops at the first
entry to fire. `Pattern` and `Rule` are both frozen dataclasses holding `op: Op | None`, a callable,
and a `matches`/`applies` method that is the same expression. Only two things differ: what "fires"
means — `PatternMatcher.rewrite` returns the first replacement that is not the input node,
`Spec.check` returns the first rule whose check returns a diagnostic string — and that `Spec` is
already generic over `(Node, Op)` while `PatternMatcher` is hard-wired to `Expr` and `ExprOp`.

**Two walk-rebuilds.** `ir/match.py`'s `rewrite` and `passes/program.py`'s `_transform` both rebuild
a hash-consed graph bottom up, and differ in three ways that do not look deliberate:

| | `rewrite` (expr) | `_transform` (program) |
| --- | --- | --- |
| traversal | iterative, over `topo()` | recursive, over `args` |
| shared subgraph | cached in `replacements`, visited once | re-walked once per reference |
| per node | retries the matcher to fixpoint | applies `fn` exactly once |

The second row is a defect, not just duplication. `ProgramNode.__new__` interns on
`(op, args, attrs, dtype)`, so program graphs share subnodes exactly as expression graphs do, and
`_transform` keeps no memo — a node reachable by k paths is rebuilt k times. Its three call sites
are `_subst_var`, `_expand_inlinables` and `_apply_pack`; the first two are the fusion pass's inner
loop, and `_expand_inlinables` calls itself from inside its own callback, so the re-walk compounds
along a chain of single-use producers.

`_walk` is a third walker, but a read-only one: seven call sites, all counting nodes or collecting
buffer names, none order-sensitive. Merging it is not obviously worth anything.

## What we are changing

1. Factor the op-indexed table once, generic over `(Node, Op)`, and build both `PatternMatcher` and
   `Spec` on it. `Spec`'s type parameters are already right, so the work is on the match side.
2. Give the walk-rebuild one implementation, parameterized by the per-dialect rebuild step — `Expr`
   carries seven fields (`op`, `args`, `type`, `name`, `value`, `attrs`, `lowering`), `ProgramNode`
   carries four — and by whether a node is retried to fixpoint.
3. Repoint `passes/program.py`'s three `_transform` call sites at it. That is where the memo lands.
4. Leave `_walk` alone unless step 2 makes it fall out for free.

## Why this is the one that can break quietly

It is the only refactoring here that can change behavior without changing a line of intent. Two
hazards, both worth writing down before starting:

- **Fixpoint semantics.** Adopting `rewrite`'s per-node retry loop for the program dialect would
  apply a program rewrite repeatedly where it applies once today, and whether each one terminates
  under repetition is a per-pass property nobody has checked. Keep single application for program
  passes and make the retry an explicit argument rather than the default.
- **Memoizing `_transform` is only sound if every callback is a pure function of its node.** All
  three are, today: each closes over a fixed substitution table or rename plan and mutates nothing
  during the walk. That is a fact to re-establish when the work starts, not to assume.

Gate: the byte-for-byte generated-C corpus in `tests/test_c_snapshot.py` and `tests/baseline/c/`.
This work is why the corpus outlived the restructure. Take a fresh snapshot before starting
(`uv run python tests/test_c_snapshot.py`) and do not regenerate it while the work is in flight —
regenerating is the defect, not the fix. A pass-ordering or match-semantics slip shows up there and
almost nowhere else.

Landing condition, unchanged from the restructuring plan: it lands only if the code shrinks.
`ir/match.py` is 78 lines and `ir/spec.py` is 71. If the generic table plus two dialect adapters is
not clearly under that, the duplication is cheaper than the abstraction, and the right outcome is
to close this section unlanded and say so.

## Sequencing

Independent of the other decided sections: this one touches `ir/` and `passes/`, while the
derivative API work touches the public derivative surface and the VMAP rename touches the operation
vocabulary. If both are queued, do this one **after** the VMAP rename — that rename already lists
`tests/test_c_snapshot.py` among the files it touches, so ordering it first means the corpus is
regenerated once instead of twice.

## Considered and rejected

- **Unifying `_walk` with `topo` as well.** The program passes use `_walk` for queries where order
  does not matter, and `topo` pays for an ordering none of the seven call sites reads. A shared
  walker here would be a name, not a saving.
- **Doing this during the restructure.** It was phase 9 for a reason: every other phase was either
  a seam change with the layout fixed or a move with no logic change, and this one is neither. It
  stayed out so that a C-snapshot diff during the restructure could only ever mean a mistake.

# Open problems

Noticed and reproduced, not scoped. Neither entry has a decision, a design or an
`internal/todo.md` item; both are here so the next session does not rediscover them. An entry
leaves this section for a `#` section of its own once someone decides what to do about it.

## Zero-input `Function`s, and the flat call seam that survives because of them

Found while landing the typed call surface, by trying to ban zero-input `Function`s and watching
the suite break. Two independent producers, both legitimate:

- **Differentiation.** `_call_jvp_function`, `_call_jvp_many_function` and
  `_call_jvp_many_const_function` in `ad/forward.py` each keep only the callee inputs the
  derivative actually depends on. A constant derivative depends on none, so the synthesized callee
  has no inputs at all — `duplicate_fw_second_x`, `scale_add_fw_y_x`,
  `bicycle_stage_interstage_fw_eq_znext` among others.
- **Solvers.** The oracle and bounds functions of any problem declared without parameters:
  `standalone_qp_oracle`, `settings_qp_bounds`.

Both lower to real C procedures taking no arguments, so neither is a mistake to delete.

What we are living with. `Function.__call__` dispatches on leaf kind, and an empty tree has no
`Expr` leaf, so it cannot be routed symbolically — it reads as an evaluation. That is the reason
`ad/forward.py` is the one module outside `function/` permitted to use `_flat_symbolic_call`,
pinned by `test_flat_call_seams_stay_inside_their_sanctioned_modules` in `tests/test_layering.py`.

What a fix looks like, and why it did not ride along with the API change. A `CALL` to a
no-argument procedure returning a constant is pure overhead; inlining the constant at the call site
would remove the AD producer, let the ban stand, and shrink the generated source. But it changes
what AD emits, so it regenerates the byte-for-byte C corpus (`tests/test_c_snapshot.py`,
`tests/baseline/c/`) — an AD and codegen change wearing an API cleanup's clothes. The solver side
needs its own answer either way: a parameterless oracle has nothing to take, so either a zero-input
signature stays legal or the descriptor stops building one.

## `vmap` and the AD entry points erase the callee's declared trees

`Function` is meant to be the unit of composition, and after the typed call surface it nearly is:
`fn(tree)` is typed on both the symbolic and the numeric side, and the derivative wrappers keep the
source's input tree, so `al.gradient(fn, "f", "x")` is a `Function[SI, NI, Expr, np.ndarray]`.
Three holes remain, all on the paths that matter most for composing:

- **`al.vmap` is untyped end to end.** `vmap(callee: Any, length: int, inputs: Any, output: int = 0)
  -> Expr` in `function/sugar.py`. The callee's declared trees are never read, the result is a bare
  flat `Expr` rather than the callee's output tree repeated, and an output is selected by integer
  index — the addressing-by-position that the declared trees removed everywhere else.
- **`al.jvp`, `al.jvp_many`, `al.vjp` and `al.vjp_many` are expression-level.** They take
  `Sequence[Expr]` and return `tuple[Expr, ...]`: typed, but tree-blind, with no Function-level
  spelling that preserves structure.
- **The callees AD synthesizes are `Any`-typed.** Every builder in `ad/forward.py` and
  `ad/reverse.py` goes through `Function._from_exprs`, whose trees come from `flat_tree`, so each
  result is a `Function[Any, Any, Any, Any]`. This already leaks into the call surface: both
  `__call__` overloads match `Any`, which is why their order in `function/model.py` is load-bearing
  and had to be commented.

Cost of leaving it: any composition passing through a map or a derivative drops out of the typed
world, and the developer experience degrades exactly where the library stakes its claim, on
preserved mapped structure. The D3 sketch anticipated the first of these — `arity.py`'s docstring
lists "`vmap` typed by its callee's trees" as left for the implementation.

Whatever lands needs both kinds of test the tree work uses, because they catch different things:
runtime structure tests beside `tests/function/test_tree.py`, and static ones in
`tests/typing/test_arity.py`, which runs under `ty check --error-on-warning` where every
`ty: ignore` marks an expected error and an unused one fails the check.
