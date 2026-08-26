> Planned work, not a frozen record. One `#` section per refactoring; add new ones alongside
> rather than editing settled ones, and remove a section once it has landed.

# DRAFT: solver problem construction, decoupled from the solver

**Draft, 2026-08-19. Not decided, and not to be implemented.** Two sessions explored this
separately and reached similar but not identical shapes; Ted is not convinced by either. What
follows records the findings so the next round starts from them; it settles nothing. Everything
under "Still open" is genuinely open, and several things under "The shape both sessions reached"
have only one session behind them.

## The problem

`@al.function` exists because CasADi makes you instantiate symbolic variables, wire them into a
graph by hand, and then wrap the result — leaving stale symbols in Python scope whose names collide
with the numbers you want afterwards. `al.nlp(...)` and `al.qp(...)` still make you do exactly
that. Four distinct costs, in rough order of how much they hurt:

**Parameter order is discovered, not declared.** `collect_free_inputs` sorts by `(name, id)`, so the
call signature depends on alphabetical accident. `docs/guide/solvers.md` has to warn that "when two
parameters happen to have the same shape, getting this wrong produces a wrong answer instead of an
error", and `Function.call` takes a positional `Sequence`, so a nested solver cannot dodge it.

**No problem/solver split, so backend-independent work is redone.** Nothing in `nlp()` branches on
`solver` while building the gradient, sparse Jacobian and sparse Lagrangian Hessian; only the
descriptor's backend and options differ. The safety filter in `benchmarks/problems/unbumpercars/filters.py`
builds a main SQP solver and an l1-globalization fallback from identical expressions and pays for
the same differentiation twice, as do the backend sweeps in `benchmarks/run.py`.

**The problem object already exists, unnamed.** That same filter builds an oracle `Function` with
outputs `("cost", "g")` and then takes it apart again — `z, _bar_x, ... = base.inputs`,
`cost, g = base.outputs` — to hand the pieces back to `al.nlp`.

**Symbols leak into scope.** `z`, `bar_x`, `physics`, `dt` sit in the enclosing scope, and
`physics`/`dt` are names you also want for numbers. Note that `@al.function` does not actually fix
this in the way its docstring claims: `Expr.__new__` interns unconditionally, so `al.sym("x", 3)` is
one process-global object and the decorator hides symbols from your namespace rather than scoping
them. Two declarations agreeing on name, shape and `diff` silently denote the same parameter.

Minor, but in the same area: `diff=False` is the user's job on every parameter symbol today, and
the decorator cannot express it without spelling a full `TensorType` — which is why decorated
`chain_link_accel` marks `mass`/`spring_d`/`rest_len` differentiable while the hand-built
`chain_ode_fn` marks the same quantities `diff=False`.

## The shape both sessions reached

A backend-free problem description, authored from expressions, and a free function that turns it
into a solver.

```python
@al.problem(vars={"z": al.var(NZ * (n + 1), lb=lb, ub=ub)}, params={"p": n_param(n)})
def race_car(z, p) -> al.Problem:
    ...                                    # body unchanged from today
    return al.Problem(minimize=cost, eq=eq, ineq=al.bounded(corridor, -w, w))

solver = al.solver(race_car, "ipopt", name=..., options={...})
```

- **`al.Problem` is one frozen dataclass**, both what a traced body returns and the hand-built
  representation. The tracer creates the symbols, calls the body, and fills in `vars`, `params` and
  `name` with `dataclasses.replace`. The hand-built path constructs the same type directly, so
  unlike today the sugar is a pure desugaring with no behaviour of its own.
- **Roles are dataclass fields, not dict keys** — `minimize`, `eq`, `ineq` — because dict keys are
  not statically checkable. Same reason `al.var(shape, lb=, ub=)` and `al.bounded(expr, lo, hi)`
  are types rather than parallel keyword arguments.
- **Declaration order is parameter order.** Inference stays for `params=None`: `benchmarks/run.py`
  builds a QP inside an `@al.function` body over the enclosing function's symbols, and that case
  has no declaration to read. When `params` is declared, a reachable undeclared input should be a
  build error rather than a silently appended argument.
- **`diff` follows from the role.** Variables are differentiable, parameters are not; the flag
  stops being the user's job.
- **`eq` and `ineq` accept a list of groups**, concatenated by alloy — this removes the manual
  `al.concat(parts)` in the chain and race-car problems and lets groups with different bounds
  coexist without padding a flat lower/upper pair.
- **Multi-block variables cost nothing.** The tracer declares one internal decision symbol and
  hands the body slices of it, which is what the benchmark bodies already write by hand. With
  `vars={"u": ..., "s": ...}` the filter's hand-concatenated `[u; s]` and hand-sliced
  `out["x"][:n_u]` become named, and `x0`/`x_lb` assembly stays plain NumPy on the Python side.
- **Derived functions cache on the problem**, so a second backend over the same problem is
  descriptor assembly only.
- **No stage structure in the API.** Multistage stays a body-level concern through `al.vmap` and
  `Function.call`, as `chain_eq_function` does now.

## Authoring: expressions, not stage functions

Settled between both sessions, against anvil's `eq_initial_fn` / `eq_interstage_fn` /
`cost_stage_fn` slot vocabulary.

Cost and constraints have to be able to share one graph. In the race-car problem the corridor
constraint reuses `e_lat` and `d_phi` computed a few lines earlier for the cost residuals; if
`minimize` and `ineq` were separate `Function`s, alloy would have to inline them (losing the
boundary) or keep them apart (losing the sharing). This is why `nlp()` already stitches everything
into one base function with outputs `("f", "g")`.

Authoring from expressions does not mean everything is inlined, because call nodes are first class:
a `Function` participates by being `.call`ed from the body, and derivatives and sparsity go through
`CALL` fine. So the user picks the compile boundary independently of the problem statement, which
is strictly more than `Function`-valued fields would allow.

Anvil needed stage functions because its multistage assembler had to know the block structure to
build block-sparse Jacobians. Alloy gets that from `vmap`, and `internal/roadmap.md` is explicit
that the multistage result was matched without introducing a multistage primitive. Baking stages
into `Problem` would undo that.

The one real case for a `Function`-valued input is an externally supplied oracle (precompiled
CasADi C). That already exists as `ExternalOracle` on `SolverDescriptor`, and it is a
solver-construction concern rather than a problem-statement one.

## Naming

Ted rejected a `al.nlp` (solver builder) / `al.NLP` (problem type) pair. The diagnosis worth
keeping: the fault is not the casing — alloy already ships `al.function` / `al.Function` and nobody
trips over it — but that the two names would return *different types*. The workable rule is that a
case pair is fine when the lowercase name is sugar that builds the CapWords thing, and confusing
otherwise. Under that rule `al.problem` / `al.Problem` is fine.

`al.solver(problem, "ipopt")` is the leading candidate for the builder and is **not agreed**. The
argument for: `al.function` builds a `Function`, `al.problem` builds a `Problem`, `al.solver`
builds a `SolverFunction` — three lowercase factories each named after its product, and the
`solver=` keyword collision disappears once the backend is positional. The argument against: it
sits close to `SolverFunction` without matching it exactly. `al.build` was considered and says
nothing about what it builds.

The backend-agnostic problem type also means one builder covers both shapes, because the plugin
registry already distinguishes QP from NLP backends. That makes today's
`al.nlp(x=..., f=..., h_eq=...)` deletable: a hand-built `Problem` expresses everything it did.

Two hard constraints from Ted: the builder is a free function, not a `Problem` method
(`problem.solver("ipopt")` is rejected), and the declaration keyword is `vars=`, not `decisions=`.
`vars` shadows the builtin only inside `problem()`'s own body, and flake8-builtins is not enabled.

## Still open

1. **The builder name.** See above.
2. **What `al.qp(P=..., c=...)` returns.** One session would leave it returning a `SolverFunction`
   as it does today, which leaves two doors to the same product, one of them skipping `Problem`.
   The other proposed it return a `Problem` instead, writing `0.5 x'Px + c'x` and the affine
   constraints over an internal decision symbol and stashing the matrices it was handed in one
   optional field for backends that want them — so `al.solver(prob, "piqp")` is the only door, and
   the extraction work below can fill the same field later. Unresolved, and the data form is what
   every current call site uses.
3. **Named constraint groups**, for "which block is violated" diagnostics. A dict reintroduces the
   string keys the dataclass exists to avoid; a `name=` field on `al.bounded(...)` gets the same
   benefit without them.
4. **Multi-block warm starts.** Per-block call-time names (`u0`, `s0`) can collide with parameter
   names; letting the existing `x0` accept a mapping avoids inventing names at all.
5. **Keyword invocation of traced bodies.** Calling the body with keyword arguments (`fn(**syms)`)
   turns a mismatch between the declared names and the body's parameters into a `TypeError` naming
   the parameter. `@al.function` has the same gap today, since `function/api.py` calls positionally,
   and the same fix applies to both.
6. **Whether any of this is worth its size**, which is Ted's standing objection and the reason this
   section is a draft. The parameter-ordering hazard and the duplicated differentiation are
   concrete; the rest is ergonomics, and the current surface is small.

## Considered and rejected

- **Operator-overloaded constraints** (`h == 0`, `l <= g <= u`, in the manner of CVXPY or JuMP).
  `Expr` has no comparison operators today, so `<=` is free, but `__eq__` is not: overloading it
  breaks the identity and hashing that `topo`, `cse` and the `seen`-set walks depend on. Half a
  domain-specific language is worse than none.
- **A returned dict of roles** (`{"minimize": ..., "eq": ...}`). No static checking of the keys.
- **An imperative model builder** (`m.minimize(...)`, `m.subject_to(...)`). More state, no more
  expressiveness than a returned dataclass.
- **A second, parametric `bounds` field on `Problem`.** Two places to write a bound, and the second
  was string-keyed. Parameter-dependent bounds — supported today, used by no call site — stay
  expressible by passing an `Expr` as `lb`/`ub` on the hand-built path.
- **`name` owned by `Problem`.** The main and fallback solvers in the filter come from one problem
  and need two artifact names, so the name belongs to the builder call. `Problem.name` survives
  only as a default taken from the body's `__name__`.
- **Removing the expression-level path.** Two ways to build a solver is the point, provided the
  sugar desugars exactly.
- **A `Function`-accepting shortcut on `@al.problem`.** The two-line body
  `cost, g = oracle.call([...]); return al.Problem(minimize=cost, ineq=al.bounded(g, lo=0.0))`
  covers the filter's hand-built oracle and keeps one path.
- **A variable-layout or stage-bounds system**, to replace index arithmetic like
  `lb[i * nz + nx : (i + 1) * nz] = -1.0`. Most of it is `np.tile(...).reshape(n + 1, NZ)` followed
  by patching columns, which needs no API. Worth deciding separately if it still hurts afterwards.

## Interactions

- **Derivative API rework**, now landed. It names `src/alloy/solvers/nlp.py` as the only consumer inside
  `src/` of the old `(wrt, of)` argument order; that caller now uses `(of, wrt)`.
- This draft assumes the landed `VMAP`/`vmap` spelling; anything written here refers to `al.vmap`.
- `internal/roadmap.md` sketches extracting `P = hess:f:x:x`, `c = grad:f:x` at `x=0` and
  `A_eq = jac:h:x` so one problem can target either PIQP or IPOPT. It is the reason the type is
  called `Problem` rather than `NLP`, it needs structural validation that the objective is
  quadratic and the constraints affine (`_jac_mask` can prove the Hessian does not depend on `x`),
  and both sessions would defer it.

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
