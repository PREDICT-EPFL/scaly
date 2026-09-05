> Planned work, not a frozen record. One `#` section per refactoring; add new ones alongside
> rather than editing settled ones, and remove a section once it has landed.

# Function templates

Follow-up D3.1 after the solver API lands on dev. The interface sketch and its design rationale
live in [the typing playground](../../typing_playground/README.md#templates). The sketch does not
lower, compile, or evaluate expressions; its tests prove the proposed interface only.

`FunctionTemplate` owns a declaration with shape holes and produces concrete `Function` instances.
The compiler continues to consume concrete Functions. This keeps unresolved shapes out of the
expression graph and lets the existing call, differentiation, and compilation machinery remain
responsible for each instance.

The production implementation still needs decisions and verification in these places:

- Specialization keys and generated C names must distinguish every input property that changes
  the graph, including nested structure and relevant `TensorType` metadata. The sketch only
  mangles flat shapes. Decide how holes obtain dtype and differentiability before copying it.
- Trace each instance once, including the bare mode that infers declarations from a call.
  The sketch currently traces bare bodies twice.
- Lift the existing derivative wrappers over concrete instances. Seed and multiplier declarations
  must use the generated input types while preserving the source tree structure; a constant
  output's differentiability flag is not the multiplier's flag.
- Keep symbolic and numerical calls typed, preserve eager checks for fully declared templates,
  and cover actual compilation, cache reuse, and composition in addition to static assertions.
- Settle typed `vmap` separately. The sketch adds a leading axis to each leaf, while the real
  builder uses flattened slices, starts, and strides. Calling a template inside a traced body
  can instantiate it there; a consumer that inspects a callee needs a concrete instance.

The existing implementation is in `src/alloy/function/{tree,model,api}.py`; the template sketch
is `typing_playground/templates.py`. Naming and declaration shorthand remain the playground's
open questions. Overload sets and constraints on shape holes remain deferred.

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

These remaining limitations have follow-up entries in `internal/todo.md`. Their implementation
choices are not settled. Move an entry to its own `#` section once its design is decided.
Automatic differentiation is abbreviated AD below.

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
pinned by `test_flat_call_seams_stay_inside_their_sanctioned_modules` in `tests/test_import_layering.py`.

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

A composition through `vmap` or the low-level derivative builders loses its declared tree types.
The public Function-level derivative wrappers preserve the source input tree. Extending that
property to mapped structure remains D3.2. The playground README records this limitation, and
`typing_playground/function.py` holds a candidate interface.

Whatever lands needs both kinds of test the tree work uses, because they catch different things:
runtime structure tests beside `tests/function/test_tree.py`, and static ones in
`tests/typing/test_arity.py`, which runs under `ty check --error-on-warning` where every
`ty: ignore` marks an expected error and an unused one fails the check.

## Compiled CasADi artifacts across worktrees

The API merge review reproduced four CasADi IPOPT test failures from cached libraries whose
runtime search paths pointed into a deleted worktree. All five tests in
`tests/benchmarks/test_casadi_ipopt.py` passed with a fresh `ALLOY_CASADI_IPOPT_CACHE`.

`benchmarks/harness/casadi_ipopt.py` keys its cache on the serialized problem, options, CasADi
version, solver library content, compiler, and optimization flag. The key omits the library search
paths supplied by `backend_compile_flags`. Identical solver libraries in a new worktree therefore
reuse an artifact whose runtime search paths still name the old one.

A fix must either include those paths in the cache identity or remove the artifact's dependency
on worktree paths. Verify relocation after the original directory disappears; a successful load
while both worktrees exist does not exercise the failure. A separate cache is the temporary
workaround, not the intended behavior.
