> Planned work, not a frozen record. One `#` section per refactoring; add new ones alongside
> rather than editing settled ones, and remove a section once it has landed.

# Open problems

These remaining limitations have follow-up issues in GitHub. The first two are now
designed in [the core compiler roadmap](core_compiler_roadmap.md#signatures-and-templates) ([#8],
[#13] and [#58]) and stay here as the record of why. Automatic differentiation is abbreviated AD
below.

## Zero-input `Function`s, and the flat call seam that survives because of them

Found while landing the typed call surface, by trying to ban zero-input `Function`s and watching
the suite break. Two independent producers, both legitimate:

- **Differentiation.** `_call_jvp_function`, `_call_jvp_many_function` and
  `_call_jvp_many_const_function` in `ad/calls.py` each keep only the callee inputs the
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
source's input tree, so `sc.gradient(fn, "f", "x")` is a `Function[SI, NI, Expr, np.ndarray]`.
Three holes remain, all on the paths that matter most for composing:

- **`sc.vmap` is untyped end to end.** `vmap(callee: Any, length: int, inputs: Any, output: int = 0)
  -> Expr` in `function/sugar.py`. The callee's declared trees are never read, the result is a bare
  flat `Expr` rather than the callee's output tree repeated, and an output is selected by integer
  index — the addressing-by-position that the declared trees removed everywhere else.
- **`sc.jvp`, `sc.jvp_many` and `sc.vjp` are expression-level.** They take
  `Sequence[Expr]` and return `tuple[Expr, ...]`: typed, but tree-blind, with no Function-level
  spelling that preserves structure.
- **The callees AD synthesizes are `Any`-typed.** Every builder in `ad/calls.py` goes
  through `Function._from_exprs`, whose trees come from `flat_tree`, so each result is a `Function[Any, Any, Any, Any]`. This already leaks into the call surface: both
  `__call__` overloads match `Any`, which is why their order in `function/model.py` is load-bearing
  and had to be commented.

A composition through `vmap` or the low-level derivative builders loses its declared tree types.
The public Function-level derivative wrappers preserve the source input tree. Extending that
property to mapped structure remains open (todo [#13]). The playground README records this
limitation, and
`typing_playground/function.py` holds a candidate interface.

Whatever lands needs both kinds of test the tree work uses, because they catch different things:
runtime structure tests beside `tests/function/test_tree.py`, and static ones in
`tests/typing/test_arity.py`, which runs under `ty check --error-on-warning` where every
`ty: ignore` marks an expected error and an unused one fails the check.

[#8]: https://github.com/PREDICT-EPFL/scaly/issues/8
[#13]: https://github.com/PREDICT-EPFL/scaly/issues/13
[#58]: https://github.com/PREDICT-EPFL/scaly/issues/58
