# Solver plugins

A solver plugin connects Scaly models to an optimization solver. It is a separate Python package
that generates a C wrapper around the solver's native interface. Read this page when implementing
a backend. The [solver guide](../guide/solvers.md) covers using an existing solver.

Scaly generates the problem's objective, constraints, and derivatives as C functions, called
oracles. The plugin's wrapper calls these functions and passes their results to the native solver.
The wrapper and oracles compile together, so a numerical solve does not call back into Python.

## Anatomy of a plugin

A plugin is a Python package that:

1. depends on `scaly`;
2. exposes a backend object through the `scaly.solvers` entry-point group;
3. when it wraps its own native solver, bundles the vendored shared library under `<pkg>/lib/` and
   its C headers under `<pkg>/include/`, built by a hatch build hook (`plugins/scaly-piqp` and
   `plugins/scaly-ipopt` are the two reference implementations). `plugins/scaly-sqp` has no
   library of its own and generates C against `scaly-piqp`'s.

```toml
# pyproject.toml
[project.entry-points."scaly.solvers"]
mysolver = "scaly_mysolver:BACKEND"
```

The entry-point name is the backend string users pass as the second argument to
`sc.solver(problem, backend)`.

## The backend protocol

`BACKEND` must satisfy `scaly.solvers.registry.SolverBackend`, which defines the common backend
metadata and the code generation hook. A backend accepts either quadratic programs, abbreviated
QP, or nonlinear programs, abbreviated NLP. Scaly's just-in-time compiler, abbreviated JIT,
uses the library and header paths when it builds the generated code.

| member | meaning |
|---|---|
| `name` | backend name; must equal the entry-point name |
| `kind` | `"qp"` or `"nlp"`, the descriptor family the solver accepts |
| `protocol_version` | the `SOLVER_PLUGIN_PROTOCOL_VERSION` the plugin was written against (validated at load; mismatch is a hard error) |
| `lib_stem` | shared-library stem: the file is `lib<stem>.dylib` / `lib<stem>.so` |
| `link_flags` | linker flags, e.g. `("-lmysolver",)` |
| `header` | C header path relative to `include_dir()`, e.g. `"mysolver/api.h"`; core emits `#include "<header>"` in solver-bearing translation units |
| `lib_dir()` / `include_dir()` | vendored library / header directories; they join the JIT's `-L`/`-I`/rpath search path |
| `render_wrapper(fun, ctx)` | the C wrapper template (below) |

NLP backends also satisfy `scaly.solvers.registry.NlpSolverBackend`. They declare `hess_triangle`
as `"lower"` or `"upper"`, and `require_backend(..., "nlp")` validates it at runtime. QP backends do
not declare this member.

Discovery, protocol-version validation, and kind checking live in `src/scaly/solvers/registry.py`.
Library and header discovery (`src/scaly/solvers/paths.py`) derives everything else from this
metadata: `solver_loadable("mysolver")`, compile/link flags, diagnostics, and the per-plugin
exact-path override env var `SCALY_MYSOLVER_LIB`.

## The codegen contract: `render_wrapper`

```python
def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]: ...
```

`fun` is the concrete graph instance being rendered. `fun.descriptor` (a `SolverDescriptor`,
`src/scaly/solvers/model.py`) carries the problem dimensions, input/output signatures,
oracle `Function`s, sparsity patterns, and user options. `ctx` supplies the C names and helpers
needed by the wrapper through `scaly.codegen.solver.SolverWrapperCtx`:

- `ctx.symbol` is the solver's mangled C identifier. Prefix every static the template declares with
  it, since multiple solvers can share one translation unit.
- `ctx.raw_symbol` is the name of the function the template must define.
- `ctx.stats_symbol` is the `scaly_solver_stats` static the template must fill on every call. Core
  declares it and exports the `<symbol>_stats(...)` accessor, and the plugin only writes the fields.
- `ctx.raw_symbol_of(fn)` gives the C symbol of an oracle/derivative `Function` or `ExternalOracle`
  from the descriptor. Scaly renders ordinary functions into the same translation unit.
  An external oracle contributes its declared source and raw symbol directly before the wrapper.

The returned lines are C source, emitted verbatim into the translation unit between the oracle
kernels and the public entry point. The entry point follows Scaly's application binary interface,
or ABI.

### Required signature

With `I = len(desc.input_signature)` and `O = len(desc.output_signature)`:

```c
static void <ctx.raw_symbol>(const double* in0, ..., const double* in{I-1},
                             double* out0, ..., double* out{O-1},
                             double* w) { ... }
```

`w` is the caller's scratch workspace. Pass it as the last argument of every oracle `_raw` call.
An oracle that needs workspace cannot accept `NULL`. Do not use it for the
wrapper's own storage. Solver workspaces and O(n²) buffers belong in `static` locals, since the
wrapper is non-reentrant by contract (see [the generated interface](../how_it_works/generated_interface.md)).

### Oracle calling convention

Every descriptor `Function` renders as
`static void <name>_raw(const double* <in0>, ..., double* <out0>, ..., double* w)` with inputs and
outputs in the Function's declared order.

### Statistics

Fill every field of `ctx.stats_symbol` on every call, including early returns after an error.
The fields have the following meanings:

| Fields | Required values |
|---|---|
| `version`, `_pad0` | `SCALY_SOLVER_STATS_VERSION` and zero |
| `status` | A `SCALY_SOLVE_*` status from `src/scaly/solvers/stats.py` |
| `native_status` | The solver's status code, cast to `int32_t` |
| `iter`, `obj` | Iteration count and objective value |
| `t_total`, `t_fe`, `t_solver`, `t_qp`, `t_globalization`, `t_glue` | Total time and the function-evaluation, solver, QP, globalization, and wrapper portions |
| The five `n_eval_*` fields | Oracle evaluation counts |
| `primal_viol` | Infinity norm of constraint violation at the returned solution |
| `step_inf` | Infinity norm of the last computed step |
| `alpha` | Last accepted line-search step length, or zero if no step was accepted |
| `merit_penalty` | Final merit penalty, or zero if the method does not use one |
| `backtracks` | Total number of rejected line-search trial points |
| `qp_iter` | QP iterations accumulated across outer iterations |

Set diagnostics that the backend cannot provide to zero. Time operations with `scaly_clock_s()`,
which Scaly emits into every generated unit that contains a solver. Maintain
`t_total ≈ t_fe + t_solver + t_qp + t_globalization + t_glue`. A direct QP backend reports its solve
in `t_qp`. An NLP backend that cannot separate its internal work reports it in `t_solver`.

Map native statuses through the solver header's enum constants rather than integer literals.
An upstream rename or renumbering then causes a compile error instead of an incorrect status.

### Options

`desc.options` is the user's `options={...}` dict as a tuple of pairs. Lower each option into the
generated C (settings-struct assignments, `AddIpopt*Option` calls, ...) and raise
`NotImplementedError` for values that cannot be lowered. Options are baked as constants. The JIT
cache key includes the source hash, so each distinct set of options requires compilation.

## Descriptor families

Core normalizes every `Problem` into one `SolverDescriptor`. Plugins consume flat buffers and static
metadata. They do not inspect `Expr` nodes or reconstruct the user's trees.

Every typed solver uses the same flattened leaf order:

```text
inputs  = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq, [parameter leaves]
outputs = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq
```

`desc.n_var_blocks` is the number of variable leaves and determines all four fixed-group offsets.
`desc.param_names` names the leaves after `2 * n_var_blocks + 2` fixed inputs. The wrapper must
scatter its flat native solution and box multipliers back into the declared variable blocks.
Equality and inequality arrays are always present, including when their sizes are zero.

### Bound convention

Core oracles use IEEE negative infinity for an absent lower bound and IEEE positive infinity for an
absent upper bound, for variable and inequality bounds in both descriptor families. A plugin must
translate those values after evaluating the oracle and before calling a solver that uses a finite
sentinel. The PIQP adapter maps them to `-PIQP_INF` and `PIQP_INF`, the macro from the vendored
header, and the IPOPT adapter maps them to `-2e19` and `2e19`. Apply the same translation on initial
setup and on every update path. Do not make a wrapper depend on the identity of `sc.NO_LB` or
`sc.NO_UB`. Substitution and code generation preserve their values, not Python object identity.

### Quadratic programs

`kind == "qp"`, selected by `sc.solver(problem, "piqp")`.

- The problem shape is `min 0.5 x' P x + c' x` subject to `A x = b`, `l <= G x <= u`, and box bounds.
- `desc.oracle` takes parameter leaves and emits `P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub`. Empty constraint blocks are omitted from the oracle but remain size-zero multiplier groups in the solver signature.
- Dense matrices are row-major. When `desc.sparse` is true, the oracle emits compact compressed sparse column values in the baked `P_sparsity`, `A_sparsity`, and `G_sparsity` order. `P_sparsity` contains the upper triangle.
- The oracle emits IEEE infinities for absent bounds, and the wrapper converts them to the QP solver's native convention.
- A QP plugin is a standalone solver only. scaly-sqp does not consume this contract for its subproblems. Its wrapper is written against PIQP's C API and links `scaly-piqp`'s library. See the [user guide](../guide/solver_backends.md#scaly-sqp).

### Nonlinear programs

`kind == "nlp"`, selected by `sc.solver(problem, "ipopt")` or `"sqp"`.

- The problem shape is `min f(x,p)` subject to `h_eq(x,p) = 0`, two-sided inequalities, and box bounds.
- Core concatenates the variable leaves into one internal `x` for the oracles. `desc.base` maps `(x, *params)` to `f` and, when constraints exist, stacked `g = [h_eq; g_ineq]`.
- `desc.grad` has the same inputs and returns the dense objective gradient. `desc.jac` returns the compact sparse Jacobian of `g` in `desc.jac_sparsity` order.
- `desc.hess` takes `(x, *params, lam:f[, lam:g])` and returns the compact Lagrangian Hessian in `desc.hess_sparsity` order.
- `desc.bounds` takes only parameter leaves and returns `x_lb, x_ub[, l_ineq, u_ineq]`.
- Core asks the backend for `hess_triangle` and hands the wrapper an oracle and pattern already cut to that layout. IPOPT selects lower, and scaly-sqp selects upper.
- Any NLP oracle may instead be an `ExternalOracle` with the same signature. Its source defines `raw_symbol` using the flat-buffer convention, and `workspace_size` contributes to root and nested workspace packing.
- The bounds oracle emits IEEE infinities for absent bounds, and the wrapper converts them to the NLP solver's native convention.

## Versioning

`SOLVER_PLUGIN_PROTOCOL_VERSION` (`src/scaly/solvers/registry.py`) covers the whole contract above:
descriptor semantics and oracle output orderings, the `SolverWrapperCtx` fields, the `_raw` calling
convention, and the `scaly_solver_stats` layout (`SCALY_SOLVER_STATS_VERSION` tracks the struct ABI
itself, and a stats change bumps both). Any breaking change to any of these bumps the protocol version,
and `get_backend` refuses plugins declaring a different version.

History:

- v1: packaging metadata only (hand-written templates in core).
- v2: plugin-owned codegen via `render_wrapper`.
- v3: external NLP oracles and the stats-v2 QP/globalization timing fields.
- v4: the stats-v3 per-solve diagnostics tail (`primal_viol`, `step_inf`, `alpha`, `merit_penalty`,
  `backtracks`, `qp_iter`), appended after `_pad0`. The struct grows from 96 to 136 bytes.
- v5: backend-selected NLP Hessian triangles and the compact oracle output convention that the
  descriptor pattern is the handed layout.
- v6: typed `Problem`/`Function` solver signatures, variable-block metadata, and the fixed
  warm-start and result order.
- v7: IEEE-infinity semantics for absent bounds in core QP and NLP oracles. Plugins normalize them
  to native solver sentinels.

## What core owns

Plugins must not duplicate any of this:

- Problem normalization and oracle assembly (`sc.problem` / `sc.solver`), including derivative
  factories and sparsity detection.
- The universal C ABI entry point, workspace packing, and the `_raw` kernel rendering (Program IR).
- The `scaly_solver_stats` struct, the `SCALY_SOLVE_*` status enum, and `scaly_clock_s`. Plugins
  fill and use them, never redefine them.
- JIT compilation, caching (keyed on source and flags), and library/header discovery.
- The typed `Solver` and `Function` call interfaces and `Function.solver_stats()`.

## Checklist for a new plugin

1. Create the package. If the solver has its own C API, add a hatch build hook that bundles the
   library and headers. Use `plugins/scaly-piqp` as a reference.
2. Define `BACKEND` with the metadata fields and `render_wrapper`. Expose it through the
   `scaly.solvers` entry point.
3. Implement the wrapper. Call the oracles, normalize infinite bounds, and drive the solver's C API.
   Use native enum constants for statuses and fill the statistics on every return path.
   Prefix static state with `ctx.symbol`.
4. Add tests under `plugins/scaly-<name>/tests/`. Compare with analytic or reference solutions,
   compile a nested solver call, and check the statistics. The PIQP and IPOPT suites provide examples.
5. Nothing in `src/scaly/` should need to change. If it does, the protocol is missing something.
   Raise it as a core issue instead of forking core.
