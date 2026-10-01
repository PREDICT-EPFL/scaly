# Solver plugins

How a solver integration plugs into scaly. Read this when writing a new solver plugin. Everything
solver-specific lives in the plugin package; nothing is added to the scaly codebase. CasADi's
`Conic`/`Nlpsol` plugins must be written inside the CasADi tree against its internal C++ headers,
whereas a scaly solver plugin is an ordinary pip-installable Python package. (Decision record:
`internal/notes/benchmark-buildout.md` §3.5.)

There is exactly one solve path (`internal/notes/benchmark-buildout.md` §3.4): a generated C
wrapper, emitted alongside the oracle kernels into a single translation unit, calling the solver's
C API directly. Plugins ship no Python solve code. They ship a vendored native library, C headers,
packaging metadata, and a Python method class that renders the C wrapper.

## Anatomy of a plugin

A plugin is a Python package that:

1. depends on `scaly-numerics` (the distribution that ships `scaly.opt`, and the core with it), and
   checks at import the extension API it was written against
   (`scaly.ext.require_ext_api(1, "scaly-mysolver")`);
2. defines a method class, a frozen dataclass subclassing `scaly.opt.external.External`, and
   declares it in the `scaly.methods` entry-point group as `opt.<name>`;
3. when it wraps its own native solver, bundles the vendored shared library under `<pkg>/lib/` and
   its C headers under `<pkg>/include/`, built by a hatch build hook (`plugins/scaly-piqp` and
   `plugins/scaly-ipopt` are the two reference implementations). `plugins/scaly-sqp` has no
   library of its own; it generates C against `scaly-piqp`'s.

```toml
# pyproject.toml
[project.entry-points."scaly.methods"]
"opt.mysolver" = "scaly_mysolver:MySolver"
```

Users then write `sc.opt.solver(problem, sc.opt.MySolver(options={...}))`, or
`sc.opt.solver(problem, "mysolver")` for the default options: `sc.opt` resolves the class by the
entry point's object name the first time it is read, without importing the plugin before.

## The method class

```python
@dataclass(frozen=True)
class MySolver(External):
  name = "opt.mysolver"
  kind = "nlp"
  hess_triangle = "lower"
  lib_stem = "mysolver"
  link_flags = ("-lmysolver",)
  header = "mysolver/api.h"

  include_dir = staticmethod(include_dir)
  lib_dir = staticmethod(lib_dir)

  def check_options(self) -> None: ...  # refuse options the solver does not take

  def render_wrapper(self, fun, ctx) -> list[str]: ...
```

| member | meaning |
|---|---|
| `name` | `"opt.<name>"`; must equal the entry-point name (checked when the registry loads it) |
| `api` | inherited: the method API of `sc.opt.NLP` it implements (`scaly.opt.method.METHOD_API`); a mismatch is refused at load |
| `kind` | `"qp"` (the method takes the quadratic normal form and refuses a problem that is not quadratic) or `"nlp"` (it takes the oracles) |
| `hess_triangle` | NLP only: `"lower"` or `"upper"`, the Hessian triangle the wrapper consumes |
| `lib_stem` | shared-library stem: the file is `lib<stem>.dylib` / `lib<stem>.so` |
| `link_flags` | linker flags, e.g. `("-lmysolver",)` |
| `header` | C header path relative to `include_dir()`, e.g. `"mysolver/api.h"`; core emits `#include "<header>"` in solver-bearing translation units |
| `lib_dir()` / `include_dir()` | vendored library / header directories; they join the JIT's `-L`/`-I`/rpath search path |
| `options` | inherited dataclass field: the solver's own options by name, compiled into the wrapper |
| `check_options()` | refuse what the solver does not take, so a bad option fails when the method is made |
| `render_wrapper(fun, ctx)` | the C wrapper template (below) |

A method may add typed fields of its own (PIQP's `sparse`). `External` implements `supports` (a QP
method proves the problem quadratic) and `build` (it builds the solver's descriptor from the
problem's normal form, `sc.opt.extract_qp` or `sc.opt.nlp_oracles`). The registry
(`src/scaly/opt/method.py`) finds methods and checks their name and API; library and header
discovery (`src/scaly/opt/external/paths.py`) derives everything else from the class attributes:
`solver_loadable("mysolver")`, compile/link flags, diagnostics, and the per-plugin exact-path
override env var `SCALY_MYSOLVER_LIB`.

## The codegen contract: `render_wrapper`

```python
def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]: ...
```

`fun` is the plain typed `Function` being rendered; `solver_descriptor(fun)`
(`scaly.opt.external.graph`) returns its `SolverDescriptor` (`src/scaly/opt/external/model.py`), which carries
the problem dimensions, input/output signatures, oracle/derivative `Function`s, sparsity patterns,
and user options. The descriptor is also `fun.extern`: it is the Function's extern callee
(`scaly.function.extern`), which is how the compiler reaches the wrapper at all. `ctx` is the
codegen kit (`scaly.opt.external.wrapper.SolverWrapperCtx`):

- `ctx.symbol` is the solver's mangled C identifier. Prefix every static the template declares with
  it, since multiple solvers can share one translation unit.
- `ctx.raw_symbol` is the name of the function the template must define. Core defines the solver's
  exported raw function around it: it calls this one, then writes the Function's `Info` outputs
  (`status`, `iter`, `obj`, `primal_viol`) from the stats the template filled.
- `ctx.stats_symbol` is the `scaly_solver_stats` static the template must fill on every call. Core
  declares it and exports the `<symbol>_stats(...)` accessor; the plugin only writes the fields.
- `ctx.raw_symbol_of(fn)` gives the C symbol of an oracle/derivative `Function` or `ExternalOracle`
  from the descriptor. Scaly Functions are rendered into the same translation unit by Program IR; an
  external oracle contributes its declared source and raw symbol directly before the wrapper.

The returned lines are C source, emitted verbatim into the translation unit between the oracle
kernels and the universal-ABI entry point.

### Required signature

With `I = len(desc.input_signature)` and `O = len(desc.output_signature)`, the solution's outputs
(the `Info` outputs are core's, after them):

```c
static void <ctx.raw_symbol>(const double* in0, ..., const double* in{I-1},
                             double* out0, ..., double* out{O-1},
                             double* w) { ... }
```

`w` is the caller's packed scratch workspace. Pass it through as the last argument of every oracle
`_raw` call; the kernels need it and crash on NULL at any nontrivial size. Do not use it for the
wrapper's own storage. Solver workspaces and O(n²) buffers belong in `static` locals, since the
wrapper is non-reentrant by contract (see [the generated interface](../how_it_works/generated_interface.md)).

### Oracle calling convention

Every descriptor `Function` renders as
`static void <name>_raw(const double* <in0>, ..., double* <out0>, ..., double* w)` with inputs and
outputs in the Function's declared order.

### Stats

Fill every field of `ctx.stats_symbol` on every call, including early-error returns:
`version = SCALY_SOLVER_STATS_VERSION`, `status` (one of the `SCALY_SOLVE_*` macros, the
method-neutral `sc.Status` codes), `native_status` (the solver's own code, cast
to `int32_t`), `iter`, `obj`, the `t_total`/`t_fe`/`t_solver`/`t_qp`/`t_globalization`/`t_glue`
timing split, the five `n_eval_*` counters, `_pad0 = 0`, and the stats-v3 diagnostics tail:
`primal_viol` (constraint violation at the returned `x`, inf norm), `step_inf` (inf norm of the last
computed step), `alpha` (last accepted line-search step length; `0.0` if no step was accepted),
`merit_penalty` (final merit penalty parameter, or zero for a globalization such as a filter that
has no merit penalty), `backtracks` (total rejected line-search trial points), and `qp_iter` (QP
iteration count accumulated across outer iterations). Fill diagnostics the backend has no concept
of with zero.

Time with `scaly_clock_s()`, which core emits into every solver-bearing unit, and maintain
`t_total ≈ t_fe + t_solver + t_qp + t_globalization + t_glue`. A direct QP backend reports its solve
in `t_qp`; an NLP backend that cannot expose its internal split reports it in `t_solver`.

Map native statuses through the vendored header's enum constants, never integer literals, so an
upstream rename or renumbering breaks at compile time instead of silently. This is what removes the
drift problem of hand-written bindings (`internal/notes/benchmark-buildout.md` §3.4).

### Options

`desc.options` is the method's `options` as a tuple of pairs. Lower each option into the
generated C (settings-struct assignments, `AddIpopt*Option` calls, ...); refuse the ones that cannot
be lowered in `check_options`, so the error comes when the method is made. Options are baked as constants; the JIT
cache key covers them through the source hash, so option sweeps recompile per point (accepted, see
`internal/notes/benchmark-buildout.md` §3.4).

## Descriptor families

Core normalizes every problem into one `SolverDescriptor`. Plugins consume flat buffers and static
metadata; they do not inspect `Expr` nodes or reconstruct the user's trees.

Every typed solver uses the same flattened leaf order:

```text
inputs  = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq, [parameter leaves]
outputs = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq      (desc.output_signature)
          info:status, info:iter, info:objective, info:primal_residual      (core's frame)
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
header; the IPOPT adapter maps them to `-2e19` and `2e19`. Apply the same translation on initial
setup and on every update path. Do not make a wrapper depend on the identity of `sc.opt.NO_LB` or
`sc.opt.NO_UB`; substitution and code generation preserve their values, not Python object identity.

### QP

`kind == "qp"`: PIQP.

- The problem shape is `min 0.5 x' P x + c' x` subject to `A x = b`, `l <= G x <= u`, and box bounds.
  The objective's constant is not part of it: fill `obj` with the solver's `0.5 x' P x + c' x`, and
  core's frame adds the constant (`desc.objective_constant`) to the stats and the `Info` objective.
- `desc.oracle` takes parameter leaves and emits `P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub`. Empty constraint blocks are omitted from the oracle but remain size-zero multiplier groups in the solver signature.
- Dense matrices are row-major. When `desc.sparse` is true, the oracle emits compact compressed sparse column values in the baked `P_sparsity`, `A_sparsity`, and `G_sparsity` order. `P_sparsity` contains the upper triangle.
- The oracle emits IEEE infinities for absent bounds; the wrapper converts them to the QP solver's native convention.
- A QP plugin is a standalone solver only. scaly-sqp does not consume this contract for its subproblems; its wrapper is written against PIQP's C API and links `scaly-piqp`'s library. See the [user guide](../guide/solver_backends.md#scaly-sqp).

### NLP

`kind == "nlp"`: IPOPT and SQP.

- The problem shape is `min f(x,p)` subject to `h_eq(x,p) = 0`, two-sided inequalities, and box bounds.
- Core concatenates the variable leaves into one internal `x` for the oracles. `desc.base` maps `(x, *params)` to `f` and, when constraints exist, stacked `g = [h_eq; g_ineq]`.
- `desc.grad` has the same inputs and returns the dense objective gradient. `desc.jac` returns the compact sparse Jacobian of `g` in `desc.jac_sparsity` order.
- `desc.hess` takes `(x, *params, lam:f[, lam:g])` and returns the compact Lagrangian Hessian in `desc.hess_sparsity` order.
- `desc.bounds` takes only parameter leaves and returns `x_lb, x_ub[, l_ineq, u_ineq]`.
- Core asks the method for `hess_triangle` and hands the wrapper an oracle and pattern already cut to that layout. IPOPT selects lower; scaly-sqp selects upper.
- Any NLP oracle may instead be an `ExternalOracle` with the same signature. Its source defines `raw_symbol` using the flat-buffer convention, and `workspace_size` contributes to root and nested workspace packing.
- The bounds oracle emits IEEE infinities for absent bounds; the wrapper converts them to the NLP solver's native convention.

## Versioning

`METHOD_API` (`src/scaly/opt/method.py`) is the method API of `sc.opt.NLP`, and it covers the whole
contract above: the method class, descriptor semantics and oracle output orderings, the
`SolverWrapperCtx` fields, the `_raw` calling convention and the `Info` outputs, and the
`scaly_solver_stats` layout (`SCALY_SOLVER_STATS_VERSION` tracks the struct ABI itself; a stats
change bumps both). Any breaking change to any of these bumps it, and the registry refuses a method
declaring a different one. It continues the plugin protocol's numbering.

History:

- v1: packaging metadata only (hand-written templates in core).
- v2: plugin-owned codegen via `render_wrapper`.
- v3: external NLP oracles and the stats-v2 QP/globalization timing fields.
- v4: the stats-v3 per-solve diagnostics tail (`primal_viol`, `step_inf`, `alpha`, `merit_penalty`,
  `backtracks`, `qp_iter`), appended after `_pad0`; the struct grows from 96 to 136 bytes.
- v5: backend-selected NLP Hessian triangles and the compact oracle output convention that the
  descriptor pattern is the handed layout.
- v6: typed `Problem`/`Function` solver signatures, variable-block metadata, and the fixed
  warm-start and result order.
- v7: IEEE-infinity semantics for absent bounds in core QP and NLP oracles; plugins normalize them
  to native solver sentinels.
- v8: the descriptor is the solver Function's extern callee, read with `solver_descriptor(fun)`
  (`fun.descriptor` is gone), and `SolverWrapperCtx` moved to `scaly.opt.external.wrapper`.
- v9 (`METHOD_API`): plugins are method classes (`External`) in the `scaly.methods` group under
  `opt.<name>` instead of backend objects in `scaly.solvers`; options are the method's, checked when
  it is made; the wrapper defines `<raw>_solve` and core's frame adds the `Info` outputs.

## What core owns

Plugins must not duplicate any of this:

- Problem normalization and oracle assembly (`sc.opt.problem`, the normal forms `sc.opt.extract_qp`
  and `sc.opt.nlp_oracles`, `sc.opt.solver`), including derivative factories and sparsity detection.
- The universal C ABI entry point, workspace packing, and the `_raw` kernel rendering (Program IR).
- The `scaly_solver_stats` struct, the `SCALY_SOLVE_*` status enum (`sc.Status`), the `Info`
  outputs, and `scaly_clock_s`. Plugins fill and use them, never redefine them.
- JIT compilation, caching (keyed on source, compiler and flags, and found again from the graph and
  the C your callee renders), and library/header discovery. The JIT asks your callee for its
  dependencies, sources, C and build requirements every time a Function holding it is built, also
  when the library is already cached, so those four must be cheap, must not change anything, and
  must give the same answer for the same callee.
- The typed `Function` call interface and `sc.opt.solver_stats(fun)`, which reads the stats accessor
  through the extern-callee protocol (`Function.callee_state`).

## Checklist for a new plugin

1. Package skeleton and, for a solver with its own C API, a hatch build hook vendoring the library
   and headers (copy `plugins/scaly-piqp`).
2. The method class (`External`) with the metadata, `check_options` and `render_wrapper`, declared
   under `opt.<name>` in the `scaly.methods` entry-point group.
3. The wrapper template: drive the solver's C API from the oracle kernels, map statuses through enum
   constants, fill the stats struct, keep all state in `ctx.symbol`-prefixed statics. Normalize IEEE
   infinite bounds before every native setup or update call.
4. Tests under `plugins/scaly-<name>/tests/`, each marked `@pytest.mark.method("opt.<name>")`:
   the conformance suite of its problem class from `scaly.testing.conformance` (a QP method runs
   `qp.check_solves` and `qp.check_refuses` as `test_conformance_piqp.py` does), correctness
   against analytic or reference solutions, a nested-solve JIT test, and a stats sanity check (see
   the piqp and ipopt test suites for the pattern). List the method in the table of
   `tests/conformance/` too, or say there why it is held to another suite: an installed method
   missing from it fails.
5. Nothing in `src/scaly/` should need to change. If it does, the protocol is missing something.
   Raise it as a core issue instead of forking core.
