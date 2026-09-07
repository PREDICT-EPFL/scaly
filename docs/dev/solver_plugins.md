# Solver plugins

How a solver integration plugs into alloy. This is the document to read when
writing a new solver plugin — everything solver-specific lives in the plugin
package; nothing needs to be added to the alloy codebase. That is the design
point: CasADi's `Conic`/`Nlpsol` plugins must be written inside the CasADi
tree against its internal C++ headers, whereas an alloy solver plugin is an
ordinary pip-installable Python package. (Decision record: `internal/notes/benchmark-buildout.md`
§3.5.)

The binding doctrine behind the interface is `internal/notes/benchmark-buildout.md` §3.4: there is
exactly one solve path — a generated C wrapper, emitted alongside the oracle
kernels into a single translation unit, calling the solver's C API directly.
Plugins therefore ship **no Python solve code**; they ship a vendored native
library, C headers, packaging metadata, and a Python function that renders
the C wrapper.

## Anatomy of a plugin

A plugin is a Python package that:

1. depends on `alloy`;
2. bundles the vendored solver shared library under `<pkg>/lib/` and its C
   headers under `<pkg>/include/` (built by a hatch build hook — see
   `plugins/*/hatch_build.py` for the two reference implementations);
3. exposes a backend object through the `alloy.solvers` entry-point group:

```toml
# pyproject.toml
[project.entry-points."alloy.solvers"]
mysolver = "alloy_mysolver:BACKEND"
```

The entry-point name is the backend string users pass as the second argument to `al.solver(problem, backend)`.

## The backend protocol

`BACKEND` must satisfy `alloy.solvers.registry.SolverBackend`. This protocol defines the common backend metadata and codegen hook:

| member | meaning |
|---|---|
| `name` | backend name; must equal the entry-point name |
| `kind` | `"qp"` or `"nlp"` — which descriptor family the solver accepts |
| `protocol_version` | the `SOLVER_PLUGIN_PROTOCOL_VERSION` the plugin was written against (validated at load; mismatch is a hard error) |
| `lib_stem` | shared-library stem: the file is `lib<stem>.dylib` / `lib<stem>.so` |
| `link_flags` | linker flags, e.g. `("-lmysolver",)` |
| `header` | C header path relative to `include_dir()`, e.g. `"mysolver/api.h"`; core emits `#include "<header>"` in solver-bearing translation units |
| `lib_dir()` / `include_dir()` | vendored library / header directories; they join the JIT's `-L`/`-I`/rpath search path |
| `render_wrapper(fun, ctx)` | the C wrapper template (below) |

NLP backends also satisfy `alloy.solvers.registry.NlpSolverBackend`. They must declare `hess_triangle` as `"lower"` or `"upper"`; `require_backend(..., "nlp")` validates it at runtime. QP backends do not declare this member.

Discovery, protocol-version validation, and kind checking live in
`src/alloy/solvers/registry.py`. Library and header discovery
(`src/alloy/solvers/paths.py`) derives everything else from this metadata:
`solver_loadable("mysolver")`, compile/link flags, diagnostics, and the
per-plugin exact-path override env var `ALLOY_MYSOLVER_LIB`.

## The codegen contract: `render_wrapper`

```python
def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]: ...
```

`fun` is the plain typed `Function` being rendered; `fun.descriptor` (a
`SolverDescriptor`, `src/alloy/solvers/model.py`) carries the
problem dimensions, input/output signatures, oracle/derivative `Function`s,
sparsity patterns, and user options. `ctx` is the codegen kit
(`alloy.codegen.solver.SolverWrapperCtx`):

- `ctx.symbol` — the solver's mangled C identifier. Prefix every static the
  template declares with it (multiple solvers can share one translation unit).
- `ctx.raw_symbol` — the name of the function the template **must define**.
- `ctx.stats_symbol` — the `alloy_solver_stats` static the template **must
  fill** on every call. Core declares it and exports the
  `<symbol>_stats(...)` accessor; the plugin only writes the fields.
- `ctx.raw_symbol_of(fn)` — the C symbol of an oracle/derivative `Function`
  or `ExternalOracle` from the descriptor. Alloy Functions are rendered into
  the same translation unit by Program IR; an external oracle contributes its
  declared source and raw symbol directly before the wrapper.

The returned lines are C source, emitted verbatim into the translation unit
between the oracle kernels and the universal-ABI entry point.

**Required signature.** With `I = len(desc.input_signature)` and
`O = len(desc.output_signature)`:

```c
static void <ctx.raw_symbol>(const double* in0, ..., const double* in{I-1},
                             double* out0, ..., double* out{O-1},
                             double* w) { ... }
```

`w` is the caller's packed scratch workspace. Pass it through as the last
argument of every oracle `_raw` call — the kernels need it and crash on NULL
at any nontrivial size. Do not use it for the wrapper's own storage; solver
workspaces and O(n²) buffers belong in `static` locals (the wrapper is
non-reentrant by contract, see [the ABI](../how_it_works/c_abi.md)).

**Oracle calling convention.** Every descriptor `Function` renders as
`static void <name>_raw(const double* <in0>, ..., double* <out0>, ..., double* w)`
with inputs and outputs in the Function's declared order.

**Stats.** Fill every field of `ctx.stats_symbol` on every call (including
early-error returns): `version = ALLOY_SOLVER_STATS_VERSION`, `status` (one
of the `ALLOY_SOLVE_*` macros — the backend-neutral enum in
`src/alloy/solvers/stats.py`), `native_status` (the solver's own code, cast
to `int32_t`), `iter`, `obj`, the `t_total`/`t_fe`/`t_solver`/`t_qp`/
`t_globalization`/`t_glue` timing split, the five `n_eval_*` counters,
`_pad0 = 0`, and the stats-v3 diagnostics tail — `primal_viol` (constraint
violation at the returned `x`, inf norm), `step_inf` (inf norm of the last
computed step), `alpha` (last accepted line-search step length; `0.0` if no
step was accepted), `merit_penalty` (final merit penalty parameter, or zero
for a globalization such as a filter that has no merit penalty),
`backtracks` (total rejected line-search trial points), and `qp_iter` (QP
iteration count accumulated across outer iterations). Diagnostics the backend
has no concept of are filled with zero. Time with `alloy_clock_s()` (emitted
by core into every solver-bearing unit); maintain `t_total ≈ t_fe + t_solver
+ t_qp + t_globalization + t_glue`. A direct QP backend reports its solve in `t_qp`;
an NLP backend that cannot expose its internal split reports it in
`t_solver`. Map native statuses through the
vendored header's enum **constants**, not integer literals, so upstream
renames/renumbers break at compile time instead of silently — this is how
the drift problem of hand-written bindings is dissolved structurally
(`internal/notes/benchmark-buildout.md` §3.4).

**Options.** `desc.options` is the user's `options={...}` dict as a tuple of
pairs. Lower each option into the generated C (settings-struct assignments,
`AddIpopt*Option` calls, ...) and raise `NotImplementedError` for values that
cannot be lowered. Options are baked as constants; the JIT cache key covers
them through the source hash, so option sweeps recompile per point (accepted,
see `internal/notes/benchmark-buildout.md` §3.4).

## Descriptor families

Core normalizes every `Problem` into one `SolverDescriptor`. Plugins consume flat buffers and static metadata; they do not inspect `Expr` nodes or reconstruct the user's trees.

Every typed solver uses the same flattened leaf order:

```text
inputs  = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq, [parameter leaves]
outputs = [variable leaves], [box-multiplier leaves], lam_eq, lam_ineq
```

`desc.n_var_blocks` is the number of variable leaves. It determines all four fixed-group offsets. `desc.param_names` names the leaves after `2 * n_var_blocks + 2` fixed inputs. The wrapper must scatter its flat native solution and box multipliers back into the declared variable blocks. Equality and inequality arrays are always present, including when their sizes are zero.

**Bound convention.** Core oracles use IEEE negative infinity for an absent lower bound and IEEE
positive infinity for an absent upper bound. This rule applies to variable and inequality bounds in
both descriptor families. A plugin must translate those values after evaluating the oracle and
before calling a solver that uses a finite sentinel. For example, the built-in PIQP adapter maps
them to `-1e30` and `1e30`, while the IPOPT adapter maps them to `-2e19` and `2e19`. Apply the same
translation on initial setup and every update path. Do not make a wrapper depend on the identity of
`al.NO_LB` or `al.NO_UB`; substitution and code generation preserve their values, not Python object
identity.

**QP** (`kind == "qp"`, selected by `al.solver(problem, "piqp")`):

- The problem shape is `min 0.5 x' P x + c' x` subject to `A x = b`, `l <= G x <= u`, and box bounds.
- `desc.oracle` takes parameter leaves and emits `P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub`. Empty constraint blocks are omitted from the oracle but remain size-zero multiplier groups in the solver signature.
- Dense matrices are row-major. When `desc.sparse` is true, the oracle emits compact compressed sparse column values in the baked `P_sparsity`, `A_sparsity`, and `G_sparsity` order. `P_sparsity` contains the upper triangle.
- The oracle emits IEEE infinities for absent bounds; the wrapper converts them to the QP solver’s native convention.
- A QP plugin is a standalone solver only. alloy-sqp does not consume this contract for its subproblems; its wrapper is written against PIQP's C API and links `alloy-piqp`'s library. See the [user guide](../guide/solver_backends.md#alloy-sqp).

**NLP** (`kind == "nlp"`, selected by `al.solver(problem, "ipopt")` or `"sqp"`):

- The problem shape is `min f(x,p)` subject to `h_eq(x,p) = 0`, two-sided inequalities, and box bounds.
- Core concatenates the variable leaves into one internal `x` for the oracles. `desc.base` maps `(x, *params)` to `f` and, when constraints exist, stacked `g = [h_eq; g_ineq]`.
- `desc.grad` has the same inputs and returns the dense objective gradient. `desc.jac` returns the compact sparse Jacobian of `g` in `desc.jac_sparsity` order.
- `desc.hess` takes `(x, *params, lam:f[, lam:g])` and returns the compact Lagrangian Hessian in `desc.hess_sparsity` order.
- `desc.bounds` takes only parameter leaves and returns `x_lb, x_ub[, l_ineq, u_ineq]`.
- Core asks the backend for `hess_triangle` and hands the wrapper an oracle and pattern already cut to that layout. IPOPT selects lower; alloy-sqp selects upper.
- Any NLP oracle may instead be an `ExternalOracle` with the same signature. Its source defines `raw_symbol` using the flat-buffer convention, and `workspace_size` contributes to root and nested workspace packing.
- The bounds oracle emits IEEE infinities for absent bounds; the wrapper converts them to the NLP solver’s native convention.

## Versioning

`SOLVER_PLUGIN_PROTOCOL_VERSION` (`src/alloy/solvers/registry.py`) covers the
whole surface above: descriptor semantics and oracle output orderings, the
`SolverWrapperCtx` fields, the `_raw` calling convention, and the
`alloy_solver_stats` layout (`ALLOY_SOLVER_STATS_VERSION` tracks the struct
ABI itself; a stats change bumps both). Any breaking change to any of these
bumps the protocol version, and `get_backend` refuses plugins declaring a
different version. History: v1 = packaging metadata only (hand-written
templates in core); v2 = plugin-owned codegen via `render_wrapper`; v3 =
external NLP oracles and the stats-v2 QP/globalization timing fields; v4 =
the stats-v3 per-solve diagnostics tail (`primal_viol`, `step_inf`, `alpha`,
`merit_penalty`, `backtracks`, `qp_iter` — appended after `_pad0`, struct
grows from 96 to 136 bytes); v5 = backend-selected NLP Hessian triangles and
the compact oracle output convention that the descriptor pattern is the
handed layout; v6 = typed `Problem`/`Function` solver signatures, variable-block metadata,
and the fixed warm-start and result order. v7 = IEEE-infinity semantics for absent bounds in
core QP and NLP oracles; plugins normalize them to native solver sentinels.

## What core owns (and plugins must not duplicate)

- Problem normalization and oracle assembly (`al.problem` / `al.solver`), including
  derivative factories and sparsity detection.
- The universal C ABI entry point, workspace packing, and the `_raw` kernel
  rendering (Program IR).
- The `alloy_solver_stats` struct, the `ALLOY_SOLVE_*` status enum, and
  `alloy_clock_s` — plugins fill/use them, never redefine them.
- JIT compilation, caching (source + flags keyed), and library/header
  discovery.
- The typed `Function` call surface and `Function.solver_stats()`.

## Checklist for a new plugin

1. Package skeleton + hatch build hook vendoring the solver lib/headers
   (copy `plugins/alloy-piqp` for a C-API solver).
2. `BACKEND` object with the metadata fields and `render_wrapper`, exposed
   via the `alloy.solvers` entry point.
3. The wrapper template: drive the solver's C API from the oracle kernels,
   map statuses via enum constants, fill the stats struct, keep all state in
   `ctx.symbol`-prefixed statics. Normalize IEEE infinite bounds before every
   native setup or update call.
4. Tests under `plugins/alloy-<name>/tests/`: correctness against analytic /
   reference solutions, a nested-solve JIT test, and a stats sanity check
   (see the piqp/ipopt test suites for the pattern).
5. Nothing in `src/alloy/` should need to change. If it does, the protocol
   is missing something — that is a core issue to raise, not a reason to
   fork core.
