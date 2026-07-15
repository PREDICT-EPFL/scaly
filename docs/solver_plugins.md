# Solver plugins

How a solver integration plugs into alloy. This is the document to read when
writing a new solver plugin — everything solver-specific lives in the plugin
package; nothing needs to be added to the alloy codebase. That is the design
point: CasADi's `Conic`/`Nlpsol` plugins must be written inside the CasADi
tree against its internal C++ headers, whereas an alloy solver plugin is an
ordinary pip-installable Python package. (Decision record: root `ROADMAP.md`
§3.5.)

The binding doctrine behind the interface is `ROADMAP.md` §3.4: there is
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

The entry-point name is the string users pass as `al.qp(..., solver=...)` /
`al.nlp(..., solver=...)`.

## The backend protocol

`BACKEND` must satisfy `alloy.solvers.registry.SolverBackend`:

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

Discovery, protocol-version validation, and kind checking live in
`src/alloy/solvers/registry.py`. The toolchain (`src/alloy/toolchain.py`)
derives everything else from this metadata: `solver_loadable("mysolver")`,
compile/link flags, diagnostics, and the per-plugin exact-path override env
var `ALLOY_MYSOLVER_LIB`.

## The codegen contract: `render_wrapper`

```python
def render_wrapper(fun: Function, ctx: SolverWrapperCtx) -> list[str]: ...
```

`fun` is the `SolverFunction` being rendered; `fun.descriptor` (a
`SolverDescriptor`, `src/alloy/solvers/solver_function.py`) carries the
problem dimensions, input/output signatures, oracle/derivative `Function`s,
sparsity patterns, and user options. `ctx` is the codegen kit
(`alloy.codegen.solver_c.SolverWrapperCtx`):

- `ctx.symbol` — the solver's mangled C identifier. Prefix every static the
  template declares with it (multiple solvers can share one translation unit).
- `ctx.raw_symbol` — the name of the function the template **must define**.
- `ctx.stats_symbol` — the `alloy_solver_stats` static the template **must
  fill** on every call. Core declares it and exports the
  `<symbol>_stats(...)` accessor; the plugin only writes the fields.
- `ctx.raw_symbol_of(fn)` — the C symbol of an oracle/derivative `Function`
  from the descriptor (they are rendered into the same translation unit by
  Program IR, before the wrapper).

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
non-reentrant by contract, see `docs/spec.md`).

**Oracle calling convention.** Every descriptor `Function` renders as
`static void <name>_raw(const double* <in0>, ..., double* <out0>, ..., double* w)`
with inputs and outputs in the Function's declared order.

**Stats.** Fill every field of `ctx.stats_symbol` on every call (including
early-error returns): `version = ALLOY_SOLVER_STATS_VERSION`, `status` (one
of the `ALLOY_SOLVE_*` macros — the backend-neutral enum in
`src/alloy/solvers/stats.py`), `native_status` (the solver's own code, cast
to `int32_t`), `iter`, `obj`, the `t_total`/`t_fe`/`t_solver`/`t_glue` timing
split, the five `n_eval_*` counters, and `_pad0 = 0`. Time with
`alloy_clock_s()` (emitted by core into every solver-bearing unit); maintain
`t_total ≈ t_fe + t_solver + t_glue`. Map native statuses through the
vendored header's enum **constants**, not integer literals, so upstream
renames/renumbers break at compile time instead of silently — this is how
the drift problem of hand-written bindings is dissolved structurally
(`ROADMAP.md` §3.4).

**Options.** `desc.options` is the user's `options={...}` dict as a tuple of
pairs. Lower each option into the generated C (settings-struct assignments,
`AddIpopt*Option` calls, ...) and raise `NotImplementedError` for values that
cannot be lowered. Options are baked as constants; the JIT cache key covers
them through the source hash, so option sweeps recompile per point (accepted,
see `ROADMAP.md` §3.4).

## Descriptor families

Core normalizes problems; plugins consume the normalized form. The plugin
does not parse user input and never sees `Expr`s — only `Function`s to call
and static metadata.

**QP** (`kind == "qp"`, built by `al.qp`):

- shape: `min ½xᵀPx + cᵀx  s.t.  A_eq x = b_eq,  l ≤ G_ineq x ≤ u,  x_lb ≤ x ≤ x_ub`.
- wrapper inputs: `x0`, `lam_eq0`, `lam_ineq0`, then `desc.param_names` in
  order. Outputs: `x`, `cost`, `lam_eq`†, `lam_ineq`†, `lam_box` (†present
  only when the constraint block exists; signed convention: positive ⇒ upper
  bound active).
- one oracle `Function` (`desc.oracle`): params in, flattened QP data out, in
  the order `P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub`
  (optional blocks present only when nonempty; row-major dense, or compact
  CSC-ordered values when `desc.sparse` with the patterns in
  `desc.P_sparsity` (upper triangle) / `A_sparsity` / `G_sparsity`).
- "no bound" is encoded as `±1e30` (`qp.PIQP_INF`, the QP family's infinity
  sentinel); clamp or translate to the solver's own convention.

**NLP** (`kind == "nlp"`, built by `al.nlp`):

- shape: `min f(x,p)  s.t.  h_eq(x,p) = 0,  l ≤ g_ineq(x,p) ≤ u,  x_lb ≤ x ≤ x_ub`.
- wrapper inputs: `x0`, `lam_eq0`, `lam_ineq0`, `lam_box0`, then params.
  Outputs: `x`, `f`, `h_eq`, `g_ineq`, `lam_eq`, `lam_ineq`, `lam_box`
  (signed multipliers; `lam_box = z_U − z_L`).
- oracle `Function`s: `desc.base` `(x, *params) → (f, g_all)` with
  `g_all = [h_eq; g_ineq]` stacked; `desc.grad` (dense objective gradient);
  `desc.jac` (compact sparse Jacobian of `g_all`, COO pattern in
  `desc.jac_sparsity`); `desc.hess` (compact sparse Lagrangian Hessian,
  inputs `(x, obj_factor, [lam,] *params)`, symmetric COO pattern in
  `desc.hess_sparsity` with `desc.hess_lower_mask` marking the lower
  triangle); `desc.bounds` `(*params) → (x_lb, x_ub[, l_ineq, u_ineq])`.
- "no bound" is `±inf` from the bounds oracle; clamp to the solver's
  convention (IPOPT: `±2e19`).

## Versioning

`SOLVER_PLUGIN_PROTOCOL_VERSION` (`src/alloy/solvers/registry.py`) covers the
whole surface above: descriptor semantics and oracle output orderings, the
`SolverWrapperCtx` fields, the `_raw` calling convention, and the
`alloy_solver_stats` layout (`ALLOY_SOLVER_STATS_VERSION` tracks the struct
ABI itself; a stats change bumps both). Any breaking change to any of these
bumps the protocol version, and `get_backend` refuses plugins declaring a
different version. History: v1 = packaging metadata only (hand-written
templates in core); v2 = plugin-owned codegen via `render_wrapper`.

## What core owns (and plugins must not duplicate)

- Problem normalization and oracle assembly (`al.qp` / `al.nlp`), including
  derivative factories and sparsity detection.
- The universal C ABI entry point, workspace packing, and the `_raw` kernel
  rendering (Program IR).
- The `alloy_solver_stats` struct, the `ALLOY_SOLVE_*` status enum, and
  `alloy_clock_s` — plugins fill/use them, never redefine them.
- JIT compilation, caching (source + flags keyed), and library/header
  discovery.
- The Python-side `SolverFunction` call surface, `last_stats`/`last_status`.

## Checklist for a new plugin

1. Package skeleton + hatch build hook vendoring the solver lib/headers
   (copy `plugins/alloy-piqp` for a C-API solver).
2. `BACKEND` object with the metadata fields and `render_wrapper`, exposed
   via the `alloy.solvers` entry point.
3. The wrapper template: drive the solver's C API from the oracle kernels,
   map statuses via enum constants, fill the stats struct, keep all state in
   `ctx.symbol`-prefixed statics.
4. Tests under `plugins/alloy-<name>/tests/`: correctness against analytic /
   reference solutions, a nested-solve JIT test, and a stats sanity check
   (see the piqp/ipopt test suites for the pattern).
5. Nothing in `src/alloy/` should need to change. If it does, the protocol
   is missing something — that is a core issue to raise, not a reason to
   fork core.
