# Solvers

`al.qp(...)` and `al.nlp(...)` build a solver you can call like any other function — and, more
usefully, embed inside a larger alloy graph. The whole thing compiles to one shared library that
links against the vendored solver, with no Python between the solve and the numbers.

Three backends ship: **PIQP** for quadratic programs, **IPOPT** and **alloy-sqp** for nonlinear
ones. This page is how to drive any of them; [Solver backends](solver_backends.md) is what each one
is and which to pick.

## Problem shapes

Both shapes name their constraint categories explicitly, rather than folding everything into
CasADi's two-sided `lba <= Ax <= uba`.

**QP**

```text
min   0.5 x' P x + c' x
s.t.  A_eq x = b_eq
      l_ineq <= G_ineq x <= u_ineq     (two-sided general inequalities)
      x_lb   <= x <= x_ub              (box)
```

**NLP**

```text
min   f(x, p)
s.t.  h_eq(x, p) = 0
      l_ineq <= g_ineq(x, p) <= u_ineq  (two-sided general inequalities)
      x_lb   <= x <= x_ub               (box)
```

Inequalities stay two-sided because PIQP supports them natively. The IPOPT path stacks
`[h_eq; g_ineq]` into one `g(x)` with bounds `[0; l_ineq] <= g <= [0; u_ineq]`, and that conversion
happens once when the solver is built, not on every call.

## Building a solver

```python
import alloy as al
import numpy as np

qp = al.qp(
    P=P_expr,           c=c_expr,
    A_eq=A_eq_expr,     b_eq=b_eq_expr,
    G_ineq=G_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr,       x_ub=ub_expr,
    solver="piqp",
)

nlp = al.nlp(
    x=x_sym, p=p_sym,
    f=f_expr,
    h_eq=h_eq_expr,
    g_ineq=g_ineq_expr, l_ineq=l_ineq_expr, u_ineq=u_ineq_expr,
    x_lb=lb_expr,       x_ub=ub_expr,
    solver="ipopt",     # or "sqp"
)
```

Every symbolic input may be:

- an alloy `Expr` over free parameters — the interesting case, and what makes the solver
  reusable across states;
- a NumPy array or Python scalar, which becomes a constant;
- or `None`, for the optional pieces (`A_eq`/`b_eq`, `G_ineq`/`l_ineq`/`u_ineq`, `x_lb`/`x_ub`).

Both builders return a `SolverFunction`: a real `Function` with the usual call surface
(`input_names`, `output_names`, positional or keyword `__call__`).

## Calling it

**Inputs**, in order:

| Name | Size | Meaning |
| --- | --- | --- |
| `x0` | `n` | initial primal iterate |
| `lam_eq0` | `p` | initial equality multipliers |
| `lam_ineq0` | `m` | initial inequality multipliers |
| `lam_box0` | `n` | initial signed box multipliers (NLP only) |
| *parameters* | — | every free parameter found in the symbolic inputs, in deterministic name order |

A QP takes the first three; an NLP takes all four. **What a backend does with them differs** —
alloy-sqp uses every one unconditionally, IPOPT always starts from `x0` but wants an option before
it will use the multipliers, and PIQP ignores the duals entirely.
[Solver backends](solver_backends.md#choosing-one) has the comparison.

One convention worth knowing here: `lam_box0` is signed, and the wrapper splits it into the
non-negative pair a solver actually wants — `z_L = max(-lam_box0, 0)` and
`z_U = max(lam_box0, 0)` — which is the inverse of how `lam_box` comes back out.

**Outputs.** QP returns `x`, `cost`, `lam_eq`, `lam_ineq`, `lam_box`. NLP returns `x`, `f`, the
constraint values `h_eq` and `g_ineq` at the optimum, and `lam_eq`, `lam_ineq`, `lam_box`.

```python
mu = al.sym("mu", 2)
qp = al.qp(P=np.eye(2), c=-mu)

for mu_val in [np.array([0.0, 0.0]), np.array([1.5, -0.3])]:
    out = qp(x0=np.zeros(2), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0), mu=mu_val)
    # out["x"] approaches mu_val each time
```

## Sign and ordering conventions

- `lam_ineq` and `lam_box` are **signed**: positive means the upper bound is active, negative means
  the lower bound is. Both solvers track non-negative `z_l` and `z_u` separately inside; the
  generated wrapper does the conversion.
- For an NLP, `lam = [lam_h; lam_g]` is the stacked multiplier vector IPOPT works with. The
  Lagrangian is built over `["f", "g"]` in that order, so `lam:g` refers to the same stacked
  vector — which is what makes the sparse Lagrangian Hessian come out right.
- Sparse Jacobian and Hessian patterns are COO `(rows, cols)` in the order `al.sparse_jacobian` and
  `al.sparse_hessian` produce. That order is not necessarily sorted; see
  [the ABI](../how_it_works/c_abi.md#sparse-outputs).

## Solve statistics

`SolverFunction.last_stats` holds a full `SolverStats` after each solve: the alloy and native
status, iteration count, objective, the `t_fe` / `t_solver` / `t_qp` / `t_globalization` / `t_glue`
timing split, and five oracle evaluation counters.

`SolverFunction.last_status` is the derived view — `SolverStatus(code, name, iter, stats)` — whose
`ok` is true for `OK` and `ACCEPTABLE`. IPOPT's `Feasible_Point_Found` maps to `ACCEPTABLE`.

Six per-solve diagnostics are reported when the backend has the concept and zero when it does not:

| Field | Meaning | SQP | PIQP | IPOPT |
|---|---|---|---|---|
| `primal_viol` | constraint violation (infinity norm) at the returned `x` | recomputed at the returned iterate | `info.primal_res` | `inf_pr` at the last iteration |
| `step_inf` | infinity norm of the last computed step | last QP step | 0 | `d_norm` |
| `alpha` | last **accepted** line-search step length; `0.0` if none was ever accepted | filter or l1 result | 0 | `alpha_pr` |
| `merit_penalty` | final merit penalty parameter | adaptive l1 penalty; `0.0` under filter globalization | 0 | 0 |
| `backtracks` | rejected line-search trial points across the solve | counted directly | 0 | `ls_trials - 1`, summed |
| `qp_iter` | QP iterations accumulated across SQP iterations | summed PIQP `info.iter` | `info.iter` | 0 |

IPOPT's callback-sourced fields record regular-mode iterations only. Restoration-phase values
describe the restoration subproblem rather than your problem, so they are skipped.

## Sparse QP

`al.qp(...)` assembles the problem for PIQP's dense interface by default. `sparse=True` derives the
structural patterns instead and bakes them in, so each solve refills values only — worth trying
when your data really is sparse, with two caveats attached. Both are in
[Solver backends](solver_backends.md#piqp).

## Nesting a solver in a graph

This is the part that is hard to do with a solver library and easy here. A `SolverFunction` is a
real `Function` whose body is one `solver_call` per output, so `solver.call([...])` returns
ordinary `Expr`s that flow into further computation:

```python
@al.function("safety_filter", {"x": (NX,), "u_ref": (NU,)})
def safety_filter(x, u_ref):
    f_x = dynamics.f.call([x])[0]
    g_x = dynamics.g.call([x])[0]
    h, h_grad = cbf(x)

    qp = al.qp(
        P=al.const(np.eye(NU)),
        c=-u_ref,
        G_ineq=h_grad @ g_x,
        l_ineq=-(h_grad @ f_x + alpha * h),
        u_ineq=al.const(np.full(NG, np.inf)),
    )
    out = qp.call([
        al.const(np.zeros(NU)),   # x0
        al.const(np.zeros(0)),    # lam_eq0
        al.const(np.zeros(NG)),   # lam_ineq0
        u_ref, x,                 # the free parameters, in sorted-name order
    ])
    return {"u": out[0]}
```

Mind the parameter order. `al.qp` finds the free parameters in its symbolic arguments and orders
them **by name**, so `u_ref` comes before `x` no matter which order you wrote them in. Read the
order off `qp.input_names` rather than guessing — when two parameters happen to have the same
shape, getting this wrong produces a wrong answer instead of an error.

`safety_filter` is now an ordinary function. Call it from Python and it compiles to one shared
library containing the oracles, the solver wrapper and the host entry — nothing interpreted in the
loop.

`solver_call` is marked non-differentiable, so derivatives through a solver are zero. Implicit
function theorem differentiation is future work.

## Shipping one in C++

Rendering a solver-bearing function is the same call as any other; the only extra thing a consumer
needs is the link flags:

```python
from alloy.codegen import render_c_module
from alloy.solvers.graph import solver_compile_flags

module = render_c_module(safety_filter, typed_buffers=False)
(out_dir / module.header_name).write_text(module.header)
(out_dir / module.source_name).write_text(module.source)

flags = solver_compile_flags(safety_filter)
# ['-I/.../alloy/include', '-L/.../alloy/lib', '-Wl,-rpath,/.../alloy/lib', '-lpiqpc']
```

Pass `rpath=False` if you will bundle the libraries into your own install tree and set the run path
yourself. Then the driver is plain C++ against the [universal ABI](../how_it_works/c_abi.md):

```cpp
#include "safety_filter.h"

int main() {
    double x[NX] = {/* state */};
    double u_ref[NU] = {/* reference */};
    double u[NU];
    const double* arg[] = {x, u_ref};
    double* res[] = {u};
    double w[(safety_filter_SZ_W > 0 ? safety_filter_SZ_W : 1)];
    return safety_filter(arg, res, nullptr, w, nullptr);
}
```

## Limitations

- **The vendored libraries have to be built.** `libpiqpc` and `libipopt` live in their plugin
  packages' `lib/` directories, built by the first `uv sync` (5–8 minutes cold). A missing library
  raises `SolverLibraryError` when a solve or a solver-bearing compile needs it.
- **Generated wrappers are single-threaded.** Wrapper state — the context struct, data buffers,
  the PIQP workspace, the latest statistics — lives in per-symbol statics. Distinct solvers are
  isolated from each other even in one translation unit, but two threads calling the *same*
  compiled solver will clobber each other.
- **Infinities differ by backend.** Each solver has its own "no bound" sentinel: `±1e30` for PIQP,
  `±2e19` for IPOPT. Omit a bound entirely and the QP builder fills in PIQP's sentinel for you. If
  you pass `±np.inf` explicitly it reaches the generated C as `INFINITY`; the IPOPT wrapper clamps
  that to its own convention, while PIQP takes it as given. Omitting a bound is the reliable way to
  say "unbounded".
- **Options are the backend's own, and are not all validated early.** `options={...}` reaches the
  solver under its own names; how a bad one fails, and what each backend defaults to, is
  per-backend.
- **No derivatives through a solve.** See above.

Per-backend limits — which backends honour a warm start, how each handles a bad option, what
alloy-sqp will and will not call a success — are on
[Solver backends](solver_backends.md#choosing-one).

For which backend to pick and what each vendors, see [Solver backends](solver_backends.md). For
what the builders assemble underneath, [how solvers work](../how_it_works/solvers.md). For writing a
new backend, [Solver plugins](../dev/solver_plugins.md).
