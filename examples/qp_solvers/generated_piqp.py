"""A problem-level front end for the generated PIQP (``scaly.solvers.ipm``), shaped like ``sc.opt.solver``.

``sc.opt.solver(problem, "piqp")`` proves a ``sc.opt.problem`` quadratic, extracts ``P``, ``c``, ``A``, ``b``,
``G`` and the bounds as expressions of the parameters, and hands their values to the vendored PIQP
library at run time. ``solver(problem, backend)`` below does the same extraction (Scaly's own
helpers) and then *generates the solver itself*: the sparsity patterns and which bounds are finite
become a ``QPStructure``, and ``ipm.Solver`` writes PIQP 0.6.2's algorithm (Ruiz equilibration,
the initial point, the Mehrotra loop with proximal updates, unscaling) over it. The result is one
plain ``Function`` from the parameters to the solution, with no library behind it.

The library will get this as ``backend="scaly"`` (plan, Tier 4, items 30 and 31); until then it
lives here, with the private extraction helpers it shares with ``build_qp``.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

import scaly as sc
from scaly.ir.expr import substitute
from scaly.solvers.ipm import INF, QPStructure, QPValues, Settings, Solver

OUTPUTS = ("x", "y", "z_l", "z_u", "status", "iter", "obj")
"""The generated Function's outputs: the stacked variables, the equality multipliers, the
inequality multipliers of the lower and upper sides, PIQP's status code (1 is solved), the
iteration count and the objective at ``x``."""


def solver(
  problem: sc.opt.NLP,
  backend: Literal["sparse", "dense"] = "sparse",
  settings: Settings | None = None,
  *,
  name: str | None = None,
) -> sc.Function:
  """The generated PIQP for ``problem``: a ``Function`` from the problem's parameters (flattened, in
  the order of ``problem.params.names``) to ``OUTPUTS``."""
  name = name or f"{problem.name}_ipm_{backend}"
  form = sc.opt.extract_qp(problem)
  P, c, A, b, G, g_lb, g_ub, x_lb, x_ub = form.P, form.c, form.A, form.b, form.G, form.g_lb, form.g_ub, form.x_lb, form.x_ub
  n, p, m = P.shape[0], A.shape[0], G.shape[0]
  params = list(form.params)

  # The patterns ``PIQP(sparse=True)`` finds, and which bounds are finite at one random probe: an
  # entry or bound that depends on a parameter counts as present whatever value it takes.
  P_sp, A_sp, G_sp = form.patterns(name)
  probe = sc.Function.from_exprs(f"{name}_probe", params, [g_lb, g_ub, x_lb, x_ub], form.param_names, ("g_lb", "g_ub", "x_lb", "x_ub"))
  rng = np.random.default_rng(0)
  bounds = probe(*probe.input_tree.unflatten(tuple(rng.standard_normal(e.shape) for e in params)))
  h_l, h_u, xl, xu = (np.asarray(v, dtype=float) for v in bounds)
  A_pat, G_pat = (np.zeros(shape, dtype=bool) if sp is None else sp for sp, shape in ((A_sp, (p, n)), (G_sp, (m, n))))
  s = QPStructure.from_patterns(P_sp, A_pat, G_pat, h_l=h_l, h_u=h_u, x_l=xl, x_u=xu)
  for got, want in ((s.P_rows, P_sp), (s.A_rows, A_sp), (s.G_rows, G_sp)):
    assert want is None or np.array_equal(got, want.rows), "pattern order differs from the gathered values"

  data = {
    "P": form.in_pattern(P, P_sp),
    "c": c,
    "A": form.in_pattern(A, A_sp) if A_sp is not None else sc.const(np.zeros(0)),
    "b": b,
    "G": form.in_pattern(G, G_sp) if G_sp is not None else sc.const(np.zeros(0)),
    "h_l": g_lb,
    "h_u": g_ub,
    "x_l": x_lb,
    "x_u": x_ub,
  }

  oracles = sc.opt.nlp_oracles(problem)
  objective = oracles.base.outputs[0]  # the problem's own objective, constants included

  def solve(inputs: Any) -> Any:
    # The data above are expressions of the problem's parameter symbols; the Function has its own.
    swap = dict(zip(params, problem.params.flatten_symbolic(inputs, name), strict=True))
    out = Solver(s, backend, settings, name=name).solve(QPValues.preprocess(s, **{k: substitute(v, swap) for k, v in data.items()}))
    obj = substitute(objective, {**swap, oracles.x: out["x"]})
    return out["x"], out["y"], out["z_l"], out["z_u"], out["status"], out["iter"], obj

  return sc.function(problem.params, output=sc.G(*OUTPUTS), name=name)(solve)


def structure_summary(problem: sc.opt.NLP) -> dict[str, Any]:
  """Sizes and non-zeros of the extracted QP, for the tables."""
  form = sc.opt.extract_qp(problem)
  n, p, m = form.P.shape[0], form.A.shape[0], form.G.shape[0]
  nnz = [0 if sp is None or not rows else len(sp.rows) for sp, rows in zip(form.patterns(f"{problem.name}_nnz"), (n, p, m), strict=True)]
  return {"n": n, "p": p, "m": m, "nnz_P_upper": nnz[0], "nnz_A": nnz[1], "nnz_G": nnz[2]}


def qp_data(problem: sc.opt.NLP) -> sc.Function:
  """The extracted QP as dense matrices, for checking any solver's answer: a ``Function`` from the
  parameters to ``((P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), f0)``, with ``f0`` the objective at
  ``x = 0``, so that the problem's objective is ``1/2 x^T P x + c^T x + f0``."""
  form = sc.opt.extract_qp(problem)
  data = (form.P, form.c, form.A, form.b, form.G, form.g_lb, form.g_ub, form.x_lb, form.x_ub)
  f0 = form.f0
  params = list(form.params)

  def evaluate(inputs: Any) -> tuple[sc.Expr, ...]:
    swap = dict(zip(params, problem.params.flatten_symbolic(inputs, "qp_data"), strict=True))
    return tuple(substitute(e, swap) for e in (*data, f0))

  def nested(inputs: Any) -> Any:
    P, c, A, b, G, g_lb, g_ub, x_lb, x_ub, f = evaluate(inputs)
    return (P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), f

  tree = sc.G(sc.G("P", "c"), sc.G("A", "b"), sc.G("G", "g_lb", "g_ub"), sc.G("x_lb", "x_ub"), "f0")
  return sc.function(problem.params, output=tree, name=f"{problem.name}_qp_data")(nested)


__all__ = ["INF", "OUTPUTS", "qp_data", "solver", "structure_summary"]
