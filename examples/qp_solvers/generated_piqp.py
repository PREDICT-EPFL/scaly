"""A problem-level front end for the generated PIQP (``scaly.solvers.ipm``), shaped like ``sc.solver``.

``sc.solver(problem, "piqp")`` proves a ``sc.problem`` quadratic, extracts ``P``, ``c``, ``A``, ``b``,
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
from scipy import sparse

import scaly as sc
from scaly.ir.expr import substitute
from scaly.solvers.ipm import INF, QPStructure, QPValues, Settings, Solver
from scaly.solvers.nlp import _lowered
from scaly.solvers.qp import _gathered, _prove_quadratic, _prove_variable_independent_bounds, _qp_data, _qp_matrix_sparsity

OUTPUTS = ("x", "y", "z_l", "z_u", "status", "iter", "obj")
"""The generated Function's outputs: the stacked variables, the equality multipliers, the
inequality multipliers of the lower and upper sides, PIQP's status code (1 is solved), the
iteration count and the objective at ``x``."""


def _mask(sp: sc.SparsityType | None, shape: tuple[int, int]) -> sparse.csc_array:
  if sp is None or shape[0] == 0:
    return sparse.csc_array(shape, dtype=bool)
  return sparse.csc_array((np.ones(len(sp.rows), dtype=bool), (np.array(sp.rows), np.array(sp.cols))), shape=shape)


def solver(
  problem: sc.Problem,
  backend: Literal["sparse", "dense"] = "sparse",
  settings: Settings | None = None,
  *,
  name: str | None = None,
) -> sc.Function:
  """The generated PIQP for ``problem``: a ``Function`` from the problem's parameters (flattened, in
  the order of ``problem.params.names``) to ``OUTPUTS``."""
  name = name or f"{problem.name}_ipm_{backend}"
  _prove_variable_independent_bounds(problem)
  cached = _lowered(problem)
  _prove_quadratic(problem, cached)
  P, c, A, b, G, g_lb, g_ub, x_lb, x_ub = _qp_data(problem, cached)
  n, p, m = P.shape[0], A.shape[0], G.shape[0]
  params = list(problem._param_symbols)

  # Patterns and finite bounds from the expressions' structure and one random probe, as
  # ``build_qp(..., sparse=True)`` finds its patterns: an entry or bound that depends on a
  # parameter counts as present whatever value it takes.
  probe_outputs = [P.vec(), A.vec(), G.vec(), g_lb, g_ub, x_lb, x_ub]
  probe = sc.Function._from_exprs(f"{name}_probe", params, probe_outputs, problem.params.names, ("P", "A", "G", "g_lb", "g_ub", "x_lb", "x_ub"))
  rng = np.random.default_rng(0)
  sample = probe.input_tree.unflatten(tuple(rng.standard_normal(e.shape) for e in params))
  p_val, a_val, g_val, *bounds = probe(*sample)
  P_sp = _qp_matrix_sparsity(P, params, p_val, triu=True)
  A_sp = _qp_matrix_sparsity(A, params, a_val) if p else None
  G_sp = _qp_matrix_sparsity(G, params, g_val) if m else None
  h_l, h_u, xl, xu = (np.asarray(v, dtype=float) for v in bounds)
  s = QPStructure.from_patterns(_mask(P_sp, (n, n)), _mask(A_sp, (p, n)), _mask(G_sp, (m, n)), h_l=h_l, h_u=h_u, x_l=xl, x_u=xu)
  for got, want in ((s.P_rows, P_sp), (s.A_rows, A_sp), (s.G_rows, G_sp)):
    assert want is None or np.array_equal(got, want.rows), "pattern order differs from the gathered values"

  data = {
    "P": _gathered(P, P_sp),
    "c": c,
    "A": _gathered(A, A_sp) if A_sp is not None else sc.const(np.zeros(0)),
    "b": b,
    "G": _gathered(G, G_sp) if G_sp is not None else sc.const(np.zeros(0)),
    "h_l": g_lb,
    "h_u": g_ub,
    "x_l": x_lb,
    "x_u": x_ub,
  }

  def solve(inputs: Any) -> Any:
    # The data above are expressions of the problem's parameter symbols; the Function has its own.
    swap = dict(zip(params, problem.params.flatten_symbolic(inputs, name), strict=True))
    out = Solver(s, backend, settings, name=name).solve(QPValues.preprocess(s, **{k: substitute(v, swap) for k, v in data.items()}))
    obj = substitute(cached["f"], {**swap, cached["x"]: out["x"]})  # the problem's own objective, constants included
    return out["x"], out["y"], out["z_l"], out["z_u"], out["status"], out["iter"], obj

  return sc.function(problem.params, output=sc.G(*OUTPUTS), name=name)(solve)


def structure_summary(problem: sc.Problem) -> dict[str, Any]:
  """Sizes and non-zeros of the extracted QP, for the tables."""
  cached = _lowered(problem)
  _prove_quadratic(problem, cached)
  P, _, A, _, G, *_ = _qp_data(problem, cached)
  params = list(problem._param_symbols)
  probe = sc.Function._from_exprs(f"{problem.name}_nnz_probe", params, [P.vec(), A.vec(), G.vec()], problem.params.names, ("P", "A", "G"))
  rng = np.random.default_rng(0)
  vals = probe(*probe.input_tree.unflatten(tuple(rng.standard_normal(e.shape) for e in params)))
  n, p, m = P.shape[0], A.shape[0], G.shape[0]
  nnz = [len(_qp_matrix_sparsity(M, params, v, triu=k == 0).rows) if M.shape[0] else 0 for k, (M, v) in enumerate(zip((P, A, G), vals))]
  return {"n": n, "p": p, "m": m, "nnz_P_upper": nnz[0], "nnz_A": nnz[1], "nnz_G": nnz[2]}


def qp_data(problem: sc.Problem) -> sc.Function:
  """The extracted QP as dense matrices, for checking any solver's answer: a ``Function`` from the
  parameters to ``((P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), f0)``, with ``f0`` the objective at
  ``x = 0``, so that the problem's objective is ``1/2 x^T P x + c^T x + f0``."""
  cached = _lowered(problem)
  _prove_quadratic(problem, cached)
  data = _qp_data(problem, cached)
  f0 = substitute(cached["f"], {cached["x"]: sc.const(np.zeros(cached["x"].shape))})
  params = list(problem._param_symbols)

  def evaluate(inputs: Any) -> tuple[sc.Expr, ...]:
    swap = dict(zip(params, problem.params.flatten_symbolic(inputs, "qp_data"), strict=True))
    return tuple(substitute(e, swap) for e in (*data, f0))

  def nested(inputs: Any) -> Any:
    P, c, A, b, G, g_lb, g_ub, x_lb, x_ub, f = evaluate(inputs)
    return (P, c), (A, b), (G, g_lb, g_ub), (x_lb, x_ub), f

  tree = sc.G(sc.G("P", "c"), sc.G("A", "b"), sc.G("G", "g_lb", "g_ub"), sc.G("x_lb", "x_ub"), "f0")
  return sc.function(problem.params, output=tree, name=f"{problem.name}_qp_data")(nested)


__all__ = ["INF", "OUTPUTS", "qp_data", "solver", "structure_summary"]
