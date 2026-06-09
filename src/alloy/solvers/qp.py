"""``al.qp(...)`` — build an opaque solver Function wrapping PIQP.

The QP shape (per ``docs/roadmap.md``) is::

    min   0.5 xᵀ P x + cᵀ x
    s.t.  A_eq x = b_eq
          l_ineq ≤ G_ineq x ≤ u_ineq
          x_lb  ≤ x ≤ x_ub

Each symbolic input may be an Alloy ``Expr`` over a set of free parameters ``p``,
or a plain numpy/python value (which becomes a constant). At call time the
parameter values are passed through to evaluate the QP data, and the QP is
solved through PIQP's dense interface.

QP inputs are (in order): ``x0``, ``lam_eq0``, ``lam_ineq0`` plus every free
parameter in deterministic name/id order. Initial dual values are accepted for
API symmetry but PIQP's dense path does not yet consume warm starts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..expr import Expr, as_expr
from ..function import Function
from ._oracle import collect_free_inputs
from ._piqp import PIQP_INF, PIQPDenseSolver
from .solver_function import SolverDescriptor, SolverFunction, SolverStatus

_SUPPORTED_SOLVERS = {"piqp"}


def _as_expr_optional(value: Any) -> Expr | None:
  if value is None:
    return None
  return as_expr(value)


def _check_shape(name: str, expr: Expr | None, expected: tuple[int, ...]) -> None:
  if expr is None:
    return
  if expr.shape != expected:
    raise ValueError(f"QP input {name!r} has shape {expr.shape}, expected {expected}")


def qp(
  *,
  P: Any,
  c: Any,
  A_eq: Any = None,
  b_eq: Any = None,
  G_ineq: Any = None,
  l_ineq: Any = None,
  u_ineq: Any = None,
  x_lb: Any = None,
  x_ub: Any = None,
  solver: str = "piqp",
  name: str | None = None,
  options: dict[str, float | int] | None = None,
) -> SolverFunction:
  if solver not in _SUPPORTED_SOLVERS:
    raise ValueError(f"unsupported QP solver {solver!r}; supported: {sorted(_SUPPORTED_SOLVERS)}")

  P_e = as_expr(P)
  c_e = as_expr(c)
  if len(P_e.shape) != 2 or P_e.shape[0] != P_e.shape[1]:
    raise ValueError(f"P must be square 2D, got shape {P_e.shape}")
  n = P_e.shape[0]
  if c_e.shape != (n,):
    raise ValueError(f"c must have shape ({n},), got {c_e.shape}")

  A_e = _as_expr_optional(A_eq)
  b_e = _as_expr_optional(b_eq)
  G_e = _as_expr_optional(G_ineq)
  l_e = _as_expr_optional(l_ineq)
  u_e = _as_expr_optional(u_ineq)
  xl_e = _as_expr_optional(x_lb)
  xu_e = _as_expr_optional(x_ub)

  if (A_e is None) != (b_e is None):
    raise ValueError("A_eq and b_eq must be provided together")
  if (G_e is None) and (l_e is not None or u_e is not None):
    raise ValueError("G_ineq is required when l_ineq/u_ineq are provided")

  p_dim = 0 if A_e is None else A_e.shape[0]
  m_dim = 0 if G_e is None else G_e.shape[0]

  _check_shape("A_eq", A_e, (p_dim, n))
  _check_shape("b_eq", b_e, (p_dim,))
  _check_shape("G_ineq", G_e, (m_dim, n))
  _check_shape("l_ineq", l_e, (m_dim,))
  _check_shape("u_ineq", u_e, (m_dim,))
  _check_shape("x_lb", xl_e, (n,))
  _check_shape("x_ub", xu_e, (n,))

  if l_e is None and m_dim:
    l_e = as_expr(np.full(m_dim, -PIQP_INF))
  if u_e is None and m_dim:
    u_e = as_expr(np.full(m_dim, PIQP_INF))
  if xl_e is None:
    xl_e = as_expr(np.full(n, -PIQP_INF))
  if xu_e is None:
    xu_e = as_expr(np.full(n, PIQP_INF))

  assert xl_e is not None and xu_e is not None
  oracle_outs: list[Expr] = [P_e.vec(), c_e]
  oracle_names = ["P", "c"]
  if p_dim:
    assert A_e is not None and b_e is not None
    oracle_outs.extend([A_e.vec(), b_e])
    oracle_names.extend(["A_eq", "b_eq"])
  if m_dim:
    assert G_e is not None and l_e is not None and u_e is not None
    oracle_outs.extend([G_e.vec(), l_e, u_e])
    oracle_names.extend(["G_ineq", "l_ineq", "u_ineq"])
  oracle_outs.extend([xl_e, xu_e])
  oracle_names.extend(["x_lb", "x_ub"])

  params = collect_free_inputs(oracle_outs)
  oracle_name = (name or "qp") + "_oracle"
  param_names: tuple[str, ...] = tuple(p.name or f"p{i}" for i, p in enumerate(params))
  oracle = Function(
    oracle_name,
    list(params),
    oracle_outs,
    list(param_names),
    oracle_names,
  )

  input_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x0", (n,)),
    ("lam_eq0", (p_dim,)),
    ("lam_ineq0", (m_dim,)),
    *((pname, pe.shape) for pname, pe in zip(param_names, params, strict=True)),
  )
  output_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x", (n,)),
    ("cost", ()),
    ("lam_eq", (p_dim,)),
    ("lam_ineq", (m_dim,)),
    ("lam_box", (n,)),
  )

  resolved_options = {"verbose": 0, **(options or {})}
  descriptor = SolverDescriptor(
    name=name or "qp_piqp",
    backend="piqp",
    n=n,
    n_eq=p_dim,
    n_ineq=m_dim,
    input_signature=input_signature,
    output_signature=output_signature,
    param_names=param_names,
    oracle=oracle,
    options=tuple(sorted(resolved_options.items())),
    oracle_output_names=tuple(oracle_names),
  )
  return SolverFunction(descriptor)


def _qp_backend(descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
  """Pure-numeric PIQP runner used by direct ``SolverFunction`` calls."""
  n, p_dim, m_dim = descriptor.n, descriptor.n_eq, descriptor.n_ineq
  oracle = descriptor.oracle
  assert oracle is not None

  # Parameters start at index 3 (after x0, lam_eq0, lam_ineq0).
  param_args = list(inputs[3:])
  data = oracle.eval_list(*param_args)
  out_by_name = dict(zip(descriptor.oracle_output_names, data, strict=True))

  P_arr = out_by_name["P"].reshape(n, n)
  c_arr = out_by_name["c"]
  A_arr = out_by_name["A_eq"].reshape(p_dim, n) if p_dim else None
  b_arr = out_by_name["b_eq"] if p_dim else None
  G_arr = out_by_name["G_ineq"].reshape(m_dim, n) if m_dim else None
  l_arr = out_by_name["l_ineq"] if m_dim else None
  u_arr = out_by_name["u_ineq"] if m_dim else None
  xl_arr = out_by_name["x_lb"]
  xu_arr = out_by_name["x_ub"]

  workspace: PIQPDenseSolver | None = descriptor.runtime.get("piqp_workspace")
  if workspace is None:
    workspace = PIQPDenseSolver(n, p_dim, m_dim, settings=dict(descriptor.options))
    descriptor.runtime["piqp_workspace"] = workspace
  workspace.update(
    P=P_arr,
    c=c_arr,
    A_eq=A_arr,
    b_eq=b_arr,
    G_ineq=G_arr,
    l_ineq=l_arr,
    u_ineq=u_arr,
    x_lb=xl_arr,
    x_ub=xu_arr,
  )
  sol = workspace.solve()
  outs = [
    sol.x,
    np.asarray(sol.primal_obj, dtype=np.float64),
    sol.lam_eq,
    sol.lam_ineq_u - sol.lam_ineq_l,
    sol.lam_box_u - sol.lam_box_l,
  ]
  return outs, SolverStatus(code=sol.status, name=sol.status_name, iter=sol.iter)
