"""Ruiz equilibration of a QP as generated code, as PIQP 0.6.2 computes it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...function import Function
from ...function.sugar import while_loop
from ...ir.expr import Expr, concat, gather, maximum, minimum, norm_inf, segment_max, segment_sum, where
from .structure import QPStructure, QPValues

MIN_SCALING, MAX_SCALING = 1e-4, 1e4


@dataclass(frozen=True, eq=False)
class Scaling:
  """The equilibration found: ``delta`` scales the variables (first ``n``), the equality rows (next
  ``p``) and the inequality rows (last ``m``); ``delta_b`` scales the box bounds; ``c`` the cost."""

  delta: Expr
  delta_b: Expr
  c: Expr


def _limit(d: Expr) -> Expr:
  """PIQP's ``limit_scaling``: below 1e-4 counts as 1 (nothing to scale), above 1e4 is capped."""
  return where(d < MIN_SCALING, 1.0, minimum(d, MAX_SCALING))


def _p_entries(s: QPStructure, p: Expr) -> tuple[Expr, np.ndarray]:
  """``|P|``'s upper entries, each off-diagonal one twice, with the column each counts for."""
  off = s.P_rows != s.P_cols
  vals = concat([p.abs(), gather(p.abs(), np.flatnonzero(off))]) if off.any() else p.abs()
  return vals, np.concatenate([s.P_cols, s.P_rows[off]])


def _abs_max_by_node(s: QPStructure, p: Expr, a: Expr, g: Expr) -> tuple[Expr, Expr]:
  """Per KKT node, the largest magnitude of the (scaled) matrices: ``(x columns, constraint rows)``."""
  p_vals, p_ids = _p_entries(s, p)
  cols = segment_max(concat([p_vals, a.abs(), g.abs()]), np.concatenate([p_ids, s.A_cols, s.G_cols]), s.n, fill=0.0)
  rows = segment_max(concat([a.abs(), g.abs()]), np.concatenate([s.A_rows, s.p + s.G_rows]), s.p + s.m, fill=0.0)
  return cols, rows


def ruiz(s: QPStructure, v: QPValues, *, scale_cost: bool = False, max_iter: int = 10, epsilon: float = 1e-3) -> Scaling:
  """PIQP's Ruiz equilibration of the KKT matrix ``[[P, A^T, G^T, D], [A], [G], [D]]`` (``D`` the box
  scaling): at most ``max_iter`` passes, each dividing every row and column by the square root of
  its largest magnitude, stopping once every factor is within ``epsilon`` of one. With
  ``scale_cost`` the cost is also scaled towards unit size after each pass; PIQP then keeps the cost
  maxima where its stopping test reads the box factors, and so does this.

  A ``while_loop`` whose params are the matrix values and ``c``, and whose carry holds the scalings
  and the last pass's factors."""
  n, p, m = s.n, s.p, s.m
  big = n + p + m
  # carry = [delta (big) | delta_b (n) | d_iter (big) | d_iter_b (n) | c]
  size = 2 * big + 2 * n + 1
  carry = Expr.sym("ruiz", (size,))
  pv, av, gv, cv = Expr.sym("P", v.P.shape), Expr.sym("A", v.A.shape), Expr.sym("G", v.G.shape), Expr.sym("c", v.c.shape)
  delta, delta_b = carry[:big], carry[big : big + n]
  d_iter, d_iter_b = carry[big + n : 2 * big + n], carry[2 * big + n : 2 * big + 2 * n]
  cost = carry[size - 1]
  dx, deq, din = delta[:n], delta[n : n + p], delta[n + p :]
  # The matrices as the loop has scaled them so far.
  p_now = cost * gather(dx, s.P_rows) * pv * gather(dx, s.P_cols)
  a_now = gather(deq, s.A_rows) * av * gather(dx, s.A_cols)
  g_now = gather(din, s.G_rows) * gv * gather(dx, s.G_cols)
  x_b = delta_b * dx
  cols, rows = _abs_max_by_node(s, p_now, a_now, g_now)
  step = 1.0 / _limit(concat([maximum(cols, x_b), rows])).sqrt()
  step_b = 1.0 / _limit(x_b).sqrt()
  new_delta, new_delta_b = delta * step, delta_b * step_b
  new_cost = cost.reshape((1,))
  if scale_cost:
    ndx = new_delta[:n]
    p_scaled = cost * gather(ndx, s.P_rows) * pv * gather(ndx, s.P_cols)
    cost_cols = segment_max(*_p_entries(s, p_scaled), n, fill=0.0)
    gamma = _limit(segment_sum(cost_cols, np.zeros(n, dtype=np.int64), 1) / float(n))
    gamma = _limit(maximum(gamma, norm_inf(cost * ndx * cv).reshape((1,))))
    new_cost = new_cost / gamma
    step_b = cost_cols  # PIQP's aliasing: the stopping test reads the cost maxima
  nxt = concat([new_delta, new_delta_b, step, step_b, new_cost])
  names = ["ruiz", "P", "A", "G", "c"]
  body = Function._from_exprs("ipm_ruiz_pass", [carry, pv, av, gv, cv], [nxt], names, ["next"])
  go = maximum(norm_inf(1.0 - d_iter), norm_inf(1.0 - d_iter_b)) > epsilon
  cond = Function._from_exprs("ipm_ruiz_go", [carry, pv, av, gv, cv], [go], names, ["go"])
  init = Expr.const(np.concatenate([np.ones(big + n), np.zeros(big + n), np.ones(1)]))
  out, _ = while_loop(cond, body, init, max_iter=max_iter, params=(v.P, v.A, v.G, v.c))
  return Scaling(delta=out[:big], delta_b=out[big : big + n], c=out[size - 1])


@dataclass(frozen=True, eq=False)
class ScaledQP:
  """The problem after equilibration, as PIQP's solver loop sees it; ``x_b`` is the box scaling
  (``x_l <= x_b * x <= x_u``, entry-wise over all ``n`` variables)."""

  values: QPValues
  x_b: Expr
  scaling: Scaling


def scale(s: QPStructure, v: QPValues, sc_: Scaling) -> ScaledQP:
  """Apply ``sc_`` to ``v``: the data the iteration works on."""
  n, p = s.n, s.p
  dx, deq, din = sc_.delta[:n], sc_.delta[n : n + p], sc_.delta[n + p :]
  values = QPValues(
    P=sc_.c * gather(dx, s.P_rows) * v.P * gather(dx, s.P_cols),
    c=sc_.c * dx * v.c,
    A=gather(deq, s.A_rows) * v.A * gather(dx, s.A_cols),
    b=v.b * deq,
    G=gather(din, s.G_rows) * v.G * gather(dx, s.G_cols),
    h_l=v.h_l * din,
    h_u=v.h_u * din,
    x_l=v.x_l * gather(sc_.delta_b, s.x_l_idx),
    x_u=v.x_u * gather(sc_.delta_b, s.x_u_idx),
  )
  return ScaledQP(values, sc_.delta_b * dx, sc_)
