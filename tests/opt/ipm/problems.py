"""The interior-point work's problem set: the reference QPs of ``scaly.testing.qp``, their value
vectors in a ``QPStructure``'s order, and the problems the decision-trace gates run."""

from __future__ import annotations

from functools import cache

from scaly.testing.qp import QP, infeasible_problems, maros_meszaros, maros_meszaros_names, mpc_qp, random_qp


def ipm_inputs(qp: QP):
  """A ``QPStructure`` for ``qp`` and its value vectors in the structure's entry order, by name."""
  from scaly.opt.ipm import QPStructure

  s = QPStructure.from_patterns(qp.P, qp.A, qp.G, h_l=qp.h_l, h_u=qp.h_u, x_l=qp.x_l, x_u=qp.x_u)
  dense = {k: getattr(qp, k).toarray() for k in ("P", "A", "G")}
  values = {
    "P": dense["P"][s.P_rows, s.P_cols],
    "A": dense["A"][s.A_rows, s.A_cols],
    "G": dense["G"][s.G_rows, s.G_cols],
    "c": qp.c,
    "b": qp.b,
    "h_l": qp.h_l,
    "h_u": qp.h_u,
    "x_l": qp.x_l,
    "x_u": qp.x_u,
  }
  return s, values


@cache
def gate_problems() -> dict[str, QP]:
  """Every problem the decision-trace gates run: the Maros–Mészáros subset, the infeasible problems,
  two MPC problems, random LPs whose primal residual falls by between 5% and 10% in some iteration
  (the proximal update's threshold), and a few random QPs."""
  problems = {name: maros_meszaros(name) for name in maros_meszaros_names()}
  problems.update({name: qp for name, (qp, _) in infeasible_problems().items()})
  problems.update({qp.name: qp for qp in (mpc_qp(4, 2, 10), mpc_qp(12, 4, 20))})
  randoms = [random_qp(40, 30, 10, seed=1, lp=True), random_qp(20, 15, 5, seed=35, lp=True), random_qp(40, 30, 10, seed=37, lp=True)]
  randoms += [random_qp(30, 20, 5, seed=seed) for seed in range(4)]
  problems.update({qp.name: qp for qp in randoms})
  return problems
