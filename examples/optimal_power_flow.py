"""AC optimal power flow on the WSCC 9-bus system: network equations from index tables, solved by IPOPT.

Dispatch three generators to serve three loads at the least cost, subject to the full AC power-flow
physics, voltage limits and line ratings:

    minimize   sum_g c2_g P_g^2 + c1_g P_g + c0_g
    subject to P_g(i) - P_d(i) = sum_{lines at i} P_line,   Q_g(i) - Q_d(i) = sum_{lines at i} Q_line,   theta_ref = 0,
               P_line^2 + Q_line^2 <= rate^2 at both ends,   0.9 <= |V| <= 1.1,   generator limits.

The network lives in index tables, not in code: every line's from- and to-bus voltages are read with
``sc.gather``, its pi-model flows are computed vectorially for all lines at once, and each bus's
balance adds up the flows of its lines with ``sc.segment_sum`` (the generators enter with
``sc.scatter``). Changing the network is changing the tables. The sparsity of the constraint
Jacobian, which a bus touches only through its lines, is found by Scaly from those tables.

The data are MATPOWER's ``case9`` (Zimmerman, Murillo-Sanchez and Thomas, IEEE Trans. Power Systems
26(1), 2011), whose published optimum is 5296.69 $/h. The solution is checked against that value
and against an independent NumPy evaluation of the power-flow mismatch with the bus admittance
matrix.

The generated C lands in ``examples/generated/optimal_power_flow/``; it links against the vendored IPOPT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "optimal_power_flow"
BASE_MVA = 100.0
N_BUS = 9
PD = np.array([0, 0, 0, 0, 90, 0, 100, 0, 125]) / BASE_MVA
QD = np.array([0, 0, 0, 0, 30, 0, 35, 0, 50]) / BASE_MVA
GEN_BUS = np.array([0, 1, 2])
P_MIN, P_MAX = np.array([10, 10, 10]) / BASE_MVA, np.array([250, 300, 270]) / BASE_MVA
Q_LIM = 300 / BASE_MVA
COST = np.array([[0.11, 5.0, 150.0], [0.085, 1.2, 600.0], [0.1225, 1.0, 335.0]])  # c2, c1, c0 with P in MW
# from, to (0-based), r, x, b (p.u.), rating (MVA)
LINES = np.array(
  [
    [0, 3, 0.0, 0.0576, 0.0, 250],
    [3, 4, 0.017, 0.092, 0.158, 250],
    [4, 5, 0.039, 0.17, 0.358, 150],
    [2, 5, 0.0, 0.0586, 0.0, 300],
    [5, 6, 0.0119, 0.1008, 0.209, 150],
    [6, 7, 0.0085, 0.072, 0.149, 250],
    [7, 1, 0.0, 0.0625, 0.0, 250],
    [7, 8, 0.032, 0.161, 0.306, 250],
    [8, 3, 0.01, 0.085, 0.176, 250],
  ]
)
FROM, TO = LINES[:, 0].astype(int), LINES[:, 1].astype(int)
Y_SERIES = 1.0 / (LINES[:, 2] + 1j * LINES[:, 3])
G_S, B_S, B_C = Y_SERIES.real, Y_SERIES.imag, LINES[:, 4]
RATE = LINES[:, 5] / BASE_MVA
N_LINE = len(LINES)
OPTIMUM = 5296.69  # $/h, MATPOWER's runopf(case9)


def line_flows(theta: sc.Expr, v: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  """``(P_ft, Q_ft, P_tf, Q_tf)`` of every line, pi model with the line charging split in two."""
  vf, vt = sc.gather(v, FROM), sc.gather(v, TO)
  dtheta = sc.gather(theta, FROM) - sc.gather(theta, TO)
  g, b, bc = sc.const(G_S), sc.const(B_S), sc.const(B_C)
  c, s = dtheta.cos(), dtheta.sin()
  p_ft = g * vf * vf - vf * vt * (g * c + b * s)
  q_ft = -(b + 0.5 * bc) * vf * vf - vf * vt * (g * s - b * c)
  p_tf = g * vt * vt - vf * vt * (g * c - b * s)
  q_tf = -(b + 0.5 * bc) * vt * vt - vf * vt * (-g * s - b * c)
  return p_ft, q_ft, p_tf, q_tf


@sc.problem(vars=sc.G(sc.L("theta", N_BUS), sc.L("v", N_BUS), sc.L("pg", 3), sc.L("qg", 3)))
def opf(variables: tuple[sc.Expr, ...]) -> sc.ProblemSpec:
  theta, v, pg, qg = variables
  p_ft, q_ft, p_tf, q_tf = line_flows(theta, v)
  ends = np.r_[FROM, TO]
  p_balance = sc.scatter(pg, GEN_BUS, N_BUS) - sc.const(PD) - sc.segment_sum(sc.concat([p_ft, p_tf]), ends, N_BUS)
  q_balance = sc.scatter(qg, GEN_BUS, N_BUS) - sc.const(QD) - sc.segment_sum(sc.concat([q_ft, q_tf]), ends, N_BUS)
  mw = BASE_MVA * pg
  cost = (sc.const(COST[:, 0]) * mw * mw + sc.const(COST[:, 1]) * mw + sc.const(COST[:, 2])).sum()
  loading = sc.concat([p_ft * p_ft + q_ft * q_ft, p_tf * p_tf + q_tf * q_tf])
  return sc.ProblemSpec(
    minimize=cost,
    eq=(theta[0:1], p_balance, q_balance),
    ineq=(sc.bounded(loading, hi=sc.const(np.r_[RATE, RATE] ** 2), name="ratings"),),
    lb=(sc.NO_LB, sc.const(np.full(N_BUS, 0.9)), sc.const(P_MIN), sc.const(np.full(3, -Q_LIM))),
    ub=(sc.NO_UB, sc.const(np.full(N_BUS, 1.1)), sc.const(P_MAX), sc.const(np.full(3, Q_LIM))),
  )


solve_opf = sc.solver(opf, "ipopt", name="opf_case9", options={"tol": 1e-10})


def mismatch(theta: np.ndarray, v: np.ndarray, pg: np.ndarray, qg: np.ndarray) -> float:
  """Power-flow mismatch from the bus admittance matrix, independently of the line formulation."""
  y = np.zeros((N_BUS, N_BUS), dtype=complex)
  for f, t, ys, bc in zip(FROM, TO, Y_SERIES, B_C, strict=True):
    y[f, f] += ys + 0.5j * bc
    y[t, t] += ys + 0.5j * bc
    y[f, t] -= ys
    y[t, f] -= ys
  voltage = v * np.exp(1j * theta)
  injection = voltage * np.conj(y @ voltage)
  gen = np.zeros(N_BUS, dtype=complex)
  gen[GEN_BUS] = pg + 1j * qg
  return float(np.abs(injection - (gen - (PD + 1j * QD))).max())


def main() -> dict:
  start = (np.zeros(N_BUS), np.ones(N_BUS), 0.5 * (P_MIN + P_MAX), np.zeros(3))
  zeros = (np.zeros(N_BUS), np.zeros(N_BUS), np.zeros(3), np.zeros(3))
  (theta, v, pg, qg), lam_box, lam_eq, lam_ineq = solve_opf((start, zeros, np.zeros(opf.n_eq), np.zeros(opf.n_ineq), ()))
  stats = solve_opf.solver_stats()
  y = Y_SERIES
  vf, vt = v[FROM] * np.exp(1j * theta[FROM]), v[TO] * np.exp(1j * theta[TO])
  s_ft = vf * np.conj((y + 0.5j * B_C) * vf - y * vt)
  s_tf = vt * np.conj((y + 0.5j * B_C) * vt - y * vf)
  return {
    "cost": float(stats.obj),
    "pg_mw": BASE_MVA * pg,
    "qg_mvar": BASE_MVA * qg,
    "v": v,
    "theta_deg": np.degrees(theta),
    "mismatch": mismatch(theta, v, pg, qg),
    "loading": np.maximum(np.abs(s_ft), np.abs(s_tf)) / RATE,
    "lmp": -lam_eq[1 : 1 + N_BUS] / BASE_MVA,  # d cost / d load, per MW
    "iterations": stats.iter,
    "status": stats.to_solver_status(),
  }


if __name__ == "__main__":
  out = main()
  print(f"IPOPT: {out['status'].name} after {out['iterations']} iterations; cost {out['cost']:.2f} $/h (MATPOWER: {OPTIMUM})")
  print(f"dispatch P = {np.array2string(out['pg_mw'], precision=2)} MW, Q = {np.array2string(out['qg_mvar'], precision=2)} MVAr")
  print(f"voltages {out['v'].min():.4f} to {out['v'].max():.4f} p.u.; most loaded line at {100 * out['loading'].max():.1f}% of its rating")
  print(f"power-flow mismatch from the bus admittance matrix: {out['mismatch']:.1e} p.u.")
  print(f"locational marginal prices from the balance multipliers ($/MWh): {np.array2string(out['lmp'], precision=2)}")
  write_module(solve_opf, GENERATED)
  print(f"generated C for the OPF solver in {GENERATED}")
