"""The AC-OPF case study's model, small: rosetta-opf's polar formulation with a phase-shifting transformer.

`examples/case_studies/acopf` writes PowerModels' reference AC-OPF (flows at both branch ends as
variables, pi model with tap ratio and phase shift, bus shunts, angle-difference and thermal limits)
over all buses and branches at once with `sc.gather` and `sc.segment_sum`, and solves it with Scaly's
IPOPT. This builds the same model for a 3-bus network with one transformer and one shunt, and checks
the solution two ways that do not share the formulation: the power-flow mismatch computed from the bus
admittance matrix, and the objective of SciPy's SLSQP on the problem written with that matrix.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize

import scaly as sc

# Buses 0 (reference, generator), 1 (load, shunt), 2 (generator, load). Branch 0-1 a line, 1-2 a
# transformer (tap 1.05, shift 3 degrees), 0-2 a line.
F, T = np.array([0, 1, 0]), np.array([1, 2, 2])
R, X, BC = np.array([0.02, 0.01, 0.03]), np.array([0.08, 0.05, 0.12]), np.array([0.04, 0.0, 0.03])
TAP, SHIFT = np.array([1.0, 1.05, 1.0]), np.array([0.0, np.radians(3.0), 0.0])
Y = 1 / (R + 1j * X)
G, B = Y.real, Y.imag
TR, TI = TAP * np.cos(SHIFT), TAP * np.sin(SHIFT)
G_FR, B_FR, G_TO, B_TO = np.zeros(3), BC / 2, np.zeros(3), BC / 2
PD, QD = np.array([0.0, 0.9, 0.5]), np.array([0.0, 0.3, 0.2])
GS, BS = np.array([0.0, 0.02, 0.0]), np.array([0.0, 0.15, 0.0])
GEN_BUS = np.array([0, 2])
PMIN, PMAX, QMIN, QMAX = np.array([0.0, 0.0]), np.array([2.0, 0.6]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])
COST = np.array([[10.0, 20.0, 0.0], [5.0, 35.0, 0.0]])
RATE = np.array([1.2, 1.0, 1.0])
ANG = np.radians(30.0)
VMIN, VMAX = 0.94, 1.06
NB, NBR, NG = 3, 3, 2
ARC_BUS = np.r_[F, T]  # arcs: all from-ends, then all to-ends
F_ARC, T_ARC = np.arange(NBR), NBR + np.arange(NBR)


@sc.problem(vars=sc.G(sc.L("va", NB), sc.L("vm", NB), sc.L("pg", NG), sc.L("qg", NG), sc.L("p", 2 * NBR), sc.L("q", 2 * NBR)))
def opf(variables):
  va, vm, pg, qg, p, q = variables
  C = sc.const
  ttm = TR**2 + TI**2
  vf, vt = sc.gather(vm, F), sc.gather(vm, T)
  dva = sc.gather(va, F) - sc.gather(va, T)
  cs, sn = vf * vt * dva.cos(), vf * vt * dva.sin()
  fc, fs = (-G * TR + B * TI) / ttm, (-B * TR - G * TI) / ttm
  tc, ts = (-G * TR - B * TI) / ttm, (-B * TR + G * TI) / ttm
  eqs = (
    sc.gather(p, F_ARC) - (C((G + G_FR) / ttm) * vf * vf + C(fc) * cs + C(fs) * sn),
    sc.gather(q, F_ARC) - (C(-(B + B_FR) / ttm) * vf * vf - C(fs) * cs + C(fc) * sn),
    sc.gather(p, T_ARC) - (C(G + G_TO) * vt * vt + C(tc) * cs - C(ts) * sn),
    sc.gather(q, T_ARC) - (C(-(B + B_TO)) * vt * vt - C(ts) * cs - C(tc) * sn),
  )
  bal_p = sc.segment_sum(p, ARC_BUS, NB) - sc.segment_sum(pg, GEN_BUS, NB) + C(PD) + C(GS) * vm * vm
  bal_q = sc.segment_sum(q, ARC_BUS, NB) - sc.segment_sum(qg, GEN_BUS, NB) + C(QD) - C(BS) * vm * vm
  rate = np.r_[RATE, RATE]
  return sc.ProblemSpec(
    minimize=(C(COST[:, 0]) * pg * pg + C(COST[:, 1]) * pg + C(COST[:, 2])).sum(),
    eq=(va[0:1], bal_p, bal_q, *eqs),
    ineq=(sc.bounded(dva, lo=C(np.full(NBR, -ANG)), hi=C(np.full(NBR, ANG))), sc.bounded(p * p + q * q, hi=C(rate**2))),
    lb=(sc.NO_LB, C(np.full(NB, VMIN)), C(PMIN), C(QMIN), C(-rate), C(-rate)),
    ub=(sc.NO_UB, C(np.full(NB, VMAX)), C(PMAX), C(QMAX), C(rate), C(rate)),
  )


def ybus() -> np.ndarray:
  Yb = np.zeros((NB, NB), dtype=complex)
  for k in range(NBR):
    tap = TR[k] + 1j * TI[k]
    f, t = F[k], T[k]
    Yb[f, f] += (Y[k] + G_FR[k] + 1j * B_FR[k]) / abs(tap) ** 2
    Yb[t, t] += Y[k] + G_TO[k] + 1j * B_TO[k]
    Yb[f, t] -= Y[k] / np.conj(tap)
    Yb[t, f] -= Y[k] / tap
  return Yb


def injection(va, vm):
  V = vm * np.exp(1j * va)
  return V * np.conj(ybus() @ V)


@pytest.mark.solver("ipopt")
def test_polar_opf_with_a_phase_shifter_matches_an_admittance_matrix_model() -> None:
  solve = sc.solver(opf, "ipopt", name="test_acopf_three_bus", options={"tol": 1e-10})
  x0 = (np.zeros(NB), np.ones(NB), np.clip(np.zeros(NG), PMIN, PMAX), np.zeros(NG), np.zeros(2 * NBR), np.zeros(2 * NBR))
  (va, vm, pg, qg, p, q), *_ = solve(
    x0, (np.zeros(NB), np.zeros(NB), np.zeros(NG), np.zeros(NG), np.zeros(2 * NBR), np.zeros(2 * NBR)), np.zeros(opf.n_eq), np.zeros(opf.n_ineq), ()
  )
  stats = solve.solver_stats()
  assert stats.to_solver_status().name == "OK"
  gen = np.zeros(NB, dtype=complex)
  np.add.at(gen, GEN_BUS, pg + 1j * qg)
  mismatch = injection(va, vm) - (gen - (PD + 1j * QD) - (GS - 1j * BS) * vm**2)
  assert np.abs(mismatch).max() < 1e-8

  # SLSQP on (va[1:], vm, pg) with the balance from the admittance matrix; qg follows from it.
  def unpack(z):
    return np.r_[0.0, z[:2]], z[2:5], z[5:7]

  def cost(z):
    pg_ = unpack(z)[2]
    return float((COST[:, 0] * pg_**2 + COST[:, 1] * pg_ + COST[:, 2]).sum())

  def balance(z):
    va_, vm_, pg_ = unpack(z)
    s = injection(va_, vm_) + (PD + 1j * QD) + (GS - 1j * BS) * vm_**2
    gp = np.zeros(NB)
    np.add.at(gp, GEN_BUS, pg_)
    return np.r_[s.real - gp, s.imag[1]]  # bus 1 has no generator: its reactive balance is an equality

  def limits(z):
    va_, vm_, _ = unpack(z)
    s = injection(va_, vm_) + (PD + 1j * QD) + (GS - 1j * BS) * vm_**2
    qg_ = s.imag[GEN_BUS]
    V = vm_ * np.exp(1j * va_)
    out = []
    for k in range(NBR):
      tap = TR[k] + 1j * TI[k]
      f, t = F[k], T[k]
      i_f = (Y[k] + 1j * B_FR[k]) / abs(tap) ** 2 * V[f] - Y[k] / np.conj(tap) * V[t]
      i_t = (Y[k] + 1j * B_TO[k]) * V[t] - Y[k] / tap * V[f]
      out += [
        RATE[k] ** 2 - abs(V[f] * np.conj(i_f)) ** 2,
        RATE[k] ** 2 - abs(V[t] * np.conj(i_t)) ** 2,
        ANG - (va_[f] - va_[t]),
        ANG + (va_[f] - va_[t]),
      ]
    return np.r_[out, qg_ - QMIN, QMAX - qg_]

  bounds = [(None, None)] * 2 + [(VMIN, VMAX)] * 3 + list(zip(PMIN, PMAX, strict=True))
  ref = optimize.minimize(
    cost, np.r_[0, 0, 1, 1, 1, 0.8, 0.4], method="SLSQP", bounds=bounds,
    constraints=[{"type": "eq", "fun": balance}, {"type": "ineq", "fun": limits}], options={"ftol": 1e-14, "maxiter": 500},
  )  # fmt: skip
  assert ref.success
  assert abs(stats.obj - ref.fun) < 1e-7 * abs(ref.fun)
  np.testing.assert_allclose(np.r_[va[1:], vm, pg], ref.x, atol=1e-5)
