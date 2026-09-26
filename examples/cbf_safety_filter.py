"""A control-barrier-function safety filter: a QP whose data come from AD, nested in the controller.

A unicycle ``(x, y, heading)`` driven by speed and turn rate ``u = (v, w)`` is steered to a goal by
a simple nominal law that ignores three circular obstacles. Its look-ahead point
``p = (x, y) + d (cos h, sin h)`` has input-affine dynamics ``p' = g(heading) u``, so for the barrier
``h_i(p) = |p - o_i|^2 - r_i^2`` the condition ``dh_i/dt + gamma h_i >= 0`` is linear in ``u``. The
filter applies the admissible input closest to the nominal one,

    minimize |u - u_nom|^2_W   subject to   grad h_i(p)^T g(heading) u + gamma h_i(p) >= 0,  |u| <= u_max.

Three features meet here:

* the Lie derivatives ``grad h_i^T g`` come from ``sc.gradient`` and ``sc.jacobian`` of the barrier
  and of the look-ahead point, so changing either needs no hand derivation;
* the filter is a ``sc.problem`` in the variables ``u`` with the state and nominal input as
  parameters. PIQP accepts it because Scaly proves the cost quadratic and the constraints affine in
  ``u`` (their coefficients depend on the parameters only);
* the solver is a ``Function``, called with ``Expr`` leaves inside ``controller``, which computes the
  nominal input and filters it. The controller, the QP oracles and the PIQP wrapper compile into one
  shared library, and ``write_module`` renders it as one C function to link against ``libpiqpc``.

The closed loop is simulated from Python, and the filter's QP solutions are checked against SciPy.

The generated C lands in ``examples/generated/cbf_safety_filter/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import optimize

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "cbf_safety_filter"
LOOKAHEAD, GAMMA, DT, STEPS = 0.3, 2.0, 0.02, 600
U_MAX = np.array([1.0, 3.0])  # speed and turn rate
WEIGHT = np.array([1.0, 0.05])  # deviating in turn rate is cheap
OBSTACLES = np.array([[1.5, 0.3, 0.5], [3.0, -0.5, 0.4], [3.3, 0.8, 0.35]])  # (x, y, radius)
GOAL = np.array([4.5, 0.0])


def lookahead(state: sc.Expr) -> sc.Expr:
  return sc.stack([state[0] + LOOKAHEAD * state[2].cos(), state[1] + LOOKAHEAD * state[2].sin()])


def input_matrix(state: sc.Expr) -> sc.Expr:
  """``g`` with ``p' = g u``: the Jacobian of ``p`` in the state times the unicycle's input matrix."""
  u = sc.sym("u", 2)
  state_rate = sc.stack([u[0] * state[2].cos(), u[0] * state[2].sin(), u[1]])
  return sc.jacobian(lookahead(state), state) @ sc.jacobian(state_rate, u)


def barriers(p: sc.Expr) -> sc.Expr:
  return sc.stack([(p[0] - ox) ** 2 + (p[1] - oy) ** 2 - r**2 for ox, oy, r in OBSTACLES])


@sc.problem(vars=sc.L("u", 2), params=sc.G(sc.L("state", 3), sc.L("u_nom", 2)))
def safety_filter(u: sc.Expr, params: tuple[sc.Expr, sc.Expr]) -> sc.ProblemSpec:
  state, u_nom = params
  p = lookahead(state)
  h = barriers(p)
  lie = sc.jacobian(h, p) @ input_matrix(state)  # (obstacles, 2)
  return sc.ProblemSpec(
    minimize=(sc.const(WEIGHT) * (u - u_nom) ** 2).sum(),
    ineq=(sc.bounded(lie @ u + GAMMA * h, lo=0.0, name="cbf"),),
    lb=sc.const(-U_MAX),
    ub=sc.const(U_MAX),
  )


filter_qp = sc.solver(
  safety_filter, "piqp", name="cbf_qp", options={"eps_abs": 1e-10, "eps_rel": 1e-10, "eps_duality_gap_abs": 1e-10, "eps_duality_gap_rel": 1e-10}
)


def nominal(state: sc.Expr) -> sc.Expr:
  """Drive the look-ahead point straight at the goal: u = g^{-1} (-k (p - goal)), saturated."""
  g = input_matrix(state)
  pdot = -1.5 * (lookahead(state) - sc.const(GOAL))
  det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
  u = sc.stack([g[1, 1] * pdot[0] - g[0, 1] * pdot[1], g[0, 0] * pdot[1] - g[1, 0] * pdot[0]]) / det
  return sc.minimum(sc.maximum(u, sc.const(-U_MAX)), sc.const(U_MAX))


@sc.function(sc.L("state", 3), sc.G(sc.L("u", 2), sc.L("u_nom", 2), sc.L("h", len(OBSTACLES))))
def controller(state: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  u_nom = nominal(state)
  zeros = sc.const(np.zeros(2))
  u, *_ = filter_qp((zeros, zeros, sc.const(np.zeros(0)), sc.const(np.zeros(len(OBSTACLES))), (state, u_nom)))
  return u, u_nom, barriers(lookahead(state))


def scipy_filter(state: np.ndarray, u_nom: np.ndarray) -> np.ndarray:
  """The same QP with SciPy's SLSQP and finite-difference-free hand derivatives, for the check."""
  x, y, th = state
  p = np.array([x + LOOKAHEAD * np.cos(th), y + LOOKAHEAD * np.sin(th)])
  g = np.array([[np.cos(th), -LOOKAHEAD * np.sin(th)], [np.sin(th), LOOKAHEAD * np.cos(th)]])
  h = np.array([(p[0] - ox) ** 2 + (p[1] - oy) ** 2 - r**2 for ox, oy, r in OBSTACLES])
  grad_h = np.array([2 * (p - [ox, oy]) for ox, oy, _ in OBSTACLES])
  a, b = grad_h @ g, GAMMA * h
  res = optimize.minimize(
    lambda u: np.sum(WEIGHT * (u - u_nom) ** 2),
    np.clip(u_nom, -U_MAX, U_MAX),
    jac=lambda u: 2 * WEIGHT * (u - u_nom),
    constraints=[{"type": "ineq", "fun": lambda u: a @ u + b, "jac": lambda u: a}],
    bounds=list(zip(-U_MAX, U_MAX, strict=True)),
    method="SLSQP",
    options={"ftol": 1e-14, "maxiter": 200},
  )
  return res.x


def main() -> dict:
  state = np.array([0.0, 0.0, 0.0])
  states, inputs, nominals, hs = [], [], [], []
  for _ in range(STEPS):
    u, u_nom, h = controller(state)
    states.append(state)
    inputs.append(u)
    nominals.append(u_nom)
    hs.append(h)
    state = state + DT * np.array([u[0] * np.cos(state[2]), u[0] * np.sin(state[2]), u[1]])
  states, inputs, nominals, hs = map(np.array, (states, inputs, nominals, hs))
  active = np.flatnonzero(np.abs(inputs - nominals).max(axis=1) > 1e-3)  # beyond the interior-point tolerance at an active bound
  sample = active[:: max(1, len(active) // 10)]
  check = max(np.abs(inputs[k] - scipy_filter(states[k], nominals[k])).max() for k in sample) if len(sample) else 0.0
  return {"states": states, "inputs": inputs, "barriers": hs, "active": active, "scipy_error": check, "stats": controller.solver_stats("cbf_qp")}


if __name__ == "__main__":
  out = main()
  x, y, heading = out["states"][-1]
  point = np.array([x + LOOKAHEAD * np.cos(heading), y + LOOKAHEAD * np.sin(heading)])
  print(f"{STEPS} steps: the filter changed the nominal input at {len(out['active'])} of them")
  print(
    f"smallest barrier value along the way {out['barriers'].min():.4f} (>= 0 is safe); the look-ahead point ends {np.linalg.norm(point - GOAL):.1e} from the goal"
  )
  print(f"the filtered inputs agree with SciPy's SLSQP on the same QP to {out['scipy_error']:.1e}")
  print(f"last QP: {out['stats'].iter} PIQP iterations, status {out['stats'].to_solver_status().name}")
  write_module(controller, GENERATED)
  print(f"generated C for the controller with the PIQP filter inside in {GENERATED}")
