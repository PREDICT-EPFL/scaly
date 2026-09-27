"""I3: implicit Runge-Kutta steps and their Jacobians, timed in C.

    uv run internal/notes/perf_2026_09_27_integrators/bench_implicit.py [--rounds 7]

The model is the cart-pole of `bench_explicit.py` (4 states, 1 input), one interval of 0.05 s. Two
pieces per variant: the step `(x, u) -> xnext`, and its Jacobians in `x` and `u`. Every implicit
variant runs three Newton iterations from `f(x)`:

- `lib_radau3`, `lib_radau3_full`, `lib_radau2`, `lib_gauss2`, `lib_sdirk3`:
  `sc.integrators.implicit`, simplified Newton unless `_full`; derivatives by the implicit function
  theorem (one LU of the 12x12 or 8x8 stage matrix, or 4x4 per stage for SDIRK);
- `hand_radau3`: the same Radau IIA(3) step written out in scaly, simplified Newton with
  `linalg.solve(..., assume="gen")`, differentiated by AD through the three iterations: what the
  rules replace;
- `casadi_sx_radau3`: the same written out in CasADi SX (`ca.solve` on the symbolic matrix),
  differentiated through the iterations. CasADi's own implicit integrators go through its Newton
  rootfinder, which cannot be code-generated, so this is the generated-code form a CasADi user has;
- `lib_rk4`: `sc.integrators.rk4` with two substeps, the explicit reference.

Also printed: each variant's error against the converged Radau IIA(3) step, and against the exact
flow (a DOP853 solve at 1e-13), so speed can be read next to accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build" / "implicit"
NX, NU, DT = 4, 1, 0.05


def scaly_variants() -> dict:
  import scaly as sc
  from scaly import integrators as si
  from scaly import linalg
  from bench_explicit import dyn

  sdyn = lambda x, u: dyn(x, u, lambda v: v.sin(), lambda v: v.cos(), sc.stack)  # noqa: E731

  @sc.function(NX, NU, output="xdot")
  def cartpole(x, u):
    return sdyn(x, u)

  steps = {
    "lib_radau3": si.implicit(cartpole, "radau_iia", stages=3, dt=DT, newton_iters=3, name="lib_radau3"),
    "lib_radau3_full": si.implicit(cartpole, "radau_iia", stages=3, dt=DT, newton_iters=3, newton="full", name="lib_radau3_full"),
    "lib_radau2": si.implicit(cartpole, "radau_iia", stages=2, dt=DT, newton_iters=3, name="lib_radau2"),
    "lib_gauss2": si.implicit(cartpole, "gauss_legendre", stages=2, dt=DT, newton_iters=3, name="lib_gauss2"),
    "lib_sdirk3": si.implicit(cartpole, "sdirk3", dt=DT, newton_iters=3, name="lib_sdirk3"),
    "lib_rk4": si.rk4(cartpole, dt=DT, steps=2, name="lib_rk4"),
  }
  tab = si.radau_iia(3)

  @sc.function(NX, NU, output="xnext", name="hand_radau3")
  def hand(x, u):
    fx = cartpole(x, u)
    jac = sc.jacobian(fx, x)
    m = sc.const(np.eye(3 * NX)) - DT * sc.concat([sc.concat([float(tab.a[i, j]) * jac for j in range(3)], axis=1) for i in range(3)], axis=0)
    k = sc.concat([fx] * 3)
    for _ in range(3):
      ks = [k[i * NX : (i + 1) * NX] for i in range(3)]
      g = sc.concat([ks[i] - cartpole(x + DT * sum(float(tab.a[i, j]) * ks[j] for j in range(3)), u) for i in range(3)])
      k = k - linalg.solve(m, g, assume="gen")
    return x + DT * sum(float(tab.b[i]) * k[i * NX : (i + 1) * NX] for i in range(3))

  steps["hand_radau3"] = hand
  def jacobians(name, step):
    @sc.function(NX, NU, output=sc.G("jx", "ju"), name=f"{name}_jac")
    def step_jac(x, u):
      y = step(x, u)
      return sc.jacobian(y, x), sc.jacobian(y, u)

    return step_jac

  return {name: {"step": step, "step_jac": jacobians(name, step)} for name, step in steps.items()}


def casadi_variant() -> dict:
  import casadi as ca

  from scaly import integrators as si
  from bench_explicit import dyn

  tab = si.radau_iia(3)
  x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
  f = ca.Function("f", [x, u], [dyn(x, u, ca.sin, ca.cos, lambda xs: ca.vertcat(*xs))])
  fx = f(x, u)
  jac = ca.jacobian(fx, x)
  m = ca.DM.eye(3 * NX) - DT * ca.vertcat(*[ca.horzcat(*[float(tab.a[i, j]) * jac for j in range(3)]) for i in range(3)])
  k = ca.repmat(fx, 3, 1)
  for _ in range(3):
    ks = [k[i * NX : (i + 1) * NX] for i in range(3)]
    g = ca.vertcat(*[ks[i] - f(x + DT * sum(float(tab.a[i, j]) * ks[j] for j in range(3)), u) for i in range(3)])
    k = k - ca.solve(m, g)
  y = x + DT * sum(float(tab.b[i]) * k[i * NX : (i + 1) * NX] for i in range(3))
  step = ca.Function("casadi_sx_radau3_step", [x, u], [y])
  step_jac = ca.Function("casadi_sx_radau3_step_jac", [x, u], [ca.jacobian(y, x), ca.jacobian(y, u)])
  return {"casadi_sx_radau3": {"step": step, "step_jac": step_jac}}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--unroll", type=int, default=None, help="sc.options(dense_unroll=...) while building, to study the factorizations' form")
  parser.add_argument("--only", help="comma-separated variants")
  args = parser.parse_args()
  if args.unroll is not None:
    import scaly as sc

    sc.set_options(dense_unroll=args.unroll)
  sys.path.insert(0, str(HERE))
  import bench_explicit as be

  from scipy.integrate import solve_ivp

  be.BUILD = BUILD
  variants = {**scaly_variants(), **casadi_variant()}
  if args.only:
    variants = {k: v for k, v in variants.items() if k in args.only.split(",")}
  metas = {}
  for name, pieces in variants.items():
    for piece, fn in pieces.items():
      build = be.build_casadi if name.startswith("casadi") else be.build_scaly
      metas[name, piece] = build(name, piece, fn, BUILD / name / piece)
  x0, u0 = be.inputs("step")
  import scaly as sc
  from scaly import integrators as si

  @sc.function(NX, NU, output="xdot")
  def cartpole(x, u):
    return be.dyn(x, u, lambda v: v.sin(), lambda v: v.cos(), sc.stack)

  converged = si.implicit(cartpole, "radau_iia", stages=3, dt=DT, tol=1e-15, max_iter=50, name="converged")(x0, u0)
  exact = solve_ivp(lambda t, x: be.dyn(x, u0, np.sin, np.cos, np.array), (0, DT), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  best: dict = {}
  for _ in range(args.rounds):
    for name in variants:
      for piece in ("step", "step_jac"):
        t, _ = be.time_cell(BUILD / name / piece, metas[name, piece], 400, 100)
        best[name, piece] = min(best.get((name, piece), np.inf), t)
  print("| variant | step ns | step Jacobians ns | C lines step / jac | error vs converged Radau | error vs exact flow |")
  print("| --- | --- | --- | --- | --- | --- |")
  for name, pieces in variants.items():
    y = np.array(pieces["step"](x0, u0)).ravel() if not name.startswith("casadi") else np.array(pieces["step"](x0, u0)).ravel()
    lines = f"{metas[name, 'step']['c_lines']} / {metas[name, 'step_jac']['c_lines']}"
    print(f"| {name} | {best[name, 'step']:.0f} | {best[name, 'step_jac']:.0f} | {lines} | {np.abs(y - converged).max():.1e} | {np.abs(y - exact).max():.1e} |")


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
