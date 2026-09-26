"""PR 8: dense Hessians through a scan, with one tangent loop carrying every seed (C-93) against the
previous behavior of one tangent loop per seed. The previous behavior is reproduced by making the
multi-seed loop rules decline, which sends ``jvp_many`` down its per-seed fallback exactly as before.

Workloads: the double-integrator single-shooting MPC cost of the user example (the scan inside a
called Function, every state penalised) and the nonlinear RK4 cart-pendulum rollout cost.

Usage: ``python pr8_multiseed.py [mpc|rk4] [new|old] N...``
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np

import scaly as sc
import scaly.ad.forward as forward
from scaly.codegen import render_c_module

from bench_common import median_us

DT = 0.1


def mpc_cost(n: int) -> sc.Function:
  a = np.array([[1.0, DT], [0.0, 1.0]])
  b = np.array([[0.5 * DT**2], [DT]])
  z, u = sc.sym("z", 2), sc.sym("u", 1)
  zn = a @ z + b @ u
  step = sc.Function._from_exprs(f"m8_step{n}", [z, u], [zn, zn, sc.stack([sc.sumsqr(z) + 0.1 * sc.sumsqr(u)])], ["z", "u"], ["zn", "zo", "c"])
  x0, big_u = sc.sym("x0", 2), sc.sym("U", n)
  _, xs, costs = sc.scan(step, x0, [(big_u, 0, 1)], length=n)
  shoot = sc.Function._from_exprs(f"m8_shoot{n}", [x0, big_u], [xs, costs], ["x0", "U"], ["X", "costs"])
  xs_c, costs_c = shoot._flat_symbolic_call([x0, big_u])
  return sc.Function._from_exprs(f"m8_cost{n}", [x0, big_u], [costs_c.sum() + 10.0 * sc.sumsqr(xs_c)], ["x0", "U"], ["f"])


def rk4_cost(n: int) -> sc.Function:
  z, u = sc.sym("z", 4), sc.sym("u", 1)

  def f(s: sc.Expr) -> sc.Expr:
    return sc.stack([s[2], s[3], u[0] - 0.1 * s[2], -9.81 * s[1].sin() - u[0] * s[1].cos()])

  k1 = f(z)
  k2 = f(z + 0.5 * DT * k1)
  k3 = f(z + 0.5 * DT * k2)
  k4 = f(z + DT * k3)
  nxt = z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  step = sc.Function._from_exprs(f"r8_step{n}", [z, u], [nxt, sc.stack([sc.sumsqr(z) + 0.01 * u[0] * u[0]])], ["z", "u"], ["zn", "c"])
  z0, us = sc.sym("z0", 4), sc.sym("us", n)
  zn_, costs = sc.scan(step, z0, [(us, 0, 1)], length=n)
  return sc.Function._from_exprs(f"r8_cost{n}", [z0, us], [costs.sum() + 10.0 * sc.sumsqr(zn_)], ["z0", "us"], ["f"])


def use_old_behavior() -> None:
  def decline(*_: object) -> sc.Expr:
    raise forward._JVPManyUnsupported("loop")

  forward._scan_jvp_many = decline  # type: ignore[assignment]
  forward._while_jvp_many = decline  # type: ignore[assignment]
  forward._contains_loop = lambda _fn: False  # type: ignore[assignment]


def main() -> None:
  workload, mode, sizes = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3:]]
  if mode == "old":
    use_old_behavior()
  for n in sizes:
    cost = mpc_cost(n) if workload == "mpc" else rk4_cost(n)
    wrt = "U" if workload == "mpc" else "us"
    point = (np.array([1.0, -0.5]) if workload == "mpc" else np.array([0.1, 0.4, -0.2, 0.05]), np.sin(np.arange(n)) * 0.3)
    t0 = time.perf_counter()
    hess = sc.hessian(cost, "f", wrt)
    body = render_c_module(hess).body
    hess(point)
    first = (time.perf_counter() - t0) * 1e3
    us = median_us(lambda: hess(point), repeat=21)
    procs = len(re.findall(r"void \w+_raw\(", body))
    print(
      f"{workload:>4} {mode:>4} {n:>5} lines={len(body.splitlines()):>6} loops={body.count('for ('):>5} procs={procs:>3} first_ms={first:>8.0f} us={us:>9.1f}",
      flush=True,
    )


if __name__ == "__main__":
  main()
