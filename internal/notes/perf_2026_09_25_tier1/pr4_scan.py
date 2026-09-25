"""PR 4: an RK4 rollout as ``scan`` against the same body chained ``N`` times, for the value and for
the reverse-mode gradient: render time, compile time, C size and run time."""

from __future__ import annotations

import time

import numpy as np

import scaly as sc
from scaly.codegen import render_c_source

from bench_common import median_us

DT = 0.1


def body() -> sc.Function:
  z, u = sc.sym("z", 4), sc.sym("u", 1)

  def f(s: sc.Expr) -> sc.Expr:
    return sc.stack([s[2], s[3], u[0] - 0.1 * s[2], -9.81 * s[1].sin() - u[0] * s[1].cos()])

  k1 = f(z)
  k2 = f(z + 0.5 * DT * k1)
  k3 = f(z + 0.5 * DT * k2)
  k4 = f(z + DT * k3)
  return sc.Function._from_exprs("b4_rk4", [z, u], [z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)], ["z", "u"], ["znext"])


def rollout(n: int, how: str) -> sc.Function:
  step = body()
  z0, us = sc.sym("z0", 4), sc.sym("us", n)
  if how == "scan":
    (zn,) = sc.scan(step, z0, [(us, 0, 1)], length=n)
  else:
    zn = z0
    for k in range(n):
      (zn,) = step._flat_symbolic_call([zn, us[k : k + 1]])
  cost = sc.sumsqr(zn)
  (grad,) = sc.vjp((cost,), (us,), (sc.const(1.0),))
  return sc.Function._from_exprs(f"b4_{how}_{n}", [z0, us], [zn, grad], ["z0", "us"], ["zN", "grad"])


def main() -> None:
  print(f"{'N':>5} {'variant':<9}{'build graph ms':>15}{'render ms':>11}{'compile+load ms':>17}{'C lines':>9}{'us/call':>10}")
  z0 = np.array([0.1, 0.4, -0.2, 0.05])
  for n in (10, 50, 200, 1000):
    us = np.sin(np.arange(n) * 0.3)
    for how in ("scan", "unrolled"):
      if how == "unrolled" and n > 200:
        continue
      t0 = time.perf_counter()
      fun = rollout(n, how)
      t1 = time.perf_counter()
      src = render_c_source(fun)
      t2 = time.perf_counter()
      fun((z0, us))
      t3 = time.perf_counter()
      print(f"{n:>5} {how:<9}{(t1 - t0) * 1e3:>15.1f}{(t2 - t1) * 1e3:>11.1f}{(t3 - t2) * 1e3:>17.1f}{src.count(chr(10)):>9}{median_us(lambda: fun((z0, us))):>10.1f}")


if __name__ == "__main__":
  main()
