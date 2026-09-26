"""PR 1 (T2-1): the step number as a loop-body input, against the two ways a time-varying step was
written before it.

A cart-pendulum RK4 rollout tracks a reference ``r(t) = sin(0.3 t)``, with ``t = k * DT`` at step
``k``. Three spellings of the same function:

``index``   ``scan(..., index=True)``: the body takes ``k`` and computes ``t`` itself.
``table``   the times as a ``float64`` constant sliced one entry per step (a stored table).
``carry``   the time as a fifth carry entry that each step advances by ``DT``.

Usage: ``python pr1_index.py N...``. Prints value, gradient and Hessian timings, source lines and
whether a table of ``N`` entries reached the C.
"""

from __future__ import annotations

import re
import sys
import time

import numpy as np

import scaly as sc
from scaly.codegen import render_c_module

from bench_common import median_us

DT = 0.05


def _dyn(z: sc.Expr, u: sc.Expr, t: sc.Expr) -> sc.Expr:
  r = (0.3 * t).sin()
  return sc.stack([z[2], z[3], u - 0.1 * z[2] + 0.5 * (r - z[0]), -9.81 * z[1].sin() - u * z[1].cos()])


def _rk4(z: sc.Expr, u: sc.Expr, t: sc.Expr) -> sc.Expr:
  k1 = _dyn(z, u, t)
  k2 = _dyn(z + 0.5 * DT * k1, u, t + 0.5 * DT)
  k3 = _dyn(z + 0.5 * DT * k2, u, t + 0.5 * DT)
  k4 = _dyn(z + DT * k3, u, t + DT)
  return z + (DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout(n: int, how: str) -> sc.Function:
  z0, us = sc.sym("z0", 4), sc.sym("us", n)
  z, u = sc.sym("z", 4), sc.sym("u", 1)
  if how == "index":
    k = sc.sym("k", (), dtype="int64")
    t = DT * k.cast("float64")
    body = sc.Function._from_exprs(f"p1i{n}", [z, k, u], [_rk4(z, u[0], t), sc.stack([sc.sumsqr(z) + 0.01 * u[0] * u[0]])], ["z", "k", "u"], ["n", "c"])
    fin, costs = sc.scan(body, z0, [(us, 0, 1)], length=n, index=True)
  elif how == "table":
    t = sc.sym("t", ())
    body = sc.Function._from_exprs(f"p1t{n}", [z, t, u], [_rk4(z, u[0], t), sc.stack([sc.sumsqr(z) + 0.01 * u[0] * u[0]])], ["z", "t", "u"], ["n", "c"])
    fin, costs = sc.scan(body, z0, [(sc.const(DT * np.arange(n)), 0, 1), (us, 0, 1)], length=n)
  else:
    zt = sc.sym("zt", 5)
    zz, t = zt[:4], zt[4]
    nxt = sc.concat([_rk4(zz, u[0], t), sc.stack([t + DT])])
    body = sc.Function._from_exprs(f"p1c{n}", [zt, u], [nxt, sc.stack([sc.sumsqr(zz) + 0.01 * u[0] * u[0]])], ["zt", "u"], ["n", "c"])
    fin5, costs = sc.scan(body, sc.concat([z0, sc.const(np.zeros(1))]), [(us, 0, 1)], length=n)
    fin = fin5[:4]
  return sc.Function._from_exprs(f"p1_{how}{n}", [z0, us], [sc.stack([sc.sumsqr(fin) + costs.sum()])], ["z0", "us"], ["f"])


def main() -> None:
  for n in (int(a) for a in sys.argv[1:]):
    point = (np.array([0.1, 0.4, -0.2, 0.05]), np.sin(np.arange(n)) * 0.3)
    ref = None
    for kind in ("value", "gradient", "hessian"):
      for how in ("index", "table", "carry"):
        base = rollout(n, how)
        fn = base if kind == "value" else getattr(sc, kind)(base, "f", "us")
        t0 = time.perf_counter()
        body = render_c_module(fn).body
        out = np.asarray(fn(point)).reshape(-1)
        first = (time.perf_counter() - t0) * 1e3
        if how == "index":
          ref = out
        err = float(np.max(np.abs(out - ref)) / max(1.0, float(np.max(np.abs(ref)))))
        table = bool(re.search(rf"static const \w+ \w+\[{n}\]", body))
        us = median_us(lambda fn=fn: fn(point), repeat=21)
        print(
          f"{kind:<9} {how:<6} {n:>5} lines={len(body.splitlines()):>5} table={table!s:<5} first_ms={first:>6.0f} us={us:>9.2f} rel_diff={err:.1e}",
          flush=True,
        )


if __name__ == "__main__":
  main()
