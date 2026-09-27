"""I4: `si.adaptive` against SciPy's `solve_ivp`, work against precision, both called from Python.

    uv run internal/notes/perf_2026_09_27_integrators/bench_adaptive.py

Van der Pol (mu = 3, forced by u = 0.3) over 2 s from (2, 0), the reference a DOP853 solve at 1e-13.
Per tolerance (rtol, atol = rtol / 100): the error and the fastest of 50 calls (after one warm call,
which compiles ours) for `si.adaptive` with `dopri5` and `bs32`, and for `solve_ivp` with `RK45`
(the same Dormand-Prince pair) and `DOP853`. This is the use a closed-loop simulation makes of a
plant model: one Python call per sampling interval. SciPy's cost is its Python stepping loop, so the
comparison is of what a user can call, not of two compiled kernels.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.integrate import solve_ivp

import scaly as sc
from scaly import integrators as si

MU, U, T = 3.0, 0.3, 2.0
X0 = np.array([2.0, 0.0])


@sc.function(2, 1, output="xdot")
def vdp(x, u):
  return sc.stack([x[1], MU * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


def rhs(t, x):
  return [x[1], MU * (1 - x[0] ** 2) * x[1] - x[0] + U]


def best(call, repeats: int = 50) -> float:
  call()
  times = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    call()
    times.append(time.perf_counter() - t0)
  return min(times) * 1e6


def main() -> None:
  exact = solve_ivp(rhs, (0, T), X0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  print("| rtol | si dopri5 err | us | si bs32 err | us | solve_ivp RK45 err | us | nfev | solve_ivp DOP853 err | us | nfev |")
  print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for tol in (1e-4, 1e-6, 1e-8, 1e-10):
    row = [f"{tol:.0e}"]
    for pair in ("dopri5", "bs32"):
      step = si.adaptive(vdp, pair, rtol=tol, atol=tol / 100, max_steps=200_000, name=f"{pair}_{int(-np.log10(tol))}")
      args = (X0, np.array([U]), np.array(T))
      row += [f"{np.abs(step(*args) - exact).max():.1e}", f"{best(lambda: step(*args)):.1f}"]
    for method in ("RK45", "DOP853"):
      sol = solve_ivp(rhs, (0, T), X0, method=method, rtol=tol, atol=tol / 100)
      row += [f"{np.abs(sol.y[:, -1] - exact).max():.1e}", f"{best(lambda: solve_ivp(rhs, (0, T), X0, method=method, rtol=tol, atol=tol / 100), 10):.0f}", str(sol.nfev)]
    print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
  main()
