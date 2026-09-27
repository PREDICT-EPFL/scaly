"""Evaluating one expensive scalar function at 300 points with ``vmap`` (Scaly).

The function is sin applied 100 000 times. In Scaly the repetition is a ``scan``, a loop in C, so
the graph has one sin node instead of 100 000; ``vmap`` then maps the function over the grid, one
loop around the call. There is no parallel variant: the generated loop is serial.

After casadi/docs/examples/python/parallel_map.py.
"""

import time

import numpy as np

import scaly as sc
from _common import show

N = 300
DEPTH = 100_000
POINTS = np.linspace(0.0, 2.0 * np.pi, N)


@sc.function(sc.L("y", 1), output=sc.L("sin_y", ...))
def sin_step(y):
  return y.sin()


@sc.function(sc.L("x", 1), output=sc.L("y", ...))
def f0(x):
  (y,) = sc.scan(sin_step, x, [], length=DEPTH)
  return y


def build(verbose: bool = False):
  @sc.function(sc.L("x", N), output=sc.L("y", ...))
  def f_map(x):
    return sc.vmap(f0, N, [(x, 0, 1)])

  f_map(POINTS)  # generate and compile

  def run():
    return {"y": f_map(POINTS)}

  return run


if __name__ == "__main__":
  t0 = time.perf_counter()
  run = build()
  print(f"built in {time.perf_counter() - t0:.2f} s")
  t0 = time.perf_counter()
  out = run()
  print(f"map: {time.perf_counter() - t0:.3f} s")
  show(out)
