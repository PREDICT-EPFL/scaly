"""PR 7: the gradient of a function of a Newton solve, through the solver's steps (a masked
``max_iter``-step backward scan) against the implicit-function rule attached by
``sc.custom_derivative`` (one division per entry at the solution)."""

from __future__ import annotations

import time

import numpy as np

import scaly as sc

from bench_common import median_us


def build(n: int) -> dict[str, sc.Function]:
  c = sc.sym("c", 2 * n)
  x, p = c[:n], c[n:]
  body = sc.Function._from_exprs(f"b7_step_{n}", [c], [sc.concat([x - (x**3 + x - p) / (3 * x * x + 1), p])], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"b7_go_{n}", [c], [sc.norm_inf(x**3 + x - p) > 1e-12], ["c"], ["go"])
  pp = sc.sym("p", n)
  final, _ = sc.while_loop(cond, body, sc.concat([sc.const(np.zeros(n)), pp]), max_iter=50)
  solve = sc.Function._from_exprs(f"b7_solve_{n}", [pp], [final[:n]], ["p"], ["x"])
  pi, xi, xb = sc.sym("p", n), sc.sym("x", n), sc.sym("xbar", n)
  rule = sc.Function._from_exprs(f"b7_vjp_{n}", [pi, xi, xb], [xb / (3 * xi * xi + 1)], ["p", "x", "xbar"], ["pbar"])
  q = sc.sym("q", n)
  out = {}
  for tag, fn in (("through steps", solve), ("implicit rule", sc.custom_derivative(solve, vjp=rule))):
    cost = sc.Function._from_exprs(f"b7_cost_{tag[0]}_{n}", [q], [(fn(q) ** 2).sum()], ["q"], ["c"])
    out[tag] = sc.gradient(cost, "c", "q")
  out["value only"] = solve
  return out


def main() -> None:
  print(f"{'n':>6} {'variant':<16}{'first call ms':>15}{'us/call':>10}")
  for n in (10, 100, 1000):
    p = np.linspace(0.5, 20.0, n)
    for tag, fun in build(n).items():
      t0 = time.perf_counter()
      fun(p)
      first = (time.perf_counter() - t0) * 1e3
      print(f"{n:>6} {tag:<16}{first:>15.1f}{median_us(lambda: fun(p)):>10.1f}")


if __name__ == "__main__":
  main()
