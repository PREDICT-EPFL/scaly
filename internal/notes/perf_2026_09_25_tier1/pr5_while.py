"""PR 5: Newton's method as a ``while_loop`` (stops when converged) against the same step as a
fixed-length ``scan`` of ``max_iter`` steps, with and without the reverse-mode gradient."""

from __future__ import annotations

import time

import numpy as np

import scaly as sc

from bench_common import median_us

MAX_ITER = 50


def functions(n: int) -> dict[str, sc.Function]:
  c = sc.sym("c", 2 * n)
  x, target = c[:n], c[n:]
  body = sc.Function._from_exprs(f"b5_step_{n}", [c], [sc.concat([x - (x**3 - target) / (3.0 * x * x), target])], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"b5_go_{n}", [c], [sc.norm_inf(x**3 - target) > 1e-12], ["c"], ["go"])
  c0 = sc.sym("c0", 2 * n)
  final, count = sc.while_loop(cond, body, c0, max_iter=MAX_ITER)
  (fixed,) = sc.scan(body, c0, [], length=MAX_ITER)
  out = {}
  for label, result in (("while_loop", final), ("scan(max_iter)", fixed)):
    cost = sc.sumsqr(result[:n])
    (grad,) = sc.vjp((cost,), (c0,), (sc.const(1.0),))
    tag = label.split("(")[0]
    out[label] = sc.Function._from_exprs(f"b5_{tag}_{n}", [c0], [result], ["c0"], ["c"])
    out[label + " + grad"] = sc.Function._from_exprs(f"b5_{tag}_g_{n}", [c0], [result, grad], ["c0"], ["c", "g"])
  out["count"] = sc.Function._from_exprs(f"b5_count_{n}", [c0], [count], ["c0"], ["n"])
  return out


def main() -> None:
  print(f"{'n':>5} {'variant':<24}{'first call ms':>15}{'us/call':>10}")
  for n in (10, 100, 1000):
    targets = np.linspace(1.0, 50.0, n)
    start = np.concatenate([np.full(n, 2.0), targets])
    funs = functions(n)
    print(f"{n:>5} {'(steps taken)':<24}{'':>15}{float(funs.pop('count')(start)):>10.0f}")
    for label, fun in funs.items():
      t0 = time.perf_counter()
      fun(start)
      first = (time.perf_counter() - t0) * 1e3
      print(f"{n:>5} {label:<24}{first:>15.1f}{median_us(lambda: fun(start)):>10.1f}")


if __name__ == "__main__":
  main()
