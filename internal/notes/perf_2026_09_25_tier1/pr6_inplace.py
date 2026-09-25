"""PR 6: a scan whose step changes 4 entries of a large carry, updated in place (one slot, 4 writes
per step) against the two-slot carry (the whole carry rewritten every step)."""

from __future__ import annotations

import numpy as np

import scaly as sc
import scaly.passes.lowering as lowering

from bench_common import median_us, timed

STEPS = 1000


def host(size: int, tag: str) -> sc.Function:
  c, x = sc.sym("c", size), sc.sym("x", 2)
  u1 = sc.index_add(c, [0, 1], x * c[2:4])
  u2 = sc.index_set(u1, [size - 1, size - 2], sc.stack([u1[0] * 0.5, u1[1] + u1[2]]))
  body = sc.Function._from_exprs(f"b6_step_{size}", [c, x], [u2], ["c", "x"], ["cn"])
  c0, xs = sc.sym("c0", size), sc.sym("xs", 2 * STEPS)
  (final,) = sc.scan(body, c0, [(xs, 0, 2)], length=STEPS)
  return sc.Function._from_exprs(f"b6_{tag}_{size}", [c0, xs], [final], ["c0", "xs"], ["c"])


def main() -> None:
  print(f"{'carry size':>11} {'variant':<12}{'first call ms':>15}{'us/call':>10}{'ns/step':>10}")
  for size in (16, 1_000, 100_000):
    start = np.linspace(0.5, 1.5, size)
    data = np.sin(np.arange(2.0 * STEPS)) * 1e-3
    for tag, donate in (("in_place", True), ("two_slot", False)):
      lowering.DONATE_CARRIES = donate
      fun = host(size, tag)
      _, first = timed(lambda: fun((start, data)))
      us = median_us(lambda: fun((start, data)))
      print(f"{size:>11} {tag:<12}{first:>15.1f}{us:>10.1f}{us * 1e3 / STEPS:>10.1f}")
  lowering.DONATE_CARRIES = True


if __name__ == "__main__":
  main()
