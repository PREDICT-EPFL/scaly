"""Python-side call overhead of a compiled Function: one leaf, one four-leaf group, four parameters.

Run in two checkouts, interleaved round by round, and compare minima:
  PYTHONPATH=<base worktree>/src uv run --no-sync internal/notes/perf_2026_09_27_templates/trivial_call.py --label base
  uv run --no-sync internal/notes/perf_2026_09_27_templates/trivial_call.py --label head
Each run prints one line per case, the fastest of ``--rounds`` rounds of ``--calls`` calls, in microseconds.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import scaly as sc


def cases() -> dict[str, tuple[object, tuple[object, ...]]]:
  @sc.function(sc.L("x", 1), output=sc.L("y", ...))
  def one(x):
    return x + 1.0

  @sc.function(sc.G(sc.L("a", 1), sc.L("b", 1), sc.L("c", 1), sc.L("d", 1)), output=sc.L("y", ...))
  def group(inputs):
    a, b, c, d = inputs
    return a + b + c + d

  v = np.zeros(1)
  out: dict[str, tuple[object, tuple[object, ...]]] = {"one leaf": (one, (v,)), "one group of 4": (group, ((v, v, v, v),))}
  try:
    four = sc.function(sc.L("a", 1), sc.L("b", 1), sc.L("c", 1), sc.L("d", 1), output=sc.L("y", ...))(lambda a, b, c, d: a + b + c + d)
    out["4 parameters"] = (four, (v, v, v, v))
  except TypeError:
    pass  # a checkout before parameter lists
  try:
    template = sc.function(sc.L(), output=sc.L("y", ...))(lambda x: x + 1.0)
    out["template, one leaf"] = (template, (v,))
  except TypeError:
    pass  # a checkout before templates
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--label", default="run")
  parser.add_argument("--calls", type=int, default=100_000)
  parser.add_argument("--rounds", type=int, default=7)
  args = parser.parse_args()
  for name, (fn, call_args) in cases().items():
    call = fn.__call__  # type: ignore[attr-defined]
    call(*call_args)
    best = float("inf")
    for _ in range(args.rounds):
      start = time.perf_counter()
      for _ in range(args.calls):
        call(*call_args)
      best = min(best, (time.perf_counter() - start) / args.calls)
    print(f"{args.label}\t{name}\t{best * 1e6:.2f}")


if __name__ == "__main__":
  main()
