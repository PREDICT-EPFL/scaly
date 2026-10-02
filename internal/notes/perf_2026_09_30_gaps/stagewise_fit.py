"""Fit the stagewise backend's weights in ``opt.ipm``'s cost model, and check the three-way choice (C-247).

    uv run internal/notes/perf_2026_09_30_gaps/stagewise_fit.py [--variant t9fit]

Reads ``perf_2026_09_27_ipm_speed/build/timing_<variant>.json`` and the cells' ``meta.json``
(``gen.py --variant <variant> --backends sparse,stagewise,dense --split`` on the ``mpc_*`` family,
with and without path rows, ``mpc_<nx>_<nu>_<N>_<r>``, and Hessians of dense blocks coupled with
their neighbours, ``btri_<K>_<B>``; then ``timing.py --variants <variant>``). The stagewise backend's time an iteration is modelled as
a non-negative combination of its counts (``scaly.opt.ipm.cost.stage_work``), fitted by
non-negative least squares on the relative error, as ``backend_fit.py`` fits the other two.

Two weights are not fitted: those of the block factorization's and the arrays' products'
multiply-adds are the rates the profile shows at blocks of 28 to 42 (``FIXED``). On a family of
multistage problems the counts that grow with a block's cube are not told apart from those that
grow with its square (the kernels' rate rises with the block over this range), and a fit of all of
them gives the cubes no weight, which would call one dense block of a hundred rows cheap.

Prints the weights, the prediction's error in and out of sample, and for each problem which
backend the model picks with the dense and sparse weights as they stand, against the fastest
measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

NAMES = ("constant", "vectors", "entries", "cells", "pairs", "arrays", "products", "factor", "solve")
FIXED = {"products": 4.0e-5, "factor": 7.0e-5}  # microseconds a multiply-add: 25 G and 14 G a second
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
BUILD = HERE.parent / "perf_2026_09_27_ipm_speed" / "build"
sys.path.insert(0, str(BUILD.parent))


def rows(variant: str) -> list[dict]:
  from gen import problem
  from scaly.opt.ipm.cost import iteration_us, stage_terms, work
  from tests.opt.ipm.problems import ipm_inputs

  timing = {r["cell"]: r["best_us"][variant] for r in json.loads((BUILD / f"timing_{variant}.json").read_text())}
  out: dict[str, dict] = {}
  for cell, us in timing.items():
    name, backend = cell.rsplit("_", 1)
    meta = json.loads((BUILD / variant / cell / "meta.json").read_text())
    row = out.setdefault(name, {"name": name})
    row[backend] = us / max(int(meta["iter"]), 1)
  for name, row in out.items():
    s, _ = ipm_inputs(problem(name))
    row["terms"] = stage_terms(s)
    w = work(s)
    row["model"] = {b: iteration_us(w, b) for b in ("dense", "sparse")}
  return [r for r in out.values() if "stagewise" in r]


def fit(data: list[dict]) -> np.ndarray:
  x = np.array([r["terms"] for r in data], dtype=float)
  y = np.array([r["stagewise"] for r in data])
  free = [k for k, name in enumerate(NAMES) if name not in FIXED]
  rest = y - sum(x[:, NAMES.index(name)] * value for name, value in FIXED.items())
  weight = 1.0 / y
  scale = np.maximum(np.abs(x[:, free]).max(axis=0), 1.0)
  coef, _ = nnls((x[:, free] / scale) * weight[:, None], rest * weight)
  out = np.zeros(len(NAMES))
  out[free] = coef / scale
  for name, value in FIXED.items():
    out[NAMES.index(name)] = value
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--variant", default="t9fit")
  args = parser.parse_args()
  data = rows(args.variant)
  coef = fit(data)
  print("stagewise weights:", ", ".join(f"{name} {v:.3e}" for name, v in zip(NAMES, coef, strict=True)))
  errors = [float(np.dot(r["terms"], coef)) / r["stagewise"] for r in data]
  print(f"predicted/measured an iteration: median {np.median(errors):.2f}, range {min(errors):.2f}-{max(errors):.2f}")
  loo = []
  for i, r in enumerate(data):
    c = fit(data[:i] + data[i + 1 :])
    loo.append(float(np.dot(r["terms"], c)) / r["stagewise"])
  print(f"leave-one-out: median {np.median(loo):.2f}, range {min(loo):.2f}-{max(loo):.2f}")
  print("| problem | sparse | stagewise | dense | model's stagewise | pick | pick / fastest |")
  print("| --- | --- | --- | --- | --- | --- | --- |")
  worst, right = 1.0, 0
  for r, c_loo in zip(data, [fit(data[:i] + data[i + 1 :]) for i in range(len(data))], strict=True):
    predicted = {**r["model"], "stagewise": float(np.dot(r["terms"], c_loo))}
    pick = min(predicted, key=predicted.__getitem__)
    measured = {b: r[b] for b in ("sparse", "stagewise", "dense") if b in r}
    fastest = min(measured.values())
    ratio = measured.get(pick, float("nan")) / fastest
    worst = max(worst, ratio) if ratio == ratio else worst
    right += ratio == 1.0
    print(
      f"| {r['name']} | {r['sparse']:.1f} | {r['stagewise']:.1f} | {r.get('dense', float('nan')):.1f} | {predicted['stagewise']:.1f} | {pick} | {ratio:.2f} |"
    )
  print(f"the model (stagewise weights left one out) picks the fastest measured on {right}/{len(data)}; its worst pick is {worst:.2f}x the fastest")


if __name__ == "__main__":
  main()
