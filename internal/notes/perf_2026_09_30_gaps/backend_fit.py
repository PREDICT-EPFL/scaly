"""Fit the weights of ``opt.ipm``'s backend cost model to the measured time per iteration (A5).

    uv run internal/notes/perf_2026_09_30_gaps/backend_fit.py

Reads ``results/backend_costs.json`` (``backend_costs.py``). Each backend's time per iteration is
modelled as a non-negative combination of the counts its iteration is made of (``scaly.opt.ipm.cost.Work``),
fitted by non-negative least squares on the relative error. Prints the weights, how often the model
picks the faster backend, the worst ratio of the pick to the better backend, and the same under
leave-one-out (each problem predicted by weights fitted without it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
STRAIGHT = 4096  # the reference machine's Target.straight_line_ops


def terms(row: dict, backend: str) -> np.ndarray:
  """The counts a backend's iteration is modelled on, in the order of ``cost.WEIGHTS``."""
  base = [1.0, row["vectors"], row["entries"]]
  if backend == "dense":
    straight = row["factor"] < STRAIGHT
    return np.array([*base, row["assembly"], 0.0 if straight else row["factor"], row["factor"] if straight else 0.0, row["solve"]], dtype=float)
  return np.array([*base, row["updates"], row["nnz_l"]], dtype=float)


def fit(rows: list[dict], backend: str) -> np.ndarray:
  x = np.array([terms(r, backend) for r in rows])
  y = np.array([r[f"{backend}_us"] / max(r[f"{backend}_iter"], 1) for r in rows])
  w = 1.0 / y  # relative error
  scale = np.maximum(np.abs(x).max(axis=0), 1.0)
  coef, _ = nnls((x / scale) * w[:, None], y * w)
  return coef / scale


def predict(row: dict, backend: str, coef: np.ndarray) -> float:
  return float(terms(row, backend) @ coef)


def evaluate(rows: list[dict], coefs: dict[str, np.ndarray] | None) -> tuple[int, float, list[str]]:
  right, worst, wrong = 0, 1.0, []
  for i, r in enumerate(rows):
    cs = coefs or {b: fit(rows[:i] + rows[i + 1 :], b) for b in ("dense", "sparse")}
    pick = min(("dense", "sparse"), key=lambda b: predict(r, b, cs[b]))
    best = min(r["dense_us"], r["sparse_us"])
    ratio = r[f"{pick}_us"] / best
    right += ratio == 1.0
    worst = max(worst, ratio)
    if ratio > 1.0:
      wrong.append(f"{r['name']} {pick} {ratio:.2f}")
  return right, worst, wrong


def main() -> None:
  rows = json.loads((HERE / "results" / "backend_costs.json").read_text())
  coefs = {b: fit(rows, b) for b in ("dense", "sparse")}
  for b, c in coefs.items():
    print(b, " ".join(f"{v:.3e}" for v in c))
  right, worst, wrong = evaluate(rows, coefs)
  print(f"in sample: faster backend on {right}/{len(rows)}, worst pick {worst:.2f}x the better; {wrong}")
  right, worst, wrong = evaluate(rows, None)
  print(f"leave-one-out: faster backend on {right}/{len(rows)}, worst pick {worst:.2f}x the better; {wrong}")
  for b in ("dense", "sparse"):
    err = [predict(r, b, coefs[b]) / (r[f"{b}_us"] / max(r[f"{b}_iter"], 1)) for r in rows]
    print(f"{b}: predicted/measured per iteration, median {np.median(err):.2f}, range {min(err):.2f}-{max(err):.2f}")


if __name__ == "__main__":
  main()
