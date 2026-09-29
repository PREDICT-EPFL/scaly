# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""Minimum-compliance truss design: assembly by run-time scatter, a dense solve, and its gradient.

A cantilever is to carry a tip load with a given volume of material. Start from a *ground
structure*: nodes on a grid, a bar between every pair of neighbours (diagonals included), and choose
the cross-section area of every bar to make the structure as stiff as possible,

    minimize C(a) = f^T u(a)   subject to   K(a) u = f,   sum_e a_e L_e = V,   a_min <= a,

where ``K(a) = sum_e a_e E / L_e g_e g_e^T`` is the stiffness matrix. The optimality-criteria method
rescales every area by ``(-dC/da_e / (eta L_e))^(1/2)`` and finds the multiplier ``eta`` that meets
the volume by bisection; thin bars shrink to ``a_min`` and the truss that remains is the design.

The analysis is one generated ``Function`` whose *inputs* include the structure itself: node
coordinates, the bar list as an ``int64`` table and the free degrees of freedom. Each bar's sixteen
stiffness entries are scattered into ``K`` with ``put_add`` at indices computed from the bar table,
so the same compiled code analyses any truss with as many nodes and bars. The reduced system is
solved by the generated dense ``cholesky``, and ``sc.gradient`` of the compliance in the areas runs
reverse mode through the solve, the scatter and the ``take``s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "truss_sizing"

NODES_X, NODES_Y, SPACING = 7, 4, 1.0
E = 1.0
A_MIN, A_MAX = 1e-4, 100.0


def ground_structure() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """``(coordinates (n, 2), bars (m, 2) int64, free dofs int64, load (2 n))``: pinned on the left, tip load."""
  coords = np.array([(i * SPACING, j * SPACING) for i in range(NODES_X) for j in range(NODES_Y)], dtype=float)
  bars = [(p, q) for p in range(len(coords)) for q in range(p + 1, len(coords)) if np.linalg.norm(coords[p] - coords[q]) <= 1.5 * SPACING]
  fixed = {2 * p + d for p in range(len(coords)) if coords[p, 0] == 0.0 for d in (0, 1)}
  free = np.array([d for d in range(2 * len(coords)) if d not in fixed], dtype=np.int64)
  load = np.zeros(2 * len(coords))
  tip = int(np.argmin(np.abs(coords[:, 0] - coords[:, 0].max()) + np.abs(coords[:, 1] - SPACING)))
  load[2 * tip + 1] = -1.0
  return coords, np.array(bars, dtype=np.int64), free, load


COORDS, BARS, FREE, LOAD = ground_structure()
NN, NB, NF = len(COORDS), len(BARS), len(FREE)
NDOF = 2 * NN


@sc.function(NB, 2 * NN, sc.L((NB, 2), dtype="int64"), sc.L(NF, dtype="int64"), NDOF, output=sc.G("compliance", "gradient", "lengths"))
def analyse(areas: sc.Expr, coords: sc.Expr, bars: sc.Expr, free: sc.Expr, load: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  start, end = bars[:, 0], bars[:, 1]
  dx = sc.take(coords, 2 * end) - sc.take(coords, 2 * start)
  dy = sc.take(coords, 2 * end + 1) - sc.take(coords, 2 * start + 1)
  length = (dx * dx + dy * dy).sqrt()
  g = sc.stack([-dx, -dy, dx, dy], axis=1) / length.reshape((NB, 1))  # (NB, 4): the bar's direction in its four dofs
  dofs = sc.stack([2 * start, 2 * start + 1, 2 * end, 2 * end + 1], axis=1)  # (NB, 4) int64
  # Sixteen entries per bar: k_e g g^T at (dofs[i], dofs[j]), summed where bars share a node.
  entries = ((E * areas / length).reshape((NB, 1, 1)) * g.reshape((NB, 4, 1)) * g.reshape((NB, 1, 4))).reshape((NB * 16,))
  where = (dofs.reshape((NB, 4, 1)) * NDOF + dofs.reshape((NB, 1, 4))).reshape((NB * 16,))
  k = sc.put_add(sc.const(np.zeros(NDOF * NDOF)), where, entries).reshape((NDOF, NDOF))
  k_free = sc.take(sc.take(k, free).T, free)  # the rows and columns of the free dofs
  f_free = sc.take(load, free)
  u_free = linalg.cho_solve(linalg.cholesky(k_free), f_free)
  compliance = (f_free * u_free).sum()
  return compliance, sc.gradient(compliance, areas), length


def reference(areas: np.ndarray) -> tuple[float, np.ndarray]:
  """Compliance and its gradient in NumPy, from the textbook ``dC/da_e = -u_e^T k_e u_e / a_e``."""
  k = np.zeros((NDOF, NDOF))
  blocks = []
  for (p, q), a in zip(BARS, areas, strict=True):
    d = COORDS[q] - COORDS[p]
    length = np.linalg.norm(d)
    g = np.r_[-d, d] / length
    dofs = [2 * p, 2 * p + 1, 2 * q, 2 * q + 1]
    unit = E / length * np.outer(g, g)
    k[np.ix_(dofs, dofs)] += a * unit
    blocks.append((dofs, unit))
  u = np.zeros(NDOF)
  u[FREE] = np.linalg.solve(k[np.ix_(FREE, FREE)], LOAD[FREE])
  return float(LOAD @ u), np.array([-u[dofs] @ unit @ u[dofs] for dofs, unit in blocks])


def optimality_criteria(mean_area: float = 0.1, iterations: int = 80, move: float = 0.3) -> dict[str, np.ndarray]:
  structure = (COORDS.reshape(-1), BARS, FREE, LOAD)
  _, _, lengths = analyse(np.full(NB, A_MAX), *structure)
  volume = mean_area * lengths.sum()
  areas = np.full(NB, volume / lengths.sum())
  history = []
  for _ in range(iterations):
    compliance, grad, _ = analyse(areas, *structure)
    history.append(float(compliance))
    lo, hi = 1e-12, 1e12
    while hi / lo > 1 + 1e-10:  # bisection on the volume multiplier
      eta = np.sqrt(lo * hi)
      trial = np.clip(
        areas * np.sqrt(np.maximum(-grad, 0.0) / (eta * lengths)), np.maximum(A_MIN, areas * (1 - move)), np.minimum(A_MAX, areas * (1 + move))
      )
      lo, hi = (eta, hi) if trial @ lengths > volume else (lo, eta)
    areas = trial
  return {"areas": areas, "lengths": lengths, "history": np.array(history), "volume": np.array(volume)}


def render(areas: np.ndarray) -> str:
  lines = []
  for e in np.argsort(-areas):
    if areas[e] < 0.1 * areas.max():
      break
    (x0, y0), (x1, y1) = COORDS[BARS[e, 0]], COORDS[BARS[e, 1]]
    lines.append(f"  ({x0:.0f},{y0:.0f}) -> ({x1:.0f},{y1:.0f})  area {areas[e]:.3f}")
  return "\n".join(lines)


if __name__ == "__main__":
  out = optimality_criteria()
  h = out["history"]
  kept = int((out["areas"] >= 0.1 * out["areas"].max()).sum())
  print(f"{NB} candidate bars, {NDOF - NF} fixed dofs; compliance {h[0]:.2f} (uniform) -> {h[-1]:.2f} after {len(h)} iterations")
  print(f"{kept} bars above a tenth of the largest area:")
  print(render(out["areas"]))
  write_module(analyse, GENERATED)
  print(f"generated C in {GENERATED}")
