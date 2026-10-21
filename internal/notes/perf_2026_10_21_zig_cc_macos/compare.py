# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Compare the Apple Clang and zig cc arms of the macOS closed-loop measurement for #82, cell by cell, and plot the ratios."""

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/zig017-macos")
step_fields = ("fe_time_ms", "solver_time_ms", "iterations")
cells = [(p, s) for p in ("race_cars", "unbumpercars", "npmpc") for s in ("ipopt", "sqp")]


def cell(arm: str, problem: str, solver: str) -> dict[str, float]:
  (path,) = (root / arm).glob(f"repeat_*/{problem}/{solver}+scaly")
  with (path / "telemetry.csv").open() as fp:
    steps = list(csv.DictReader(fp))[1:]
  out = {f: statistics.mean(float(s[f]) for s in steps) for f in step_fields}
  out["build_ms"] = json.loads((path / "summary.json").read_text())["build_ms"]
  return out


ratios = {}
print("| Problem | Solver | Mean per step after the first | Apple Clang | zig cc | zig / Apple Clang |")
print("|---|---|---|---:|---:|---:|")
for problem, solver in cells:
  clang, zig = cell("appleclang", problem, solver), cell("zig", problem, solver)
  for f in clang:
    flag = " ⚠ iterations differ" if f == "iterations" and abs(zig[f] - clang[f]) > 1e-9 else ""
    print(f"| {problem} | {solver} | {f} | {clang[f]:.4g} | {zig[f]:.4g} | {zig[f] / clang[f]:.3f}{flag} |")
  ratios[problem, solver] = (zig["fe_time_ms"] / clang["fe_time_ms"], zig["solver_time_ms"] / clang["solver_time_ms"])

fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
width = 0.38
for i, (label, color) in enumerate((("Function evaluation", "#2a78d6"), ("Solver", "#eb6834"))):
  xs = [k + (i - 0.5) * width for k in range(len(cells))]
  ys = [ratios[c][i] for c in cells]
  ax.bar(xs, [y - 1 for y in ys], width * 0.92, bottom=1, color=color, label=label, zorder=2)
  for x, y in zip(xs, ys):
    ax.text(x, y + (0.003 if y >= 1 else -0.003), f"{y:.3f}", ha="center", va="bottom" if y >= 1 else "top", fontsize=7, color="#52514e")
ax.axhline(1.0, color="#0b0b0b", linewidth=1, zorder=3)
ax.set_xticks(range(len(cells)), [f"{p}\n{s}" for p, s in cells], fontsize=8)
ax.set_ylabel("zig cc / Apple Clang\n(mean time per step, below 1 favours zig)", fontsize=8)
ax.set_ylim(0.9, 1.1)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6, zorder=0)
for side in ("top", "right"):
  ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=2)
ax.set_title("zig cc 0.17 against Apple Clang 21, M4 Max, closed loops with Scaly oracles", fontsize=9)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("ratios.png"))
