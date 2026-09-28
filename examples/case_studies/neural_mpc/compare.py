"""Run the Real-time Neural MPC grid in fresh processes and tabulate it next to Table II.

    examples/case_studies/neural_mpc/baseline/setup.sh
    ACADOS_SOURCE_DIR=... DYLD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib \\
      uv run --with torch --with $ACADOS_SOURCE_DIR/interfaces/acados_template \\
      examples/case_studies/neural_mpc/compare.py --out examples/case_studies/neural_mpc/results/grid.json

For each network of Table II and each column, `--processes` fresh processes (an empty Scaly cache, torch
on one thread), each running the authors' 50-step loop after 20 warm-up steps; the table keeps the
process with the lowest mean step time. Columns:

  naive    CasADi's expanded network in acados (the paper's "Naive")
  rtn      PyTorch's Taylor surrogate fed to acados (the paper's "RTN-MPC")
  dropin   Scaly's exact network in acados, replacing CasADi's generated sensitivities
  taylor   RTN-MPC with Scaly computing the surrogate instead of PyTorch
  naive0   the naive column with the authors' zeroed output layer, run at two sizes to show that
           CasADi 3.8 then removes the network from the generated code

A process that runs longer than `--timeout` (a compile that does not finish, say) is recorded as such.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

SIZES = [(0, 0), (2, 16), (2, 128), (5, 16), (5, 128), (12, 32), (12, 512)]
PAPER_I7 = {  # Table II, i7 CPU, Hz: naive, RTN-MPC (the "None" cell spans both rows)
  (0, 0): (4262, 4262),
  (2, 16): (2228, 1096),
  (2, 128): (116, 1071),
  (5, 16): (1139, 885),
  (5, 128): (31, 784),
  (12, 32): (168, 588),
  (12, 512): (11, 507),
}
COLUMNS = ("naive", "rtn", "dropin", "taylor")


def run(column: str, layers: int, width: int, timeout: float) -> dict:
  if column in ("naive", "rtn", "naive0"):
    cmd = [sys.executable, str(HERE / "baseline" / "run_acados.py"), "--layers", str(layers), "--width", str(width), "--mode", column.rstrip("0")]
    if column == "naive0":
      cmd += ["--out-scale", "0"]
  else:
    cmd = [sys.executable, str(HERE / "run_scaly.py"), "--layers", str(layers), "--width", str(width), "--mode", column]
  load = wait_for_quiet()
  env = {**os.environ, "OMP_NUM_THREADS": "1"}
  env.pop("SCALY_CACHE_DIR", None)
  try:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    return {"column": column, "layers": layers, "width": width, "failed": f"exceeded {timeout:.0f} s", "load": load}
  if proc.returncode:
    return {"column": column, "layers": layers, "width": width, "failed": proc.stderr[-800:], "load": load}
  return {"column": column, **json.loads(proc.stdout.strip().splitlines()[-1]), "load": load}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--processes", type=int, default=3)
  ap.add_argument("--timeout", type=float, default=1800.0)
  ap.add_argument("--only", nargs="*", help="columns to run")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  results = json.loads(args.out.read_text()) if args.out.exists() else {}
  jobs = [(c, layers, width) for layers, width in SIZES for c in COLUMNS if (layers, width) != (0, 0) or c == "naive"]
  jobs += [("naive0", 2, 128), ("naive0", 5, 128)]
  for column, layers, width in jobs:
    key = f"{column}:{layers}x{width}"
    if key in results or (args.only and column not in args.only):
      continue
    runs = [run(column, layers, width, args.timeout) for _ in range(args.processes)]
    ok = [r for r in runs if "failed" not in r]
    best = min(ok, key=lambda r: r["mean"]) if ok else runs[0]
    best["means"] = [r["mean"] for r in ok]
    results[key] = best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(key, {k: best.get(k) for k in ("hz_mean", "mean", "t_lin_median", "t_setup", "failed")}, flush=True)
  print(table(results))
  print(json.dumps(agreement(results), indent=1))


def table(results: dict) -> str:
  rows = [
    "| Network | Parameters | Paper naive, Hz | Paper RTN, Hz | Naive (CasADi), Hz | RTN (PyTorch), Hz | Scaly in acados, Hz | Scaly Taylor, Hz | Linearization naive / Scaly, µs |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for layers, width in SIZES:
    params = 0 if layers == 0 else (2 * width + width) + (layers - 1) * (width * width + width) + (2 * width + 2)
    paper = PAPER_I7[(layers, width)]

    def hz(column: str) -> str:
      r = results.get(f"{column}:{layers}x{width}")
      if r is None:
        return "–"
      return "failed" if "failed" in r else f"{r['hz_mean']:.0f}"

    def lin(column: str) -> str:
      r = results.get(f"{column}:{layers}x{width}")
      return "–" if r is None or "failed" in r else f"{1e6 * r['t_lin_median']:.0f}"

    name = "none" if layers == 0 else f"{layers} x {width}"
    rows.append(
      f"| {name} | {params:,} | {paper[0]} | {paper[1]} | {hz('naive')} | {hz('rtn')} | {hz('dropin')} | {hz('taylor')} | {lin('naive')} / {lin('dropin')} |"
    )
  return "\n".join(rows)


def agreement(results: dict) -> dict:
  """Largest closed-loop state difference of each column from the naive (exact CasADi) column."""
  out = {}
  for layers, width in SIZES[1:]:
    ref = results.get(f"naive:{layers}x{width}")
    if ref is None or "failed" in ref:
      continue
    for column in COLUMNS[1:]:
      r = results.get(f"{column}:{layers}x{width}")
      if r and "failed" not in r:
        out[f"{column}:{layers}x{width}"] = float(np.max(np.abs(np.array(r["states"]) - np.array(ref["states"]))))
  return out


if __name__ == "__main__":
  main()
