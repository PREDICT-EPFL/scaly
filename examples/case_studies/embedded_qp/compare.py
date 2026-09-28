"""The oscillating-masses MPC of qoco-benchmarks, every solver, every horizon, in fresh processes.

    examples/case_studies/embedded_qp/baseline/setup.sh
    uv run --with qoco --with qocogen --with clarabel --with cvxpy --with pandas --with osqp \\
      examples/case_studies/embedded_qp/compare.py --out examples/case_studies/embedded_qp/results/grid.json

Per horizon: the QOCO side (`baseline/run_qoco.py`: QOCO, Clarabel, OSQP at 1e-7 and 1e-3, and QOCOGEN
generated from the first instance) and the Scaly side (`run_scaly.py`: the generated PIQP and the
PIQP library), each on the same `--instances` instances, each in its own process with an empty JIT
cache after the quiet-machine gate. QOCOGEN is attempted beyond the benchmark's own limit (it generates
only up to horizon 56) until one generation plus compile exceeds `--qocogen-budget`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

HORIZONS = [8, 20, 32, 44, 56, 76, 96, 116, 136, 156, 200]


def run(cmd: list[str], timeout: float) -> tuple[bool, str]:
  load = wait_for_quiet()
  env = {**os.environ, "OMP_NUM_THREADS": "1"}
  env.pop("SCALY_CACHE_DIR", None)
  try:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    return False, f"exceeded {timeout:.0f} s (load {load:.1f})"
  return proc.returncode == 0, proc.stderr[-800:]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
  ap.add_argument("--instances", type=int, default=5)
  ap.add_argument("--qocogen-budget", type=float, default=1800.0)
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  results = json.loads(args.out.read_text()) if args.out.exists() else {}
  qocogen_ok = all(r.get("qocogen_ok", True) for r in results.values())
  for horizon in args.horizons:
    key = str(horizon)
    if key in results:
      continue
    row: dict = {"horizon": horizon}
    with tempfile.TemporaryDirectory(prefix="embedded-qp-") as tmp:
      out = Path(tmp) / "qoco.json"
      cmd = [sys.executable, str(HERE / "baseline" / "run_qoco.py"), "--horizon", str(horizon), "--instances", str(args.instances)]
      cmd += ["--work", str(Path(tmp) / "qocogen"), "--out", str(out)] + ([] if qocogen_ok else ["--no-qocogen"])
      t0 = time.perf_counter()
      ok, err = run(cmd, args.qocogen_budget + 600)
      if not ok and qocogen_ok and err.startswith("exceeded"):  # QOCOGEN's generation or compile ran out: record it, rerun without it
        row["qocogen_failed"] = err
        qocogen_ok = False
        ok, err = run(cmd + ["--no-qocogen"], 3600)
      row["qoco_side"] = json.loads(out.read_text()) if ok else {"failed": err}
      row["qoco_side_wall"] = time.perf_counter() - t0
      row["qocogen_ok"] = qocogen_ok
      for solver in ("scaly_sparse", "piqp_sparse"):
        sout = Path(tmp) / f"{solver}.json"
        ok, err = run([sys.executable, str(HERE / "run_scaly.py"), "--horizon", str(horizon), "--solver", solver, "--instances", str(args.instances), "--out", str(sout)], 3600)
        row[solver] = json.loads(sout.read_text()) if ok else {"failed": err}
    results[key] = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, default=float))
    print(horizon, summary(row), flush=True)
  print(table(results))


def median(values: list[float]) -> float:
  return float(np.median(values)) if values else float("nan")


def summary(row: dict) -> dict:
  out = {}
  q = row["qoco_side"]
  if "rows" in q:
    for name in ("qoco", "clarabel", "osqp", "osqp_1e-3", "qocogen"):
      times = [r[name]["solve_time" if name != "qocogen" else "solve_s"] for r in q["rows"] if name in r]
      if times:
        out[name] = 1e6 * median(times)
  for name in ("scaly_sparse", "piqp_sparse"):
    if "instances" in row[name]:
      out[name] = 1e6 * median([i["solve_s"] for i in row[name]["instances"]])
  return {k: round(v, 1) for k, v in out.items()}


def table(results: dict) -> str:
  cols = [("qocogen", "QOCOGEN"), ("scaly_sparse", "Scaly PIQP (generated)"), ("piqp_sparse", "PIQP library"), ("qoco", "QOCO"), ("clarabel", "Clarabel"), ("osqp", "OSQP 1e-7"), ("osqp_1e-3", "OSQP 1e-3")]
  rows = ["| Horizon | Variables | " + " | ".join(f"{c}, µs" for _, c in cols) + " |", "| ---: | ---: | " + " | ".join("---:" for _ in cols) + " |"]
  for key in sorted(results, key=int):
    row = results[key]
    s = summary(row)
    t = int(key)
    rows.append(f"| {t} | {4 * t + 8 * (t + 1)} | " + " | ".join(f"{s[c]:.1f}" if c in s else "–" for c, _ in cols) + " |")
  return "\n".join(rows)


if __name__ == "__main__":
  main()
