"""The DiffMPC study's grid: Scaly against the recorded baselines, per problem of the paper's Table 3.

    uv run examples/case_studies/diffmpc/baseline/run_baselines.py --out examples/case_studies/diffmpc/results/baselines.json
    uv run examples/case_studies/diffmpc/compare.py --out examples/case_studies/diffmpc/results/scaly.json
    uv run examples/case_studies/diffmpc/compare.py --out examples/case_studies/diffmpc/results/scaly.json --report

Runs `run_scaly.py` for each problem in a fresh process behind the quiet-machine gate, then tabulates
the median over seeds of the forward (one episode) and backward (its gradient) times against the
baselines' in `baselines.json`, and how each baseline's gradient compares with Scaly's full and
truncated ones (for trajax, the truncated one of its one-knot-longer problem).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

NAMES = {
  "mpcpytorch": "mpc.pytorch",
  "diffmpc": "DiffMPC",
  "trajax": "trajax",
  "per_solve_implicit": "Scaly",
  "per_solve_ad": "Scaly, AD through Riccati",
  "hoisted_implicit": "Scaly, Riccati hoisted",
}


def run(problems: list[int], seeds: int, out: Path) -> None:
  rows = json.loads(out.read_text())["rows"] if out.exists() else []
  have = {r["problem"] for r in rows}
  work = Path(tempfile.mkdtemp(prefix="diffmpc-compare-"))
  for problem in problems:
    if problem in have:
      continue
    load = wait_for_quiet()
    res = work / f"scaly_{problem}.json"
    subprocess.run(["uv", "run", str(HERE / "run_scaly.py"), "--problem", str(problem), "--seeds", str(seeds), "--out", str(res)], check=True)
    for r in json.loads(res.read_text())["rows"]:
      r["load"] = load
      rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seeds, "rows": rows}, indent=1))


def relative(a, b) -> float:
  a, b = np.asarray(a), np.asarray(b)
  return float(np.abs(a - b).max() / np.abs(b).max())


def table(scaly: list[dict], baselines: list[dict], threads: str = "1") -> str:
  lines = [
    "| Problem | nx, nu, T, batch | Implementation | Forward, ms | Backward, ms | Scaly faster: forward | backward | Gradient vs Scaly's full | vs truncated |",
    "|" + "---|" * 9,
  ]
  sys.path.insert(0, str(HERE))
  from scaly_impl import PROBLEMS

  for problem in sorted({r["problem"] for r in scaly}):
    p = PROBLEMS[problem]
    ours = {r["variant"]: r for r in scaly if r["problem"] == problem}
    ref = ours["per_solve_implicit"]
    f0, b0 = np.median(ref["forward_s"]), np.median(ref["backward_s"])
    size = f"{p.nx}, {p.nu}, {p.horizon}, {p.batch}"
    for r in [x for x in baselines if x["problem"] == problem and x["threads"] == threads]:
      f, b = np.median(r["forward_s"]), np.median(r["backward_s"])
      full = max(relative(g, s) for g, s in zip(r["gradient"], ref["gradient"]))
      trunc = relative(r["gradient"][0], ref["gradient_truncated_long_seed0" if r["implementation"] == "trajax" else "gradient_truncated_seed0"])
      lines.append(
        f"| {problem} | {size} | {NAMES[r['implementation']]} | {1e3 * f:.1f} | {1e3 * b:.1f} | {f / f0:.1f}x | {b / b0:.1f}x | {full:.0e} | {trunc:.0e} |"
      )
    for v in ("per_solve_implicit", "per_solve_ad", "hoisted_implicit"):
      r = ours[v]
      lines.append(f"| {problem} | {size} | {NAMES[v]} | {1e3 * np.median(r['forward_s']):.2f} | {1e3 * np.median(r['backward_s']):.1f} | | | | |")
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--problems", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
  ap.add_argument("--seeds", type=int, default=5)
  ap.add_argument("--baselines", type=Path, default=HERE / "results" / "baselines.json")
  ap.add_argument("--threads", default="1")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true")
  args = ap.parse_args()
  if not args.report:
    run(args.problems, args.seeds, args.out)
  print(table(json.loads(args.out.read_text())["rows"], json.loads(args.baselines.read_text())["rows"], args.threads))


if __name__ == "__main__":
  main()
