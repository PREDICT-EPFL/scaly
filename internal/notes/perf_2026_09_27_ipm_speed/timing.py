"""Time the solvers ``gen.py`` built, from C, interleaving variants, against vendored PIQP.

    uv run internal/notes/perf_2026_09_27_ipm_speed/timing.py --variants base,new [--rounds 5] [--budget 0.1] [--piqp]

Each round calls every variant of every (problem, backend) once through ``time_entry.c`` (the
universal entry called in a loop; no Python in the timed region) for about
``--budget`` seconds; a variant's time is the fastest single solve over all rounds, so variants
see the same machine state (Spotlight, clocks) in turn. ``--piqp`` adds vendored PIQP 0.6.2's own
solve timer, the fastest of as many in-process repeats. Prints a Markdown table and the geometric
mean of each variant's time over the first's; writes ``build/timing_<variants>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
BUILD = HERE / "build"
DRIVER_SRC = HERE / "time_entry.c"


def driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < DRIVER_SRC.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", "-o", str(exe), str(DRIVER_SRC)], check=True)
  return exe


def call(cell: Path, reps: int) -> tuple[float, float, np.ndarray]:
  meta = json.loads((cell / "meta.json").read_text())
  out = subprocess.run(
    [str(driver()), str(cell / "lib.so"), meta["symbol"], str(cell / "inputs.bin"), str(reps), str(cell / "outputs.bin")], capture_output=True, text=True, check=True
  )
  best, median = (float(v) * 1e-3 for v in out.stdout.split())
  return best, median, np.fromfile(cell / "outputs.bin", dtype="<f8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--variants", required=True)
  parser.add_argument("--rounds", type=int, default=5)
  parser.add_argument("--budget", type=float, default=0.1, help="seconds per call of the driver")
  parser.add_argument("--piqp", action="store_true")
  parser.add_argument("--only", help="comma-separated problem names")
  parser.add_argument("--backends", default="sparse,dense")
  args = parser.parse_args()
  variants = args.variants.split(",")
  cells = sorted({p.name for p in (BUILD / variants[0]).iterdir() if (p / "meta.json").exists()})
  cells = [c for c in cells if all((BUILD / v / c / "meta.json").exists() for v in variants)]
  cells = [c for c in cells if c.rsplit("_", 1)[1] in args.backends.split(",")]
  if args.only:
    keep = set(args.only.split(","))
    cells = [c for c in cells if c.rsplit("_", 1)[0] in keep]
  metas = {(v, c): json.loads((BUILD / v / c / "meta.json").read_text()) for v in variants for c in cells}
  metas_order = sorted(cells, key=lambda c: (c.rsplit("_", 1)[1], metas[(variants[0], c)]["n"] + metas[(variants[0], c)]["p"] + metas[(variants[0], c)]["m"]))
  best = {(v, c): float("inf") for v in variants for c in cells}
  med: dict[tuple[str, str], list[float]] = {(v, c): [] for v in variants for c in cells}
  xs: dict[tuple[str, str], np.ndarray] = {}
  reps = {}
  for c in cells:  # repeats per driver call from one probe of the first variant
    b, _, _ = call(BUILD / variants[0] / c, 3)
    reps[c] = int(min(5000, max(5, args.budget / max(b * 1e-6, 1e-7))))
  for r in range(args.rounds):
    for c in cells:
      order = variants if r % 2 == 0 else variants[::-1]
      for v in order:
        b, m, out = call(BUILD / v / c, reps[c])
        best[(v, c)] = min(best[(v, c)], b)
        med[(v, c)].append(m)
        xs[(v, c)] = out
    print(f"round {r + 1}/{args.rounds} done", file=sys.stderr, flush=True)
  piqp = {}
  if args.piqp:
    from gen import problem
    from tests.solvers.ipm import piqp_trace

    for c in cells:
      name, backend = c.rsplit("_", 1)
      tr = piqp_trace.run(problem(name), dense=backend == "dense", repeat=max(10, min(200, reps[c])))
      piqp[c] = {"solve_us": tr.info["solve_time_min"] * 1e6, "setup_us": tr.info["setup_time"] * 1e6, "iter": int(tr.info["iter"]), "status": tr.status}
  head = "| problem | backend | n / p / m | iter | " + " | ".join(f"{v} (us)" for v in variants) + " | " + " | ".join(f"{v}/{variants[0]}" for v in variants[1:])
  if piqp:
    head += " | PIQP solve (us) | PIQP iter | " + " | ".join(f"{v}/PIQP" for v in variants)
  print(head + " |")
  print("|" + " --- |" * (head.count("|") + 1 - 1))
  rows = []
  for c in metas_order:
    name, backend = c.rsplit("_", 1)
    m0 = metas[(variants[0], c)]
    iters = "/".join(str(metas[(v, c)]["iter"]) for v in variants)
    line = f"| {name} | {backend} | {m0['n']} / {m0['p']} / {m0['m']} | {iters} | " + " | ".join(f"{best[(v, c)]:.1f}" for v in variants)
    line += " | " + " | ".join(f"{best[(v, c)] / best[(variants[0], c)]:.3f}" for v in variants[1:])
    row = {"cell": c, "best_us": {v: best[(v, c)] for v in variants}, "median_us": {v: float(np.median(med[(v, c)])) for v in variants}}
    row["x_agree"] = {v: float(np.max(np.abs(xs[(v, c)][: m0["n"]] - xs[(variants[0], c)][: m0["n"]]), initial=0.0)) for v in variants[1:]}
    if piqp:
      p = piqp[c]
      line += f" | {p['solve_us']:.1f} | {p['iter']} | " + " | ".join(f"{best[(v, c)] / p['solve_us']:.2f}" for v in variants)
      row["piqp"] = p
    print(line + " |")
    rows.append(row)
  for backend in args.backends.split(","):
    sel = [r for r in rows if r["cell"].endswith("_" + backend)]
    if not sel:
      continue
    for v in variants[1:]:
      gm = float(np.exp(np.mean([np.log(r["best_us"][v] / r["best_us"][variants[0]]) for r in sel])))
      print(f"{backend}: geometric mean {v}/{variants[0]} = {gm:.3f} over {len(sel)}")
    if piqp:
      for v in variants:
        gm = float(np.exp(np.mean([np.log(r["best_us"][v] / r["piqp"]["solve_us"]) for r in sel])))
        big = [r for r in sel if r["piqp"]["solve_us"] >= 50.0]
        gm_big = float(np.exp(np.mean([np.log(r["best_us"][v] / r["piqp"]["solve_us"]) for r in big]))) if big else float("nan")
        print(f"{backend}: geometric mean {v}/PIQP solve = {gm:.3f} over {len(sel)}; {gm_big:.3f} over {len(big)} with PIQP >= 50 us")
  (BUILD / f"timing_{'_'.join(variants)}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
