"""The two IPM backends' work per iteration, from a problem's structure alone, against their measured time.

    uv run internal/notes/perf_2026_09_30_gaps/backend_costs.py --timing ../perf_2026_09_27_ipm_speed/build/timing_<variant>.json

For each of the 55 problems of the IPM speed study: the counts a backend's iteration is made of,
read off the ``QPStructure`` and the sparse KKT matrix's symbolic factorization, the quantities
``opt.ipm``'s generation-time choice (A5) weighs; and, from the timing file, each backend's
measured time per iteration. Writes ``results/backend_costs.json``, which the fit reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "perf_2026_09_27_ipm_speed"))


def features(name: str) -> dict:
  import dataclasses

  from gen import problem

  from scaly.opt.ipm.cost import work
  from tests.opt.ipm.problems import ipm_inputs

  s, _ = ipm_inputs(problem(name))
  return {"name": name, "n": s.n, "p": s.p, "m": s.m, **dataclasses.asdict(work(s))}


def main() -> None:
  from gen import all_names

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--timing", type=Path, required=True)
  args = parser.parse_args()
  rows_in = json.loads(args.timing.read_text())
  variant = next(iter(rows_in[0]["best_us"]))
  build = args.timing.parent / variant
  best = {r["cell"]: r for r in rows_in}
  rows = []
  for name in all_names():
    row = features(name)
    for backend in ("sparse", "dense"):
      cell = best.get(f"{name}_{backend}")
      meta = json.loads((build / f"{name}_{backend}" / "meta.json").read_text())
      row[f"{backend}_us"] = cell["best_us"][variant] if cell else None
      row[f"{backend}_iter"] = meta["iter"]
      row["piqp_us"] = row.get("piqp_us") or {}
      if cell and "piqp" in cell:
        row["piqp_us"][backend] = cell["piqp"]["solve_us"]
    rows.append(row)
    print(name, row, flush=True)
  (HERE / "results").mkdir(exist_ok=True)
  (HERE / "results" / "backend_costs.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
