"""A/B timing from C: each case rendered from several source trees, compiled with the JIT's flags, timed interleaved.

    AB_TREES=<dir> uv run --no-sync python internal/notes/perf_2026_09_30_gaps/select_ab/ab.py cases <tree>,<tree> <case>,<case> [rounds]

``<dir>/base_<tree>/scaly`` is a copy of a checkout's ``src/scaly`` (``git archive <rev> src``), put
first on the path to render that tree's C. ``cases.py`` holds the shapes: selects under masks that
are rare, random or always true, piecewise functions on random and sorted data, running sums and
dot products over a select, selects on a flag. Written by the Tier 5 review, whose findings set
C-217's rule; ``../results/select_ahead_cases.txt`` is its table (before C-217, its first rule,
the rule kept).
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

S = Path(__file__).resolve().parent
REPO = S.parents[3]
TREES = Path(os.environ.get("AB_TREES", S / "trees"))
DRIVER_SRC = REPO / "internal/notes/perf_2026_09_30_codegen/time_entry.c"
OUT = S / "out"
FLAGS = ["-O2", "-mcpu=native", "-fno-math-errno"]


def driver():
  exe = OUT / "time_entry"
  if not exe.exists():
    OUT.mkdir(exist_ok=True)
    subprocess.run(["cc", "-O2", str(DRIVER_SRC), "-o", str(exe)], check=True)
  return exe


def main():
  module = sys.argv[1]
  variants = sys.argv[2].split(",")  # names of base_<v> trees under the scratch dir
  cases = sys.argv[3].split(",")
  rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 9
  budget = 0.05
  exe = driver()
  for case in cases:
    cells = []
    for v in variants:
      tree, _, flag = v.partition("+")
      tag = v.replace("+", "_").replace("=", "")
      c, b = OUT / f"{module}_{case}_{tag}.c", OUT / f"{module}_{case}_{tag}.bin"
      env = {**os.environ, "PYTHONPATH": str(TREES / f"base_{tree}")}
      if flag:
        env[flag.split("=")[0]] = flag.split("=")[1]
      r = subprocess.run(["uv", "run", "--no-sync", "python", str(S / "render.py"), case, str(c), str(b), str(S), module], env=env, capture_output=True, text=True, cwd=REPO)
      if r.returncode:
        print(case, v, "RENDER FAILED", r.stderr[-400:]); cells = []; break
      meta = json.loads(r.stdout.strip().splitlines()[-1])
      assert f"base_{tree}" in meta["scaly"], meta
      lib = c.with_suffix(".so")
      subprocess.run(["cc", *FLAGS, "-fPIC", "-shared", str(c), "-lm", "-o", str(lib)], check=True)
      cells.append((v, lib, meta["symbol"], b))
    if not cells:
      continue
    t = time.perf_counter()
    while time.perf_counter() - t < 0.3:
      subprocess.run([str(exe), str(cells[0][1]), cells[0][2], str(cells[0][3]), "20", "/dev/null", "50"], capture_output=True, check=True)
    batches = {}
    for v, lib, sym, b in cells:
      per = float(subprocess.run([str(exe), str(lib), sym, str(b), "5", "/dev/null", "10"], capture_output=True, text=True, check=True).stdout.split()[0])
      batches[v] = max(1, int(budget * 1e9 / max(per, 1.0) / 20))
    best = {v: float("inf") for v, *_ in cells}
    meds = {v: [] for v, *_ in cells}
    outs = {}
    for _ in range(rounds):
      for v, lib, sym, b in cells:
        o = OUT / f"{module}_{case}_{v.replace('+', '_').replace('=', '')}.out"
        r = subprocess.run([str(exe), str(lib), sym, str(b), "20", str(o), str(batches[v])], capture_output=True, text=True, check=True)
        lo, med = map(float, r.stdout.split())
        best[v] = min(best[v], lo); meds[v].append(med)
        outs[v] = np.fromfile(o)
    first = cells[0][0]
    diff = max(float(np.nanmax(np.abs(outs[v] - outs[first]))) if outs[v].size else 0.0 for v in outs)
    print(f"{case:22}", "  ".join(f"{v} {best[v]:10.1f} ns med {np.median(meds[v]):10.1f} ({best[v] / best[first]:5.3f})" for v in best), f" |diff| {diff:.1e}", flush=True)


main()
