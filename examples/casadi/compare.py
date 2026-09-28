"""Run every CasADi/Scaly pair in this directory and compare results, code size, setup time and run time.

    uv run examples/casadi/compare.py                 # every pair, 3 fresh processes per variant
    uv run examples/casadi/compare.py rocket race_car --processes 5
    uv run examples/casadi/compare.py --dir examples/interp/pairs   # the pairs in another directory

There are up to three variants per pair: the CasADi file as written (oracles evaluated by CasADi's
virtual machine, unless the original compiles them), the same file with ``CASADI_JIT=1`` when it
calls ``casadi_jit()`` (``expand`` and ``jit`` with the compiler flags Scaly's JIT uses), and the
Scaly file. Each variant runs in its own fresh processes, with an empty Scaly JIT cache, one BLAS and
OpenMP thread, and the libraries (NumPy, SciPy, CasADi, Scaly) imported before the clock starts.
In each process:

- setup is the time from importing the example module to ``build()`` returning plus the first
  ``run()``, minus one steady-state ``run()``: modelling, derivative construction, solver creation
  and, for Scaly (and the CasADi examples that generate code), C generation and compilation;
- run is the median over repeated ``run()`` calls after the first.

The reported numbers are medians over the processes. Code size counts the lines of each file that
hold code, not comments, docstrings or blank lines. Results agree when every array the two sides
return matches to ``RTOL`` relative to the array's largest entry; keys named ``iter*`` are
iteration counts and are reported, not compared.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import tokenize
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = HERE  # where the pairs are: this directory, or --dir
RTOL = 1e-6  # agreement: max |casadi - scaly| <= RTOL * max(max |casadi|, FLOOR), per array
FLOOR = 1e-2  # arrays smaller than this (multipliers of inactive constraints, a zero optimum) compare absolutely
JIT_CALL = re.compile(r"casadi_jit\((expand=False)?\)")


def pairs() -> list[str]:
  return sorted(p.name.removesuffix("_casadi.py") for p in PAIRS.glob("*_casadi.py") if (PAIRS / p.name.replace("_casadi", "_scaly")).exists())


def code_lines(path: Path) -> int:
  """Lines holding a token other than a comment, a docstring or layout."""
  lines: set[int] = set()
  prev = tokenize.INDENT
  for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
    if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
      prev = tok.type if tok.type != tokenize.COMMENT else prev
      continue
    is_docstring = tok.type == tokenize.STRING and prev in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE)
    if not is_docstring:
      lines.update(range(tok.start[0], tok.end[0] + 1))
    prev = tok.type
  return len(lines)


def variants(name: str) -> list[str]:
  jit = JIT_CALL.search((PAIRS / f"{name}_casadi.py").read_text()) is not None
  return ["casadi", *(["casadi_jit"] if jit else []), "scaly"]


def worker(name: str, side: str, min_time: float, max_reps: int) -> None:
  """One fresh process: import, build, run until ``min_time`` has passed; JSON on the last line of stdout."""
  import numpy  # noqa: F401  (library imports stay outside the clock)
  import scipy.linalg  # noqa: F401

  if side.startswith("casadi"):
    import casadi  # noqa: F401
    import casadi.tools  # noqa: F401
  else:
    import scaly  # noqa: F401
    import scaly.codegen  # noqa: F401
  sys.path.insert(0, str(HERE))
  sys.path.insert(0, str(PAIRS))  # a pair directory's own _common, if it has one, comes first
  import _common  # noqa: F401

  t0 = time.perf_counter()
  module = __import__(f"{name}_{side.removesuffix('_jit')}")
  run = module.build()
  t_build = time.perf_counter() - t0
  t0 = time.perf_counter()
  out = run()
  t_first = time.perf_counter() - t0
  times: list[float] = []
  start = time.perf_counter()
  while len(times) < 3 or (time.perf_counter() - start < min_time and len(times) < max_reps):
    t0 = time.perf_counter()
    run()
    times.append(time.perf_counter() - t0)
  t_run = statistics.median(times)
  record = {"build": t_build, "first": t_first, "run": t_run, "setup": t_build + t_first - t_run, "reps": len(times)}
  record["out"] = {k: np.asarray(v, dtype=float).reshape(-1).tolist() for k, v in out.items()}
  print(json.dumps(record))


def measure(name: str, side: str, processes: int, min_time: float, max_reps: int) -> dict:
  records = []
  for _ in range(processes):
    with tempfile.TemporaryDirectory(prefix="casadi-examples-") as work:  # the Scaly JIT cache, and where CasADi's JIT writes
      env = {**os.environ, "SCALY_CACHE_DIR": work, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
      env["CASADI_JIT"] = "1" if side == "casadi_jit" else "0"
      cmd = [sys.executable, __file__, "--worker", name, side, "--dir", str(PAIRS), "--min-time", str(min_time), "--max-reps", str(max_reps)]
      proc = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=work)
    if proc.returncode != 0:
      if side == "casadi_jit":  # a JIT build that fails (gcc out of memory, say) is a result, not an error
        print(f"{name}_{side} failed:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return {"failed": proc.stderr.strip().splitlines()[-1][-300:] if proc.stderr.strip() else f"exit {proc.returncode}"}
      raise RuntimeError(f"{name}_{side} failed:\n{proc.stderr[-3000:]}")
    records.append(json.loads(proc.stdout.strip().splitlines()[-1]))
  summary = {key: statistics.median(r[key] for r in records) for key in ("build", "first", "run", "setup")}
  return {**summary, "reps": records[0]["reps"], "out": records[0]["out"], "loc": code_lines(PAIRS / f"{name}_{side.removesuffix('_jit')}.py")}


def agreement(a: dict[str, list[float]], b: dict[str, list[float]]) -> dict:
  worst, worst_key, missing, iterations = 0.0, "", sorted(set(a) ^ set(b)), {}
  for key in sorted(set(a) & set(b)):
    x, y = np.asarray(a[key]), np.asarray(b[key])
    if key.startswith("iter"):
      iterations[key] = (x.tolist(), y.tolist())
      continue
    if x.shape != y.shape:
      return {"ok": False, "max_rel": float("inf"), "key": key, "missing": missing, "iterations": iterations}
    finite = np.isfinite(x) & np.isfinite(y)
    scale = max(FLOOR, float(np.max(np.abs(x[finite]), initial=0.0)))
    err = float(np.max(np.abs(x - y)[finite], initial=0.0)) / scale
    if not np.array_equal(np.isfinite(x), np.isfinite(y)):
      err = float("inf")
    if err > worst:
      worst, worst_key = err, key
  return {"ok": worst <= RTOL and not missing, "max_rel": worst, "key": worst_key, "missing": missing, "iterations": iterations}


def fmt_time(t: float) -> str:
  return f"{t * 1e6:.0f} µs" if t < 1e-3 else f"{t * 1e3:.1f} ms" if t < 1.0 else f"{t:.2f} s"


def table(results: dict[str, dict]) -> str:
  """Times as CasADi · CasADi with JIT · Scaly; n/a where the CasADi file has no JIT variant."""
  rows = [
    "| Example | Agree (max rel. diff) | Iterations CasADi / Scaly | Code lines CasADi / Scaly | Setup CasADi · JIT · Scaly | Run CasADi · JIT · Scaly |",
    "| --- | --- | --- | ---: | ---: | ---: |",
  ]
  for name, r in results.items():
    c, s, agree = r["casadi"], r["scaly"], r["agreement"]
    j = r.get("casadi_jit")
    ok = f"{'yes' if agree['ok'] else '**no**'} ({agree['max_rel']:.0e}{', ' + agree['key'] if agree['key'] else ''})"
    its = [(int(sum(x)), int(sum(y))) for x, y in agree["iterations"].values()]
    if len(its) > 2:  # many solves: the totals
      its = [(sum(x for x, _ in its), sum(y for _, y in its))]
    its = "; ".join(f"{x} / {y}" for x, y in its) or "n/a"

    def three(key: str) -> str:
      jit = "n/a" if j is None else "failed" if "failed" in j else fmt_time(j[key])
      return " · ".join([fmt_time(c[key]), jit, fmt_time(s[key])])

    rows.append(f"| `{name}` | {ok} | {its} | {c['loc']} / {s['loc']} | {three('setup')} | {three('run')} |")
  return "\n".join(rows)


def machine() -> str:
  import platform

  import casadi

  import scaly

  cpu = platform.processor() or platform.machine()
  try:
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
  except (OSError, StopIteration):
    pass
  cc = subprocess.run([os.environ.get("SCALY_CC", "cc"), "--version"], capture_output=True, text=True).stdout.splitlines()[0]
  return (
    f"{cpu}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}; Python {platform.python_version()}, "
    f"CasADi {getattr(casadi, '__version__', '?')}, Scaly {scaly.__dict__.get('__version__', 'from this checkout')}, NumPy {np.__version__}; {cc}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("names", nargs="*", help="pairs to run (default: all)")
  parser.add_argument("--dir", type=Path, default=HERE, help="the directory holding the pairs (default: this one)")
  parser.add_argument("--processes", type=int, default=3)
  parser.add_argument("--min-time", type=float, default=1.0, help="seconds of repeated runs per process")
  parser.add_argument("--max-reps", type=int, default=200)
  parser.add_argument("--json", type=Path, default=None, help="also write the raw results here, after every pair")
  parser.add_argument("--resume", action="store_true", help="keep the pairs already in --json and run the rest")
  parser.add_argument("--worker", nargs=2, metavar=("NAME", "SIDE"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  global PAIRS
  PAIRS = args.dir.resolve()
  if args.worker:
    worker(args.worker[0], args.worker[1], args.min_time, args.max_reps)
    return
  results = json.loads(args.json.read_text())["results"] if args.resume and args.json and args.json.exists() else {}
  for name in args.names or pairs():
    if name in results:
      continue
    print(f"{name} ...", file=sys.stderr, flush=True)
    r = {side: measure(name, side, args.processes, args.min_time, args.max_reps) for side in variants(name)}
    r["agreement"] = agreement(r["casadi"]["out"], r["scaly"]["out"])
    if "out" in r.get("casadi_jit", {}) and not agreement(r["casadi"]["out"], r["casadi_jit"]["out"])["ok"]:
      r["agreement"]["ok"] = False  # the JIT variant must reproduce the interpreted one
    results[name] = r
    if args.json:
      args.json.write_text(json.dumps({"machine": machine(), "results": results}, indent=1))
  results = {name: results[name] for name in sorted(results)}
  for name, r in results.items():  # code size from the files as they are now, also for resumed pairs
    for side in ("casadi", "scaly"):
      r[side]["loc"] = code_lines(PAIRS / f"{name}_{side}.py")
  print(f"Machine: {machine()}\n")
  print(table(results))
  if not all(r["agreement"]["ok"] for r in results.values()):
    sys.exit(1)


if __name__ == "__main__":
  main()
