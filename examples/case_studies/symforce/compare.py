"""The SymForce study: its Table IV problem, and a scaling sweep, SymForce against Scaly.

    examples/case_studies/symforce/baseline/setup.sh
    uv run examples/case_studies/symforce/compare.py --out examples/case_studies/symforce/results/sweep.json
    uv run examples/case_studies/symforce/compare.py --out examples/case_studies/symforce/results/sweep.json --report

The paper row (5 poses, 20 landmarks) uses the example's checked-in generated code, as SymForce's
benchmark does, with SymForce built twice: as released (its internal timers on) and with the timers
compiled out. Scaly runs grouped (a pose's matching factors in one body) and per factor.

The sweep regenerates the example for each size with SymForce's own generator (`gen_symforce.py`),
builds `bench_symforce.cc` against the no-timer build at the example's flags (`-O3 -mcpu=native`), and
times the same sizes in Scaly. SymForce's fixed variant (the whole problem's linearization as one
generated function) is generated while one generation fits `--fixed-budget`; the dynamic variant runs at
every size. Every process runs behind the quiet-machine gate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "baseline"
TP = BASE / "third_party"
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

SIZES = [5, 10, 20, 50, 100, 200, 500, 1000]
EXAMPLE = Path("symforce/examples/robot_3d_localization")


def env() -> dict[str, str]:
  out = {}
  for line in (BASE / "env.sh").read_text().splitlines():
    if line.startswith("export "):
      k, v = line[len("export ") :].split("=", 1)
      out[k] = v.strip('"')
  return out


def compile_harness(build: Path, include_first: Path | None, sources: list[Path], out: Path, fixed: bool) -> float:
  e = env()
  sf = Path(e["SYMFORCE"])
  inc = [f"-I{include_first}"] if include_first else []
  cmd = [
    "c++", "-std=c++17", "-O3", "-mcpu=native", "-DNDEBUG", *inc, f"-I{sf}", f"-I{sf / 'gen' / 'cpp'}", f"-I{build / 'lcmtypes' / 'cpp'}",
    f"-I{sf / 'third_party' / 'eigen_lcm' / 'lcmtypes' / 'eigen_lcm_lcm' / 'cpp'}", f"-I{sf / 'third_party' / 'skymarshal' / 'include'}",
    f"-I{e['EIGEN_INCLUDE']}", f"-I{build / '_deps' / 'fmtlib-src' / 'include'}", f"-I{build / '_deps' / 'spdlog-src' / 'include'}",
    f"-I{build / '_deps' / 'metis-src' / 'include'}", "-DSPDLOG_FMT_EXTERNAL",
  ]  # fmt: skip
  if (build / "no_tic_toc").exists():
    cmd += [f"-I{build / 'no_tic_toc'}", "-DSYMFORCE_TIC_TOC_HEADER=<noop_tic_toc.h>"]
  if not fixed:
    cmd.append("-DNO_FIXED")
  libs = [f"-L{build / 'symforce' / 'opt'}", f"-L{build}", "-lsymforce_opt", "-lsymforce_gen", "-lsymforce_cholesky"]
  if include_first is None:
    libs = [f"-L{build / 'symforce' / 'examples'}", "-lsymforce_examples", *libs]
  libs += [
    str(build / "_deps" / "spdlog-build" / "libspdlog.a"),
    str(build / "_deps" / "fmtlib-build" / "libfmt.a"),
    str(build / "_deps" / "metis-build" / "libmetis" / "libmetis.a"),
  ]
  rpath = [f"-Wl,-rpath,{build / d}" for d in ("symforce/examples", "symforce/opt", ".")]
  t0 = time.perf_counter()
  subprocess.run([*cmd, *map(str, sources), *libs, *rpath, "-o", str(out)], check=True, capture_output=True)
  return time.perf_counter() - t0


def run_scaly(measurements: Path, symforce_json: Path | None, out: Path, per_factor: bool = False) -> dict:
  cmd = ["uv", "run", str(HERE / "run_scaly.py"), "--measurements", str(measurements), "--out", str(out)]
  if symforce_json:
    cmd += ["--symforce", str(symforce_json)]
  if per_factor:
    cmd.append("--per-factor")
  wait_for_quiet()
  subprocess.run(cmd, check=True, capture_output=True)
  return json.loads(out.read_text())


def paper(work: Path) -> dict:
  """Table IV's problem with the checked-in generated code."""
  e = env()
  row = {}
  for label, build in (("released", Path(e["SYMFORCE_BUILD"])), ("no_timers", Path(e["SYMFORCE_BUILD_NO_TIMERS"]))):
    exe = work / f"bench_{label}"
    compile_harness(build, None, [BASE / "bench_symforce.cc"], exe, fixed=True)
    wait_for_quiet()
    subprocess.run([str(exe), str(work / f"symforce_{label}.json")], check=True, capture_output=True)
    row[f"symforce_{label}"] = json.loads((work / f"symforce_{label}.json").read_text())
  meas = Path(e["SYMFORCE"]) / EXAMPLE / "gen" / "measurements.cc"
  row["scaly"] = run_scaly(meas, work / "symforce_no_timers.json", work / "scaly5.json")
  row["scaly_per_factor"] = run_scaly(meas, work / "symforce_no_timers.json", work / "scaly5_pf.json", per_factor=True)
  return row


def sweep_size(n: int, work: Path, fixed: bool, budget: float) -> tuple[dict, bool]:
  e = env()
  sf, build = Path(e["SYMFORCE"]), Path(e["SYMFORCE_BUILD_NO_TIMERS"])
  d = work / str(n)
  ex = d / EXAMPLE
  shutil.rmtree(d, ignore_errors=True)
  ex.mkdir(parents=True)
  for f in ("common.h", "run_dynamic_size.cc", "run_dynamic_size.h", "run_fixed_size.cc", "run_fixed_size.h"):
    shutil.copy(sf / EXAMPLE / f, ex / f)
  common = ex / "common.h"
  common.write_text(common.read_text().replace("kNumPoses = 5;", f"kNumPoses = {n};"))
  gen_cmd = [e["SYMFORCE_PYTHON"], str(BASE / "gen_symforce.py"), str(sf), str(n), str(ex)]
  row: dict = {"n_poses": n}
  if fixed:
    try:
      done = subprocess.run(gen_cmd, check=True, capture_output=True, text=True, timeout=budget)
      row["symforce_generate"] = json.loads(done.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
      row["fixed_skipped"] = f"generation exceeded {budget:.0f} s"
      fixed = False
  if not fixed:
    done = subprocess.run([*gen_cmd, "--no-fixed"], check=True, capture_output=True, text=True)
    row["symforce_generate"] = json.loads(done.stdout.strip().splitlines()[-1])
  sources = [BASE / "bench_symforce.cc", ex / "run_dynamic_size.cc", ex / "gen" / "measurements.cc"] + ([ex / "run_fixed_size.cc"] if fixed else [])
  exe = d / "bench"
  row["symforce_compile_s"] = compile_harness(build, d, sources, exe, fixed)
  row["symforce_binary_bytes"] = exe.stat().st_size
  wait_for_quiet()
  subprocess.run([str(exe), str(d / "symforce.json")], check=True, capture_output=True)
  row["symforce"] = json.loads((d / "symforce.json").read_text())
  ref = d / "symforce_ref.json"
  sym = row["symforce"]
  ref.write_text(json.dumps({"fixed": sym["fixed"] if "fixed" in sym else sym["dynamic"]}))
  row["scaly"] = run_scaly(ex / "gen" / "measurements.cc", ref, d / "scaly.json")
  return row, fixed


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
  ap.add_argument("--fixed-budget", type=float, default=1200.0)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true")
  args = ap.parse_args()
  res = json.loads(args.out.read_text()) if args.out.exists() else {"sweep": []}
  if not args.report:
    work = Path(tempfile.mkdtemp(prefix="symforce-"))
    if "paper" not in res:
      res["paper"] = paper(work)
      args.out.parent.mkdir(parents=True, exist_ok=True)
      args.out.write_text(json.dumps(res, indent=1))
    fixed = all("fixed_skipped" not in r for r in res["sweep"])
    have = {r["n_poses"] for r in res["sweep"]}
    for n in args.sizes:
      if n in have:
        continue
      row, fixed = sweep_size(n, work, fixed, args.fixed_budget)
      res["sweep"].append(row)
      args.out.write_text(json.dumps(res, indent=1))
      s, c = row["symforce"], row["scaly"]
      fx = s.get("fixed", {})
      print(
        f"n={n:5d} SymForce fixed {fx.get('solve_us', float('nan')):9.1f} us (gen {row['symforce_generate']['t_generate']:.1f} s, "
        f"{row['symforce_generate']['linearization_bytes'] / 1e6:.1f} MB), dynamic {s['dynamic']['solve_us']:9.1f} us; Scaly {c['solve']['best_us']:9.1f} us "
        f"(C {c['solve']['c_bytes'] / 1e3:.0f} kB); iterations {c['solve']['iterations']}",
        flush=True,
      )
  print(json.dumps(res.get("paper", {}).get("scaly", {}).get("vs_symforce", {})))


if __name__ == "__main__":
  main()
