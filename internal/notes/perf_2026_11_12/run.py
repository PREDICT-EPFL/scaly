"""Run the fixed before/after compiler grid, then retime its binaries in alternating order."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REVISIONS = {"before": "e7f80e1f0db7f9434b5138411a86d06fa35a0028", "after": "5335c46d9bd0ce96137833ddb2ba737d68758716"}
GRID = {
  "chain": ([3, 5, 9], "casadi_map_sx"),
  "chain_jac": ([3, 5, 9], "casadi_map_sx"),
  "race_cars": ([5, 50, 400], "casadi_sx"),
  "race_cars_jac": ([5, 50, 400], "casadi_sx"),
  "npmpc": ([6, 12, 200], "casadi_mx"),
  "npmpc_jac": ([6, 12, 200], "casadi_mx"),
  "npmpc_decoder": ([16, 64, 256], "casadi_mx"),
  "npmpc_decoder_jac": ([16, 64, 256], "casadi_mx"),
  "unbumpercars": ([2, 8, 32], "casadi_mx"),
}


def run(args):
  root, source = args.root.resolve(), args.source.resolve()
  frozen = root / "frozen"
  bench = frozen / "benchmarks/run.py"
  env = {
    **os.environ,
    "SCALY_VECTOR_LIBM": "glibc",
    "SCALY_CC": "gcc",
    "CC": "gcc",
    "CXX": "clang++",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "SCALY_STUDY_SOURCE": str(source),
    "SCALY_IPOPT_LIB": str(frozen / "native/ipopt/libipopt.so"),
    "SCALY_PIQP_LIB": str(frozen / "native/piqp/libpiqpc.so"),
    "SCALY_FATROP_LIB": str(frozen / "native/fatrop/libfatrop.so"),
    "SCALY_REQUIRE_SOLVERS": "piqp,ipopt,sqp",
    "PYTHONPATH": os.pathsep.join((str(frozen), str(source / "src"))),
  }
  env.pop("VIRTUAL_ENV", None)
  records = []
  mode = "preflight" if args.preflight else "study"
  manifest_path = root / f"{mode}.json"
  manifest = {
    "revisions": REVISIONS,
    "grid": GRID,
    "commands": records,
    "environment": {
      k: env[k]
      for k in env
      if k.startswith("SCALY_") or k in ("CC", "CXX", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONHASHSEED", "PYTHONPATH")
    },
  }

  def execute(command, log=None):
    print("$ " + shlex.join(map(str, command)), flush=True)
    started = time.monotonic()
    with log.open("w") if log else open(os.devnull, "w") as sink:
      result = subprocess.run(list(map(str, command)), cwd=source, env=env, stdout=sink if log else None, stderr=subprocess.STDOUT)
    records.append(
      {
        "command": list(map(str, command)),
        "returncode": result.returncode,
        "seconds": time.monotonic() - started,
        "finished": datetime.now(timezone.utc).isoformat(),
        "log": str(log) if log else None,
      }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
      raise RuntimeError(f"command exited {result.returncode}; see {log or 'the study log'}")

  def uv(*command, log=None):
    execute(["uv", "run", "--frozen", "--project", source, *command], log)

  def select(side):
    execute(["git", "switch", "--detach", REVISIONS[side]])
    execute(["uv", "sync", "--frozen"])
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=source).strip() == b""
    libraries = {
      str(path.relative_to(frozen)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(frozen.glob("native/*/*.so*")) if path.is_file()
    }
    manifest.setdefault("native_libraries", {})[side] = libraries
    if len(manifest["native_libraries"]) == 2:
      assert manifest["native_libraries"]["before"] == manifest["native_libraries"]["after"]
    for name, expected in json.loads((frozen / "frozen.json").read_text()).items():
      assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == expected, name

  def retime():
    rows = []
    for workload, (sizes, control) in GRID.items():
      for size in sizes:
        for backend, category, count in (("scaly", "sweep", 3), (control, "controls", 1)):
          for build in range(1, count + 1):
            paths = {
              side: root
              / side
              / category
              / workload
              / f"repeat_{build}"
              / workload
              / f"{backend}_{'M' if workload.startswith('chain') else 'C' if workload == 'unbumpercars' else 'W' if workload.startswith('npmpc_decoder') else 'N'}{size}"
              for side in REVISIONS
            }
            if not all((path / "benchmark").exists() for path in paths.values()):
              continue
            for repeat in range(4):
              for side in ("before", "after") if repeat % 2 == 0 else ("after", "before"):
                path = paths[side]
                assert all(p.read_text().strip() == "performance" for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
                assert Path("/sys/devices/system/cpu/cpufreq/boost").read_text().strip() == "0"
                command = [str(path / "benchmark"), "--benchmark_min_time=0.5s", "--benchmark_format=json"]
                result = subprocess.run(command, cwd=path, capture_output=True, text=True, check=True, timeout=60, env=env)
                target = root / "retiming" / workload / f"{backend}_{size}" / f"build_{build}_{side}_{repeat + 1}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix(".json").write_text(result.stdout)
                target.with_suffix(".log").write_text(result.stderr)
                assert "correctness ok" in result.stdout + result.stderr
                record = json.loads(result.stdout[result.stdout.index("{") :])["benchmarks"][0]
                assert record["time_unit"] == "ns"
                rows.append(
                  dict(workload=workload, size=size, backend=backend, build=build, repetition=repeat + 1, side=side, runtime_ns=record["real_time"])
                )
    with (root / "retiming.csv").open("w") as fp:
      writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)

  measuring = Path.home() / ".scaly-measuring"
  with measuring.open("x") as lock:
    lock.write(f"post-run-2026-11-12 {mode} pid:{os.getpid()} out:{root}\n")
  try:
    for side in REVISIONS:
      select(side)
      env["SCALY_CACHE_DIR"] = str(root / "caches" / mode / side)
      env["SCALY_CASADI_IPOPT_CACHE"] = str(root / "caches" / mode / side / "casadi-ipopt")
      if args.preflight:
        check = root / "checks" / side
        check.mkdir(parents=True, exist_ok=True)
        uv(bench, "smoke", log=check / "smoke.log")
        uv(
          bench,
          "sweep",
          "--workloads",
          "chain",
          "--sizes",
          "3",
          "--backends",
          "scaly,casadi_map_sx",
          "--repetitions",
          "1",
          "--headline",
          "--boost",
          "off",
          "--benchmark-min-time",
          "0.5s",
          "--out",
          check / "chain.csv",
          log=check / "toolchain.log",
        )
        continue
      for workload, (sizes, control) in GRID.items():
        for backend, category, repetitions in (("scaly", "sweep", 3), (control, "controls", 1)):
          target = root / side / category / workload / f"{workload}.csv"
          target.parent.mkdir(parents=True, exist_ok=True)
          uv(
            bench,
            "sweep",
            "--workloads",
            workload,
            "--sizes",
            ",".join(map(str, sizes)),
            "--backends",
            backend,
            "--repetitions",
            str(repetitions),
            "--order-seed",
            "0",
            "--headline",
            "--boost",
            "off",
            "--benchmark-min-time",
            "0.5s",
            "--out",
            target,
          )
      for problem in ("chain", "race_cars", "npmpc", "unbumpercars"):
        for solver in ("ipopt", "sqp"):
          target = root / side / "closed-loop" / problem / solver
          oracles = "scaly" if (problem, solver) == ("chain", "ipopt") else "scaly,casadi"
          uv(
            bench,
            "closed-loop",
            "--problem",
            problem,
            "--solver",
            solver,
            "--oracle",
            oracles,
            "--repetitions",
            "1",
            "--order-seed",
            "0",
            "--headline",
            "--boost",
            "off",
            "--out-dir",
            target,
          )
      uv(bench, "report", root / side)
    if not args.preflight:
      retime()
      (root / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
  finally:
    measuring.unlink()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  parser.add_argument("source", type=Path)
  parser.add_argument("--preflight", action="store_true")
  run(parser.parse_args())
