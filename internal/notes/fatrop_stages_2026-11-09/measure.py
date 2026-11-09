"""Record native function-evaluation time for both Fatrop paths on the builder's chain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.harness.provenance import collect, require_headline_settings
from benchmarks.problems.chain import ChainParams, chain_ocp, initial_state
import scaly as sc


def worker(args: argparse.Namespace) -> None:
  ocp = chain_ocp(args.masses, args.horizon)
  problem = ocp.stage_problem()
  if args.path == "whole":
    problem = replace(problem, stage_functions=None)
  solve = sc.solver(problem, "fatrop", options={"tol": 1e-10})
  start = initial_state(args.masses)
  params = (start, *(np.array([value]) for value in ChainParams(dt=1.0 / args.horizon).array()))
  guess = ocp.pack(start, 0.0, stage_form=True)
  solve(params, x0=guess)
  samples = []
  for _ in range(30):
    variables, *_ = solve(params, x0=guess)
    stats = solve.stats()
    assert stats.to_solver_status().ok, stats
    samples.append(asdict(stats))
  args.out.mkdir(parents=True)
  result = {
    "path": args.path,
    "M": args.masses,
    "N": args.horizon,
    "samples": samples,
    "solution": np.concatenate([variables[0].ravel(), variables[1]]).tolist(),
    "provenance": collect(ROOT, "gcc", sys.argv),
  }
  (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
  require_headline_settings(False)
  assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "requires a clean checkout"
  lock = Path.home() / ".scaly-measuring"
  with lock.open("x") as stream:
    stream.write(f"fatrop-stages pid:{os.getpid()} out:{args.out.resolve()}\n")
  try:
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for masses in (3, 5):
      for horizon in (10, 25):
        order = ["whole", "stage"]
        random.Random(masses + horizon).shuffle(order)
        pair = []
        for path in order:
          output = args.out / f"M{masses}_N{horizon}_{path}"
          cache = args.out.resolve() / "cache" / output.name
          cache.mkdir(parents=True)
          command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--path",
            path,
            "--masses",
            str(masses),
            "--horizon",
            str(horizon),
            "--out",
            str(output),
          ]
          with (args.out / f"{output.name}.log").open("w") as log:
            subprocess.run(command, env={**os.environ, "SCALY_CACHE_DIR": str(cache)}, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
          pair.append(json.loads((output / "result.json").read_text()))
        np.testing.assert_allclose(pair[0]["solution"], pair[1]["solution"], rtol=0.0, atol=1e-8)
        iterations = {sample["iter"] for result in pair for sample in result["samples"]}
        assert len(iterations) == 1, iterations
        for result in pair:
          summary = {"M": masses, "N": horizon, "path": result["path"], "iter": next(iter(iterations))}
          for field in ("t_fe", "t_total", "t_solver", "t_glue"):
            times = np.array([sample[field] * 1e3 for sample in result["samples"]])
            summary[field + "_median_ms"] = float(np.median(times))
            summary[field + "_std_ms"] = float(np.std(times, ddof=1))
          summary["counts"] = {field: result["samples"][0][field] for field in result["samples"][0] if field.startswith("n_eval")}
          results.append(summary)
          print(json.dumps(summary), flush=True)
    require_headline_settings(False)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
  finally:
    lock.unlink()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--path", choices=("whole", "stage"))
  parser.add_argument("--masses", type=int)
  parser.add_argument("--horizon", type=int)
  arguments = parser.parse_args()
  worker(arguments) if arguments.worker else run(arguments)
