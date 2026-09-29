"""Run the NP-MPC study: the authors' controllers and Scaly's on the same problems, in fresh processes.

    examples/case_studies/np_mpc/baseline/setup.sh
    uv run examples/case_studies/np_mpc/compare.py all

  reference  the authors' Python controller over one lockstep episode per method (system 3, the
             100-step context of seed 0), writing the OCP instances its solves started from
  replay     every implementation solves those instances from the recorded guesses, `--repeats` fresh
             processes each, the fastest solve per instance kept
  lockstep   whole episodes, the authors' controller against Scaly's, with and without their delay
  grid       the adaptation study, the authors' controller and Scaly's, with and without the delay
  realtime   the authors' real-time setup: their server and simulator, their clients and Scaly's
  all        all of the above, in that order

Everything lands in `results/`; the instances and the YAMLs the C++ controllers read in
`baseline/third_party/work/`. Each timed process first waits for a quiet machine (`_common.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from _common import wait_for_quiet  # noqa: E402

import protocol as pr  # noqa: E402

BASELINE = HERE / "baseline"
TP = BASELINE / "third_party"
VENV_PY = TP / "venv" / "bin" / "python"
BUILD = TP / "build"
UPSTREAM = TP / "neural_process_mpc"
WORK = TP / "work"
RESULTS = HERE / "results"
METHODS = ("neural", "equation")
ENV = {
  **os.environ,
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "MKL_NUM_THREADS": "1",
  "MPLBACKEND": "Agg",
  "NPMPC_UPSTREAM": str(UPSTREAM),
}


def casadi_ipopt_dir() -> Path:
  """A directory whose `libipopt.dylib` is CasADi's wheel IPOPT, for running Scaly's generated solver
  against that build through the loader (the generated wrapper links `@rpath/libipopt.dylib`)."""
  site = Path(subprocess.check_output([str(VENV_PY), "-c", "import casadi, os; print(os.path.dirname(casadi.__file__))"], text=True).strip())
  swap = WORK / "casadi_ipopt"
  swap.mkdir(parents=True, exist_ok=True)
  for lib in site.glob("lib*.dylib"):
    target = swap / ("libipopt.dylib" if lib.name == "libipopt.3.dylib" else lib.name)
    if not target.exists():
      target.symlink_to(lib)
  return swap


def upstream_cmd(*args: str) -> list[str]:
  return [str(VENV_PY), str(BASELINE / "upstream.py"), *args]


def scaly_cmd(*args: str) -> list[str]:
  return [sys.executable, str(HERE / "run_scaly.py"), *args]


def replay_cmd(impl: str, method: str, inst: Path, out: Path) -> tuple[list[str], dict]:
  io = ["--instances", str(inst), "--out", str(out)]
  env: dict = {}
  if impl.startswith("casadi-python-"):
    cmd = upstream_cmd("replay", "--casadi", impl.removeprefix("casadi-python-"), *io)
  elif impl.startswith("casadi-cpp-"):
    variant = impl.removeprefix("casadi-cpp-")
    cmd = [str(BUILD / "replay_casadi"), "--method", method, *io] + ([f"--{variant}"] if variant != "published" else [])
  elif impl.startswith("laopt-"):
    solver = "sqp" if "sqp" in impl else "ipopt"
    binary = "replay_laopt_scaly_ipopt" if impl.endswith("scalybuild") else "replay_laopt"
    cmd = [str(BUILD / binary), "--solver", solver, "--method", method, *io]
  else:
    _, form, solver, *build = impl.split("-")
    cmd = scaly_cmd("replay", "--form", form, "--solver", solver, *io)
    if build == ["casadibuild"]:
      env["DYLD_LIBRARY_PATH"] = str(casadi_ipopt_dir())
  return cmd, env


REPLAYS = (
  "casadi-python-published",
  "casadi-python-expand",
  "casadi-python-jit",
  "casadi-cpp-published",
  "casadi-cpp-expand",
  "casadi-cpp-jit",
  "laopt-ipopt",
  "laopt-ipopt-scalybuild",
  "laopt-sqp",
  "scaly-opti-ipopt",
  "scaly-opti-ipopt-casadibuild",
  "scaly-laopt-ipopt",
  "scaly-laopt-sqp",
)


def run(cmd: list[str], env: dict | None = None, timeout: float = 3600.0) -> tuple[bool, str, float]:
  load = wait_for_quiet()
  proc = subprocess.run(cmd, env={**ENV, **(env or {})}, capture_output=True, text=True, timeout=timeout)
  return proc.returncode == 0, proc.stderr[-2000:], load


def solve_ms(step: dict) -> float:
  return float(step["solve_ms"]) if "solve_ms" in step else 1e3 * float(step["solve_s"])


def reference(args: argparse.Namespace) -> None:
  """The authors' controller over one episode per method, with its instances."""
  WORK.mkdir(parents=True, exist_ok=True)
  RESULTS.mkdir(parents=True, exist_ok=True)
  for method in METHODS:
    out = RESULTS / f"lockstep_casadi-python-published_{method}.json"
    ok, err, _ = run(upstream_cmd("lockstep", "--method", method, "--instances", str(WORK / f"instances_{method}.json"), "--out", str(out)))
    if not ok:
      raise SystemExit(f"reference {method} failed:\n{err}")
    print(f"reference {method}: {out.name}", flush=True)


def replay(args: argparse.Namespace) -> None:
  """Every implementation over the reference instances; the fastest of `--repeats` processes per solve."""
  results_path = RESULTS / "replay.json"
  results = json.loads(results_path.read_text()) if results_path.exists() else {}
  for method in METHODS:
    inst = WORK / f"instances_{method}.json"
    ref = json.loads((RESULTS / f"lockstep_casadi-python-published_{method}.json").read_text())
    ref_u, ref_it = np.array(ref["plan_u"]), np.array([s["iter"] for s in ref["steps"]])
    for impl in args.only or REPLAYS:
      key = f"{impl}:{method}"
      if key in results and not args.force:
        continue
      runs, loads, failed = [], [], None
      for r in range(args.repeats):
        out = WORK / f"replay_{impl}_{method}_{r}.json"
        cmd, env = replay_cmd(impl, method, inst, out)
        if not Path(cmd[0]).exists() and not cmd[0].endswith("python"):
          failed = f"missing {cmd[0]}"
          break
        ok, err, load = run(cmd, env)
        if not ok:
          failed = err
          break
        runs.append(json.loads(out.read_text()))
        loads.append(load)
      if failed:
        results[key] = {"implementation": impl, "method": method, "failed": failed}
        print(f"{key}: FAILED {failed[-300:]}", flush=True)
        results_path.write_text(json.dumps(results))
        continue
      times = np.array([[solve_ms(s) for s in run_["steps"]] for run_ in runs])
      best = times.min(axis=0)
      first = runs[0]["steps"]
      u = np.array([np.asarray(s["u"], np.float64).reshape(-1) for s in first])
      it = np.array([int(s["iter"]) for s in first])
      c_ms = (
        np.array([[1e3 * s["t_total"] for s in run_["steps"]] for run_ in runs]).min(axis=0)
        if "t_total" in first[0] and impl.startswith("scaly")
        else None
      )
      results[key] = {
        "implementation": impl,
        "method": method,
        "repeats": len(runs),
        "loads": loads,
        "ipopt": runs[0].get("ipopt"),
        "setup_ms": [run_.get("setup_ms") for run_ in runs],
        "first_solve_ms": float(times[:, 0].min()),
        "solve_ms": best[1:].tolist(),  # instance 0 is left out: CasADi builds its solver in the first solve
        "c_ms": None if c_ms is None else c_ms[1:].tolist(),
        "iter": it.tolist(),
        "qp_iter": [int(s.get("qp_iter", 0)) for s in first],
        "converged": [bool(s["converged"]) for s in first],
        "max_du_vs_reference": float(np.abs(u - ref_u).max()),
        "du_vs_reference": np.abs(u - ref_u).max(axis=1).tolist(),
        "iter_equal_reference": int((it == ref_it).sum()),
        "iter_consistent_across_repeats": all([int(s["iter"]) for s in run_["steps"]] == it.tolist() for run_ in runs),
      }
      row = results[key]
      print(
        f"{key}: median {np.median(row['solve_ms']):.2f} ms, mean {np.mean(row['solve_ms']):.2f}, iter {np.mean(it):.2f} "
        f"({row['iter_equal_reference']}/100 as the reference), converged {sum(row['converged'])}/100, max|du| {row['max_du_vs_reference']:.1e}",
        flush=True,
      )
      results_path.write_text(json.dumps(results))


def lockstep(args: argparse.Namespace) -> None:
  """Whole episodes: the authors' controller (CasADi as published) and Scaly's (both forms, both
  methods of Scaly's), with the delay and without."""
  for delay in (True, False):
    flag = [] if delay else ["--no-delay"]
    tag = "delay" if delay else "nodelay"
    for method in METHODS:
      jobs = [] if delay else [("casadi-python-published", upstream_cmd("lockstep", "--method", method, *flag))]  # delayed: the reference
      jobs += [
        (f"scaly-{form}-{solver}", scaly_cmd("lockstep", "--method", method, "--form", form, "--solver", solver, *flag))
        for form, solver in (("opti", "ipopt"), ("laopt", "ipopt"), ("laopt", "sqp"))
      ]
      for impl, cmd in jobs:
        out = RESULTS / f"lockstep_{impl}_{method}_{tag}.json"
        if out.exists() and not args.force:
          continue
        ok, err, _ = run([*cmd, "--out", str(out)])
        print(f"lockstep {impl} {method} {tag}: {'ok' if ok else 'FAILED ' + err[-300:]}", flush=True)


def grid(args: argparse.Namespace) -> None:
  for delay in (True, False):
    flag = [] if delay else ["--no-delay"]
    tag = "delay" if delay else "nodelay"
    for impl, cmd in (("casadi-python-published", upstream_cmd("grid", *flag)), ("scaly-opti-ipopt", scaly_cmd("grid", *flag))):
      out = RESULTS / f"grid_{impl}_{tag}.json"
      if out.exists() and not args.force:
        continue
      ok, err, _ = run([*cmd, "--out", str(out)])
      print(f"grid {impl} {tag}: {'ok' if ok else 'FAILED ' + err[-300:]}", flush=True)


def realtime(args: argparse.Namespace) -> None:
  """The authors' real-time setup: their `run_realtime.py` for their own agents, then Scaly's client
  against the same server."""
  out = RESULTS / "realtime_upstream.json"
  if not out.exists() or args.force:
    ok, err, _ = run([str(VENV_PY), str(BASELINE / "run_realtime.py"), "--repeats", str(args.repeats), "--out", str(out)], timeout=7200)
    print(f"realtime upstream: {'ok' if ok else 'FAILED ' + err[-500:]}", flush=True)
    if ok:
      embed_trajectories(out)
  rows = []
  for method in METHODS:
    for form, solver, wait in (("laopt", "ipopt", 0.0), ("laopt", "sqp", 0.0), ("laopt", "sqp", 4.3), ("opti", "ipopt", 0.0)):
      impl = f"scaly-{form}-{solver}" + (f"-wait{wait:g}ms" if wait else "")
      for r in range(args.repeats):
        dump = f"{impl}_{method}_{r}".replace("-", "_").replace(".", "p")
        client_out = WORK / f"realtime_{dump}.json"
        config = UPSTREAM / "model" / "furuta_mpc.json"
        server = subprocess.Popen(
          [str(VENV_PY), "scripts/run_qube_server.py", "--no-plot", "--dump", dump, "--port", "56211", "--mpc-config", str(config)],
          cwd=UPSTREAM,
          env=ENV,
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
        wait_for_quiet()
        client = subprocess.run(
          [*scaly_cmd_rt(method, form, solver, client_out), "--min-solve-ms", str(wait)],
          env=ENV,
          capture_output=True,
          text=True,
          timeout=600,
        )
        server.wait(timeout=60)
        if client.returncode:
          rows.append({"implementation": impl, "method": method, "failed": client.stderr[-500:]})
          continue
        log = np.load(UPSTREAM / "logs" / f"{dump}.npz")
        summary = json.loads(client_out.read_text())
        rows.append(
          {**{k: v for k, v in summary.items() if k != "steps"}, **realtime_metrics(log), "implementation": impl, "repeat": r, "dump": dump}
        )
        print(
          f"realtime {impl} {method} #{r}: cost {rows[-1]['cost']:.1f}, settles {rows[-1]['settle_time']}, {summary['solves']} solves, mean {summary['solve_ms_mean']:.2f} ms",
          flush=True,
        )
  (RESULTS / "realtime_scaly.json").write_text(json.dumps(pr.jsonable(rows)))


def scaly_cmd_rt(method: str, form: str, solver: str, out: Path) -> list[str]:
  return [
    sys.executable,
    str(HERE / "realtime_scaly.py"),
    "--method",
    method,
    "--form",
    form,
    "--solver",
    solver,
    "--port",
    "56211",
    "--out",
    str(out),
  ]


def embed_trajectories(path: Path) -> None:
  """Copy each real-time run's states, sampled at the controller's period, from its log in
  `third_party/` into the results file, so the notebook needs no baseline tree."""
  doc = json.loads(path.read_text())
  for agent in doc["agents"].values():
    for r in agent["runs"]:
      log = np.load(r["trajectory"], allow_pickle=True)
      if "x0" in log.files:  # the Python deploy's own dump: one measured state per MPC step
        n = int(log["k"]) + 1
        r["states"], r["inputs"] = log["x0"][:n].tolist(), log["u"][:n, 0, 0].tolist()
      else:
        m = realtime_metrics(log)
        r["states"], r["inputs"] = m["states"], m["inputs"]
  path.write_text(json.dumps(doc))


def realtime_metrics(log: np.lib.npyio.NpzFile) -> dict:
  """Their server's log resampled at the controller's period, scored as `compute_metrics` scores their runs."""
  t, x, u = log["t"] - log["t"][0], log["x"], log["u"]
  idx = np.searchsorted(t, np.arange(0.0, t[-1] + 1e-9, pr.DT))
  idx = idx[idx < len(t)]
  states, inputs = x[idx], u[idx]
  settle = pr.settle_step(states)
  return {
    "cost": pr.realized_cost(states, inputs),
    "settle_step": settle,
    "settle_time": None if settle is None else settle * pr.DT,
    "states": states.tolist(),
    "inputs": inputs.tolist(),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("stage", choices=["reference", "replay", "lockstep", "grid", "realtime", "all"])
  ap.add_argument("--repeats", type=int, default=3)
  ap.add_argument("--only", nargs="*", help="replay implementations to run")
  ap.add_argument("--force", action="store_true", help="rerun what results/ already has")
  args = ap.parse_args()
  if not VENV_PY.exists():
    raise SystemExit(f"run {BASELINE / 'setup.sh'} first")
  RESULTS.mkdir(exist_ok=True)
  stages = ["reference", "replay", "lockstep", "grid", "realtime"] if args.stage == "all" else [args.stage]
  for stage in stages:
    t0 = time.perf_counter()
    {"reference": reference, "replay": replay, "lockstep": lockstep, "grid": grid, "realtime": realtime}[stage](args)
    print(f"== {stage}: {time.perf_counter() - t0:.0f} s", flush=True)
  shutil.rmtree(WORK / "casadi_ipopt", ignore_errors=True)


if __name__ == "__main__":
  main()
