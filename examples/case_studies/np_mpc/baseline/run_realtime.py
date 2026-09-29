"""The upstream's real-time closed-loop runs, as published, on its simulated Furuta pendulum.

Run with the upstream's Python after setup.sh (it needs the venv's torch and casadi):

  examples/case_studies/np_mpc/baseline/third_party/venv/bin/python \\
    examples/case_studies/np_mpc/baseline/run_realtime.py --repeats 3 --out realtime.json

Agents (select with --agents, comma-separated; all by default):
  cpp-{np,eq}-laopt-sqp          run_furuta_{np,eq}_laopt_client as published: laOPT SQP with PIQP,
                                 answering no earlier than kMinSolveMs = 4.3 ms after a solve starts
  cpp-{np,eq}-laopt-ipopt        the same client with kSolver = SolverType::IPOPT, the only change (a
                                 copy of its source made by CMakeLists.txt), against CasADi's IPOPT
  cpp-{np,eq}-laopt-ipopt-scaly  that IPOPT client against Scaly's vendored IPOPT 3.14.19
  cpp-{np,eq}-casadi             run_furuta_{np,eq}_casadi_client (Opti + IPOPT, the wheel's)
  py-np-casadi                   scripts/deploy_cnp.sh: PYTHONPATH=src python -m npmpc.main --system furuta
                                 --folder model/ --deploy furuta_mpc.json --model furuta_np.json, which
                                 first runs a 2 s context experiment to estimate z, then the MPC
  py-eq-casadi                   the same with a copy of furuta_mpc.json whose "method" is "equation"

Each C++ agent runs against scripts/run_qube_server.py --no-plot --dump <name> started from the clone,
with its default MPC config (model/mpc_config.yaml, model/mpc_config_equation.yaml as shipped); the
server's .npz log (500 Hz states, the client's reported solve times) is moved to --runs-dir. The
Python deploy runs inside a thin wrapper that times each MPC Opti.solve() call with perf_counter, as
run_controller's own solve_times do (it only logs their mean); its .npz dump is copied likewise.

Closed-loop cost and settling time follow FurutaRuntime.compute_metrics (src/npmpc/mpc/runtime/
furuta.py): the realized stage + interstage cost over consecutive states at the controller's dt, and
the first step after which theta stays within 10 deg of upright. The Python agent logs one state per
MPC step (paced to dt); the C++ server log is sampled every dt from its first row, taking the torque
in effect at each sample. Solve-time statistics leave out the first solve, which the laOPT clients
repeat until converged and the Python agent makes in warm_start before its loop; it is reported as
first_solve_ms.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TP = HERE / "third_party"
UPSTREAM = Path(os.environ.get("NPMPC_UPSTREAM", TP / "neural_process_mpc"))
BUILD = Path(os.environ.get("NPMPC_BUILD", TP / "build"))
WORK = Path(os.environ.get("NPMPC_WORK", TP / "work"))

CPP_AGENTS = {
  "cpp-np-laopt-sqp": "run_furuta_np_laopt_client",
  "cpp-np-laopt-ipopt": "run_furuta_np_laopt_ipopt_client",
  "cpp-np-laopt-ipopt-scaly": "run_furuta_np_laopt_ipopt_client_scaly_ipopt",
  "cpp-np-casadi": "run_furuta_np_casadi_client",
  "cpp-eq-laopt-sqp": "run_furuta_eq_laopt_client",
  "cpp-eq-laopt-ipopt": "run_furuta_eq_laopt_ipopt_client",
  "cpp-eq-laopt-ipopt-scaly": "run_furuta_eq_laopt_ipopt_client_scaly_ipopt",
  "cpp-eq-casadi": "run_furuta_eq_casadi_client",
}
PY_AGENTS = {"py-np-casadi": "neural", "py-eq-casadi": "equation"}

# Runs `python -m npmpc.main <args>` with every Opti.solve() made by MPCController (warm_start and
# run_controller) timed; the records are written to argv[1] at exit.
PY_WRAPPER = r"""
import atexit, json, runpy, sys, time
import casadi
import npmpc.mpc.controller as controller

out, args = sys.argv[1], sys.argv[2:]
records, phase = [], [None]
solve = casadi.Opti.solve

def timed_solve(self, *a):
  t0 = time.perf_counter()
  ok = False
  try:
    sol = solve(self, *a)
    ok = True
    return sol
  finally:
    ms = (time.perf_counter() - t0) * 1e3
    if phase[0] is not None:
      try:
        stats = self.stats()
        it, status = int(stats.get('iter_count', -1)), str(stats.get('return_status', ''))
      except Exception:
        it, status = -1, ''
      records.append({'phase': phase[0], 'solve_ms': ms, 'converged': ok, 'iter': it, 'status': status})

def in_phase(name, fn):
  def wrapped(self, *a, **k):
    previous, phase[0] = phase[0], name
    try:
      return fn(self, *a, **k)
    finally:
      phase[0] = previous
  return wrapped

casadi.Opti.solve = timed_solve
controller.MPCController.warm_start = in_phase('warm_start', controller.MPCController.warm_start)
controller.MPCController.run_controller = in_phase('run', controller.MPCController.run_controller)
atexit.register(lambda: open(out, 'w').write(json.dumps(records)))
sys.argv = ['npmpc.main'] + args
runpy.run_module('npmpc.main', run_name='__main__', alter_sys=True)
"""


def cost_weights():
  return json.loads((UPSTREAM / "model/furuta_mpc.json").read_text())["cost"]


def closed_loop_metrics(x, u, dt, cost, theta_tol_deg=10.0):
  """FurutaRuntime.compute_metrics on states x (n, 4) spaced dt apart and the inputs u (n,) applied
  from each of them."""
  n = len(x)
  total = 0.0
  for k in range(n - 1):
    xk, dx = x[k], x[k + 1] - x[k]
    total += (
      cost["x"][0] * 2.0 * (1.0 - np.cos(xk[0]))
      + cost["x"][1] * xk[1] ** 2
      + cost["x"][2] * xk[2] ** 2
      + cost["x"][3] * xk[3] ** 2
      + cost["u"][0] * u[k] ** 2
      + sum(cost["x_diff"][i] * dx[i] ** 2 for i in range(4))
    )
  theta_err = np.abs((x[:, 0] + np.pi) % (2.0 * np.pi) - np.pi)
  within = theta_err < np.deg2rad(theta_tol_deg)
  if n == 0 or not within.any() or not within[-1]:
    settle_step = None
  else:
    violations = np.where(~within)[0]
    settle_step = int(violations[-1] + 1) if violations.size else 0
  return {"cost": float(total), "settle_step": settle_step, "settling_time": settle_step * dt if settle_step is not None else None, "n_steps": n}


def solve_stats(ms):
  ms = np.asarray(ms, float)
  if ms.size == 0:
    return None
  return {"mean": float(ms.mean()), "median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)), "max": float(ms.max())}


def wait_for_line(proc, pattern, timeout):
  """Reads proc's stdout until a line matches; returns the lines read."""
  lines, deadline = [], time.monotonic() + timeout
  while time.monotonic() < deadline:
    line = proc.stdout.readline()
    if not line:
      if proc.poll() is not None:
        break
      continue
    lines.append(line)
    if re.search(pattern, line):
      return lines
  raise RuntimeError("server did not start:\n" + "".join(lines))


def run_cpp(agent, repeat, runs_dir, env):
  binary = BUILD / CPP_AGENTS[agent]
  name = f"np_mpc_realtime_{agent}_r{repeat}"
  server = subprocess.Popen(
    [sys.executable, "-u", "scripts/run_qube_server.py", "--no-plot", "--dump", name],
    cwd=UPSTREAM,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )
  try:
    server_out = wait_for_line(server, r"listening on", 60)
    client = subprocess.run([str(binary)], cwd=UPSTREAM, env=env, capture_output=True, text=True, timeout=300)
    rest, _ = server.communicate(timeout=60)
    server_out += rest.splitlines(keepends=True)
  finally:
    if server.poll() is None:
      server.kill()
  if client.returncode != 0:
    raise RuntimeError(f"{agent}: client exited {client.returncode}:\n{client.stderr}")
  dump = UPSTREAM / "logs" / f"{name}.npz"
  trajectory = runs_dir / dump.name
  shutil.move(dump, trajectory)
  (runs_dir / f"{name}.client.log").write_text(client.stdout + client.stderr)
  (runs_dir / f"{name}.server.log").write_text("".join(server_out))

  log = np.load(trajectory)
  t, x, u, dt = log["t"], log["x"], log["u"], float(log["dt"])
  samples = np.searchsorted(t, t[0] + dt * np.arange(int((t[-1] - t[0]) / dt + 1e-9) + 1))
  metrics = closed_loop_metrics(x[samples], u[samples], dt, cost_weights())

  out = client.stdout
  solve_ms = log["solve_ms"]
  if "laopt" in agent:
    summary = re.search(r"^Solves: (\d+) \((\d+) converged\)", out, re.M)
    n, n_converged = (int(summary.group(1)), int(summary.group(2))) if summary else (0, 0)
    initial = re.search(r"^Initial solve: (\d+) solve\(\) calls", out, re.M)
    extra = {"initial_solve_calls": int(initial.group(1)) if initial else None}
  else:
    steps = re.findall(r"^step \d+ \|.*? ms \| (converged|NOT converged) \(", out, re.M)
    n, n_converged = len(steps), steps.count("converged")
    prepare = re.search(r"^Warm-up solve from the config x0: ([\d.]+) ms", out, re.M)
    extra = {"prepare_ms": float(prepare.group(1)) if prepare else None}
  if len(solve_ms) != n:
    raise RuntimeError(f"{agent}: {len(solve_ms)} solve times in the log, {n} solves in the client output")
  return {
    **metrics,
    "n_solves": int(len(solve_ms)),
    "converged_fraction": n_converged / n if n else None,
    "first_solve_ms": float(solve_ms[0]) if len(solve_ms) else None,
    "solve_ms": solve_stats(solve_ms[1:]),
    "trajectory": str(trajectory),
    "sampling": f"server log (every {float(log['state_period']) * 1e3:g} ms) sampled every dt",
    **extra,
  }


def run_python(agent, repeat, runs_dir, env):
  method = PY_AGENTS[agent]
  deploy = "furuta_mpc.json"
  config = json.loads((UPSTREAM / "model" / deploy).read_text())
  if method != config["method"]:
    config["method"] = method
    deploy = WORK / f"furuta_mpc_{method}.json"
    deploy.write_text(json.dumps(config, indent=4))
  name = f"np_mpc_realtime_{agent}_r{repeat}"
  timing = runs_dir / f"{name}.solves.json"
  proc = subprocess.run(
    [sys.executable, "-c", PY_WRAPPER, str(timing), "--system", "furuta", "--folder", "model/", "--deploy", str(deploy), "--model", "furuta_np.json"],
    cwd=UPSTREAM,
    env={**env, "PYTHONPATH": "src"},
    capture_output=True,
    text=True,
    timeout=600,
  )
  (runs_dir / f"{name}.log").write_text(proc.stdout + proc.stderr)
  if proc.returncode != 0:
    raise RuntimeError(f"{agent}: exited {proc.returncode}:\n{proc.stderr[-4000:]}")
  dump = UPSTREAM / "logs" / f"{config['save_path']}.npz"
  trajectory = runs_dir / f"{name}.npz"
  shutil.copy(dump, trajectory)

  log = np.load(trajectory, allow_pickle=True)
  k, dt = int(log["k"]), float(log["dt"])
  metrics = closed_loop_metrics(log["x0"][: k + 1], log["u"][: k + 1, 0, 0], dt, config["cost"])
  upstream = log["metrics"].item() if "metrics" in log else {}
  records = json.loads(timing.read_text())
  run = [r for r in records if r["phase"] == "run"]
  first = [r for r in records if r["phase"] == "warm_start"]
  z = re.search(r"Estimated z: (\[.*?\])", proc.stdout + proc.stderr)
  return {
    **metrics,
    "n_solves": len(run),
    "converged_fraction": float(np.mean([r["converged"] for r in run])) if run else None,
    "first_solve_ms": float(sum(r["solve_ms"] for r in first)),
    "warm_start_solve_calls": len(first),
    "solve_ms": solve_stats([r["solve_ms"] for r in run]),
    "iter_mean": float(np.mean([r["iter"] for r in run])) if run else None,
    "step_ms": solve_stats(1e3 * log["mpc_t"][: k + 1]),
    "upstream_metrics": {"cost": upstream.get("cost"), "settling_time": upstream.get("convergence_time")},
    "estimated_z": json.loads(z.group(1)) if z else None,
    "deploy_config": str(UPSTREAM / "model" / deploy) if isinstance(deploy, str) else str(deploy),
    "trajectory": str(trajectory),
    "sampling": "one state per MPC step",
  }


def summarize(runs):
  def median(key, sub=None):
    vals = [(r[key][sub] if sub else r[key]) for r in runs if r.get(key) is not None]
    return float(np.median(vals)) if vals else None

  return {
    "cost": median("cost"),
    "settling_time": median("settling_time"),
    "solve_ms_mean": median("solve_ms", "mean"),
    "solve_ms_median": median("solve_ms", "median"),
    "solve_ms_p95": median("solve_ms", "p95"),
    "solve_ms_max": median("solve_ms", "max"),
    "converged_fraction": median("converged_fraction"),
    "n_solves": median("n_solves"),
    "settled_runs": sum(r["settling_time"] is not None for r in runs),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--agents", default=",".join([*CPP_AGENTS, *PY_AGENTS]))
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--out", type=Path, default=WORK / "realtime.json")
  parser.add_argument("--runs-dir", type=Path, default=WORK / "realtime")
  args = parser.parse_args()
  agents = [a.strip() for a in args.agents.split(",") if a.strip()]
  unknown = [a for a in agents if a not in CPP_AGENTS and a not in PY_AGENTS]
  if unknown:
    parser.error(f"unknown agents {unknown}; known: {[*CPP_AGENTS, *PY_AGENTS]}")
  args.out, args.runs_dir = args.out.resolve(), args.runs_dir.resolve()
  args.runs_dir.mkdir(parents=True, exist_ok=True)
  env = {**os.environ, "MPLBACKEND": "Agg"}

  results = {"upstream": str(UPSTREAM), "build": str(BUILD), "repeats": args.repeats, "agents": {}}
  for agent in agents:
    runs = []
    for repeat in range(args.repeats):
      run = run_cpp(agent, repeat, args.runs_dir, env) if agent in CPP_AGENTS else run_python(agent, repeat, args.runs_dir, env)
      runs.append(run)
      solve = run["solve_ms"] or {}
      print(
        f"{agent} #{repeat}: cost {run['cost']:.2f}, settled {run['settling_time']}, {run['n_solves']} solves, "
        f"converged {run['converged_fraction']}, solve mean {solve.get('mean', float('nan')):.2f} ms "
        f"p95 {solve.get('p95', float('nan')):.2f} ms",
        flush=True,
      )
    results["agents"][agent] = {"summary": summarize(runs), "runs": runs}
    args.out.write_text(json.dumps(results, indent=1))
  print(f"Wrote {args.out}")


if __name__ == "__main__":
  main()
