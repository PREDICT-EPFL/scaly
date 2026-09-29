"""The authors' Python controller, from their code at the pinned commit, in this study's harnesses.

Runs under the upstream's pinned environment (PyTorch 2.5.1, CasADi 3.7.0; `setup.sh` builds it):

    . examples/case_studies/np_mpc/baseline/env.sh
    $NPMPC_PYTHON examples/case_studies/np_mpc/baseline/upstream.py lockstep --method neural --out run.json

Nothing of the controller is rewritten. `build` assembles it from their classes exactly as
`MPCController.__init__` does (`FurutaNPMPC`/`FurutaMPC`, `_build_optimization`, `configure_solver`),
without the runtime that would start their real-time simulator process; z comes from their encoder
through `FurutaTraining.estimate_z`, predictions from their `integrate`, the terminal weight from their
`compute_terminal_P` (torch autograd, SciPy's Riccati solver).

  lockstep  one episode of `protocol.lockstep`, optionally writing its OCP instances for the replays
  grid      the adaptation study: systems x context sizes x context seeds, plus the equation baseline
  replay    their Opti on an instance file, solve by solve
  export    their weights and MPC config YAMLs, which the C++ controllers read

`--casadi` picks CasADi's evaluation: `published` (their options: MX, interpreted), `expand` (SX,
interpreted) or `jit` (MX, generated C compiled by `cc -O3`, compile time reported separately).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
UPSTREAM = Path(os.environ.get("NPMPC_UPSTREAM", HERE / "third_party" / "neural_process_mpc"))
sys.path[:0] = [str(UPSTREAM / "src"), str(UPSTREAM / "scripts"), str(STUDY)]
os.environ.setdefault("MPLBACKEND", "Agg")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(var, "1")

import casadi as ca  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from npmpc.mpc.controller import MPCController  # noqa: E402
from npmpc.mpc.problem import PROBLEM_REGISTRY  # noqa: E402
from npmpc.mpc.solver import configure_solver  # noqa: E402
from npmpc.nps.NeuralProcess import NeuralProcess  # noqa: E402
from npmpc.training.furuta import FurutaTraining  # noqa: E402
from npmpc.utils import utils  # noqa: E402

import protocol as pr  # noqa: E402

MODEL_DIR = UPSTREAM / "model"
PUBLISHED = {"ipopt.max_iter": 50, "ipopt.tol": 1e-6, "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}  # configure_solver('ipopt')
VARIANTS = {"published": {}, "expand": {"expand": True}, "jit": {"jit": True, "compiler": "shell", "jit_options": {"flags": "-O3"}}}
CONTEXT_SIZES = (1, 20, 50, 100)


def load_model() -> NeuralProcess:
  utils.set_folder(str(MODEL_DIR))
  return NeuralProcess(json.loads((MODEL_DIR / "furuta_np.json").read_text()))


def mpc_params(method: str, *, z: np.ndarray | None = None, p: np.ndarray | None = None, steps: int = pr.SIM_STEPS) -> dict:
  """Their deploy config, `model/furuta_mpc.json`, for one method and model parameter."""
  params = json.loads((MODEL_DIR / "furuta_mpc.json").read_text())
  params["method"] = method
  params["experiment_options"]["sim_steps"] = steps
  if method == "neural":
    params["z"] = [float(v) for v in z]
  else:
    params["p"] = [[float(v) for v in p]]
  return params


def build(model: NeuralProcess | None, params: dict, casadi: str = "published") -> MPCController:
  """`MPCController.__init__` without its runtime (which would start their simulator)."""
  ctl = MPCController.__new__(MPCController)
  ctl.params = params
  cls = PROBLEM_REGISTRY[(params["system"], params["method"])]
  ctl.mpc = cls(model, params) if params["method"] == "neural" else cls(params)
  ctl.problem = ca.Opti()
  ctl._build_optimization()
  configure_solver(ctl.problem, params["solver"])
  if VARIANTS[casadi]:
    ctl.problem.solver("ipopt", {**PUBLISHED, **VARIANTS[casadi]})
  return ctl


def solver(ctl: MPCController):
  """One `problem.solve()` as `run_controller` makes it: parameter, primal guess, the previous duals."""
  opti = ctl.problem
  state = {"lam_g": None}

  def solve(x0: np.ndarray, x_guess: np.ndarray, u_guess: np.ndarray, s_guess: np.ndarray) -> dict:
    opti.set_value(ctl.x0, np.asarray(x0).reshape(1, -1))
    opti.set_initial(ctl.x, np.asarray(x_guess))
    opti.set_initial(ctl.u, np.asarray(u_guess).reshape(-1, 1))
    opti.set_initial(ctl.slack, np.asarray(s_guess).reshape(1, -1))
    if state["lam_g"] is not None:
      opti.set_initial(opti.lam_g, state["lam_g"])
    t0 = time.perf_counter()
    try:
      sol, ok = opti.solve(), True
    except RuntimeError:
      sol, ok = opti.debug, False
    wall = time.perf_counter() - t0
    stats = opti.stats()
    state["lam_g"] = sol.value(opti.lam_g)
    fe = sum(v for k, v in stats.items() if k.startswith("t_wall_nlp_"))
    return {
      "x": np.asarray(sol.value(ctl.x), np.float64),
      "u": np.asarray(sol.value(ctl.u), np.float64).reshape(-1),
      "slack": np.asarray(sol.value(ctl.slack), np.float64).reshape(-1),
      "converged": ok,
      "status": stats.get("return_status", ""),
      "iter": int(stats.get("iter_count", -1)),
      "objective": float(sol.value(opti.f)),
      "solve_s": wall,
      "t_total": float(stats.get("t_wall_total", float("nan"))),
      "t_fe": float(fe),
    }

  return solve


def rollout(ctl: MPCController):
  """Their runtime's `integrate_fn`: the controller's model, in float32 torch."""

  def run(x0: np.ndarray, u: np.ndarray) -> np.ndarray:
    with torch.no_grad():
      return ctl.mpc.integrate(x0=torch.Tensor(np.asarray(x0)), u=np.asarray(u, np.float64).reshape(-1, 1), dt=pr.DT).numpy().astype(np.float64)

  return run


def estimate_z(model: NeuralProcess, states: np.ndarray, inputs: np.ndarray, n: int) -> np.ndarray:
  """Their `FurutaTraining.estimate_z` on the first `n` steps of a context run."""
  traj = {"x": torch.tensor(states, dtype=torch.float32)[None], "u": torch.tensor(inputs, dtype=torch.float32)[:, None][None]}
  return FurutaTraining(model).estimate_z(traj, n).numpy().reshape(-1).astype(np.float64)


def episode(model: NeuralProcess, method: str, system: str, ctx: int, seed: int, *, steps: int, delay: bool, casadi: str) -> dict:
  p_plant = pr.SYSTEMS[system]
  z = None
  if method == "neural":
    inputs = pr.context_inputs(seed)
    z = estimate_z(model, pr.context_run(p_plant, inputs), inputs, ctx)
  params = mpc_params(method, z=z, p=p_plant, steps=steps)
  ctl, t_build = pr.timed(lambda: build(model, params, casadi))
  run = pr.lockstep(solver(ctl), rollout(ctl), p_plant, steps=steps, delay=delay)
  run.update(
    implementation=f"casadi-python-{casadi}",
    method=method,
    system=system,
    ctx=ctx,
    seed=seed,
    delay=delay,
    theta=(z if method == "neural" else p_plant).tolist(),
    terminal_P=params["cost"]["terminal_P"],
    build_s=t_build,
  )
  return run


def export_yamls(method: str, theta: np.ndarray, out_dir: Path) -> dict[str, str]:
  """Their `export_np_weights` and `export_mpc_config` (with the terminal weight their problem class computes)."""
  from export_mpc_config import export_mpc_config
  from export_np_weights import export_np_weights

  out_dir.mkdir(parents=True, exist_ok=True)
  weights = out_dir / "np_weights.yaml"
  config = out_dir / f"mpc_config_{method}.yaml"
  params = mpc_params(method, z=theta if method == "neural" else None, p=theta if method == "equation" else None)
  with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(params, f)
  with contextlib.redirect_stdout(io.StringIO()):
    export_np_weights(str(MODEL_DIR / "model.pth"), str(weights))
    export_mpc_config(str(MODEL_DIR / "furuta_np.json"), f.name, str(config), [float(v) for v in theta], method)
  os.unlink(f.name)
  return {"weights": str(weights), "mpc_config": str(config)}


def summary(run: dict) -> dict:
  """The per-episode numbers the grid keeps."""
  times = np.array([s["solve_s"] for s in run["steps"]])
  return {
    key: run[key]
    for key in ("implementation", "method", "system", "ctx", "seed", "delay", "theta", "cost", "settle_step", "settle_time", "max_abs_phi", "build_s")
  } | {
    "solve_ms_mean": float(1e3 * times.mean()),
    "solve_ms_median": float(1e3 * np.median(times)),
    "solve_ms_max": float(1e3 * times.max()),
    "iter_mean": float(np.mean([s["iter"] for s in run["steps"]])),
    "failures": int(sum(not s["converged"] for s in run["steps"])),
    "first_solve_s": float(run["warm_start"]["solve_s"]),
    "warm_start_iter": int(run["warm_start"]["iter"]),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  lk = sub.add_parser("lockstep")
  lk.add_argument("--method", choices=["neural", "equation"], default="neural")
  lk.add_argument("--system", choices=sorted(pr.SYSTEMS), default="sys3")
  lk.add_argument("--ctx", type=int, default=pr.CONTEXT_STEPS)
  lk.add_argument("--seed", type=int, default=0)
  lk.add_argument("--steps", type=int, default=pr.SIM_STEPS)
  lk.add_argument("--no-delay", action="store_true")
  lk.add_argument("--casadi", choices=sorted(VARIANTS), default="published")
  lk.add_argument("--instances", type=Path, help="also write the run's OCP instances here, with the YAMLs beside it")
  lk.add_argument("--out", type=Path, required=True)
  gr = sub.add_parser("grid")
  gr.add_argument("--seeds", type=int, default=5)
  gr.add_argument("--steps", type=int, default=pr.SIM_STEPS)
  gr.add_argument("--no-delay", action="store_true")
  gr.add_argument("--out", type=Path, required=True)
  rp = sub.add_parser("replay")
  rp.add_argument("--instances", type=Path, required=True)
  rp.add_argument("--casadi", choices=sorted(VARIANTS), default="published")
  rp.add_argument("--out", type=Path, required=True)
  ex = sub.add_parser("export")
  ex.add_argument("--method", choices=["neural", "equation"], required=True)
  ex.add_argument("--theta", type=float, nargs=4, required=True)
  ex.add_argument("--dir", type=Path, required=True)
  args = ap.parse_args()

  model = load_model()
  if args.cmd == "lockstep":
    run = episode(model, args.method, args.system, args.ctx, args.seed, steps=args.steps, delay=not args.no_delay, casadi=args.casadi)
    if args.instances:
      paths = export_yamls(args.method, np.array(run["theta"]), args.instances.parent)
      doc = {
        "method": args.method,
        "horizon": pr.HORIZON,
        "dt": pr.DT,
        **paths,
        "theta": run["theta"],
        "terminal_P": run["terminal_P"],
        "source": run["implementation"],
        "instances": pr.instances(run),
      }
      args.instances.write_text(json.dumps(doc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pr.jsonable(run if args.instances else pr.compact(run))))
  elif args.cmd == "grid":
    rows = []
    for system in sorted(pr.SYSTEMS):
      rows.append(summary(episode(model, "equation", system, 0, 0, steps=args.steps, delay=not args.no_delay, casadi="published")))
      for ctx in CONTEXT_SIZES:
        for seed in range(args.seeds):
          rows.append(summary(episode(model, "neural", system, ctx, seed, steps=args.steps, delay=not args.no_delay, casadi="published")))
          print(f"{system} ctx={ctx} seed={seed}: cost {rows[-1]['cost']:.1f}, settles {rows[-1]['settle_step']}", file=sys.stderr, flush=True)
    args.out.write_text(json.dumps(pr.jsonable({"implementation": "casadi-python-published", "rows": rows})))
  elif args.cmd == "replay":
    doc = json.loads(args.instances.read_text())
    theta = np.array(doc["theta"])
    params = mpc_params(doc["method"], z=theta, p=theta)
    ctl, t_build = pr.timed(lambda: build(model, params, args.casadi))
    solve = solver(ctl)
    steps = []
    for inst in doc["instances"]:
      out = solve(np.array(inst["x0"]), np.array(inst["x_guess"]), np.array(inst["u_guess"]).reshape(-1), np.zeros(pr.NX))
      steps.append(out)
    result = {"implementation": f"casadi-python-{args.casadi}", "method": doc["method"], "setup_ms": 1e3 * t_build, "steps": steps}
    args.out.write_text(json.dumps(pr.jsonable(result)))
  else:
    print(json.dumps(export_yamls(args.method, np.array(args.theta), args.dir)))


if __name__ == "__main__":
  main()
