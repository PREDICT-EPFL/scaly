"""The Scaly side of the NP-MPC study, in the same harnesses as `baseline/upstream.py`.

  uv run examples/case_studies/np_mpc/run_scaly.py lockstep --method neural --out run.json
  uv run examples/case_studies/np_mpc/run_scaly.py replay --instances inst.json --form opti --solver ipopt --out o.json

lockstep  one episode of `protocol.lockstep`: z from Scaly's encoder on the context run, the terminal
          weight from Scaly's linearization and SciPy's Riccati solver, predictions from Scaly's step
grid      the adaptation study, systems x context sizes x context seeds plus the equation baseline,
          every neural episode on one compiled solver
replay    solve an instance file (written by `upstream.py lockstep --instances`) with its z or p and
          its terminal weight, so the NLP is the upstream's to the last bit of P
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as pr  # noqa: E402
import scaly_impl as si  # noqa: E402

CONTEXT_SIZES = (1, 20, 50, 100)


class Study:
  """The model Functions and generated solvers, built once and shared by every episode."""

  def __init__(self) -> None:
    self.cnp = si.load_cnp()
    self.models = {kind: si.Model(kind, self.cnp) for kind in ("neural", "equation")}
    self.encoder = si.encoder_function(self.cnp)
    self.controllers: dict[tuple[str, str, str], si.Controller] = {}
    self.compile_s: dict[str, float] = {}

  def controller(self, method: str, form: str, solver: str) -> si.Controller:
    key = (method, form, solver)
    if key not in self.controllers:
      c = si.Controller(self.models[method], form, solver)
      self.compile_s["_".join(key)] = c.compile(np.zeros(4) if method == "neural" else pr.SYSTEMS["sys3"], np.eye(pr.NX))
      self.controllers[key] = c
    return self.controllers[key]

  def latent(self, system: str, ctx: int, seed: int) -> np.ndarray:
    inputs = pr.context_inputs(seed)
    x_ctx, y_ctx = pr.context_pairs(pr.context_run(pr.SYSTEMS[system], inputs), inputs, ctx)
    return si.encode(self.encoder, x_ctx, y_ctx)

  def episode(self, method: str, system: str, ctx: int, seed: int, *, form: str, solver: str, steps: int, delay: bool) -> dict:
    model = self.models[method]
    theta = self.latent(system, ctx, seed) if method == "neural" else pr.SYSTEMS[system]
    P, t_weight = pr.timed(lambda: model.terminal_weight(theta))
    c = self.controller(method, form, solver)
    c.reset()
    run = pr.lockstep(
      lambda x0, xg, ug, sg: c.solve(x0, xg, ug, sg, theta, P),
      lambda x0, u: model.rollout(theta, x0, u),
      pr.SYSTEMS[system],
      steps=steps,
      delay=delay,
    )
    run.update(
      implementation=f"scaly-{form}-{solver}",
      method=method,
      system=system,
      ctx=ctx,
      seed=seed,
      delay=delay,
      theta=np.asarray(theta).tolist(),
      terminal_P=P.tolist(),
      build_s=t_weight,  # per latent code only the terminal weight is new; the solver is compiled once
      compile_s=self.compile_s["_".join((method, form, solver))],
    )
    return run


def summary(run: dict) -> dict:
  times = np.array([s["solve_s"] for s in run["steps"]])
  c_times = np.array([s["t_total"] for s in run["steps"]])
  return {
    key: run[key]
    for key in ("implementation", "method", "system", "ctx", "seed", "delay", "theta", "cost", "settle_step", "settle_time", "max_abs_phi", "build_s")
  } | {
    "solve_ms_mean": float(1e3 * times.mean()),
    "solve_ms_median": float(1e3 * np.median(times)),
    "solve_ms_max": float(1e3 * times.max()),
    "c_ms_mean": float(1e3 * c_times.mean()),
    "iter_mean": float(np.mean([s["iter"] for s in run["steps"]])),
    "failures": int(sum(not s["converged"] for s in run["steps"])),
    "first_solve_s": float(run["warm_start"]["solve_s"]),
    "warm_start_iter": int(run["warm_start"]["iter"]),
  }


def replay(study: Study, doc: dict, form: str, solver: str) -> dict:
  theta, P = np.array(doc["theta"]), np.array(doc["terminal_P"])
  c = study.controller(doc["method"], form, solver)
  steps = []
  for inst in doc["instances"]:
    c.reset()  # each instance depends on its own guess only, as the C++ replays zero laOPT's multipliers
    steps.append(c.solve(np.array(inst["x0"]), np.array(inst["x_guess"]), np.array(inst["u_guess"]).reshape(-1), np.zeros(pr.NX), theta, P))
  return {
    "implementation": f"scaly-{form}-{solver}",
    "method": doc["method"],
    "setup_ms": 1e3 * study.compile_s["_".join((doc["method"], form, solver))],
    "steps": steps,
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
  lk.add_argument("--form", choices=si.FORMS, default="opti")
  lk.add_argument("--solver", choices=["ipopt", "sqp"], default="ipopt")
  lk.add_argument("--out", type=Path, required=True)
  gr = sub.add_parser("grid")
  gr.add_argument("--seeds", type=int, default=5)
  gr.add_argument("--steps", type=int, default=pr.SIM_STEPS)
  gr.add_argument("--no-delay", action="store_true")
  gr.add_argument("--form", choices=si.FORMS, default="opti")
  gr.add_argument("--solver", choices=["ipopt", "sqp"], default="ipopt")
  gr.add_argument("--out", type=Path, required=True)
  rp = sub.add_parser("replay")
  rp.add_argument("--instances", type=Path, required=True)
  rp.add_argument("--form", choices=si.FORMS, default="opti")
  rp.add_argument("--solver", choices=["ipopt", "sqp"], default="ipopt")
  rp.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  study = Study()
  if args.cmd == "lockstep":
    run = study.episode(args.method, args.system, args.ctx, args.seed, form=args.form, solver=args.solver, steps=args.steps, delay=not args.no_delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pr.jsonable(pr.compact(run))))
  elif args.cmd == "grid":
    rows = []
    kw = {"form": args.form, "solver": args.solver, "steps": args.steps, "delay": not args.no_delay}
    for system in sorted(pr.SYSTEMS):
      rows.append(summary(study.episode("equation", system, 0, 0, **kw)))
      for ctx in CONTEXT_SIZES:
        for seed in range(args.seeds):
          rows.append(summary(study.episode("neural", system, ctx, seed, **kw)))
          print(f"{system} ctx={ctx} seed={seed}: cost {rows[-1]['cost']:.1f}, settles {rows[-1]['settle_step']}", file=sys.stderr, flush=True)
    doc = {"implementation": f"scaly-{args.form}-{args.solver}", "compile_s": study.compile_s, "rows": rows}
    args.out.write_text(json.dumps(pr.jsonable(doc)))
  else:
    out = replay(study, json.loads(args.instances.read_text()), args.form, args.solver)
    args.out.write_text(json.dumps(pr.jsonable(out)))


if __name__ == "__main__":
  main()
