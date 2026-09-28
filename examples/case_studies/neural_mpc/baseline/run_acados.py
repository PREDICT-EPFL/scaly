"""The paper's side of the Real-time Neural MPC study (Salzmann et al., RA-L 2023, Table II).

    uv run --with torch --with <acados_template> examples/case_studies/neural_mpc/baseline/run_acados.py --layers 2 --width 16 --mode naive

A port of ml-casadi's `examples/mpc_mlp_naive_example.py` and `examples/mpc_mlp_cnn_example.py`
(pinned by `setup.sh`): a double integrator `x = (s, s_dot)`, `u = s_ddot` with `|u| <= 10`, plus an
untrained MLP residual `f_D(x)` (2 inputs, 2 outputs, tanh, output layer zeroed as the authors do),
integrated by acados' ERK (RK4, one step), N = 10 over 1 s, SQP-RTI with Gauss-Newton and full
condensing HPIPM, tracking a sine on `s`. Two ways to put the network in:

  naive  the MLP expanded into the CasADi model, which acados code-generates with its sensitivities
  rtn    RTN-MPC: a first-order Taylor surrogate `f_a + J_a (x - a)` whose `a, f_a, J_a` are
         parameters, computed by PyTorch at the previous solution after every solve

The timed loop is the authors': 50 control steps, each setting the reference and the initial state,
one RTI, reading the solution and (rtn) evaluating the network and its Jacobian and setting the
parameters. The paper reports 1/mean; this prints the mean, the median and the minimum. Unlike the
authors' naive script, both modes get the same 20 warm-up steps before timing. Torch runs one thread.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N, T_HORIZON, A_MAX, STEPS, WARMUP = 10, 1.0, 10.0, 50, 20
OUT_SCALE = 1e-6


def network(layers: int, width: int, out_scale: float = OUT_SCALE, seed: int = 0):
  """ml-casadi's MultiLayerPerceptron(2, width, 2, layers, 'Tanh'), untrained, its output bias zeroed and
  its output weights `out_scale * N(0, 1) / sqrt(width)`. The authors zero the output weights too
  (`out_scale=0`); with CasADi 3.8 that lets the expanded model drop the whole network, so acados'
  generated dynamics contain no network at all. A tiny nonzero scale keeps the dynamics nominal to
  1e-6 while making both columns evaluate every layer."""
  import torch

  sys.path.insert(0, str(HERE / "third_party" / "ml-casadi"))
  import ml_casadi.torch as mc

  torch.manual_seed(seed)
  net = mc.nn.MultiLayerPerceptron(2, width, 2, layers, "Tanh")
  with torch.no_grad():
    net.output_layer.bias.fill_(0.0)
    net.output_layer.weight.copy_(out_scale * torch.randn_like(net.output_layer.weight) / width**0.5)
  return net


def model(net, mode: str):
  import casadi as cs
  from acados_template import AcadosModel

  s, s_dot, s_ddot, u = (cs.MX.sym(n, 1) for n in ("s", "s_dot", "s_dot_dot", "u"))
  x, x_dot = cs.vertcat(s, s_dot), cs.vertcat(s_dot, s_ddot)
  if mode == "naive":
    residual, p, p0 = net(x), cs.vertcat([]), np.array([])
  else:
    residual = net.approx(x)
    p = net.sym_approx_params(order=1, flat=True)
    p0 = net.approx_params(np.array([0, 0]), flat=True, order=1)
  f_expl = cs.vertcat(s_dot, u) + residual
  m = AcadosModel()
  m.f_impl_expr, m.f_expl_expr = x_dot - f_expl, f_expl
  m.x, m.xdot, m.u, m.p, m.name = x, x_dot, u, p, "wr"
  return m, p0


def ocp(m, p0):
  from acados_template import AcadosOcp

  o = AcadosOcp()
  o.model = m
  o.solver_options.N_horizon = N
  o.solver_options.tf = T_HORIZON
  o.cost.cost_type = o.cost.cost_type_e = "LINEAR_LS"
  o.cost.W = np.array([[1.0]])
  o.cost.Vx = np.array([[1.0, 0.0]])
  o.cost.Vu = np.zeros((1, 1))
  o.cost.Vx_e = np.array([[1.0, 0.0]])
  o.cost.W_e = np.array([[0.0]])
  o.cost.yref, o.cost.yref_e = np.zeros(1), np.zeros(1)
  o.constraints.x0 = np.zeros(2)
  o.constraints.lbu, o.constraints.ubu, o.constraints.idxbu = np.array([-A_MAX]), np.array([A_MAX]), np.array([0])
  o.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
  o.solver_options.hessian_approx = "GAUSS_NEWTON"
  o.solver_options.integrator_type = "ERK"
  o.solver_options.nlp_solver_type = "SQP_RTI"
  o.parameter_values = p0
  return o


def control_loop(solver, taylor=None, steps: int = STEPS):
  """The authors' loop. `taylor(xs) -> params` is the RTN-MPC update, None for naive. Returns per-step
  wall times, the closed-loop states, and acados' per-step linearization and QP times."""
  ts = 1.0 / N
  xt = np.array([1.0, 0.0])
  times, states, t_lin, t_qp = [], [], [], []
  for i in range(steps):
    now = time.perf_counter()
    t = np.linspace(i * ts, i * ts + 1.0, 10)
    yref = np.sin(0.5 * t + np.pi / 2)
    for k, ref in enumerate(yref):
      solver.set(k, "yref", np.array([ref]))
    solver.set(0, "lbx", xt)
    solver.set(0, "ubx", xt)
    solver.solve()
    xt = solver.get(1, "x")
    xs = np.stack([solver.get(k, "x") for k in range(N)])
    if taylor is not None:
      params = taylor(xs)
      for k in range(N):
        solver.set(k, "p", params[k])
    times.append(time.perf_counter() - now)
    states.append(xt)
    t_lin.append(float(solver.get_stats("time_lin")))
    t_qp.append(float(solver.get_stats("time_qp")))
  return np.array(times), np.array(states), np.array(t_lin), np.array(t_qp)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--layers", type=int, default=0, help="0: no network")
  ap.add_argument("--width", type=int, default=0)
  ap.add_argument("--mode", choices=("naive", "rtn"), required=True)
  ap.add_argument("--out-scale", type=float, default=OUT_SCALE, help="0 reproduces the authors' zeroed output layer")
  args = ap.parse_args()
  import torch

  torch.set_num_threads(1)
  from acados_template import AcadosOcpSolver

  work = Path(tempfile.mkdtemp(prefix="neural-mpc-acados-"))
  os.chdir(work)
  t0 = time.perf_counter()
  if args.layers == 0:
    import casadi as cs

    class Zero:
      def __call__(self, x):
        return cs.MX.zeros(2, 1)

    net, mode = Zero(), "naive"
  else:
    net, mode = network(args.layers, args.width, args.out_scale), args.mode
  m, p0 = model(net, mode)
  solver = AcadosOcpSolver(ocp(m, p0), json_file=str(work / "ocp.json"), verbose=False)
  t_setup = time.perf_counter() - t0
  taylor = None
  if mode == "rtn":

    def taylor(xs):
      return net.approx_params(xs, flat=True)

  control_loop(solver, taylor, WARMUP)
  solver.reset()
  times, states, t_lin, t_qp = control_loop(solver, taylor)
  print(
    json.dumps(
      {
        "layers": args.layers,
        "width": args.width,
        "mode": "none" if args.layers == 0 else args.mode,
        "out_scale": args.out_scale,
        "t_setup": t_setup,
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "min": float(times.min()),
        "hz_mean": float(1.0 / times.mean()),
        "t_lin_median": float(np.median(t_lin)),
        "t_qp_median": float(np.median(t_qp)),
        "states": states.tolist(),
      }
    )
  )


if __name__ == "__main__":
  main()
