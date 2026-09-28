"""The Scaly side of the Real-time Neural MPC study, one variant per process. Prints one JSON line.

    ACADOS_SOURCE_DIR=... uv run --with torch --with <acados_template> examples/case_studies/neural_mpc/run_scaly.py --layers 5 --width 128 --mode dropin

`dropin` is the naive column with Scaly's network code inside acados; `taylor` is the RTN-MPC column
with Scaly's value and Jacobian in place of PyTorch's. Both reuse the baseline's OCP and control loop
unchanged (`baseline/run_acados.py`), and the same network: the baseline builds it with torch, saves
its state dict, and Scaly reads that file without torch.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "baseline"))


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--layers", type=int, required=True)
  ap.add_argument("--width", type=int, required=True)
  ap.add_argument("--mode", choices=("dropin", "taylor"), required=True)
  ap.add_argument("--out-scale", type=float, default=None)
  args = ap.parse_args()
  import torch

  torch.set_num_threads(1)
  from acados_template import AcadosOcpSolver
  from run_acados import OUT_SCALE, STEPS, WARMUP, control_loop, model, network, ocp

  work = Path(tempfile.mkdtemp(prefix="neural-mpc-scaly-"))
  os.environ.setdefault("SCALY_CACHE_DIR", str(work / "cache"))
  os.chdir(work)
  net = network(args.layers, args.width, OUT_SCALE if args.out_scale is None else args.out_scale)
  torch.save(net.state_dict(), work / "net.pt")

  t0 = time.perf_counter()
  import scaly_impl

  params = scaly_impl.load(work / "net.pt")
  if args.mode == "dropin":
    m, p0 = model(net, "naive")  # acados lays out the OCP from CasADi's model; its model sources are then replaced
    o = ocp(m, p0)
    o.code_gen_options.json_file = str(work / "ocp.json")
    AcadosOcpSolver.generate(o, verbose=False)
    code_dir = Path(o.code_gen_options.code_export_directory)
    scaly_impl.install_dropin(params, code_dir, n_p=0)
    AcadosOcpSolver.build(str(code_dir), verbose=False)
    solver = AcadosOcpSolver(o, generate=False, build=False, check_reuse_possible=False, verbose=False)
    taylor = None
  else:
    m, p0 = model(net, "rtn")
    o = ocp(m, p0)
    o.code_gen_options.json_file = str(work / "ocp.json")
    solver = AcadosOcpSolver(o, verbose=False)
    batch = scaly_impl.taylor_function(params)
    batch(np.zeros(2 * scaly_impl.N))  # compile before the clock starts, as the baseline's torch warm-up does

    def taylor(xs):
      return np.asarray(batch(xs.reshape(-1))).reshape(scaly_impl.N, -1)

  t_setup = time.perf_counter() - t0
  control_loop(solver, taylor, WARMUP)
  solver.reset()
  times, states, t_lin, t_qp = control_loop(solver, taylor, STEPS)
  print(
    json.dumps(
      {
        "layers": args.layers,
        "width": args.width,
        "mode": args.mode,
        "out_scale": OUT_SCALE if args.out_scale is None else args.out_scale,
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
