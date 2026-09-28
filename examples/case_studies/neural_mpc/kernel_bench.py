"""The network kernels of the Real-time Neural MPC study, timed on their own.

    ACADOS_SOURCE_DIR=... uv run --with torch --with <acados_template> examples/case_studies/neural_mpc/kernel_bench.py --out results/kernels.json

The closed-loop frequencies of `compare.py` include acados' Python interface, which dominates for small
networks. This takes the two computations that change between the columns and times them alone:

  vde_forw  the forward sensitivity function acados' ERK integrator calls 4 times per shooting interval
            (40 times per RTI): CasADi's, as acados generates it for the naive model (harvested from its
            code directory), against Scaly's drop-in; both compiled with the same flags and timed from
            C through the shared entry point (`../fatrop_chain/time_kernel.c`), best of 2000 calls
  taylor    RTN-MPC's surrogate at the N = 10 shooting nodes: PyTorch's `approx_params` in float32 (what
            the paper runs) and in float64, timed in Python, against Scaly's `taylor_params` timed
            from C and through its Python call
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "baseline"))
NATIVE = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
CFLAGS = ["-O2", NATIVE, "-fno-math-errno"]
SIZES = [(2, 16), (2, 128), (5, 16), (5, 128), (12, 32), (12, 512)]


def time_c(lib: Path, symbol: str, inputs: list[np.ndarray], outputs: list[int], sizes: tuple[int, int, int, int], work: Path, repeats: int = 2000):
  harness = work / "time_kernel"
  if not harness.exists():
    subprocess.run(["cc", "-O2", "-o", str(harness), str(HERE.parent / "fatrop_chain" / "time_kernel.c")], check=True)
  blob = np.array([len(inputs), len(outputs), *sizes, *(x.size for x in inputs), *outputs], dtype=np.int64).tobytes()
  blob += b"".join(np.ascontiguousarray(x, dtype=np.float64).tobytes() for x in inputs)
  (work / "in.bin").write_bytes(blob)
  out = subprocess.run(
    [str(harness), str(lib), symbol, str(work / "in.bin"), str(repeats), str(work / "out.bin")], capture_output=True, text=True, check=True
  )
  best, median = (float(v) * 1e-9 for v in out.stdout.split())
  return best, median, np.frombuffer((work / "out.bin").read_bytes(), dtype=np.float64)


def casadi_vde_forw(net, work: Path) -> Path:
  """acados' own generated `wr_expl_vde_forw.c` for the naive model."""
  from acados_template import AcadosOcpSolver
  from run_acados import model, ocp

  m, p0 = model(net, "naive")
  o = ocp(m, p0)
  cwd = Path.cwd()
  os.chdir(work)
  try:
    AcadosOcpSolver.generate(o, verbose=False)
    code_dir = Path(o.code_gen_options.code_export_directory).resolve()
  finally:
    os.chdir(cwd)
  return code_dir / "wr_model" / "wr_expl_vde_forw.c"


def work_sizes(lib: Path, symbol: str) -> tuple[int, int, int, int]:
  """`(sz_w, sz_iw, sz_arg, sz_res)` from the CasADi-layer `<symbol>_work` query (built with `casadi_int` as `int`)."""
  import ctypes

  so = ctypes.CDLL(str(lib))
  arg, res, iw, w = (ctypes.c_int() for _ in range(4))
  getattr(so, f"{symbol}_work")(ctypes.byref(arg), ctypes.byref(res), ctypes.byref(iw), ctypes.byref(w))
  return max(w.value, 1), iw.value, arg.value, res.value


def compile_so(src: Path, lib: Path, defines: list[str]) -> float:
  t0 = time.perf_counter()
  subprocess.run(["cc", *CFLAGS, *defines, "-shared", "-fPIC", "-o", str(lib), str(src)], check=True, capture_output=True)
  return time.perf_counter() - t0


def bench(layers: int, width: int, work: Path) -> dict:
  import torch

  from run_acados import network

  import scaly_impl
  from scaly.codegen import write_module

  torch.set_num_threads(1)
  net = network(layers, width)
  torch.save(net.state_dict(), work / "net.pt")
  params = scaly_impl.load(work / "net.pt")
  rng = np.random.default_rng(0)
  x, u = rng.standard_normal(2), rng.standard_normal(1)
  sx, sp = rng.standard_normal(4), rng.standard_normal(2)
  row: dict = {"layers": layers, "width": width}

  # vde_forw: CasADi (acados' code) against Scaly, same inputs, same flags.
  c_src = casadi_vde_forw(net, work)
  c_lib = work / "casadi_vde.so"
  row["casadi_vde_compile"] = compile_so(c_src, c_lib, [])
  row["casadi_vde_source_bytes"] = c_src.stat().st_size
  fn = scaly_impl.acados_functions(params)["wr_expl_vde_forw"]
  module = write_module(fn, work / "scaly_vde", casadi=True)
  s_lib = work / "scaly_vde.so"
  row["scaly_vde_compile"] = compile_so(work / "scaly_vde" / module.source_name, s_lib, ["-Dcasadi_int=int"])
  row["scaly_vde_source_bytes"] = (work / "scaly_vde" / module.source_name).stat().st_size
  inputs = [x, sx, sp, u, np.zeros(0)]
  c_best, _, c_out = time_c(c_lib, "wr_expl_vde_forw", inputs, [2, 4, 2], work_sizes(c_lib, "wr_expl_vde_forw"), work)
  s_best, _, s_out = time_c(s_lib, "wr_expl_vde_forw", inputs, [2, 4, 2], work_sizes(s_lib, "wr_expl_vde_forw"), work)
  row["casadi_vde_s"], row["scaly_vde_s"] = c_best, s_best
  row["vde_agreement"] = float(np.max(np.abs(c_out - s_out)))

  # taylor: torch float32, torch float64, Scaly.
  xs = rng.standard_normal((scaly_impl.N, 2))
  # ml-casadi casts the points to float32 (`get_approx_params_list`), so the paper's surrogate is float32;
  # the float64 row calls ml-casadi's own `batched_jacobian` on the network in double precision.
  from ml_casadi.torch.autograd import batched_jacobian

  net64 = network(layers, width).double()
  runs = {
    "torch32": lambda: net.approx_params(xs, flat=True),
    "torch64": lambda: batched_jacobian(net64, torch.tensor(xs, dtype=torch.float64), return_func_output=True),
  }
  for label, run in runs.items():
    for _ in range(20):
      run()
    reps = []
    for _ in range(200):
      t0 = time.perf_counter()
      run()
      reps.append(time.perf_counter() - t0)
    row[f"{label}_taylor_s"] = min(reps)
  batch = scaly_impl.taylor_function(params)
  module = write_module(batch, work / "scaly_taylor")
  t_lib = work / "scaly_taylor.so"
  compile_so(work / "scaly_taylor" / module.source_name, t_lib, [])
  best, _, out = time_c(t_lib, "taylor_params", [xs.reshape(-1)], [scaly_impl.N * 8], (max(module.workspace_size, 1), 0, 4, 4), work)
  row["scaly_taylor_s"] = best
  batch(xs.reshape(-1))
  reps = []
  for _ in range(200):
    t0 = time.perf_counter()
    batch(xs.reshape(-1))
    reps.append(time.perf_counter() - t0)
  row["scaly_taylor_python_s"] = min(reps)
  row["taylor_agreement"] = float(np.max(np.abs(out.reshape(scaly_impl.N, -1) - net.approx_params(xs.astype(np.float32), flat=True))))
  return row


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()
  sys.path.insert(0, str(HERE.parent))
  from _common import wait_for_quiet

  rows = []
  for layers, width in SIZES:
    load = wait_for_quiet()
    with tempfile.TemporaryDirectory(prefix="neural-kernels-") as work:
      os.environ["SCALY_CACHE_DIR"] = str(Path(work) / "cache")
      row = bench(layers, width, Path(work))
    row["load"] = load
    rows.append(row)
    print({k: (f"{1e6 * v:.1f} us" if k.endswith("_s") else v) for k, v in row.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
