"""The kernels of the 2026-09-30 generated-code speed review, from tiny to large and structured.

    uv run internal/notes/perf_2026_09_30_codegen/corpus.py --variant base [--kernels a,b] [--jobs 6]

To build another checkout's variant, put its ``src`` first on the path:
``PYTHONPATH=<worktree>/src uv run --no-sync .../corpus.py --variant <name>``.

Per kernel, in a fresh process with an empty JIT cache: the generated C (``kernel.c``, the module's
self-contained body), the input blob ``time_entry.c`` reads, the JIT's own outputs for the same
inputs (``expected.bin``, the reference every compiled variant is held to), and ``meta.json``
(symbol, sizes, graph-build, generation and compile times, C lines, workspace). Everything goes to
``build/<variant>/<kernel>/``. ``timing.py`` compiles and times the variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
BUILD = HERE / "build"

Built = tuple  # (Function, list of input arrays)


def _fn(name: str, inputs: list, outputs: list):
  import scaly as sc

  return sc.Function.from_exprs(name, inputs, outputs, [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


# ---- tiny ------------------------------------------------------------------------------------------


def rosen_hess() -> Built:
  import scaly as sc

  x = sc.sym("x", 2)
  f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
  return _fn("rosen_hess", [x], [sc.hessian(f, x)]), [np.array([0.3, -0.7])]


def _cartpole(x, u):
  import scaly as sc

  mc, mp, ell, g = 1.0, 0.1, 0.5, 9.81
  th, dx, dth = x[1], x[2], x[3]
  s, c = th.sin(), th.cos()
  den = mc + mp * s * s
  ddx = (u[0] + mp * s * (ell * dth * dth + g * c)) / den
  ddth = (-u[0] * c - mp * ell * dth * dth * c * s - (mc + mp) * g * s) / (ell * den)
  return sc.stack([dx, dth, ddx, ddth])


def _rk4(f, x, u, h):
  k1 = f(x, u)
  k2 = f(x + 0.5 * h * k1, u)
  k3 = f(x + 0.5 * h * k2, u)
  k4 = f(x + h * k3, u)
  return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def cartpole_jac() -> Built:
  import scaly as sc

  xu = sc.sym("xu", 5)
  nxt = _rk4(_cartpole, xu[0:4], xu[4:5], 0.05)
  return _fn("cartpole_jac", [xu], [nxt, sc.jacobian(nxt, xu)]), [np.array([0.1, 0.4, -0.2, 0.3, 0.7])]


def _mlp_weights(widths, seed=0):
  rng = np.random.default_rng(seed)
  return [(rng.normal(size=(o, i)) / np.sqrt(i), 0.1 * rng.normal(size=o)) for i, o in zip(widths[:-1], widths[1:])]


def mlp_small_jac() -> Built:
  import scaly as sc
  from scaly.nn import layers as nn

  x = sc.sym("x", 4)
  y = nn.mlp(x, _mlp_weights((4, 32, 32, 2)))
  return _fn("mlp_small_jac", [x], [y, sc.jacobian(y, x)]), [np.array([0.2, -0.1, 0.5, 0.3])]


def quat_jac() -> Built:
  import scaly as sc

  q, v = sc.sym("q", 4), sc.sym("v", 3)
  # Hamilton product q v q*, scalar last, written out.
  x, y, z, w = q[0], q[1], q[2], q[3]
  r = sc.stack(
    [
      (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y - z * w) * v[1] + 2 * (x * z + y * w) * v[2],
      2 * (x * y + z * w) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z - x * w) * v[2],
      2 * (x * z - y * w) * v[0] + 2 * (y * z + x * w) * v[1] + (1 - 2 * (x * x + y * y)) * v[2],
    ]
  )
  q0 = np.array([0.1, 0.2, 0.3, 0.9])
  return _fn("quat_jac", [q, v], [r, sc.jacobian(r, q)]), [q0 / np.linalg.norm(q0), np.array([1.0, -2.0, 0.5])]


# ---- dense and elementwise -------------------------------------------------------------------------


def _spd(n, seed=0):
  rng = np.random.default_rng(seed)
  m = rng.normal(size=(n, n))
  return m @ m.T + n * np.eye(n)


def _chol_solve(n: int) -> Built:
  import scaly as sc
  from scaly import linalg

  a, b = sc.sym("a", (n, n)), sc.sym("b", n)
  x = linalg.cho_solve(linalg.cholesky(a), b)
  return _fn(f"chol_solve_{n}", [a, b], [x]), [_spd(n), np.linspace(-1, 1, n)]


def chol_solve_16() -> Built:
  return _chol_solve(16)


def chol_solve_64() -> Built:
  return _chol_solve(64)


def chol_solve_200() -> Built:
  return _chol_solve(200)


def matvec_256() -> Built:
  import scaly as sc

  a, x = sc.sym("a", (256, 256)), sc.sym("x", 256)
  rng = np.random.default_rng(1)
  return _fn("matvec_256", [a, x], [a @ x]), [rng.normal(size=(256, 256)), rng.normal(size=256)]


def matmul_48() -> Built:
  import scaly as sc

  a, b = sc.sym("a", (48, 48)), sc.sym("b", (48, 48))
  rng = np.random.default_rng(2)
  return _fn("matmul_48", [a, b], [a @ b]), [rng.normal(size=(48, 48)), rng.normal(size=(48, 48))]


def mlp_big_fwd() -> Built:
  import scaly as sc
  from scaly.nn import layers as nn

  widths = (16, 256, 256, 4)
  n = nn.size(widths)
  x, pw = sc.sym("x", 16), sc.sym("pw", n)
  y = nn.mlp(x, nn.unpack(pw, widths))
  return _fn("mlp_big_fwd", [x, pw], [y]), [np.linspace(-1, 1, 16), nn.pack(_mlp_weights(widths))]


def elementwise_1e5() -> Built:
  import scaly as sc

  n = 100_000
  x, y = sc.sym("x", n), sc.sym("y", n)
  s = (x.sin() * y + (-x * x).exp()).sum()
  m = (x - y).abs().max()
  z = sc.maximum(sc.minimum(x, 1.0), -1.0) * y
  rng = np.random.default_rng(3)
  return _fn("elementwise_1e5", [x, y], [s, m, z]), [rng.normal(size=n), rng.normal(size=n)]


# ---- structured: scan, Riccati, sparse LDL ----------------------------------------------------------


def rollout_grad_100() -> Built:
  import scaly as sc

  horizon = 100
  c, uk = sc.sym("c", 5), sc.sym("uk", 1)  # carry: state (4) and running cost
  x = c[0:4]
  nxt = _rk4(_cartpole, x, uk, 0.05)
  cost = c[4] + (nxt * nxt).sum() + 0.1 * uk[0] * uk[0]
  body = _fn("rollout_step", [c, uk], [sc.concat([nxt, sc.stack([cost])])])
  u, x0 = sc.sym("u", horizon), sc.sym("x0", 4)
  final, *_ = sc.scan(body, sc.concat([x0, 0.0 * x0[0:1]]), [(u, 0, 1)], length=horizon)
  total = final[4]
  return _fn("rollout_grad_100", [u, x0], [total, sc.gradient(total, u)]), [0.3 * np.sin(np.arange(horizon) / 7.0), np.array([0.0, 0.2, 0.0, 0.0])]


def riccati_50() -> Built:
  import scaly as sc
  from scaly.linalg.stagewise import Riccati

  n, nx, nu = 50, 6, 3
  rng = np.random.default_rng(4)
  d = {
    "A": np.eye(nx) + 0.1 * rng.normal(size=(n, nx, nx)),
    "B": rng.normal(size=(n, nx, nu)),
    "Q": np.stack([_spd(nx, k) for k in range(n)]),
    "R": np.stack([_spd(nu, 100 + k) for k in range(n)]),
    "QN": _spd(nx, 999),
    "x0": rng.normal(size=nx),
    "q": rng.normal(size=(n, nx)),
    "r": rng.normal(size=(n, nu)),
    "c": rng.normal(size=(n, nx)),
    "qN": rng.normal(size=nx),
  }
  s = {k: sc.sym(k, v.shape) for k, v in d.items()}
  fac = Riccati(s["A"], s["B"], s["Q"], s["R"], s["QN"], N=n)
  xs, us, lams = fac.solve(s["x0"], s["q"], s["r"], s["c"], s["qN"])
  return _fn("riccati_50", list(s.values()), [xs, us, lams]), list(d.values())


def _laplacian(k: int):
  from scipy import sparse

  t = sparse.diags_array([-np.ones(k - 1), 4.0 * np.ones(k), -np.ones(k - 1)], offsets=[-1, 0, 1])
  e = sparse.eye_array(k)
  lap = sparse.kron(e, t) + sparse.kron(sparse.diags_array([-np.ones(k - 1), -np.ones(k - 1)], offsets=[-1, 1]), e)
  return sparse.csc_array(sparse.tril(lap))


def _mpc_kkt(stages: int, nx: int, nu: int):
  from scipy import sparse

  rng = np.random.default_rng(stages)
  z = stages * (nx + nu) + nx
  h = sparse.eye_array(z)
  rows = []
  for k in range(stages):
    blk = sparse.lil_array((nx, z))
    off = k * (nx + nu)
    blk[:, off : off + nx + nu] = rng.standard_normal((nx, nx + nu))
    blk[:, off + nx + nu : off + 2 * nx + nu] = -np.eye(nx)
    rows.append(blk)
  c = sparse.vstack(rows)
  return sparse.csc_array(sparse.tril(sparse.bmat([[h, c.T], [c, -1e-6 * sparse.eye_array(c.shape[0])]])))


def _sparse_ldl(name: str, t) -> Built:
  import scaly as sc
  from scaly.linalg import SparseLDL, SparseMatrix

  t.sort_indices()
  mat = SparseMatrix.symbol("K", t)
  fact = SparseLDL(mat, name=name)
  b = sc.sym("b", t.shape[0])
  vals = np.asarray(t.tocsr()[mat.coordinates()]).reshape(-1)
  return _fn(name, [mat.values, b], [fact.solve(b)]), [vals, np.linspace(-1, 1, t.shape[0])]


def sparse_ldl_grid30() -> Built:
  return _sparse_ldl("sparse_ldl_grid30", _laplacian(30))


def sparse_ldl_mpc50() -> Built:
  return _sparse_ldl("sparse_ldl_mpc50", _mpc_kkt(50, 6, 3))


# ---- the generated IPM ------------------------------------------------------------------------------

IPM_ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


def _ipm(qp_name: str, backend: str) -> Built:
  import scaly as sc
  from scaly.codegen.abi import c_ident
  from scaly.opt.ipm import QPValues, Solver
  from tests.opt.ipm.problems import ipm_inputs, maros_meszaros, mpc_qp

  if qp_name.startswith("mpc_"):
    nx, nu, horizon = (int(v) for v in qp_name.split("_")[1:])
    qp = mpc_qp(nx, nu, horizon, name=qp_name)
  else:
    qp = maros_meszaros(qp_name)
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in IPM_ORDER}
  name = f"ipm_{c_ident(qp_name)}_{backend}"
  res = Solver(s, backend, name=name).solve(QPValues.preprocess(s, **syms))
  keys = ["x", "status", "iter"]
  fn = sc.Function.from_exprs(name, [syms[k] for k in IPM_ORDER], [res[k] for k in keys], list(IPM_ORDER), keys)
  return fn, [np.asarray(values[k], dtype=float) for k in IPM_ORDER]


def ipm_hs118_dense() -> Built:
  return _ipm("HS118", "dense")


def ipm_qafiro_sparse() -> Built:
  return _ipm("QAFIRO", "sparse")


def ipm_mpc_12_4_20_sparse() -> Built:
  return _ipm("mpc_12_4_20", "sparse")


def ipm_cvxqp1_dense() -> Built:
  return _ipm("CVXQP1_S", "dense")


# ---- the benchmark problems' oracles (bench/harness/sweep.py builds them and their inputs) ----------


def _bench(workload: str, size: int) -> Callable[[], Built]:
  def build() -> Built:
    from bench.harness import sweep

    with tempfile.TemporaryDirectory() as tmp:
      info = sweep.build_kernel(workload, size, "scaly", Path(tmp))
      paths, _ = sweep._samples(workload, size, info, Path(tmp), load_harvested=False)
      args = [np.fromfile(paths[name], dtype=np.float64) for name, _ in info["inputs"]]
    return info["callable"], args

  build.__name__ = f"{workload}_{size}"
  return build


KERNELS: dict[str, Callable[[], Built]] = {
  f.__name__: f
  for f in (
    rosen_hess,
    cartpole_jac,
    mlp_small_jac,
    quat_jac,
    chol_solve_16,
    chol_solve_64,
    chol_solve_200,
    matvec_256,
    matmul_48,
    mlp_big_fwd,
    elementwise_1e5,
    rollout_grad_100,
    riccati_50,
    sparse_ldl_grid30,
    sparse_ldl_mpc50,
    _bench("race_cars", 40),
    _bench("race_cars_jac", 40),
    _bench("race_cars", 200),
    _bench("chain", 5),
    _bench("chain", 9),
    _bench("npmpc", 12),
    _bench("unbumpercars", 8),
    ipm_hs118_dense,
    ipm_qafiro_sparse,
    ipm_cvxqp1_dense,
    ipm_mpc_12_4_20_sparse,
  )
}
TIERS = {
  "tiny": ("rosen_hess", "cartpole_jac", "mlp_small_jac", "quat_jac", "chol_solve_16"),
  "medium": (
    "chol_solve_64",
    "matvec_256",
    "matmul_48",
    "mlp_big_fwd",
    "rollout_grad_100",
    "riccati_50",
    "race_cars_40",
    "race_cars_jac_40",
    "chain_5",
    "npmpc_12",
    "ipm_hs118_dense",
    "ipm_qafiro_sparse",
  ),
  "large": (
    "chol_solve_200",
    "elementwise_1e5",
    "sparse_ldl_grid30",
    "sparse_ldl_mpc50",
    "race_cars_200",
    "chain_9",
    "unbumpercars_8",
    "ipm_cvxqp1_dense",
    "ipm_mpc_12_4_20_sparse",
  ),
}


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def one(name: str, out: Path) -> dict:
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident

  t0 = time.perf_counter()
  fn, args = KERNELS[name]()
  fn = fn.concrete
  build = time.perf_counter() - t0
  t0 = time.perf_counter()
  module = render_c_module(fn)
  generate = time.perf_counter() - t0
  out.mkdir(parents=True, exist_ok=True)
  (out / "kernel.c").write_text(module.body)
  flat = [np.ravel(np.asarray(a, dtype=float)) for a in args]
  t0 = time.perf_counter()
  shaped = tuple(np.asarray(a, dtype=float).reshape(x.shape) for a, x in zip(args, fn.inputs, strict=True))
  result = fn(*fn.input_tree.unflatten(shaped))
  first_call = time.perf_counter() - t0
  outs = result if isinstance(result, (tuple, list)) else (result,)
  expected = np.concatenate([np.ravel(np.asarray(o, dtype=float)) for o in outs])
  (out / "expected.bin").write_bytes(expected.astype("<f8").tobytes())
  (out / "inputs.bin").write_bytes(blob(flat, [int(e.size) for e in fn.outputs], int(module.workspace_size)))
  meta = {
    "kernel": name,
    "symbol": c_ident(fn.name),
    "inputs": [int(a.size) for a in flat],
    "outputs": [int(e.size) for e in fn.outputs],
    "build_s": build,
    "generate_s": generate,
    "first_call_s": first_call,
    "link_flags": list(module.link_flags),
    "c_lines": sum(1 for line in module.body.splitlines() if line.strip()),
    "c_bytes": len(module.body),
    "workspace": int(module.workspace_size),
  }
  (out / "meta.json").write_text(json.dumps(meta, indent=1))
  return meta


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--variant", required=True)
  parser.add_argument("--kernels", help="comma-separated names or tiers (tiny, medium, large); default all")
  parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
  parser.add_argument("--one", nargs=2, metavar=("KERNEL", "OUT"), help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.one:
    print(json.dumps(one(args.one[0], Path(args.one[1]))))
    return
  names: list[str] = []
  for item in (args.kernels or ",".join(KERNELS)).split(","):
    names += list(TIERS.get(item, (item,)))
  base = BUILD / args.variant

  def run(name: str) -> str:
    with tempfile.TemporaryDirectory(prefix="scaly-cache-") as cache:
      env = {**os.environ, "SCALY_CACHE_DIR": cache}
      t = time.perf_counter()
      done = subprocess.run(
        [sys.executable, __file__, "--variant", args.variant, "--one", name, str(base / name)], capture_output=True, text=True, env=env
      )
    if done.returncode:
      tail = done.stderr.strip().splitlines()[-1][-300:] if done.stderr.strip() else done.returncode
      return f"{name:>24} FAILED {tail}"
    m = json.loads(done.stdout.strip().splitlines()[-1])
    return f"{name:>24} {time.perf_counter() - t:6.1f} s  build {m['build_s']:.2f} s gen {m['generate_s']:.2f} s  {m['c_lines']:6d} lines  w {m['workspace']}"

  with ThreadPoolExecutor(args.jobs) as pool:
    for line in pool.map(run, names):
      print(line, flush=True)


if __name__ == "__main__":
  main()
