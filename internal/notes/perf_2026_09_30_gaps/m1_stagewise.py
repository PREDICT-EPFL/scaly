"""M1 (Tier 9): what a stage-structured KKT backend has to beat, on the ``mpc_<nx>_<nu>_<N>`` QPs.

Three parts, each printed as a table:

``piqp``    vendored PIQP's solve time with each of its KKT solvers (dense Cholesky, sparse LDL', the
            sparse LDL' of the condensed matrix, and its multistage backend, which is a
            block-tridiagonal Cholesky of that matrix on BLASFEO), on the problem with its variables
            in stage order ``x_0, u_0, x_1, ...``, which the multistage backend reads its blocks from.
``counts``  the multiply-adds of one factorization: the scalar sparse LDL' of the whole KKT matrix,
            as the generated sparse backend orders it; a dense Cholesky of the condensed matrix;
            and a block recursion over the stages, with and without the zeros inside its blocks.
``blocks``  a prototype of the block recursion as generated code: the factorization of the condensed
            matrix in block-tridiagonal form and one solve with it, each a ``scan`` over the stages
            on the generated Cholesky, triangular solve and product, timed beside the dense
            Cholesky and the sparse LDL' of the same problem.

    uv run internal/notes/perf_2026_09_30_gaps/m1_stagewise.py [piqp|counts|blocks] [--problems 4x2x10,...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
BUILD = HERE / "build" / "m1"
PROBLEMS = "4x2x10,6x2x40,12x4x20,26x2x25,27x6x30,39x3x25"
KKT_SOLVERS = {"dense": 0, "sparse_ldlt": 1, "sparse_ldlt_cond": 4, "multistage": 5}


def stage_order(nx: int, nu: int, horizon: int) -> np.ndarray:
  """``mpc_qp`` has every state before every input; this is ``x_0, u_0, x_1, u_1, ..., x_N``."""
  order: list[int] = []
  for k in range(horizon):
    order += [*range(k * nx, (k + 1) * nx), *range((horizon + 1) * nx + k * nu, (horizon + 1) * nx + (k + 1) * nu)]
  return np.array([*order, *range(horizon * nx, (horizon + 1) * nx)])


def staged(nx: int, nu: int, horizon: int):
  """The problem with its variables in stage order."""
  from scaly.testing.qp import make_qp, mpc_qp

  qp = mpc_qp(nx, nu, horizon)
  o = stage_order(nx, nu, horizon)
  P = sparse.csc_array(qp.P)[o][:, o]
  return make_qp(f"mpc_{nx}_{nu}_{horizon}_staged", P, qp.c[o], A=sparse.csc_array(qp.A)[:, o], b=qp.b, x_l=qp.x_l[o], x_u=qp.x_u[o])


def piqp_driver() -> Path:
  """The suite's trace driver with one more setting read from the environment: ``SCALY_TRACE_KKT``,
  PIQP's ``kkt_solver`` as its enum's integer."""
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler
  from scaly.opt.external.paths import backend_compile_flags

  src = (ROOT / "tests" / "opt" / "ipm" / "piqp_trace.c").read_text()
  anchor = "  if (header[9] >= 0) settings.preconditioner_scale_cost = (piqp_int)header[9];\n"
  assert anchor in src
  src = src.replace(anchor, anchor + '  if (getenv("SCALY_TRACE_KKT")) settings.kkt_solver = (piqp_kkt_solver)atoi(getenv("SCALY_TRACE_KKT"));\n')
  BUILD.mkdir(parents=True, exist_ok=True)
  c, exe = BUILD / "piqp_kkt.c", BUILD / "piqp_kkt"
  if not exe.exists() or not c.exists() or c.read_text() != src:
    c.write_text(src)
    cc = find_c_compiler()
    assert cc is not None
    subprocess.run([cc.cc, str(c), *compile_flags(), *backend_compile_flags(["piqp"]), "-lm", "-o", str(exe)], check=True)
  return exe


def piqp(names: list[tuple[int, int, int]], repeat: int) -> None:
  from tests.opt.ipm import piqp_trace

  exe = piqp_driver()
  print("| problem | n / p | " + " | ".join(f"{k} (us)" for k in KKT_SOLVERS) + " | iterations | multistage, us an iteration |")
  print("| --- | --- | " + " | ".join("---" for _ in KKT_SOLVERS) + " | --- | --- |")
  for nx, nu, horizon in names:
    qp = staged(nx, nu, horizon)
    cells, iters = [], []
    for kind, code in KKT_SOLVERS.items():
      path = BUILD / f"{qp.name}_{kind}.bin"
      piqp_trace.write_problem(qp, path, dense=kind == "dense")
      env = {**os.environ, "SCALY_TRACE_REPEAT": str(repeat), "SCALY_TRACE_KKT": str(code)}
      out = subprocess.run([str(exe), str(path)], capture_output=True, text=True, env=env)
      path.unlink()
      if out.returncode:
        cells.append("failed")
        continue
      trace = piqp_trace.parse(out.stdout)
      cells.append(f"{trace.info['solve_time_min'] * 1e6:.1f}")
      iters.append(int(trace.info["iter"]))
    each = float(cells[-1]) / iters[-1] if cells[-1] != "failed" else float("nan")
    print(
      f"| mpc_{nx}_{nu}_{horizon} | {qp.n} / {qp.A.shape[0]} | " + " | ".join(cells) + f" | {'/'.join(str(i) for i in iters)} | {each:.1f} |",
      flush=True,
    )


def counts(names: list[tuple[int, int, int]]) -> None:
  from scaly.opt.ipm.kkt import kkt_symbolic
  from scaly.testing.qp import mpc_qp
  from tests.opt.ipm.problems import ipm_inputs

  print("| problem | KKT order | nnz(L), sparse LDL' | its multiply-adds | dense Cholesky of the condensed | blocks, dense | blocks, zeros skipped |")
  print("| --- | --- | --- | --- | --- | --- | --- |")
  for nx, nu, horizon in names:
    s, _ = ipm_inputs(mpc_qp(nx, nu, horizon))
    sym = kkt_symbolic(s)
    per_column = np.diff(sym.l_ptr)  # entries below the diagonal in each column of L
    ldl = int((per_column * (per_column + 1) // 2).sum())
    n = s.n
    nz = nx + nu
    dense = n**3 // 6
    # The condensed matrix in levels (u_{k-1}, x_k): the block below a diagonal block is nonzero in
    # its last nx columns only, so with x last in a block the solve is against the trailing nx x nx
    # triangle and the update has rank nx.
    blocks_dense = horizon * (nz**3 // 6 + nz * nz * nz // 2 + nz * nz * nz // 2)
    blocks = horizon * (nz**3 // 6 + nz * nx * nx // 2 + nz * nz * nx // 2)
    print(f"| mpc_{nx}_{nu}_{horizon} | {sym.n} | {int(per_column.sum())} | {ldl} | {dense} | {blocks_dense} | {blocks} |")


def condensed_blocks(
  nx: int, nu: int, horizon: int, rho: float = 1e-6, delta: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csc_array]:
  """The condensed matrix ``P + rho I + A' A / delta + W`` of the problem (``W`` a positive diagonal,
  as the box bounds give) in levels: block 0 is ``x_0`` padded to the level size with an identity,
  block ``k`` is ``(u_{k-1}, x_k)``. Returns the diagonal blocks ``(N + 1, nz, nz)``, the blocks
  below them ``(N, nz, nz)``, a right-hand side, and the matrix itself in that order, padding
  included."""
  from scaly.testing.qp import mpc_qp

  qp = mpc_qp(nx, nu, horizon)
  rng = np.random.default_rng(0)
  n = qp.n
  c = (sparse.csc_array(qp.P) + sparse.diags_array(rho + rng.uniform(0.1, 10.0, n)) + (qp.A.T @ qp.A) / delta).toarray()
  nz = nx + nu
  pad = np.arange(n, n + nu)
  levels = [np.r_[pad, np.arange(nx)]]
  for k in range(1, horizon + 1):
    levels.append(np.r_[(horizon + 1) * nx + (k - 1) * nu + np.arange(nu), k * nx + np.arange(nx)])
  order = np.concatenate(levels)
  full = np.eye(n + nu)
  full[:n, :n] = c
  full = full[np.ix_(order, order)]
  diag = np.stack([full[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] for k in range(horizon + 1)])
  below = np.stack([full[(k + 1) * nz : (k + 2) * nz, k * nz : (k + 1) * nz] for k in range(horizon)])
  band = np.abs(np.subtract.outer(np.arange(n + nu) // nz, np.arange(n + nu) // nz)) <= 1
  assert not np.any(full[~band]), "the levels are a block-tridiagonal partition"
  return diag, below, rng.standard_normal(n + nu), sparse.csc_array(full)


def _zeros(shape):
  from scaly.ir.expr import Expr

  return Expr.const(np.zeros(shape))


def best(fn, args, seconds: float = 0.3) -> float:
  fn._flat_numerical_call(*args)
  end, times = time.perf_counter() + seconds, []
  while time.perf_counter() < end or len(times) < 5:
    t0 = time.perf_counter()
    fn._flat_numerical_call(*args)
    times.append(time.perf_counter() - t0)
  return min(times)


def blocks(names: list[tuple[int, int, int]]) -> None:
  import scaly as sc
  from scaly.linalg import SparseLDL, SparseMatrix, cholesky, solve_triangular
  from scaly.testing.qp import mpc_qp
  from tests.opt.ipm.problems import ipm_inputs

  print(
    "| problem | block | dense Cholesky (us) | sparse LDL' of the KKT (us) | blocks, dense (us) | blocks, x-part (us) | its solve (us) | G multiply-adds a second, x-part | error |"
  )
  print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
  for nx, nu, horizon in names:
    nz = nx + nu
    diag, below, rhs, full = condensed_blocks(nx, nu, horizon)
    n = full.shape[0]
    tag = f"m1_{nx}_{nu}_{horizon}"

    def recursion(x_part: bool, tag=tag, nz=nz, nx=nx, nu=nu, horizon=horizon):
      """``L_k L_k' = D_k - E_{k-1} E_{k-1}'`` and ``E_k = B_k L_k^{-T}`` over the levels. With
      ``x_part`` the block below is read as its last ``nx`` columns, the only nonzero ones."""
      carry = sc.sym("S", (nz * nz,))  # E_{k-1} E_{k-1}', what the previous level takes from this one's diagonal
      d, b = sc.sym("D", (nz, nz)), sc.sym("B", (nz, nz))
      factor = cholesky(d - carry.reshape((nz, nz)))
      if x_part:
        tail = factor[nu:, nu:]
        e_x = solve_triangular(tail, b[:, nu:].T, lower=True).T  # (nz, nx): E's nonzero columns
        e = sc.concat([_zeros((nz, nu)), e_x], axis=1)
        update = e_x @ e_x.T
      else:
        e = solve_triangular(factor, b.T, lower=True).T
        update = e @ e.T
      step = sc.Function.from_exprs(
        f"{tag}_{'x' if x_part else 'd'}_step", [carry, d, b], [update.reshape((nz * nz,)), factor, e], ["S", "D", "B"], ["S_next", "L", "E"]
      ).concrete
      d_all, b_all = sc.sym("D_all", (horizon + 1, nz, nz)), sc.sym("B_all", (horizon, nz, nz))
      b_padded = sc.concat([b_all.reshape((horizon * nz * nz,)), _zeros((nz * nz,))])
      _, ls, es = sc.scan(
        step, _zeros((nz * nz,)), [(d_all.reshape(((horizon + 1) * nz * nz,)), 0, nz * nz), (b_padded, 0, nz * nz)], length=horizon + 1
      )
      return sc.Function.from_exprs(f"{tag}_{'x' if x_part else 'd'}_factor", [d_all, b_all], [ls, es], ["D_all", "B_all"], ["L", "E"]).concrete

    factors = {x_part: recursion(x_part) for x_part in (False, True)}
    times = {x_part: best(fn, (diag, below)) for x_part, fn in factors.items()}
    ls, es = (np.asarray(v).reshape((horizon + 1, nz, nz)) for v in factors[True]._flat_numerical_call(diag, below))
    lower = np.zeros((n, n))
    for k in range(horizon + 1):
      lower[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = np.tril(ls[k])
      if k < horizon:
        lower[(k + 1) * nz : (k + 2) * nz, k * nz : (k + 1) * nz] = es[k]
    dense_matrix = full.toarray()
    error = float(np.abs(lower @ lower.T - dense_matrix).max() / np.abs(dense_matrix).max())

    # One solve: forward over the levels, then backward.
    l_sym, e_sym = sc.sym("L", (horizon + 1, nz, nz)), sc.sym("E", (horizon + 1, nz, nz))
    r_sym = sc.sym("r", (n,))
    carry, lk, ek, rk = sc.sym("c", (nz,)), sc.sym("Lk", (nz, nz)), sc.sym("Ek", (nz, nz)), sc.sym("rk", (nz,))
    y = solve_triangular(lk, rk - carry, lower=True)
    forward = sc.Function.from_exprs(f"{tag}_fwd", [carry, lk, ek, rk], [ek @ y, y], ["c", "Lk", "Ek", "rk"], ["c_next", "y"]).concrete
    flat_l, flat_e = l_sym.reshape(((horizon + 1) * nz * nz,)), e_sym.reshape(((horizon + 1) * nz * nz,))
    _, ys = sc.scan(forward, _zeros((nz,)), [(flat_l, 0, nz * nz), (flat_e, 0, nz * nz), (r_sym, 0, nz)], length=horizon + 1)
    x = solve_triangular(lk.T, rk - ek.T @ carry, lower=False)
    backward = sc.Function.from_exprs(f"{tag}_bwd", [carry, lk, ek, rk], [x, x], ["c", "Lk", "Ek", "rk"], ["c_next", "x"]).concrete
    # E_k couples level k with level k + 1, so the backward step at level k reads E_k and x_{k+1}; the last E is zero.
    last = horizon * nz * nz
    _, xs = sc.scan(backward, _zeros((nz,)), [(flat_l, last, -nz * nz), (flat_e, last, -nz * nz), (ys, horizon * nz, -nz)], length=horizon + 1)
    solve = sc.Function.from_exprs(f"{tag}_solve", [l_sym, e_sym, r_sym], [xs], ["L", "E", "r"], ["x"]).concrete
    l_val, e_val = factors[True]._flat_numerical_call(diag, below)
    e_val = np.asarray(e_val).reshape((horizon + 1, nz, nz))
    x_val = (
      np.asarray(solve._flat_numerical_call(np.asarray(l_val).reshape((horizon + 1, nz, nz)), e_val, rhs)[0])
      .reshape((horizon + 1, nz))[::-1]
      .reshape(-1)
    )
    error = max(error, float(np.abs(dense_matrix @ x_val - rhs).max() / np.abs(rhs).max()))
    solve_time = best(solve, (np.asarray(l_val).reshape((horizon + 1, nz, nz)), e_val, rhs))

    a_sym = sc.sym("A", (n, n))
    dense_fn = sc.Function.from_exprs(f"{tag}_dense", [a_sym], [cholesky(a_sym)], ["A"], ["L"]).concrete
    dense_time = best(dense_fn, (dense_matrix,))

    s, _ = ipm_inputs(mpc_qp(nx, nu, horizon))
    rng = np.random.default_rng(1)
    kkt = sparse.block_array(
      [
        [sparse.diags_array(rng.uniform(1.0, 2.0, s.n)), sparse.csc_array((np.ones(s.A_rows.size), (s.A_rows, s.A_cols)), shape=(s.p, s.n)).T],
        [None, -1e-4 * sparse.eye_array(s.p)],
      ]
    )
    upper = sparse.triu(kkt, format="csc")
    values = sc.sym("v", (upper.nnz,))
    ldl = SparseLDL(SparseMatrix.from_scipy(upper).with_values(values), name=f"{tag}_kkt")
    ldl_fn = sc.Function.from_exprs(f"{tag}_ldl", [values], [ldl.values], ["v"], ["f"]).concrete
    ldl_time = best(ldl_fn, (upper.data,))

    work = horizon * (nz**3 // 6 + nz * nx * nx // 2 + nz * nz * nx // 2)
    print(
      f"| mpc_{nx}_{nu}_{horizon} | {nz} | {dense_time * 1e6:.1f} | {ldl_time * 1e6:.1f} | {times[False] * 1e6:.1f} | {times[True] * 1e6:.1f} | {solve_time * 1e6:.1f}"
      f" | {work / times[True] / 1e9:.1f} | {error:.1e} |",
      flush=True,
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("part", nargs="?", default="all", choices=["all", "piqp", "counts", "blocks"])
  parser.add_argument("--problems", default=PROBLEMS)
  parser.add_argument("--repeat", type=int, default=50)
  args = parser.parse_args()
  names = [(int(a), int(b), int(c)) for a, b, c in (name.split("x") for name in args.problems.split(","))]
  if args.part in ("all", "piqp"):
    piqp(names, args.repeat)
  if args.part in ("all", "counts"):
    counts(names)
  if args.part in ("all", "blocks"):
    blocks(names)


if __name__ == "__main__":
  main()
