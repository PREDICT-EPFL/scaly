"""PR 5 (T2-5): symbolic analysis. For each test matrix and ordering: fill (``nnz(L)`` below the
diagonal, against SuperLU's own with the same ordering where it applies), update lanes (the
left-looking multiply-adds), elimination-tree height, fundamental supernodes, widest tables, the
segments the cost model picks and the analysis time.

Matrices: 2-D grid Laplacians, a random sparse symmetric matrix, and an MPC-shaped KKT matrix
(``N`` stages of ``nx = 4``, ``nu = 2`` with dynamics couplings), stage-ordered.

Usage: ``python pr5_symbolic.py``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from scaly.linalg import analyze


def grid(k: int) -> sparse.csr_array:
  t = sparse.diags_array([-np.ones(k - 1), 4 * np.ones(k), -np.ones(k - 1)], offsets=[-1, 0, 1])
  e = sparse.diags_array([np.ones(k - 1), np.ones(k - 1)], offsets=[-1, 1])
  return sparse.csr_array(sparse.kron(sparse.eye_array(k), t) - sparse.kron(e, sparse.eye_array(k)))


def mpc_kkt(stages: int, nx: int = 4, nu: int = 2) -> sparse.csr_array:
  """``[[H, C^T], [C, 0]]`` in stage order: per stage the state, the input, then the next dynamics row block."""
  rng = np.random.default_rng(stages)
  z = stages * (nx + nu) + nx
  h = sparse.block_diag([sparse.eye_array(nx + nu)] * stages + [sparse.eye_array(nx)])
  rows = []
  for k in range(stages):
    blk = sparse.lil_array((nx, z))
    off = k * (nx + nu)
    blk[:, off : off + nx] = rng.standard_normal((nx, nx))
    blk[:, off + nx : off + nx + nu] = rng.standard_normal((nx, nu))
    blk[:, off + nx + nu : off + 2 * nx + nu] = -np.eye(nx)
    rows.append(blk)
  c = sparse.vstack(rows)
  kkt = sparse.bmat([[h, c.T], [c, -1e-6 * sparse.eye_array(c.shape[0])]], format="csr")
  # Stage order: interleave each stage's variables with its dynamics rows.
  order = []
  for k in range(stages):
    order += list(range(k * (nx + nu), (k + 1) * (nx + nu))) + list(range(z + k * nx, z + (k + 1) * nx))
  order += list(range(stages * (nx + nu), z))
  p = np.array(order)
  return sparse.csr_array(kkt[p][:, p])


def main() -> None:
  rng = np.random.default_rng(0)
  rand = sparse.random_array((3000, 3000), density=3.0 / 3000, random_state=rng, format="csr")
  cases = {"grid 50x50": grid(50), "grid 100x100": grid(100), "random n=3000": sparse.csr_array(abs(rand) + abs(rand).T + sparse.eye_array(3000)), "mpc N=100": mpc_kkt(100), "mpc N=1000": mpc_kkt(1000)}
  for name, a in cases.items():
    coo = sparse.coo_array(sparse.tril(a))
    for method in ("natural", "rcm", "mmd"):
      t0 = time.perf_counter()
      try:
        s = analyze(a.shape, coo.row, coo.col, method)
      except ValueError as err:
        print(f"{name:<14} {method:<7} refused: {str(err)[:90]}", flush=True)
        continue
      t1 = time.perf_counter()
      segs = s.segments()
      t2 = time.perf_counter()
      st = s.stats()
      ref = ""
      if method == "mmd":
        m = sparse.csc_array(abs(a) + sparse.diags_array(np.full(a.shape[0], a.shape[0] + 1.0)))
        lu = splu(m, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        ref = f" superlu_nnzL={lu.L.nnz - a.shape[0]}"
      padded = sum(g.length * g.u for g in segs)
      print(
        f"{name:<14} {method:<7} n={st['n']:>6} nnzL={st['nnz_l']:>8} lanes={st['update_lanes']:>9} height={st['height']:>5} snodes={st['supernodes']:>5} "
        f"max_u={st['max_update']:>6} segs={len(segs):>4} padded_u/lanes={padded / max(st['update_lanes'], 1):.2f} analyze={1e3 * (t1 - t0):>6.0f}ms segments={1e3 * (t2 - t1):>5.0f}ms{ref}",
        flush=True,
      )


if __name__ == "__main__":
  main()
