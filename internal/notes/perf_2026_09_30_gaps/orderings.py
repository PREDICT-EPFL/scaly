"""C-226 (Tier 7): does the ordering alone move a problem's path? The sparse IPM on one problem under
each ordering scaly has (natural, reverse Cuthill-McKee, minimum degree) and under minimum degree on
the graph relabelled at random, which breaks its ties another way: the fill, the factorization's
multiply-adds, the status and the iteration count.

  uv run internal/notes/perf_2026_09_30_gaps/orderings.py QRECIPE [relabellings]

Run it on the tree before C-226 (a worktree's src first on PYTHONPATH, --no-sync) to see
the stall; after it every ordering takes PIQP's count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import scaly as sc
from scaly.ir.expr import Expr
from scaly.linalg.symbolic import analyze, ordering
from scaly.opt.ipm import QPValues, Solver
from scaly.opt.ipm import kkt
from tests.opt.ipm.problems import ipm_inputs, maros_meszaros


def run(name, label, perm=None, method=None):
  qp = maros_meszaros(name)
  s, values = ipm_inputs(qp)
  kernels = kkt.Kernels(s, "sparse")
  mats, _ = kernels.matrices(Expr.sym("D", (sum(kernels.d_sizes),)))
  matrix = kernels.kkt_matrix(mats, Expr.sym("x_reg", (s.n,)), Expr.sym("delta_reg", ()), Expr.sym("z_reg_ir", (s.m,)))
  rows, cols = matrix.coordinates()
  sym = analyze(matrix.shape, rows, cols, method or "mmd", perm=perm)
  kkt._SYMBOLIC[s] = sym
  order = list(values)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in order}
  fname = f"ord_{name.lower()}_{label}"
  res = Solver(s, "sparse", name=fname).solve(QPValues.preprocess(s, **syms))
  keys = ["x", "status", "iter"]
  fn = sc.Function.from_exprs(fname, [syms[k] for k in order], [res[k] for k in keys], order, keys).concrete
  out = fn._flat_numerical_call(*[np.asarray(values[k], dtype=float).reshape(np.shape(values[k])) for k in order])
  print(
    f"{name:10s} {label:12s} n {matrix.shape[0]:5d} nnz(L) {sym.nnz_l:7d} lanes {sym.update_lanes:9d} status {float(np.asarray(out[1]).ravel()[0]):.0f} iter {float(np.asarray(out[2]).ravel()[0]):.0f}",
    flush=True,
  )
  return matrix.shape[0], rows, cols


if __name__ == "__main__":
  name = sys.argv[1]
  n, rows, cols = run(name, "mmd", method="mmd")
  for method in ("natural", "rcm"):
    try:
      run(name, method, method=method)
    except Exception as e:
      print(method, "failed:", repr(e)[:200])
  rng = np.random.default_rng(0)
  base = ordering(n, rows, cols, "mmd")
  for k in range(int(sys.argv[2]) if len(sys.argv) > 2 else 3):
    # the same elimination order on a relabelled graph: MMD's ties break differently
    relabel = rng.permutation(n)
    inv = np.argsort(relabel)
    perm = inv[ordering(n, relabel[rows], relabel[cols], "mmd")]
    try:
      run(name, f"mmd_tie{k}", perm=perm)
    except Exception as e:
      print("tie", k, "failed:", repr(e)[:200])
