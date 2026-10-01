"""The dense IPM backend's condensed matrix assembled two ways, and its Cholesky: time per call from Python.

    uv run internal/notes/perf_2026_09_30_gaps/assembly.py PRIMALC2 DUAL3 ex_dense LOTSCHD DUALC2

``sparse`` is every product through the patterns' index tables (the form before C-216), ``dense``
every matrix as a dense array (the prototype C-216's rule, ``kkt.dense_rows``, came from).
``results/assembly_c216.txt`` holds its output; the rule's floors came from the solver itself
(``results/ipm_dense_c216_rule1.md`` and ``_rule2.md``, the rule's first two versions).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "internal/notes/perf_2026_09_27_ipm_speed"))

import numpy as np  # noqa: E402
from gen import problem  # noqa: E402

import scaly as sc  # noqa: E402
from scaly.ir.expr import put_add  # noqa: E402
from scaly.linalg import cholesky  # noqa: E402
from scaly.opt.ipm.kkt import Matrices  # noqa: E402
from tests.opt.ipm.problems import ipm_inputs  # noqa: E402


def best(fn, args, floor=0.0):
  fn._flat_numerical_call(*args)
  t = []
  for _ in range(7):
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.05:
      fn._flat_numerical_call(*args)
      n += 1
    t.append((time.perf_counter() - t0) / n)
  return min(t) * 1e6 - floor


x0 = sc.sym("x0", 1)
empty = sc.Function.from_exprs("asm_empty", [x0], [x0 * 2.0], ["x"], ["y"])
floor = best(empty, [np.ones(1)])
print(f"call overhead {floor:.2f} us")
for name in sys.argv[1:]:
  s, values = ipm_inputs(problem(name))
  rng = np.random.default_rng(0)
  P, A, G = sc.sym("P", len(s.P_rows)), sc.sym("A", len(s.A_rows)), sc.sym("G", len(s.G_rows))
  xr, dr, zr = sc.sym("xr", s.n), sc.sym("dr", ()), sc.sym("zr", max(s.m, 1))
  mats = Matrices(s, P, A, G)
  c = mats.full_P().add_diagonal(xr)
  if s.p:
    c = c + (mats.A.T @ mats.A) * (1.0 / dr)
  if s.m:
    c = c + mats.G.T @ mats.G.scale_rows(1.0 / zr[: s.m])
  sparse_c = c.to_dense()
  n = s.n
  dense = put_add(mats.full_P().to_dense().reshape((n * n,)), np.arange(n) * (n + 1), xr).reshape((n, n))
  if s.p:
    Ad = mats.A.to_dense()
    dense = dense + (Ad.T @ Ad) * (1.0 / dr)
  if s.m:
    Gd = mats.G.to_dense()
    dense = dense + (Gd.T * (1.0 / zr[: s.m])) @ Gd
  ins = [P, A, G, xr, dr, zr]
  names = ["P", "A", "G", "xr", "dr", "zr"]
  f_sparse = sc.Function.from_exprs(f"asm_sparse_{name}", ins, [sparse_c], names, ["c"])
  f_dense = sc.Function.from_exprs(f"asm_dense_{name}", ins, [dense], names, ["c"])
  cm = sc.sym("c", (n, n))
  f_chol = sc.Function.from_exprs(f"asm_chol_{name}", [cm], [cholesky(cm)], ["c"], ["l"])
  args = [
    np.asarray(values["P"], float),
    np.asarray(values["A"], float),
    np.asarray(values["G"], float),
    np.full(n, 1e-3),
    np.array(1e-3),
    np.full(max(s.m, 1), 0.5),
  ]
  cs, cd = (np.asarray(f._flat_numerical_call(*args)[0]).reshape(n, n) for f in (f_sparse, f_dense))
  err = float(np.max(np.abs(cs - cd)) / max(1.0, float(np.max(np.abs(cs)))))
  spd = cs + n * np.eye(n) * (1.0 + np.abs(cs).max())
  dens = (len(s.P_rows) * 2 / max(1, n * n), len(s.A_rows) / max(1, s.p * n), len(s.G_rows) / max(1, s.m * n))
  print(
    f"{name:10s} n {n:4d} p {s.p:4d} m {s.m:4d} density P {dens[0]:.2f} A {dens[1]:.2f} G {dens[2]:.2f} | sparse {best(f_sparse, args, floor):8.1f} us  dense {best(f_dense, args, floor):8.1f} us  chol {best(f_chol, [spd], floor):8.1f} us  |diff| {err:.1e}",
    flush=True,
  )
