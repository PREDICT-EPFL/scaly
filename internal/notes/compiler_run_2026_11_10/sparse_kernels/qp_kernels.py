"""Sparse kernels on a Maros-Meszaros QP: KKT assembly, products and derivatives in generated C.

Run from a checkout of main: uv run internal/notes/compiler_run_2026_11_10/sparse_kernels/qp_kernels.py
"""

from pathlib import Path

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function

ROOT = Path(__file__).resolve().parents[4]
with np.load(ROOT / "tests/integration/data/maros_meszaros/CVXQP1_S.npz") as raw:

  def matrix(key):
    return sparse.csc_array((raw[f"{key}_data"], raw[f"{key}_indices"], raw[f"{key}_indptr"]), shape=tuple(raw[f"{key}_shape"]))

  P, A = matrix("P"), matrix("A")
m, n = A.shape


def leaf(name, mat):
  pattern = sc.SparsityPattern(mat.shape, *map(tuple, mat.tocoo().coords)).canonicalize()[0]
  return sc.arg(name, sc.TensorType(mat.shape, pattern=pattern), layout="csc")


def diagonal(values, size):
  return sc.pack(values, sc.SparsityPattern((size, size), tuple(range(size)), tuple(range(size))))


@sc.function(leaf("P", P), leaf("A", A), sc.arg("x", n), sc.arg("y", m), sc.arg("w", m), sc.arg("sigma", ()), sc.arg("rho", ()), name="qp_kernels")
def qp(P, A, x, y, w, sigma, rho):
  kkt = sc.concat(
    (
      sc.concat((P + sigma * sc.const(sparse.eye_array(n, format="csc")), A.T), axis=1),
      sc.concat((A, -(1.0 / rho) * sc.const(sparse.eye_array(m, format="csc"))), axis=1),
    ),
    axis=0,
  )
  return kkt, P @ x, A @ x, A.T @ y, A.T @ diagonal(w, m) @ A


@sc.function(leaf("A", A), sc.arg("x", n), sc.arg("y", m), outputs=sc.arg("f"), name="bilinear")
def bilinear(A, x, y):
  return y @ (A @ x)


rng = np.random.default_rng(0)
x, y, w = rng.standard_normal(n), rng.standard_normal(m), rng.uniform(0.5, 2.0, m)
sigma, rho = 1e-6, 0.1
kkt, px, ax, aty, atwa = qp(P, A, x, y, w, sigma, rho)

reference_kkt = sparse.block_array([[P + sigma * sparse.eye_array(n), A.T], [A, -(1 / rho) * sparse.eye_array(m)]], format="csc")
print(f"CVXQP1_S: P {P.shape} nnz {P.nnz}, A {A.shape} nnz {A.nnz}")
print(f"KKT {kkt.shape} nnz {kkt.nnz} (dense would be {kkt.shape[0] * kkt.shape[1]}), max |K - SciPy| = {abs(kkt - reference_kkt).max():.1e}")
for label, got, want in (("P x", px, P @ x), ("A x", ax, A @ x), ("A^T y", aty, A.T @ y)):
  print(f"{label:<6} max error {np.max(np.abs(got - want)):.1e}")
reference_atwa = (A.T @ sparse.diags_array(w) @ A).tocsc()
print(f"A^T diag(w) A: nnz {atwa.nnz} (SciPy {reference_atwa.nnz}), max error {abs(atwa - reference_atwa).max():.1e}")

grad = sc.gradient(bilinear, of="f", wrt="A", name="grad_bilinear")
g = sparse.csc_array(grad(A, x, y))
coo = A.tocoo()
expected = sparse.csc_array((y[coo.coords[0]] * x[coo.coords[1]], coo.coords), shape=A.shape)
print(f"d(y'Ax)/dA: nnz {g.nnz} (A has {A.nnz}), max error {abs(g - expected).max():.1e}")

eps, direction = 1e-6, rng.standard_normal(A.nnz)
plus, minus = A.copy(), A.copy()
plus.data += eps * direction
minus.data -= eps * direction
fd = (float(bilinear(plus, x, y)) - float(bilinear(minus, x, y))) / (2 * eps)
analytic = float(np.sum(sparse.csc_array((direction, A.indices, A.indptr), shape=A.shape).multiply(g)))
print(f"directional derivative in A's values: finite difference {fd:.8f}, gradient {analytic:.8f}")

for fn, label in ((qp, "qp"), (grad, "gradient")):
  program = lower_function(fn)
  buffers = [
    int(np.prod(node.attrs["shape"])) for node in walk_program(program) if node.op == ProgramOp.BUFFER and node.attrs["address_space"] == "private"
  ]
  source = render_c_source(fn)
  print(f"{label:<9} largest private buffer {max(buffers, default=0)} entries (m*n = {m * n}, n*n = {n * n}); C source {len(source)} bytes")
