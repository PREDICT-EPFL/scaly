"""The Bratu problem: a nonlinear PDE by Newton's method on its automatically sparse Jacobian.

Solid-fuel ignition gives the Bratu problem on the unit square,

    -Laplace(u) = lambda exp(u)   in (0, 1)^2,   u = 0 on the boundary,

which has a solution only for ``lambda`` below a turning point near 6.808. With the five-point
stencil on an ``n x n`` interior grid the residual is written as slices of a zero-padded array, the
way one would write it in NumPy. Nothing about the stencil is declared, yet:

* ``sc.jacobian_sparsity`` finds the five-point pattern from the graph alone, and
  ``sc.column_coloring`` colors it with the same handful of colors (7, greedily) whatever the grid
  size, so the compact Jacobian (``sc.sparse_jacobian``) costs seven forward sweeps rather than
  ``n^2``;
* ``SparseMatrix.from_sparse_jacobian`` turns the compact values into a sparse matrix that the
  generated sparse ``L D L^T`` (``linalg.SparseLDL``, with its fill-reducing ordering chosen when
  the graph is built) factors inside the Newton loop;
* the Newton loop is a ``sc.while_loop`` with ``lambda`` as a loop parameter, so the whole solve is
  one C function. Natural-parameter continuation in ``lambda`` calls it repeatedly, warm-started.

``fact.inertia()`` confirms that the Jacobian stays positive definite on the lower solution
branch. The generated C lands in ``examples/generated/bratu_newton/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "bratu_newton"
N_GRID = 40
TOL, MAX_NEWTON = 1e-10, 30
LAMBDAS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 6.7, 6.78])


def residual(u: sc.Expr, lam: sc.Expr, n: int) -> sc.Expr:
  h = 1.0 / (n + 1)
  grid = u.reshape((n, n))
  col, row = sc.const(np.zeros((n, 1))), sc.const(np.zeros((1, n + 2)))
  padded = sc.concat([row, sc.concat([col, grid, col], axis=1), row])
  lap = (4 * grid - padded[:-2, 1:-1] - padded[2:, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:]) / h**2
  return (lap - lam * grid.exp()).reshape((n * n,))


def newton_functions(n: int) -> tuple[sc.Function, sc.Function]:
  nn = n * n

  @sc.function(sc.G(sc.L("u", nn), sc.L("lam", ())), sc.G(sc.L("u_next", nn), sc.L("inertia", 3)))
  def newton_step(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr]:
    u, lam = inputs
    f = residual(u, lam, n)
    jac = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_jacobian(f, u))
    fact = linalg.SparseLDL(jac, name="bratu")
    return u - fact.solve(f), fact.inertia()

  @sc.function(sc.G(sc.L("carry", nn + 1), sc.L("lam", ())), sc.L("carry_next", nn + 1))
  def body(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    carry, lam = inputs
    u_next, _ = newton_step((carry[:nn], lam))
    return sc.concat([u_next, sc.norm_inf(residual(u_next, lam, n)).reshape((1,))])

  @sc.function(sc.G(sc.L("carry", nn + 1), sc.L("lam", ())), sc.L("go_on", ...))
  def not_converged(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    carry, _ = inputs
    return sc.greater(carry[nn], TOL)

  @sc.function(sc.G(sc.L("u0", nn), sc.L("lam", ())), sc.G(sc.L("u", nn), sc.L("residual", ()), sc.L("iterations", ()), sc.L("inertia", 3)))
  def bratu_solve(inputs: tuple[sc.Expr, sc.Expr]) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    u0, lam = inputs
    start = sc.concat([u0, sc.norm_inf(residual(u0, lam, n)).reshape((1,))])
    carry, n_iter = sc.while_loop(not_converged, body, start, max_iter=MAX_NEWTON, params=(lam,))
    _, inertia = newton_step((carry[:nn], lam))
    return carry[:nn], carry[nn], n_iter, inertia

  return newton_step, bratu_solve


def coloring_table(sizes: tuple[int, ...] = (10, 20, 40, 80)) -> list[tuple[int, int, int]]:
  """``(n^2, nnz, colors)`` of the residual Jacobian for a few grid sizes."""
  table = []
  for n in sizes:
    u, lam = sc.sym("u", n * n), sc.sym("lam", ())
    pattern = sc.jacobian_sparsity(residual(u, lam, n), u)
    table.append((n * n, pattern.nnz, int(max(sc.column_coloring(pattern))) + 1))
  return table


def reference_solve(lam: float, n: int = N_GRID) -> np.ndarray:
  """Newton with SciPy's sparse LU, for the check."""
  h = 1.0 / (n + 1)
  t = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), dtype=float)
  lap = (sparse.kron(sparse.eye(n), t) + sparse.kron(t, sparse.eye(n))) / h**2
  u = np.zeros(n * n)
  for _ in range(MAX_NEWTON):
    f = lap @ u - lam * np.exp(u)
    if np.abs(f).max() < TOL:
      break
    u -= spla.spsolve((lap - sparse.diags(lam * np.exp(u))).tocsc(), f)
  return u


def main() -> dict:
  _, bratu_solve = newton_functions(N_GRID)
  u = np.zeros(N_GRID * N_GRID)
  rows = []
  for lam in LAMBDAS:
    u, res, iterations, inertia = bratu_solve((u, np.array(lam)))
    rows.append((lam, float(u.max()), float(res), int(iterations), tuple(int(v) for v in inertia)))
  u_ref = reference_solve(6.0)
  u6, *_ = bratu_solve((np.zeros(N_GRID * N_GRID), np.array(6.0)))
  return {"continuation": rows, "reference_error": np.abs(u6 - u_ref).max(), "coloring": coloring_table(), "solver": bratu_solve}


if __name__ == "__main__":
  out = main()
  print("the residual's Jacobian, found from the graph:")
  for size, nnz, colors in out["coloring"]:
    print(f"  {size:>5} unknowns: {nnz:>6} nonzeros, {colors} colors")
  print(f"continuation in lambda on a {N_GRID} x {N_GRID} grid (one generated Newton solve per step, warm-started):")
  for lam, umax, res, iterations, inertia in out["continuation"]:
    print(f"  lambda = {lam:4.2f}: max u = {umax:.4f}, |F| = {res:.1e} after {iterations} Newton steps, pivots (+, -, 0) = {inertia}")
  print(f"lambda = 6 from zero against a SciPy sparse-LU Newton: {out['reference_error']:.1e}")
  write_module(out["solver"], GENERATED)
  print(f"generated C in {GENERATED}")
