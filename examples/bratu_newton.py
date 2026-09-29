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
* the equations are an ``sc.roots.root`` with ``lambda`` its parameter, and ``sc.roots.Newton`` with
  ``linear="sparse_ldl"`` solves them: each step turns the compact Jacobian into a
  ``SparseMatrix`` that the generated sparse ``L D L^T`` (``linalg.SparseLDL``, with its
  fill-reducing ordering chosen when the graph is built) factors;
* the Newton iteration is a ``while_loop`` in the generated code, so the whole solve is one C
  function. Natural-parameter continuation in ``lambda`` calls it repeatedly, warm-started.

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


def bratu(n: int) -> sc.roots.Root:
  """The discretized PDE on an ``n x n`` grid as a system of equations in ``u`` with parameter ``lambda``."""

  @sc.roots.root(vars=sc.L("u", n * n), params=sc.L("lam", ()), name="bratu")
  def equations(u: sc.Expr, lam: sc.Expr) -> sc.Expr:
    return residual(u, lam, n)

  return equations


def newton_function(n: int) -> sc.Function:
  """The Newton solve from a warm start, and the pivot signs of ``L D L^T`` of the Jacobian at the solution."""
  solve = sc.roots.solver(bratu(n), sc.roots.Newton(tol=TOL, max_iter=MAX_NEWTON, linear="sparse_ldl"), name="bratu_newton")

  @sc.function(n * n, (), output=sc.G("u", "residual", "iterations", "inertia"))
  def bratu_solve(u0: sc.Expr, lam: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
    u, info = solve(u0, lam)
    jac = linalg.SparseMatrix.from_sparse_jacobian(sc.sparse_jacobian(residual(u, lam, n), u))
    return u, info.residual, info.iter, linalg.SparseLDL(jac, name="bratu").inertia()

  return bratu_solve


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
  bratu_solve = newton_function(N_GRID)
  u = np.zeros(N_GRID * N_GRID)
  rows = []
  for lam in LAMBDAS:
    u, res, iterations, inertia = bratu_solve(u, np.array(lam))
    rows.append((lam, float(u.max()), float(res), int(iterations), tuple(int(v) for v in inertia)))
  u_ref = reference_solve(6.0)
  u6, *_ = bratu_solve(np.zeros(N_GRID * N_GRID), np.array(6.0))
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
