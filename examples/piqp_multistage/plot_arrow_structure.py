"""Draw the arrow structure that ``extract_arrow_structure`` finds on ``Psi`` and on its Cholesky factor ``L``.

    uv run examples/piqp_multistage/plot_arrow_structure.py [out.png]

Generates the factorization of ``Psi`` for a random multistage QP, factors it, checks
``L L^T = Psi`` and that both stay inside the blocks, and saves the figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

# piqp_multistage and piqp_sparse are packages in examples/, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scaly as sc  # noqa: E402

from piqp_multistage import MultistageKKT  # noqa: E402
from piqp_multistage.problems import QP, mpc_qp  # noqa: E402
from piqp_sparse.data import Structure, make_data  # noqa: E402
from piqp_sparse.utils import F64, INF  # noqa: E402

# categorical slots 1 to 4 of the dataviz palette, one per kind of block
COLORS = {"D": "#2a78d6", "B": "#eb6834", "E": "#1baf7a", "corner": "#eda100"}
LABELS = {"D": "D_i, diagonal", "B": "B_i, coupling to the next block", "E": "E_i, arrow", "corner": "D_N, corner of the arrow"}


def factor(kkt: MultistageKKT, qp: QP) -> sparse.csc_array:
  """``L`` from the generated factorization, with ``delta``, ``x_reg`` and ``z_reg`` at one."""
  s = kkt.structure
  _, P, A, G = Structure.from_matrices(qp.P, qp.A, qp.G)
  vector = lambda name, size: sc.arg(name, sc.TensorType((size,), F64))  # noqa: E731
  const = lambda value, size: sc.const(np.full(size, value))  # noqa: E731

  @sc.function(vector("P", s.P.nnz), vector("A", s.A.nnz), vector("G", s.G.nnz), name="piqp_multistage_factor")
  def piqp_multistage_factor(P: sc.Expr, A: sc.Expr, G: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    n, p, m = s.n, s.p, s.m
    data = make_data(s, P, const(0.0, n), A, const(0.0, p), G, const(-1.0, m), const(1.0, m), const(-INF, n), const(INF, n))
    AtA = kkt.kkt.init_AtA(data)
    L, ok = kkt.update_scalings_and_factor(data, AtA, sc.const(1.0), const(1.0, n), const(1.0, m))
    return L, ok

  L, ok = piqp_multistage_factor(P, A, G)
  assert bool(np.asarray(ok)), "a pivot of Psi is unusable"
  return sparse.csc_array(L)


def psi(qp: QP) -> sparse.csc_array:
  """``Psi`` with ``delta``, ``x_reg`` and ``z_reg`` at one, in NumPy."""
  n = qp.P.shape[0]
  return sparse.csc_array(qp.P + sparse.eye_array(n) + qp.A.T @ qp.A + qp.G.T @ qp.G)


def draw(ax, matrix: sparse.csc_array, kkt: MultistageKKT, title: str) -> None:
  from matplotlib.patches import Rectangle

  for block in kkt.blocks:
    kind = "corner" if block.kind == "D" and block.index == len(kkt.block_info) - 1 else block.kind
    r, c = block.rows, block.cols
    ax.add_patch(Rectangle((c.start - 0.5, r.start - 0.5), len(c), len(r), facecolor=COLORS[kind], alpha=0.22, edgecolor=COLORS[kind], linewidth=1.0))
  lower = sparse.tril(matrix).tocoo()
  ax.scatter(lower.col, lower.row, s=1.5, marker="s", color="#0b0b0b", linewidths=0)
  n = matrix.shape[0]
  ax.set_xlim(-0.5, n - 0.5)
  ax.set_ylim(n - 0.5, -0.5)
  ax.set_aspect("equal")
  ax.set_title(f"{title}, {lower.nnz} entries in the lower triangle", fontsize=10, color="#0b0b0b")
  ax.tick_params(labelsize=8, colors="#52514e")
  for spine in ax.spines.values():
    spine.set_color("#c3c2b7")


def main(out: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.patches import Patch

  qp = mpc_qp(horizon=12, nx=6, nu=3, n_global=3, seed=0)
  structure, *_ = Structure.from_matrices(qp.P, qp.A, qp.G)
  kkt = MultistageKKT(structure)

  Psi, L = psi(qp), factor(kkt, qp)
  assert np.allclose((L @ L.T).toarray(), Psi.toarray(), atol=1e-9 * abs(Psi).max())
  mask = kkt.block_mask()
  for name, matrix in (("Psi", Psi), ("L", L)):
    lower = sparse.tril(matrix).tocoo()
    assert mask[lower.row, lower.col].all(), f"{name} has entries outside the blocks"

  print(f"n = {structure.n}, {len(kkt.block_info) - 1} blocks and an arrow {kkt.arrow_width} wide")
  for info in kkt.block_info:
    print(f"  start {info.start:3d}  diag {info.diag_size:3d}  off-diag {info.off_diag_size:3d}")

  fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), facecolor="#fcfcfb")
  draw(axes[0], Psi, kkt, "Psi")
  draw(axes[1], L, kkt, "Cholesky factor L, natural order")
  fig.legend(handles=[Patch(facecolor=COLORS[k], alpha=0.5, edgecolor=COLORS[k], label=LABELS[k]) for k in COLORS], loc="lower center", ncol=4, frameon=False, fontsize=9)
  fig.suptitle(f"Arrow structure of a 12-stage MPC problem with 3 shared parameters (n = {structure.n})", fontsize=12, color="#0b0b0b")
  fig.tight_layout(rect=(0, 0.05, 1, 0.97))
  fig.savefig(out, dpi=150)
  print(f"saved {out}")


if __name__ == "__main__":
  main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("arrow_structure.png"))
