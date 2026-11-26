"""Draw the arrow structure that ``extract_arrow_structure`` finds on ``Psi`` and on its Cholesky factor ``L``.

    uv run examples/piqp_multistage/plot_arrow_structure.py [out_dir] [piqp/tests/data]

For each problem, generates the factorization of ``Psi``, factors it, checks ``L L^T = Psi`` and
that both stay inside the blocks, and saves a figure with three panels: ``Psi``, ``L``, and ``L``
with the entries that differ from the previous stage's in red. The problems are random
multistage QPs and, given the directory, PIQP's own multistage test problems.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy import sparse

# piqp_multistage and piqp_sparse are packages in examples/, one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scaly as sc  # noqa: E402

from piqp_multistage import MultistageKKT  # noqa: E402
from piqp_multistage.problems import QP, banded_qp, mpc_qp  # noqa: E402
from piqp_sparse.data import Structure, make_data  # noqa: E402
from piqp_sparse.utils import F64, INF  # noqa: E402

# categorical slots 1 to 4 of the dataviz palette, one per kind of block, and red for a change
COLORS = {"D": "#2a78d6", "B": "#eb6834", "E": "#1baf7a", "corner": "#eda100"}
LABELS = {"D": "D_i, diagonal", "B": "B_i, coupling to the next block", "E": "E_i, arrow", "corner": "D_N, corner of the arrow"}
INK, MUTED, CHANGED = "#0b0b0b", "#a9a8a2", "#e34948"


def _symmetric(P: sparse.sparray) -> sparse.csr_array:
  """``P`` as PIQP reads it: its upper triangle and the mirror of it."""
  upper = sparse.triu(sparse.csr_array(P))
  return sparse.csr_array(upper + sparse.triu(upper, k=1).T)


def x_reg(qp: QP) -> float:
  """A regularization that makes ``Psi`` positive definite even for an indefinite ``P``, by Gershgorin's bound."""
  P = _symmetric(qp.P)
  diagonal = P.diagonal()
  off = np.asarray(abs(P).sum(axis=1)).ravel() - abs(diagonal)
  return 1.0 + max(0.0, float(np.max(off - diagonal, initial=0.0)))


def factor(kkt: MultistageKKT, qp: QP) -> sparse.csc_array:
  """``L`` from the generated factorization, with ``delta`` and ``z_reg`` at one and ``x_reg`` from `x_reg`."""
  s = kkt.structure
  _, P, A, G = Structure.from_matrices(qp.P, qp.A, qp.G)
  vector = lambda name, size: sc.arg(name, sc.TensorType((size,), F64))  # noqa: E731
  const = lambda value, size: sc.const(np.full(size, value))  # noqa: E731
  reg = x_reg(qp)

  @sc.function(vector("P", s.P.nnz), vector("A", s.A.nnz), vector("G", s.G.nnz), name="piqp_multistage_factor")
  def piqp_multistage_factor(P: sc.Expr, A: sc.Expr, G: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    n, p, m = s.n, s.p, s.m
    data = make_data(s, P, const(0.0, n), A, const(0.0, p), G, const(-1.0, m), const(1.0, m), const(-INF, n), const(INF, n))
    AtA = kkt.kkt.init_AtA(data)
    L, ok = kkt.update_scalings_and_factor(data, AtA, sc.const(1.0), const(reg, n), const(1.0, m))
    return L, ok

  L, ok = piqp_multistage_factor(P, A, G)
  assert bool(np.asarray(ok)), "a pivot of Psi is unusable"
  return sparse.csc_array(L)


def psi(qp: QP) -> sparse.csc_array:
  """``Psi`` as `factor` assembles it, in NumPy."""
  n = qp.P.shape[0]
  return sparse.csc_array(_symmetric(qp.P) + x_reg(qp) * sparse.eye_array(n) + qp.A.T @ qp.A + qp.G.T @ qp.G)


# Stages.


def _origin(kkt: MultistageKKT, kind: str, i: int) -> tuple[int, int]:
  """The top left corner of block ``kind`` of stage ``i``, whether or not the stage stores that block."""
  start, diag, _ = kkt.block_info[i]
  rows = {"D": start, "B": start + diag, "E": kkt.structure.n - kkt.arrow_width}[kind]
  return rows, start


def stage_entries(kkt: MultistageKKT, L: sparse.csc_array) -> list[set[tuple[str, int, int]]]:
  """For each stage, a block other than the corner, the stored entries of ``L`` in its ``D_i``, ``B_i`` and ``E_i``, as ``(kind, row, col)`` within each block."""
  stored = np.tril(sparse.csc_array(L).toarray() != 0)
  stages: list[set[tuple[str, int, int]]] = [set() for _ in range(len(kkt.block_info) - 1)]
  for block in kkt.blocks:
    if block.index == len(kkt.block_info) - 1:
      continue
    rows, cols = np.nonzero(stored[block.rows.start : block.rows.stop, block.cols.start : block.cols.stop])
    stages[block.index] |= {(block.kind, int(r), int(c)) for r, c in zip(rows, cols, strict=True)}
  return stages


def stage_changes(kkt: MultistageKKT, L: sparse.csc_array) -> tuple[np.ndarray, np.ndarray, list[int]]:
  """The entries of each stage of ``L`` absent from the previous stage, those of the previous stage absent from it, as global ``(row, col)`` pairs, and the count per stage."""
  stages = stage_entries(kkt, L)
  added: list[tuple[int, int]] = []
  missing: list[tuple[int, int]] = []
  counts = [0]
  for i in range(1, len(stages)):
    new, gone = stages[i] - stages[i - 1], stages[i - 1] - stages[i]
    counts.append(len(new) + len(gone))
    for target, entries in ((added, new), (missing, gone)):
      for kind, r, c in entries:
        # an entry of a block the previous stage has but this one lacks is drawn where it would sit
        r0, c0 = _origin(kkt, kind, i)
        target.append((r0 + r, c0 + c))
  as_array = lambda xs: np.asarray(xs, dtype=np.int64).reshape(-1, 2)  # noqa: E731
  return as_array(added), as_array(missing), counts


# Figures.


def _frame(ax, kkt: MultistageKKT, n: int, title: str) -> None:
  from matplotlib.patches import Rectangle

  for block in kkt.blocks:
    kind = "corner" if block.kind == "D" and block.index == len(kkt.block_info) - 1 else block.kind
    r, c = block.rows, block.cols
    ax.add_patch(Rectangle((c.start - 0.5, r.start - 0.5), len(c), len(r), facecolor=COLORS[kind], alpha=0.22, edgecolor=COLORS[kind], linewidth=0.8))
  ax.set_xlim(-0.5, n - 0.5)
  ax.set_ylim(n - 0.5, -0.5)
  ax.set_aspect("equal")
  ax.set_title(title, fontsize=10, color=INK)
  ax.tick_params(labelsize=8, colors="#52514e")
  for spine in ax.spines.values():
    spine.set_color("#c3c2b7")


RASTER = 300  # from this size on, entries are pixels of an image, as markers would vanish
MISSING_FILL = "#f4a6a5"  # the previous stage's entries in an image, which cannot draw a hollow marker


def _marker(n: int) -> float:
  return max(0.6, min(6.0, 2.0 * (120 / n) ** 2))


def _entries(ax, rows: np.ndarray, cols: np.ndarray, n: int, color: str, *, scale: float = 1.0, hollow: bool = False) -> None:
  """Mark the entries ``(rows[k], cols[k])``: square markers on a small matrix, pixels of an image on a large one."""
  if n < RASTER:
    if hollow:
      ax.scatter(cols, rows, s=scale * _marker(n), marker="s", facecolors="none", edgecolors=color, linewidths=0.6)
    else:
      ax.scatter(cols, rows, s=scale * _marker(n), marker="s", color=color, linewidths=0)
    return
  from matplotlib.colors import to_rgba

  image = np.zeros((n, n, 4))
  keep = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
  image[rows[keep], cols[keep]] = to_rgba(MISSING_FILL if hollow else color)
  ax.imshow(image, interpolation="nearest", extent=(-0.5, n - 0.5, n - 0.5, -0.5), zorder=3)


def draw(ax, matrix: sparse.csc_array, kkt: MultistageKKT, title: str) -> None:
  lower = sparse.tril(matrix).tocoo()
  n = matrix.shape[0]
  _frame(ax, kkt, n, f"{title}, {lower.nnz} entries")
  _entries(ax, lower.row, lower.col, n, INK)


def draw_changes(ax, L: sparse.csc_array, kkt: MultistageKKT) -> int:
  added, missing, counts = stage_changes(kkt, L)
  lower = sparse.tril(L).tocoo()
  n = L.shape[0]
  changed = sum(counts)
  _frame(ax, kkt, n, f"L, {changed} entries differ from the previous stage")
  _entries(ax, lower.row, lower.col, n, MUTED)
  _entries(ax, added[:, 0], added[:, 1], n, CHANGED, scale=2.5)
  _entries(ax, missing[:, 0], missing[:, 1], n, CHANGED, scale=2.5, hollow=True)
  return changed


def problems(piqp_data: Path | None) -> Iterator[tuple[str, str, QP]]:
  """``(file stem, title, QP)`` for every problem the port was compared on."""
  for horizon, nx, nu, n_global, seed in [(12, 6, 3, 3, 0), (5, 3, 2, 0, 0), (8, 4, 2, 2, 1), (20, 6, 3, 3, 2), (10, 2, 1, 0, 3), (40, 5, 2, 1, 4)]:
    title = f"mpc_qp: {horizon} stages, nx = {nx}, nu = {nu}, {n_global} shared parameters, seed {seed}"
    yield f"mpc_qp_{horizon}_{nx}_{nu}_{n_global}_{seed}", title, mpc_qp(horizon, nx, nu, n_global, seed=seed)
  for n, bandwidth, density, n_global, seed in [(16, 4, 0.3, 2, 3), (16, 2, 0.3, 0, 2)]:
    title = f"banded_qp: n = {n}, bandwidth {bandwidth}, density {density}, {n_global} shared, seed {seed}"
    yield f"banded_qp_{n}_{bandwidth}_{density}_{n_global}_{seed}", title, banded_qp(n, bandwidth, density, n_global, seed=seed)
  if piqp_data is None:
    return
  from scipy import io

  for path in sorted(piqp_data.glob("*.mat")):
    mat = io.loadmat(path)
    if mat["P"].shape[0] > 2000:
      continue
    matrix = lambda key: sparse.csc_array(mat[key]) if mat[key].size else sparse.csc_array(mat[key].shape)  # noqa: E731
    vector = lambda key: np.asarray(mat[key], dtype=np.float64).ravel()  # noqa: E731
    qp = QP(matrix("P"), vector("c"), matrix("A"), vector("b"), matrix("G"), vector("h_l"), vector("h_u"), vector("x_l"), vector("x_u"))
    yield path.stem, f"PIQP's {path.stem}", qp


def plot(stem: str, title: str, qp: QP, out_dir: Path) -> Path:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch

  structure, *_ = Structure.from_matrices(qp.P, qp.A, qp.G)
  kkt = MultistageKKT(structure)
  Psi, L = psi(qp), factor(kkt, qp)
  assert np.allclose((L @ L.T).toarray(), Psi.toarray(), atol=1e-9 * abs(Psi).max())
  mask = kkt.block_mask()
  for name, matrix in (("Psi", Psi), ("L", L)):
    lower = sparse.tril(matrix).tocoo()
    assert mask[lower.row, lower.col].all(), f"{name} has entries outside the blocks"

  fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), facecolor="#fcfcfb")
  draw(axes[0], Psi, kkt, "Psi")
  draw(axes[1], L, kkt, "Cholesky factor L, natural order")
  changed = draw_changes(axes[2], L, kkt)
  handles: list[Patch | Line2D] = [Patch(facecolor=COLORS[k], alpha=0.5, edgecolor=COLORS[k], label=LABELS[k]) for k in COLORS]
  handles += [
    Line2D([], [], marker="s", linestyle="", color=CHANGED, label="in this stage, not the previous"),
    Line2D([], [], marker="s", linestyle="", markerfacecolor="none", markeredgecolor=CHANGED, label="in the previous stage, not this")
    if structure.n < RASTER
    else Line2D([], [], marker="s", linestyle="", color=MISSING_FILL, label="in the previous stage, not this"),
  ]
  fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)
  stages = len(kkt.block_info) - 1
  fig.suptitle(f"{title}\nn = {structure.n}, {stages} stages and an arrow {kkt.arrow_width} wide", fontsize=12, color=INK)
  fig.tight_layout(rect=(0, 0.05, 1, 0.97))
  out = out_dir / f"{stem}.png"
  fig.savefig(out, dpi=130 if structure.n < RASTER else 220)
  plt.close(fig)
  print(f"{stem:32s} n = {structure.n:4d}  stages = {stages:3d}  arrow = {kkt.arrow_width:3d}  changed entries = {changed:5d}  -> {out}")
  return out


def main(out_dir: Path, piqp_data: Path | None) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for stem, title, qp in problems(piqp_data):
    plot(stem, title, qp, out_dir)


if __name__ == "__main__":
  main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("arrow_structure"), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
