"""The multistage KKT system, PIQP 0.7.0's ``sparse::MultistageKKT``, up to the factorization.

The condensed matrix is ``piqp_sparse``'s,

    Psi = P + diag(x_reg) + A^T A / delta + G^T diag(z_reg)^-1 G,

kept in the given order of the variables instead of an AMD ordering. For a problem written stage
after stage, ``Psi`` is then block tridiagonal with an arrow: the variables coupled to every stage,
such as a parameter shared by all stages, collect in a dense border at the end. PIQP's
``extract_arrow_structure`` finds that structure from the pattern of ``Psi`` alone:

    [D_1                                  ]
    [B_1 D_2                              ]
    [    B_2 D_3                          ]
    [            ...                      ]
    [                B_N-3 D_N-2          ]
    [                      B_N-2 D_N-1    ]
    [E_1 E_2 E_3 ... E_N-3 E_N-2 E_N-1 D_N]

and a Cholesky factor of ``Psi`` in this order keeps it: the fill of ``L`` stays inside the blocks.
As in PIQP, the structure depends only on the patterns and is decided when the solver is
generated. Here the factorization is still the general sparse one, ``scaly.linalg.cholesky`` in
natural order; the block recursion over the stages comes next.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import scaly as sc
from scaly import linalg

from piqp_sparse.data import Data, Structure
from piqp_sparse.kkt import KKT
from piqp_sparse.utils import all_true

if TYPE_CHECKING:
  from scaly.ir.types import CanonicalPattern


class BlockInfo(NamedTuple):
  """PIQP's ``BlockInfo``: a diagonal block of ``diag_size`` variables from ``start``, coupled to the first ``off_diag_size`` variables of the next block."""

  start: int
  diag_size: int
  off_diag_size: int


class Factor(NamedTuple):
  """``Psi``'s factorization ``Psi = L L^T`` in the given order and whether every pivot is usable."""

  L: sc.Expr
  ok: sc.Expr


class Block(NamedTuple):
  """One block of the lower triangle: ``kind`` is ``"D"``, ``"B"`` or ``"E"``, ``index`` counts from zero, and the block spans ``rows`` by ``cols``."""

  kind: str
  index: int
  rows: range
  cols: range


# PIQP counts flops in `usize`, an unsigned 64-bit integer, and compares differences of such
# counts. The port keeps that arithmetic, wrap-around included, so that it takes PIQP's decisions.
_USIZE = 1 << 64


def _usize(x: int) -> int:
  return x % _USIZE


def flops_gemm(m: int, n: int, k: int) -> int:
  """``A * B``, ``A`` in ``R^{m x k}``, ``B`` in ``R^{k x n}``."""
  return _usize(2 * _usize(m) * _usize(n) * _usize(k))


def flops_trsm(m: int, n: int) -> int:
  """``A^-1 * B``, ``A`` in ``R^{m x m}`` triangular, ``B`` in ``R^{m x n}``."""
  return _usize(_usize(m) * _usize(m) * _usize(n))


def flops_syrk(n: int, k: int) -> int:
  """``C + A * A^T``, ``C`` in ``R^{n x n}`` triangular, ``A`` in ``R^{n x k}``."""
  return _usize(_usize(n) * _usize(n) * _usize(k))


def flops_potrf(n: int) -> int:
  """``chol(A)``, ``A`` in ``R^{n x n}``."""
  return _usize(_usize(n) * _usize(n) * _usize(n)) // 3


@dataclass
class _Cursor:
  """PIQP's ``block_structure_info``: the block being grown and the arrow so far."""

  prev_diag_block_size: int = 0
  diag_block_start: int = 0
  diag_block_size: int = 0
  off_diag_block_size: int = 0
  arrow_width: int = 0


def extract_arrow_structure(pattern: CanonicalPattern) -> tuple[BlockInfo, ...]:
  """PIQP's ``extract_arrow_structure``: the blocks of a symmetric pattern, the last one the corner of the arrow.

  ``pattern`` is the upper triangle of ``Psi``, row-major, with its diagonal stored. PIQP walks
  the lower triangle by columns, which visits the same entries in the same order. Row by row it
  grows the current diagonal block to cover each entry right of it, or, when that costs fewer
  flops, widens the arrow so that the entry falls in the border instead. It closes a block once
  the diagonal part is at least twice the coupling to the next one, once the next row would grow
  it, or at the arrow. Blocks split in two are merged at the end, which saves kernel calls and no
  flops.
  """
  n = pattern.shape[0]
  rows = np.asarray(pattern.rows, dtype=np.int64)
  cols = np.asarray(pattern.cols, dtype=np.int64)
  ptr = np.searchsorted(rows, np.arange(n + 1))
  row_cols = [cols[ptr[i] : ptr[i + 1]].tolist() for i in range(n)]

  # flop count of the tridiagonal factorization
  flops_tridiag = 0
  # flop counts of the arrow's factorization per column of the arrow, without and only with the syrk operations
  flops_arrow_normalized_no_syrk = 0
  flops_arrow_normalized_syrk = 0

  def next_block_structure(row: int, current: _Cursor) -> _Cursor:
    nxt = replace(current)
    for col in row_cols[row]:
      if col >= nxt.diag_block_start and col + nxt.arrow_width < n:
        current_block_size = nxt.diag_block_size + nxt.off_diag_block_size
        new_block_size = max(col - nxt.diag_block_start + 1, current_block_size)
        # the diagonal block is limited to the current block's height, and at least half the block
        max_diag_block_size = row - nxt.diag_block_start + 1
        new_min_diag_block_size = max(nxt.diag_block_size, (new_block_size + 1) // 2)
        new_diag_block_size = max(new_min_diag_block_size, max_diag_block_size)
        new_off_diag_block_size = new_block_size - new_diag_block_size
        remaining_width = n - nxt.diag_block_start - nxt.diag_block_size - nxt.off_diag_block_size
        new_arrow_width = min(max(nxt.arrow_width, n - col), remaining_width)

        # L_i = chol(D_i - C_{i-1} C_{i-1}^T), C_i = B_i L_i^-T
        flops_tridiag_new = (
          flops_tridiag
          + flops_syrk(new_diag_block_size, nxt.prev_diag_block_size)
          + flops_potrf(new_diag_block_size)
          + flops_trsm(new_diag_block_size, new_off_diag_block_size)
        )

        # the dense kernels work on blocks at least 4 wide, so the arrow is counted in multiples of 4
        arrow_width_flops = (nxt.arrow_width + 3) // 4 * 4
        new_arrow_width_flops = (new_arrow_width + 3) // 4 * 4
        flops_arrow = (
          arrow_width_flops * flops_arrow_normalized_no_syrk
          + arrow_width_flops * arrow_width_flops * flops_arrow_normalized_syrk
          + flops_potrf(arrow_width_flops)
        )
        flops_arrow_new = (
          new_arrow_width_flops * flops_arrow_normalized_no_syrk
          + new_arrow_width_flops * new_arrow_width_flops * flops_arrow_normalized_syrk
          # F_i = (E_i - F_{i-1} C_{i-1}^T) L_i^-T
          + flops_gemm(new_arrow_width_flops, nxt.prev_diag_block_size, new_diag_block_size)
          + flops_trsm(new_diag_block_size, new_arrow_width_flops)
          # L_N = chol(D_N - sum F_i F_i^T)
          + flops_syrk(new_arrow_width_flops, new_diag_block_size)
          + flops_potrf(new_arrow_width_flops)
        )

        # grow the diagonal block, or give the entry to the arrow
        if _usize(_usize(flops_tridiag_new) - _usize(flops_tridiag)) <= _usize(_usize(flops_arrow_new) - _usize(flops_arrow)):
          nxt.diag_block_size = new_diag_block_size
          nxt.off_diag_block_size = new_off_diag_block_size
        else:
          nxt.arrow_width = new_arrow_width
    return nxt

  def close_block(cursor: _Cursor) -> None:
    cursor.diag_block_start += cursor.diag_block_size
    cursor.prev_diag_block_size = cursor.diag_block_size
    cursor.diag_block_size = cursor.off_diag_block_size
    cursor.off_diag_block_size = 0

  blocks: list[BlockInfo] = []
  cur = _Cursor()
  for i in range(n):
    cur = next_block_structure(i, cur)
    if i + 1 < cur.diag_block_start + cur.diag_block_size:
      continue

    hit_optimal_ratio = cur.diag_block_size >= 2 * cur.off_diag_block_size
    at_end = i + 1 >= n - cur.arrow_width

    def next_block_grows(i: int = i, cur: _Cursor = cur) -> bool:
      following = next_block_structure(i + 1, cur)
      return following.diag_block_size + following.off_diag_block_size > cur.diag_block_size + cur.off_diag_block_size

    if hit_optimal_ratio or at_end or next_block_grows():
      blocks.append(BlockInfo(cur.diag_block_start, cur.diag_block_size, cur.off_diag_block_size))
      flops_tridiag += flops_syrk(cur.diag_block_size, cur.prev_diag_block_size + 1)
      flops_tridiag += flops_potrf(cur.diag_block_size)
      flops_tridiag += flops_trsm(cur.diag_block_size, cur.off_diag_block_size)
      flops_arrow_normalized_no_syrk += flops_gemm(1, cur.prev_diag_block_size, cur.diag_block_size)
      flops_arrow_normalized_no_syrk += flops_trsm(cur.diag_block_size, 1)
      flops_arrow_normalized_syrk += flops_syrk(1, cur.diag_block_size)
      close_block(cur)

    # the last block before the arrow
    if at_end and cur.diag_block_size > 0:
      assert cur.off_diag_block_size == 0
      blocks.append(BlockInfo(cur.diag_block_start, cur.diag_block_size, cur.off_diag_block_size))
      close_block(cur)

    if at_end:
      break

  # merge the blocks split in two; PIQP's loop steps past a merged block, and so does this one
  i = 0
  while i < len(blocks) - 1:
    if blocks[i].off_diag_size == blocks[i + 1].diag_size and blocks[i + 1].off_diag_size == 0:
      blocks[i] = BlockInfo(blocks[i].start, blocks[i].diag_size + blocks[i].off_diag_size, 0)
      del blocks[i + 1]
    i += 1

  # the last block is the corner of the arrow
  blocks.append(BlockInfo(cur.diag_block_start, cur.arrow_width, 0))
  assert len(blocks) >= 2
  return tuple(blocks)


@dataclass(frozen=True, eq=False)
class MultistageKKT:
  """The multistage KKT system of one QP structure."""

  structure: Structure

  @cached_property
  def kkt(self) -> KKT:
    """``piqp_sparse``'s condensed system, which holds the pattern and the assembly of ``Psi``."""
    return KKT(self.structure)

  @cached_property
  def pattern(self) -> CanonicalPattern:
    """``Psi``'s upper triangle, in the given order of the variables."""
    return self.kkt.pattern

  @cached_property
  def block_info(self) -> tuple[BlockInfo, ...]:
    """PIQP's ``block_info``, which its constructor computes first."""
    return extract_arrow_structure(self.pattern)

  @property
  def arrow_width(self) -> int:
    return self.block_info[-1].diag_size

  @cached_property
  def blocks(self) -> tuple[Block, ...]:
    """The blocks of PIQP's ``BlockKKT``: ``D_i`` on the diagonal, ``B_i`` below ``D_i``, ``E_i`` in the arrow's rows, and the corner ``D_N``.

    As in PIQP's ``construct_kkt_fac``, there is a ``B_i`` for every block but the last two, and
    it spans only the first ``off_diag_size`` rows of the next block.
    """
    n, info = self.structure.n, self.block_info
    arrow = range(n - self.arrow_width, n)
    out = []
    for i, (start, diag, off) in enumerate(info):
      cols = range(start, start + diag)
      out.append(Block("D", i, cols, cols))
      if i < len(info) - 2 and off:
        out.append(Block("B", i, range(start + diag, start + diag + off), cols))
      if i < len(info) - 1 and len(arrow):
        out.append(Block("E", i, arrow, cols))
    return tuple(out)

  def block_mask(self) -> np.ndarray:
    """Which entries of the lower triangle the blocks cover."""
    n = self.structure.n
    mask = np.zeros((n, n), dtype=bool)
    for block in self.blocks:
      mask[block.rows.start : block.rows.stop, block.cols.start : block.cols.stop] = True
    return np.tril(mask)

  # Values.

  def update_scalings_and_factor(self, data: Data, AtA: sc.Expr, delta: sc.Expr, x_reg: sc.Expr, z_reg: sc.Expr) -> Factor:
    """Assemble ``Psi`` as ``piqp_sparse`` does and factor it in the given order, as PIQP's ``factor_kkt`` does by blocks.

    Success means every pivot is usable, as PIQP's ``blasfeo_dpotrf_l`` reports.
    """
    Psi = sc.pack(self.kkt.assemble(data, AtA, delta, x_reg, z_reg), self.pattern)
    L, status = linalg.cholesky(Psi, ordering="natural")
    return Factor(L, all_true(sc.equal(status, 0)))
