"""Assemble the sparse interior-point system without its factorization or solve."""

import numpy as np

import scaly as sc
from scaly.ir.program import ProgramOp, walk_program
from tests.passes.lowering.test_fusion import _lowered, _privates, needs_cc


def kkt_assembly():
  # Upper triangle of the sparse system from devrush, opt/ipm/kkt.py, Kernels.kkt_matrix.
  # P lacks its middle diagonal entry. A and G are in column order before transposition.
  rows = np.array([0, 1, 0, 2, 0, 2, 3, 1, 2, 4, 0, 1, 5, 2, 6])
  cols = np.array([0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])
  slots = [0, 13, 1, 2, 3, 5, 10, 4, 6, 10, 7, 8, 11, 9, 12]
  primal_diagonal = (rows == cols) & (cols < 3)

  @sc.function(sc.arg("data", 10), sc.arg("xr", 3), sc.arg("dr", ()), sc.arg("zr", 2), name="kkt_assembly")
  def assemble(data, xr, dr, zr):
    blocks = sc.concat((data[:3], data[3:7], data[7:], (-dr).reshape((1,)), -zr, sc.const([0.0])))
    base = sc.gather(blocks, slots)
    diagonal = sc.gather(xr, np.where(primal_diagonal, rows, 0))
    return sc.where(primal_diagonal, base + diagonal, base).block()

  return assemble, rows, cols


@needs_cc
def test_kkt_assembly_is_one_loop():
  assemble, rows, cols = kkt_assembly()
  data = np.linspace(-1, 2, 10)
  xr = np.array([0.1, 0.3, 0.2])
  dr = np.array(0.4)
  zr = np.array([0.8, 1.2])
  p = np.array([[data[0], 0.0, data[1]], [0.0, 0.0, 0.0], [0.0, 0.0, data[2]]])
  a = np.array([[data[3], 0.0, data[5]], [0.0, data[4], data[6]]])
  g = np.array([[data[7], data[8], 0.0], [0.0, 0.0, data[9]]])
  upper = np.block([[p + np.diag(xr), a.T, g.T], [np.zeros((2, 3)), -dr * np.eye(2), np.zeros((2, 2))], [np.zeros((2, 5)), -np.diag(zr)]])
  np.testing.assert_array_equal(assemble(data, xr, dr, zr), upper[rows, cols])

  proc = _lowered(assemble)
  assert sum(n.op == ProgramOp.FOR for n in walk_program(proc)) == 1
  assert not _privates(proc)
