"""The sparse Hessian of a sum over overlapping stage windows, whose assembly pack_masks rewrites, against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.function.model import as_concrete
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function

NX = 3


def _cost(stages: int) -> sc.Function:
  @sc.function(sc.arg("w", 2 * NX), outputs=sc.arg("c"), name="overlap_stage")
  def stage(w):
    x, y = w[:NX], w[NX:]
    return (x.sin() * y.cos()).sum() + 0.1 * (x[0] * x[1] * y[2]).exp()

  @sc.function(sc.arg("z", NX * (stages + 1)), outputs=sc.arg("cost"), name=f"overlap_cost_N{stages}")
  def cost(z):
    return sc.vmap(stage, stages)(sc.window(z, 0, NX)).vec().sum()

  return cost


def _dense_hessian(z: np.ndarray, stages: int) -> np.ndarray:
  """Each stage reads x = z[3i:3i+3] and y = z[3i+3:3i+6]; its Hessian is added on that 6 x 6 block."""
  dense = np.zeros((z.size, z.size))
  for i in range(stages):
    w = z[NX * i : NX * i + 2 * NX]
    x, y = w[:NX], w[NX:]
    block = np.zeros((2 * NX, 2 * NX))
    for k in range(NX):
      block[k, k] += -np.sin(x[k]) * np.cos(y[k])
      block[NX + k, NX + k] += -np.sin(x[k]) * np.cos(y[k])
      block[k, NX + k] += -np.cos(x[k]) * np.sin(y[k])
      block[NX + k, k] += -np.cos(x[k]) * np.sin(y[k])
    # 0.1 * exp(p) with p = x0 * x1 * y2 on the variables (0, 1, 5) of the window
    e = 0.1 * np.exp(x[0] * x[1] * y[2])
    idx = [0, 1, NX + 2]
    grad = np.array([x[1] * y[2], x[0] * y[2], x[0] * x[1]])
    hp = np.array([[0, y[2], x[1]], [y[2], 0, x[0]], [x[1], x[0], 0]])
    block[np.ix_(idx, idx)] += e * (np.outer(grad, grad) + hp)
    dense[NX * i : NX * i + 2 * NX, NX * i : NX * i + 2 * NX] += block
  return dense


@pytest.mark.parametrize("stages", [8, 30])
def test_overlapping_stage_hessian_matches_numpy_through_packed_masks(stages):
  hess = sc.sparse_hessian(_cost(stages), "cost", "z")
  observed = {}
  lower_function(hess, observe=lambda name, prog: observed.__setitem__(name, prog))
  assert any(n.op == ProgramOp.BIT_SELECT for n in walk_program(observed["pass:pack_masks"])), "the assembly should take the packed path"

  pattern = as_concrete(hess).output_sparsities[0]
  assert pattern is not None
  rows, cols = np.asarray(pattern.rows), np.asarray(pattern.cols)
  z = np.random.default_rng(stages).uniform(-1.0, 1.0, NX * (stages + 1))
  dense = _dense_hessian(z, stages)
  np.testing.assert_allclose(np.asarray(hess(z)), dense[rows, cols], rtol=1e-12, atol=1e-12)
  covered = np.zeros_like(dense, dtype=bool)
  covered[rows, cols] = True
  assert not np.any(dense[~covered]), "a nonzero of the reference is missing from the pattern"
