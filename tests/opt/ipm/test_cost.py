"""``opt.ipm.cost``: the counts an iteration of each backend is made of, and the choice between them."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.target import PRESETS
from scaly.opt.ipm import QPStructure, choose_backend
from scaly.opt.ipm.cost import DENSE_WEIGHTS, Work, iteration_us, kkt_symbolic, work
from scaly.opt.ipm.kkt import Kernels
from scaly.testing.qp import maros_meszaros, mpc_qp

from .problems import ipm_inputs


def _structure() -> QPStructure:
  P = sparse.csc_array(np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]]))
  A = np.array([[1.0, 1.0, 0.0]])
  G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
  return QPStructure.from_patterns(P, A, G, h_l=[0.0, -1.0, -1.0], h_u=[1.0, 2.0, 1.0], x_l=[-1.0, -np.inf, 0.0], x_u=[1.0, np.inf, np.inf])


def test_the_counts_of_an_iteration() -> None:
  w = work(_structure())
  assert (w.vectors, w.entries) == (3 + 1 + 3, 4 + 2 + 6)
  assert w.assembly == 2**2 + (2**2 + 1**2 + 3**2)  # each row of A and G, its nonzeros squared
  assert (w.factor, w.solve) == (27 // 3, 9)


@pytest.mark.parametrize("name", ["HS118", "DUALC1", "QAFIRO", "CVXQP1_S"])
def test_the_sparse_counts_are_those_of_the_factorization_the_backend_builds(name: str) -> None:
  """The model analyses the KKT pattern the sparse backend factors, not a copy of it."""
  s, _ = ipm_inputs(maros_meszaros(name))
  kernels = Kernels(s, "sparse")
  _ = kernels._factorization
  built = kernels._ldl.symbolic
  assert (kkt_symbolic(s).update_lanes, kkt_symbolic(s).nnz_l) == (built.update_lanes, built.nnz_l)
  assert (work(s).updates, work(s).nnz_l) == (built.update_lanes, built.nnz_l)


@pytest.mark.parametrize(
  ("name", "backend"),
  [
    ("DUALC1", "dense"),  # 214 inequality rows over 9 variables: the condensed matrix is small
    ("DUALC8", "dense"),
    ("CVXQP1_S", "sparse"),
    ("QAFIRO", "sparse"),
    ("DUAL3", "sparse"),
  ],
)
def test_the_reference_machine_takes_the_faster_backend(name: str, backend: str) -> None:
  """On the M3, as measured (``notes/perf_2026_09_30_gaps/backend_costs.json``)."""
  s, _ = ipm_inputs(maros_meszaros(name))
  assert choose_backend(s, PRESETS["apple-m3"]) == backend


def test_a_long_horizon_is_sparse() -> None:
  s, _ = ipm_inputs(mpc_qp(12, 4, 20))
  assert choose_backend(s, PRESETS["apple-m3"]) == "sparse"


def test_the_target_weighs_the_dense_factor() -> None:
  """Straight-line under the target's budget, and vectorized loops by its width; under portable
  rounding every target weighs it as the reference machine."""
  w = Work(vectors=40, entries=200, assembly=300, factor=20**3 // 3, solve=400, updates=900, nnz_l=300)
  m3, generic, avx512 = (iteration_us(w, "dense", PRESETS[t]) for t in ("apple-m3", "generic", "x86-64-v4"))
  assert generic > m3 > avx512  # 2666 operations: straight-line on the M3 and AVX-512, loops for scalar C (682)
  # The terms: the factor in the straight-line slot under the budget, in the loop slot (widened by
  # the vector width, as the solve is) over it.
  assert m3 == pytest.approx(np.dot(DENSE_WEIGHTS, (1, 40, 200, 300, 0, 2666, 400)))
  assert generic == pytest.approx(np.dot(DENSE_WEIGHTS, (1, 40, 200, 300, 2 * 2666, 0, 2 * 400)))
  portable = dataclasses.replace(PRESETS["generic"], rounding="portable")
  assert iteration_us(w, "dense", portable) == m3
  assert iteration_us(w, "sparse", PRESETS["generic"]) == iteration_us(w, "sparse", PRESETS["apple-m3"])


def test_the_default_method_chooses_and_an_explicit_backend_does_not(monkeypatch) -> None:
  from scaly.opt.ipm import method

  seen = []
  real = method.choose_backend
  monkeypatch.setattr(method, "choose_backend", lambda s: seen.append(s) or real(s))

  @sc.opt.problem(vars=sc.L("x", 3), params=sc.L("t", 3))
  def prob(x, t):
    return sc.opt.ProblemSpec(minimize=((x - t) ** 2).sum(), ineq=(sc.opt.bounded(x, -1.0, 1.0),))

  solve = sc.opt.solver(prob, sc.opt.IPM(), name="ipm_auto_backend")
  assert len(seen) == 1
  x = solve.numerical_call(np.zeros(3), np.zeros(3), np.zeros(0), np.zeros(3), np.array([0.5, 2.0, -3.0]))[0]
  np.testing.assert_allclose(x, [0.5, 1.0, -1.0], atol=1e-7)
  for sparse_flag in (True, False):
    sc.opt.solver(prob, sc.opt.IPM(sparse=sparse_flag), name=f"ipm_fixed_backend_{sparse_flag}")
  assert len(seen) == 1
