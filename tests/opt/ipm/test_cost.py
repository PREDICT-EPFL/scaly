"""``opt.ipm.cost``: the counts an iteration of each backend is made of, and the choice between them."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ir.target import PRESETS
from scaly.linalg.symbolic import TooMuchWork
from scaly.opt.ipm import QPStructure, choose_backend
from scaly.opt.ipm import cost
from scaly.opt.ipm.cost import DENSE_WEIGHTS, STAGEWISE_WEIGHTS, Work, iteration_us, stage_terms, stage_work, stagewise_us, work
from scaly.opt.ipm.kkt import Kernels, kkt_symbolic
from scaly.opt.ipm.stages import stages
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


def test_dense_rows_count_as_the_dense_product_they_are_multiplied_by() -> None:
  """A matrix the dense backend multiplies as a dense array (``kkt.dense_rows``) costs a quarter of
  ``rows * n^2``, not its rows' nonzeros squared; one below the rule's floors keeps the latter."""
  n, free = 16, np.full(16, np.inf)
  P = sparse.eye_array(n, format="csc")

  def counted(rows_a: int, rows_g: int) -> int:
    s = QPStructure.from_patterns(
      P, np.ones((rows_a, n)), np.ones((rows_g, n)), h_l=np.full(rows_g, -np.inf), h_u=np.zeros(rows_g), x_l=-free, x_u=free
    )
    return work(s).assembly

  assert counted(20, 0) == 20 * n * n // 4
  assert counted(15, 0) == 15 * n * n  # 3 840 products: under the rule's 4 096, through the tables
  assert counted(15, 24) == 15 * n * n + 24 * n * n // 4


@pytest.mark.parametrize("name", ["HS118", "DUALC1", "QAFIRO", "CVXQP1_S"])
def test_the_sparse_counts_are_those_of_the_factorization_the_backend_builds(name: str) -> None:
  """The model and the sparse backend share one analysis of the KKT pattern, made once."""
  s, _ = ipm_inputs(maros_meszaros(name))
  counted = work(s)
  kernels = Kernels(s, "sparse")
  _ = kernels._factorization
  ldl = kernels._ldl
  assert ldl is not None and ldl.symbolic is kkt_symbolic(s)
  assert (counted.updates, counted.nnz_l) == (ldl.symbolic.update_lanes, ldl.symbolic.nnz_l)


@pytest.mark.parametrize(
  ("name", "backend"),
  [
    ("DUALC1", "dense"),  # 214 inequality rows over 9 variables: the condensed matrix is small
    ("DUALC8", "dense"),
    ("CVXQP1_S", "sparse"),
    ("QAFIRO", "sparse"),
    ("DUAL3", "dense"),  # a dense Hessian of order 111: since the factor went into blocks
    ("DUAL1", "dense"),
    ("PRIMALC1", "sparse"),
    ("HS118", "sparse"),
    ("HS21", "sparse"),  # the smallest problems: the constants decide
    ("HS76", "sparse"),
    ("GENHS28", "sparse"),
  ],
)
def test_the_reference_machine_takes_the_faster_backend(name: str, backend: str) -> None:
  """On the M3, as measured (``internal/notes/perf_2026_09_30_gaps/results/backend_costs.json``)."""
  s, _ = ipm_inputs(maros_meszaros(name))
  assert choose_backend(s) == backend


@pytest.mark.parametrize(
  ("stage", "backend"),
  [
    ((4, 2, 10), "sparse"),  # blocks of 6: the sparse factorization's scalar code is the faster
    ((12, 4, 20), "stagewise"),  # blocks of 16: 0.90 of the sparse backend's time
    ((26, 2, 25), "stagewise"),  # blocks of 28: 0.70
    ((27, 6, 30), "stagewise"),
  ],
)
def test_a_multistage_problem_takes_the_stagewise_backend_from_blocks_that_pay(stage: tuple[int, int, int], backend: str) -> None:
  """On the M3, as measured (``internal/notes/perf_2026_09_30_gaps/results/stagewise_fit.md``)."""
  s, _ = ipm_inputs(mpc_qp(*stage))
  assert choose_backend(s) == backend


def test_the_stagewise_counts_of_an_iteration() -> None:
  """A horizon of 25 with 26 states and 2 inputs: 26 blocks of 28 slots, 26 of them coupling."""
  s, _ = ipm_inputs(mpc_qp(26, 2, 25))
  w = stage_work(s)
  assert (w.vectors, w.entries) == (726 + 676, s.P_rows.size + s.A_rows.size)
  assert w.cells == 26 * 28 * 28 + 25 * 28 * 26
  assert w.arrays == 26 * 26 * 28 and w.products == 26 * 26 * 28 * 28  # an array of 26 rows by 28 columns a block
  # A dynamics row: 28 entries in its array and one beside it, so 29 products the array does not make; an initial-state row: its square.
  assert w.pairs == 650 * 29 + 26
  assert w.factor == 26 * (28**3 // 6) + 25 * (28 * 26 * 26 // 2 + 28 * 28 * 26)
  assert w.solve == 26 * (28 * 28 + 2 * 28 * 26)
  assert stage_terms(s) == (1.0, w.vectors, w.entries, w.cells, w.pairs, w.arrays, w.products, w.factor, w.solve)
  assert stagewise_us(s) == pytest.approx(np.dot(STAGEWISE_WEIGHTS, stage_terms(s)))
  with pytest.raises(ValueError, match="stagewise_us"):
    iteration_us(work(s), "stagewise")


def test_one_or_two_blocks_are_not_stages(monkeypatch) -> None:
  """A dense Hessian is one block: the stagewise backend would be the dense one behind another
  assembly, and it is no candidate, however little the model would have it cost."""
  s, _ = ipm_inputs(maros_meszaros("DUAL1"))
  assert stages(s).K == 1
  monkeypatch.setattr(cost, "STAGEWISE_WEIGHTS", (0.0,) * 9)
  assert choose_backend(s) == "dense"
  staged, _ = ipm_inputs(mpc_qp(4, 2, 10))
  assert choose_backend(staged) == "stagewise"  # with the same zero weights, where there are stages
  shortest, _ = ipm_inputs(mpc_qp(4, 2, 2))  # a horizon of two: three blocks, the fewest that count
  assert stages(shortest).K == 3 and choose_backend(shortest) == "stagewise"
  single, _ = ipm_inputs(mpc_qp(4, 2, 1))
  assert stages(single).K == 2 and choose_backend(single) != "stagewise"


def test_the_choice_is_the_same_for_every_target() -> None:
  """It decides the graph, built before a target is known, from weights measured on one machine."""
  s, _ = ipm_inputs(maros_meszaros("DUAL3"))
  choices = set()
  for name in PRESETS:
    with sc.target(name):
      choices.add(choose_backend(s))
  assert choices == {"dense"}


def test_the_dense_factor_is_weighed_as_the_reference_machine_generates_it() -> None:
  """Straight-line code under the M3's budget (4 096 operations), loops over it."""
  small = Work(vectors=40, entries=200, assembly=300, factor=20**3 // 3, solve=400, updates=900, nnz_l=300)
  large = Work(vectors=40, entries=200, assembly=300, factor=30**3 // 3, solve=900, updates=900, nnz_l=300)
  assert iteration_us(small, "dense") == pytest.approx(np.dot(DENSE_WEIGHTS, (1, 40, 200, 300, 0, 2666, 400)))
  assert iteration_us(large, "dense") == pytest.approx(np.dot(DENSE_WEIGHTS, (1, 40, 200, 300, 9000, 0, 900)))


def test_a_sparse_factor_too_large_to_generate_leaves_the_dense_backend(monkeypatch) -> None:
  def refuse(s):
    raise TooMuchWork("too many update multiply-adds")

  monkeypatch.setattr(cost, "kkt_symbolic", refuse)
  assert choose_backend(_structure()) == "dense"
  # A problem with stages still has the stagewise backend to weigh against the dense one.
  s, _ = ipm_inputs(mpc_qp(12, 4, 20))
  assert choose_backend(s) == "stagewise"


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
