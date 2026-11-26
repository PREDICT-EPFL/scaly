"""Sparse ``cholesky``, ``ldl`` and ``solve(assume="pos"|"sym")`` within the static-loop budget lower
to straight-line code: no loop and no index table in the factorization's procedure, every value in
a C local, the factor's flop count and no more, the status without an early return, and the same
results as the looped schedule that a pattern over the budget keeps."""

from __future__ import annotations

import re

import numpy as np
import pytest
import scipy.linalg
from scipy import sparse
from scipy.sparse.linalg import spsolve

import scaly as sc
from scaly import linalg
from scaly.codegen import render_c_source
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.linalg import symbolic
from scaly.linalg.symbolic import ORDERINGS, LDLSymbolic, Ordering
from scaly.passes.lowering import calls, lower_function

from .test_sparse import STATUS, _pattern, _sparse_type, _symmetric, _values
from .test_symbolic import _arrow, _banded, _multistage, _random

# Each fits the static-loop budget for every routine and ordering.
PATTERNS = {
  "random": lambda: _random(7, 0.3, 7),
  "banded": lambda: _banded(8, 2),
  "arrow, hub first": lambda: _arrow(5, hub_first=True),
  "arrow, hub last": lambda: _arrow(10, hub_first=False),
  "block-tridiagonal": lambda: _multistage(2, 2, 1, 6),
}
ROUTINES = ("cholesky", "ldl", "pos", "sym")
_KIND = {"cholesky": "cholesky", "ldl": "ldl", "pos": "cholesky", "sym": "ldl"}
_FLOATING = {ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.SQRT, ProgramOp.NEG}


def _routine(a: np.ndarray, routine: str, ordering: Ordering, name: str = "run") -> sc.Function:
  n = a.shape[0]
  if routine in ("pos", "sym"):

    @sc.function(sc.arg("K", _sparse_type(a)), sc.arg("b", n), outputs=sc.group(sc.arg("x", n), sc.arg("status", STATUS)), name=name)
    def solve(K, b):
      return linalg.solve(K, b, assume=routine, ordering=ordering)

    return solve

  @sc.function(sc.arg("K", _sparse_type(a)), name=name)
  def factor(K):
    return getattr(linalg, routine)(K, ordering=ordering)

  return factor


def _symbol(routine: str, ordering: Ordering) -> str:
  """The prefix of the procedure that factors, and for a solve also substitutes."""
  return f"solve_{routine}_sparse_{ordering}_" if routine in ("pos", "sym") else f"sparse_{routine}_{ordering}_"


def _procedure(fn: sc.Function, prefix: str) -> tuple[ProgramNode, str]:
  """The procedure named from ``prefix`` in the optimized program, and its generated C function."""
  (proc,) = [proc for proc in lower_function(fn).args if proc.op == ProgramOp.PROC and proc.attrs["name"].startswith(prefix)]
  match = re.search(rf"static inline void {proc.attrs['name']}_raw\(.*?\n}}\n", render_c_source(fn), re.S)
  assert match is not None
  return proc, match.group(0)


def _generated(fn: sc.Function) -> str:
  """The generated C functions, entry point included, without the shared preamble."""
  source = render_c_source(fn)
  return source[source.index('extern "C"') :]


def _operations(proc: ProgramNode) -> dict[ProgramOp, int]:
  """The floating operations the procedure computes for its outputs other than the status, each value once.

  A value, not an occurrence in the C: the scheduler leaves a product of two locals with two uses
  at both uses (``schedule_values``), leaving the common subexpression to the C compiler.
  """
  body = proc.args[proc.attrs["param_count"] :]
  assigned = {stmt.attrs["target"]: stmt.args[0] for stmt in body if stmt.op == ProgramOp.ASSIGN}
  pending = [
    value
    for stmt in body
    if stmt.op in (ProgramOp.STORE, ProgramOp.STORE_PAIR) and stmt.args[0].attrs["buffer"] != "status"
    for value in stmt.args[1:]
  ]
  seen: set[ProgramNode] = set()
  counts: dict[ProgramOp, int] = {}
  while pending:
    node = pending.pop()
    if node in seen:
      continue
    seen.add(node)
    if node.op == ProgramOp.VAR:
      pending.append(assigned[node.attrs["name"]])
      continue
    if node.op in _FLOATING and node.dtype.is_floating:
      counts[node.op] = counts.get(node.op, 0) + 1
    pending.extend(node.args)
  return counts


def _flops(analysis: LDLSymbolic, routine: str) -> int:
  """The left-looking factorization's flop count for the analysis' ordering, with a solve's substitutions.

  Column ``k`` updates column ``j`` for each entry ``L[j, k]``: a product and a subtraction for each
  entry of column ``k`` from row ``j`` down, and for LDL^T one product ``d[k] L[j, k]`` per update.
  Each entry of ``L`` is one division, and each Cholesky pivot one square root. The substitutions
  take a product and a subtraction per entry of ``L`` each way, and a division per pivot each way
  for Cholesky, once between them for LDL^T.
  """
  n, kind = analysis.n, _KIND[routine]
  l_ptr, l_rows = np.asarray(analysis.l_ptr), np.asarray(analysis.l_rows)
  flops = analysis.nnz_l + (n if kind == "cholesky" else 0)
  for k in range(n):
    rows = l_rows[l_ptr[k] : l_ptr[k + 1]]
    for j in rows:
      flops += 2 * (1 + int(np.count_nonzero(rows > j))) + (kind == "ldl")
  if routine in ("pos", "sym"):
    flops += 4 * analysis.nnz_l + (2 * n if kind == "cholesky" else n)
  return flops


def _stored(a: np.ndarray, pattern: CanonicalPattern) -> sparse.csc_array:
  """``a`` on ``pattern``, keeping the entries it sets to zero."""
  rows, cols = np.asarray(pattern.rows), np.asarray(pattern.cols)
  return sparse.csc_array((a[rows, cols], (rows, cols)), shape=pattern.shape)


def _results(fn: sc.Function, a: np.ndarray, routine: str, b: np.ndarray, pattern: CanonicalPattern | None = None) -> list[np.ndarray]:
  K = sparse.csc_array(a) if pattern is None else _stored(a, pattern)
  out = fn(K, b) if routine in ("pos", "sym") else fn(K)
  return [part.toarray() if sparse.issparse(part) else np.asarray(part) for part in out]


@pytest.mark.parametrize("ordering", ORDERINGS)
@pytest.mark.parametrize("routine", ROUTINES)
@pytest.mark.parametrize("name", list(PATTERNS))
def test_a_pattern_within_the_budget_factors_in_straight_line_code(name: str, routine: str, ordering: Ordering, monkeypatch) -> None:
  rng = np.random.default_rng(20)
  mask = PATTERNS[name]()
  n = mask.shape[0]
  a = _values(mask, rng, negative=n // 3 if routine == "sym" else 0)
  b = rng.normal(size=n)
  analysis = symbolic.ldl(_pattern(a), ordering)
  fn = _routine(a, routine, ordering)

  proc, _ = _procedure(fn, _symbol(routine, ordering))
  # The entry point reads K and writes L in CSC order straight from the procedure.
  generated = _generated(fn)
  assert "for (" not in generated and "static const" not in generated
  # Every value lives in a C local: the procedure allocates no buffer.
  assert proc.attrs.get("scalarized") and not any(stmt.op == ProgramOp.BUFFER for stmt in proc.args[proc.attrs["param_count"] :])
  operations = _operations(proc)
  assert sum(operations.values()) == _flops(analysis, routine)
  assert ProgramOp.NEG not in operations
  if routine in ("cholesky", "ldl"):
    # Each stored entry of L is computed once.
    assert operations[ProgramOp.DIV] == analysis.nnz_l

  unrolled = _results(fn, a, routine, b)
  monkeypatch.setattr(calls, "STATIC_LOOP_BUDGET", 0)
  looped_fn = _routine(a, routine, ordering, name="looped")
  assert "for (" in _procedure(looped_fn, _symbol(routine, ordering))[1]
  for value, looped in zip(unrolled, _results(looped_fn, a, routine, b), strict=True):
    np.testing.assert_allclose(value, looped, rtol=1e-14, atol=1e-15)

  perm = np.asarray(analysis.perm)
  permuted = _symmetric(a)[np.ix_(perm, perm)]
  if routine == "cholesky":
    np.testing.assert_allclose(unrolled[0], scipy.linalg.cholesky(permuted, lower=True), rtol=1e-12, atol=1e-13)
  elif routine == "ldl":
    expected_l, expected_d, _ = scipy.linalg.ldl(permuted, lower=True)
    np.testing.assert_allclose(unrolled[0], expected_l, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(unrolled[1], np.diag(expected_d), rtol=1e-12, atol=1e-13)
  else:
    np.testing.assert_allclose(unrolled[0], spsolve(sparse.csc_array(_symmetric(a)), b), rtol=1e-12, atol=1e-13)
  assert int(unrolled[-1]) == 0


@pytest.mark.parametrize("layouts", [("csc", "csc"), ("csr", "csr"), ("csr", "csc"), ("coordinates", "csr")])
def test_every_layout_reads_and_writes_the_procedure_directly(layouts: tuple[str, str]) -> None:
  a = _values(PATTERNS["arrow, hub first"](), np.random.default_rng(24))
  n = a.shape[0]
  pattern = _pattern(a)
  reversed_order = sc.SparsityPattern(pattern.shape, pattern.rows[::-1], pattern.cols[::-1])
  given, returned = layouts
  layout = reversed_order if given == "coordinates" else given

  @sc.function(
    sc.arg("K", _sparse_type(a), layout=layout),
    outputs=sc.group(sc.arg("L", (n, n), pattern=..., layout=returned), sc.arg("d", n), sc.arg("status", STATUS)),
  )
  def factor(K):
    return linalg.ldl(K, ordering="amd")

  generated = _generated(factor)
  assert "for (" not in generated and "static const" not in generated
  rows, cols = np.asarray(pattern.rows), np.asarray(pattern.cols)
  values = a[rows, cols]
  K = values[::-1] if given == "coordinates" else sparse.csr_array(a) if given == "csr" else sparse.csc_array(a)
  L, d, status = factor(K)
  assert isinstance(L, sparse.csc_array | sparse.csr_array) and L.format == returned
  expected = _results(_routine(a, "ldl", "amd"), a, "ldl", np.zeros(n))
  np.testing.assert_array_equal(L.toarray(), expected[0])
  np.testing.assert_array_equal(d, expected[1])
  assert not np.asarray(status).any()


@pytest.mark.parametrize("routine", ROUTINES)
def test_an_unusable_pivot_sets_the_status_without_an_early_return(routine: str) -> None:
  a = _values(PATTERNS["arrow, hub last"](), np.random.default_rng(21))
  n = a.shape[0]
  iperm = symbolic.ldl(_pattern(a), "amd").iperm
  fn = _routine(a, routine, "amd")
  proc, source = _procedure(fn, _symbol(routine, "amd"))
  assert not any(node.op == ProgramOp.IF for node in walk_program(proc))
  assert "return" not in source and "if (" not in source

  # Column 3 touches only the hub, so its pivot is its own diagonal entry; with two unusable pivots
  # the status names the first in the reordered numbering.
  b = np.ones(n)
  zero = a.copy()
  zero[3, 3] = zero[5, 5] = 0.0
  assert int(_results(fn, zero, routine, b, _pattern(a))[-1]) == min(iperm[3], iperm[5]) + 1
  negative = a.copy()
  negative[3, 3] = -1.0
  assert int(_results(fn, negative, routine, b)[-1]) == (iperm[3] + 1 if _KIND[routine] == "cholesky" else 0)


@pytest.mark.parametrize("budget", [calls.STATIC_LOOP_BUDGET, 0], ids=["straight-line", "looped"])
def test_back_substitution_subtracts_each_term_from_the_row_in_turn(budget: int, monkeypatch) -> None:
  """``x[0] = (z[0] - L[1, 0] x[1]) - L[2, 0] x[2]``, QDLDL's order, not ``z[0] - (L[1, 0] x[1] + L[2, 0] x[2])``.

  Floating subtraction does not associate, so each order overflows on an input where the other does
  not: here ``z[0] = 1e308`` and both terms ``1e308`` stay finite, and ``z[0] = -1e308`` with terms
  ``1e308`` then ``-1e308`` overflows.
  """
  monkeypatch.setattr(calls, "STATIC_LOOP_BUDGET", budget)
  # K = L D L^T with L[1, 0] = L[2, 0] = 1, L[2, 1] = 0 and D = diag(1e-300, 1, 1), so z[0] = b[0] / 1e-300.
  d0 = 1e-300
  a = np.array([[d0, 0.0, 0.0], [d0, d0 + 1.0, 0.0], [d0, d0, d0 + 1.0]])
  fn = _routine(a != 0.0, "sym", "natural")
  x, status = _results(fn, a, "sym", np.array([1e8, 1e308, 1e308]))
  np.testing.assert_array_equal(x, [-1e308, 1e308, 1e308])
  x, status = _results(fn, a, "sym", np.array([-1e8, 1e308, -1e308]))
  np.testing.assert_array_equal(x, [-np.inf, 1e308, -1e308])
  assert int(status) == 0


@pytest.mark.parametrize("routine", ROUTINES)
def test_one_size_over_the_budget_keeps_the_loops(routine: str) -> None:
  """Banded LDL^T of order 8 fits the budget in natural order and order 9 does not; Cholesky, with
  no products ``d[k] L[j, k]``, switches one order later."""
  kind = _KIND[routine]
  last = 8 if kind == "ldl" else 9
  for n in (last, last + 1):
    a = _values(_banded(n, 2), np.random.default_rng(22), negative=1 if routine == "sym" else 0)
    fn = _routine(a, routine, "natural")
    source = render_c_source(fn)
    looped = re.search(rf"static inline void sparse_{kind}_column_\w*_raw\(", source) is not None
    assert looped == (n > last)
    assert ("for (" in _procedure(fn, _symbol(routine, "natural"))[1]) == (n > last)
    if routine in ("pos", "sym"):
      b = np.linspace(-1.0, 1.0, n)
      np.testing.assert_allclose(_results(fn, a, routine, b)[0], np.linalg.solve(_symmetric(a), b), rtol=1e-12, atol=1e-13)


def _stage_matrix() -> np.ndarray:
  """The lower triangle of a 12x12 stage KKT matrix within the budget: diagonal costs on four states
  and two inputs, four dynamics rows coupling each state to the next and two inputs, two input bounds."""
  m = np.eye(12, dtype=bool)
  m[6:10, 0:4] = np.eye(4, dtype=bool)
  m[7, 4] = m[9, 5] = True
  m[10:12, 4:6] = np.eye(2, dtype=bool)
  return np.tril(m)


def test_a_stage_solve_inside_a_scan_gives_the_same_code_at_every_horizon() -> None:
  mask = _stage_matrix()
  pattern = _pattern(mask)
  n = mask.shape[0]

  @sc.function(sc.arg("c", n), sc.arg("k", pattern.nnz), sc.arg("b", n))
  def stage(c, k, b):
    x, status = linalg.solve(sc.pack(k, pattern), b + 0.5 * c, assume="sym", ordering="amd")
    return x, x, status

  rng = np.random.default_rng(23)
  stages, sizes = set(), []
  for horizon in (10, 100, 1000):

    @sc.function(sc.arg("ks", (horizon, pattern.nnz)), sc.arg("bs", (horizon, n)), name="horizon")
    def run(ks, bs):
      _, (xs, statuses) = sc.scan(stage, sc.const(np.zeros(n)), (ks, bs), length=horizon)
      return xs, statuses

    # The solve expands into the stage, which keeps no loop or table of its own.
    _, body = _procedure(run, "stage")
    assert "for (" not in body and "static const" not in body
    stages.add(body)
    sizes.append(len(render_c_source(run)))
    if horizon == 10:
      matrices = [_values(mask, rng, negative=6) for _ in range(horizon)]
      ks = np.stack([m[np.asarray(pattern.rows), np.asarray(pattern.cols)] for m in matrices])
      bs = rng.normal(size=(horizon, n))
      xs, statuses = run(ks, bs)
      c = np.zeros(n)
      for m, b, x in zip(matrices, bs, xs, strict=True):
        c = np.linalg.solve(_symmetric(m), b + 0.5 * c)
        np.testing.assert_allclose(x, c, rtol=1e-12, atol=1e-13)
      assert not np.asarray(statuses).any()
  # The stage is the same code at every horizon; the rest differs only in loop bounds, buffer sizes
  # and whether the stacked outputs fit the frame.
  assert len(stages) == 1
  assert max(sizes) - min(sizes) < 0.02 * min(sizes)
