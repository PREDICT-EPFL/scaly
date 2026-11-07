"""Loop nests: NumPy differential tests per op and shape class, the access maps lowering emits, and
the interface a sparse kernel is written against."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import jvp_many
from scaly.codegen.toolchain import find_c_compiler
from scaly.function.concrete import ConcreteFunction
from scaly.ir import program as p
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from scaly.passes.lowering import ctx, lower_function, main_proc
from scaly.passes.program._common import affine_coefficients, inline_producer
from scaly.utils.names import NameScope

_HAVE_CC = find_c_compiler() is not None
needs_cc = pytest.mark.skipif(not _HAVE_CC, reason="no C compiler available for JIT numeric check")


def _compiled(name: str, shapes: tuple[tuple[int, ...], ...], body: Callable[..., Expr]) -> sc.Function:
  @sc.function(sc.group(*(sc.arg(f"x{k}", s) for k, s in enumerate(shapes))), outputs=sc.arg("y"), name=name)
  def fn(inputs):
    return body(*inputs)

  return fn


def _check(name: str, shapes: tuple[tuple[int, ...], ...], body: Callable[..., Expr], reference: Callable[..., np.ndarray]) -> None:
  """Integer-valued inputs keep each product and sum exact, so the C must equal NumPy bit for bit."""
  rng = np.random.default_rng(0)
  values = tuple(rng.integers(-8, 9, s).astype(np.float64) for s in shapes)
  fn = _compiled(name, shapes, body)
  fn.recompile()
  np.testing.assert_array_equal(fn(values), reference(*values))


def _lowered(fn: sc.Function | ConcreteFunction) -> ProgramNode:
  """The main procedure as lowering emits it, before any program pass."""
  seen: dict[str, ProgramNode] = {}
  lower_function(fn, observe=seen.__setitem__)
  return main_proc(seen["lowered"])


def _divides(node: ProgramNode) -> bool:
  return any(n.op in (ProgramOp.DIV, ProgramOp.MOD) for v in walk_program(node) if v.op == ProgramOp.VIEW for n in walk_program(v.args[0]))


def _loops(stmt: ProgramNode) -> list[ProgramNode]:
  """The ranges of a nest down to its first statement that is not a lone loop, outermost first."""
  ranges = [stmt.args[0]]
  while len(stmt.args) == 2 and stmt.args[1].op == ProgramOp.FOR:
    stmt = stmt.args[1]
    ranges.append(stmt.args[0])
  return ranges


def _nests(proc: ProgramNode) -> list[ProgramNode]:
  return [s for s in proc.args[proc.attrs["param_count"] :] if s.op == ProgramOp.FOR]


def _strides(stmt: ProgramNode, buffer: str) -> list[dict[str, int]]:
  """The coefficients of the nest's loop variables in each view of ``buffer``."""
  names = frozenset(r.attrs["name"] for r in _loops(stmt))
  views = [v for v in walk_program(stmt) if v.op == ProgramOp.VIEW and v.attrs["buffer"] == buffer]
  return [{k: c for k, c in (affine_coefficients(v.args[0], names) or {}).items() if k} for v in views]


_ELEMENTWISE = {
  "scalar": ((), ()),
  "same": ((3, 4), (3, 4)),
  "scalar_operand": ((3, 4), ()),
  "leading": ((2, 3, 4), (3, 4)),
  "trailing": ((3, 4), (3, 1)),
  "middle": ((2, 3, 4), (2, 1, 4)),
  "outer": ((3, 1), (1, 4)),
  "rank5": ((2, 1, 3, 1, 2), (2, 2, 1, 2, 1)),
  "singleton_output": ((1, 3), (1, 3)),
  "empty_rows": ((0, 3), (3,)),
  "empty_columns": ((2, 0), (2, 0)),
}


@needs_cc
@pytest.mark.parametrize("case", _ELEMENTWISE)
def test_elementwise_matches_numpy(case, isolated_cache) -> None:
  _check(f"nest_ew_{case}", _ELEMENTWISE[case], lambda a, b: a * b - a, lambda a, b: a * b - a)


_MOVEMENT: dict[str, tuple[tuple[tuple[int, ...], ...], Callable[..., Expr], Callable[..., np.ndarray]]] = {
  "transpose2": (((3, 4),), lambda a: a.T, lambda a: a.T),
  "transpose3": (((2, 3, 4),), lambda a: a.transpose((2, 0, 1)), lambda a: a.transpose((2, 0, 1))),
  "transpose5": (((2, 1, 3, 2, 4),), lambda a: a.transpose((4, 2, 0, 3, 1)), lambda a: a.transpose((4, 2, 0, 3, 1))),
  "slice_steps": (((5, 6),), lambda a: a[1:4, ::2], lambda a: a[1:4, ::2]),
  "slice_int": (((4, 5, 3),), lambda a: a[:, 2, 1:], lambda a: a[:, 2, 1:]),
  "slice_last": (((4, 5),), lambda a: a[:, -1], lambda a: a[:, -1]),
  "stack0": (((3, 4), (3, 4)), lambda a, b: sc.stack([a, b], axis=0), lambda a, b: np.stack([a, b], axis=0)),
  "stack1": (((3, 4), (3, 4)), lambda a, b: sc.stack([a, b], axis=1), lambda a, b: np.stack([a, b], axis=1)),
  "stack_last": (((3, 4), (3, 4)), lambda a, b: sc.stack([a, b], axis=2), lambda a, b: np.stack([a, b], axis=2)),
  "concat0": (((2, 4), (3, 4)), lambda a, b: sc.concat([a, b], axis=0), lambda a, b: np.concatenate([a, b], axis=0)),
  "concat1": (((3, 1), (3, 4)), lambda a, b: sc.concat([a, b], axis=1), lambda a, b: np.concatenate([a, b], axis=1)),
  "concat_empty": (((3, 0), (3, 2)), lambda a, b: sc.concat([a, b], axis=1), lambda a, b: np.concatenate([a, b], axis=1)),
  "gather": (((3, 4),), lambda a: sc.gather(a, np.array([[11, 0], [5, 5], [2, 7]])), lambda a: a.reshape(-1)[np.array([[11, 0], [5, 5], [2, 7]])]),
  "transpose_reshape": (((2, 3, 4),), lambda a: a.transpose((2, 0, 1)).reshape((4, 6)), lambda a: a.transpose((2, 0, 1)).reshape((4, 6))),
  "chain": (
    ((2, 3, 4), (3,)),
    lambda a, b: (a.transpose((2, 0, 1)).reshape((4, 6))[:, ::2].T * 2.0).reshape((3, 4)).T * b,
    lambda a, b: (a.transpose((2, 0, 1)).reshape((4, 6))[:, ::2].T * 2.0).reshape((3, 4)).T * b,
  ),
}


@needs_cc
@pytest.mark.parametrize("case", _MOVEMENT)
def test_movement_matches_numpy(case, isolated_cache) -> None:
  shapes, body, reference = _MOVEMENT[case]
  _check(f"nest_mv_{case}", shapes, body, reference)


_STRIDED_SLICES = {"slice_steps", "slice_int", "slice_last", "chain"}


@pytest.mark.parametrize("case", sorted(set(_MOVEMENT) - {"gather"} - _STRIDED_SLICES))
def test_dense_kernels_index_without_division(case) -> None:
  """Every dense kernel reads and writes at strides times coordinates, except a gather's table and a
  strided slice copy."""
  shapes, body, _ = _MOVEMENT[case]
  assert not _divides(_lowered(_compiled(f"nest_div_{case}", shapes, body)))


@pytest.mark.parametrize("case", sorted(_ELEMENTWISE))
def test_broadcasts_index_without_division(case) -> None:
  assert not _divides(_lowered(_compiled(f"nest_bdiv_{case}", _ELEMENTWISE[case], lambda a, b: a * b)))


def test_strided_slice_copies_in_one_flat_loop() -> None:
  """Lanes over a nest of short rows split each row's store across cache lines, so until lanes handle
  nests a strided slice copy is one flat loop whose source coordinates divide."""
  (copy,) = _nests(_lowered(_compiled("nest_flat_slice", ((7, 27),), lambda a: a[:, 9:18])))
  (rng,) = _loops(copy)
  assert int(rng.args[1].attrs["value"]) == 63
  assert _strides(copy, "y") == [{rng.attrs["name"]: 1}]
  assert _divides(copy)


def test_contiguous_axes_merge_into_one_loop() -> None:
  """No broadcast, one flat loop; a broadcast operand keeps its axes, read with stride zero."""
  same = _nests(_lowered(_compiled("nest_merge_same", ((2, 3, 4), (2, 3, 4)), lambda a, b: a + b)))
  assert [[int(r.args[1].attrs["value"]) for r in _loops(s)] for s in same] == [[24]]
  (nest,) = _nests(_lowered(_compiled("nest_merge_middle", ((2, 3, 4), (2, 1, 4)), lambda a, b: a + b)))
  outer, middle, inner = (r.attrs["name"] for r in _loops(nest))
  assert [int(r.args[1].attrs["value"]) for r in _loops(nest)] == [2, 3, 4]
  assert _strides(nest, "x1") == [{outer: 4, inner: 1}]
  assert _strides(nest, "x0") == [{outer: 12, middle: 4, inner: 1}]


def test_singleton_axes_get_coordinate_zero() -> None:
  """A size-one axis gets no loop; a domain with nothing left runs once over its first element."""
  (nest,) = _nests(_lowered(_compiled("nest_singleton", ((1, 3, 1), (1, 3, 1)), lambda a, b: a * b)))
  assert [int(r.args[1].attrs["value"]) for r in _loops(nest)] == [3]
  (unit,) = _nests(_lowered(_compiled("nest_unit", ((1, 1), (1, 1)), lambda a, b: a * b)))
  ((rng,),) = [_loops(unit)]
  assert int(rng.args[1].attrs["value"]) == 1
  assert {v.args[0] for v in walk_program(unit) if v.op == ProgramOp.VIEW} == {p.var(rng.attrs["name"])}


@pytest.mark.parametrize(
  ("shapes", "body"),
  [
    (((0, 3), (3,)), lambda a, b: a * b),
    (((3, 0),), lambda a: a.T),
    (((4, 0),), lambda a: a[1:3]),
    (((0, 2), (0, 2)), lambda a, b: sc.stack([a, b], axis=1)),
  ],
)
def test_empty_extent_emits_no_store(shapes, body) -> None:
  proc = _lowered(_compiled(f"nest_empty_{len(shapes)}_{shapes[0][0]}", shapes, body))
  assert not [n for n in walk_program(proc) if n.op in (ProgramOp.STORE, ProgramOp.FOR)]


def test_concat_skips_an_empty_piece() -> None:
  """The empty piece emits nothing; the other fills the whole output, so its copy is one flat loop."""
  (nest,) = _nests(_lowered(_compiled("nest_concat_empty", ((3, 0), (3, 2)), lambda a, b: sc.concat([a, b], axis=1))))
  (rng,) = _loops(nest)
  assert int(rng.args[1].attrs["value"]) == 6
  assert _strides(nest, "x1") == _strides(nest, "y") == [{rng.attrs["name"]: 1}]


def _leading_seed_matvec() -> ConcreteFunction:
  """Three forward directions of ``sin(A @ x)``: the tangent has a leading seed axis."""
  a, x, s = Expr.sym("A", (4, 5)), Expr.sym("x", (5,)), Expr.sym("S", (3, 5))
  return ConcreteFunction._from_exprs("nest_seeded_matvec", [a, x, s], [jvp_many((a @ x).sin(), x, s)], ["A", "x", "S"], ["t"])


@needs_cc
def test_leading_seed_matvec_matches_numpy(isolated_cache) -> None:
  rng = np.random.default_rng(1)
  av, xv, sv = rng.standard_normal((4, 5)), rng.standard_normal(5), rng.standard_normal((3, 5))
  fn = _leading_seed_matvec()
  fn.recompile()
  np.testing.assert_allclose(fn(av, xv, sv), np.cos(av @ xv)[None, :] * (sv @ av.T), rtol=1e-14, atol=1e-14)


def test_leading_seed_matvec_access_maps() -> None:
  """The seed axis stays the outer loop of the tangent's nest, and every read is strided, not divided.

  The seed-axis transposes are copies with strided reads, which lowering-time fusion can turn into
  index remaps; the broadcast primal reads with stride zero on the seed loop."""
  proc = _lowered(_leading_seed_matvec())
  assert not _divides(proc)
  s_t, product_t, tangent = (s for s in _nests(proc) if [r.attrs["kind"] for r in _loops(s)] == [RangeKind.GLOBAL] * 2)
  seed, entry = (r.attrs["name"] for r in _loops(tangent))
  assert [int(r.args[1].attrs["value"]) for r in _loops(tangent)] == [3, 4]
  primal, product = sorted({v.attrs["buffer"] for v in walk_program(tangent) if v.op == ProgramOp.VIEW} - {"t"})
  assert _strides(tangent, primal) == [{entry: 1}]
  assert _strides(tangent, product) == _strides(tangent, "t") == [{seed: 4, entry: 1}]
  # S.T and the product's transpose back to the seed axis read with the source's row stride.
  outer, inner = (r.attrs["name"] for r in _loops(s_t))
  assert _strides(s_t, "S") == [{outer: 1, inner: 5}]
  outer, inner = (r.attrs["name"] for r in _loops(product_t))
  assert [stride for b in {v.attrs["buffer"] for v in walk_program(product_t) if v.op == ProgramOp.VIEW} for stride in _strides(product_t, b)] in (
    [{outer: 1, inner: 3}, {outer: 4, inner: 1}],
    [{outer: 4, inner: 1}, {outer: 1, inner: 3}],
  )


def test_reshape_reads_a_materialized_transpose_in_place() -> None:
  """A reshape after a transpose is not affine in the reshape's coordinates. Lowering materializes
  the transpose with strided reads, and the reshape aliases that copy, so no index divides."""
  fn = _compiled("nest_pin_transpose", ((2, 3, 4), (6,)), lambda a, b: a.transpose((2, 0, 1)).reshape((4, 6)) * b)
  proc = _lowered(fn)
  assert not _divides(proc)
  copy, product = _nests(proc)
  # Output axes 1 and 2 are source axes 0 and 1, contiguous in both, so the copy is a 4 x 6 nest.
  outer, inner = (r.attrs["name"] for r in _loops(copy))
  assert [int(r.args[1].attrs["value"]) for r in _loops(copy)] == [4, 6]
  assert _strides(copy, "x0") == [{outer: 1, inner: 4}]
  copied = next(v.attrs["buffer"] for v in walk_program(copy) if v.op == ProgramOp.VIEW and v.attrs["buffer"] != "x0")
  outer, inner = (r.attrs["name"] for r in _loops(product))
  assert _strides(product, copied) == [{outer: 6, inner: 1}]


def test_fusion_keeps_a_transpose_and_inlines_a_broadcast() -> None:
  """Until lowering-time fusion, a transposed copy stays a buffer; a broadcast product and a scalar
  chain, whose unit loops index by their variable, are inlined."""

  def privates(fn: sc.Function) -> int:
    from scaly.codegen import render_c_module

    proc = main_proc(render_c_module(fn, lanes=1).program)
    return sum(s.op == ProgramOp.BUFFER and s.attrs["address_space"] == "private" and "alias_of" not in s.attrs for s in proc.args)

  assert privates(_compiled("nest_keep_transpose", ((3, 4),), lambda a: a.T.sin())) == 1
  assert privates(_compiled("nest_inline_broadcast", ((3, 4), (4,)), lambda a, b: (a * b).sin())) == 0
  assert privates(_compiled("nest_inline_scalar", ((), ()), lambda a, b: (a * b).sin() * a)) == 0


def _csr_matvec(lctx: ctx.LowerCtx, node: Expr) -> None:
  """A constant sparse matrix times a vector as a CSR row loop, written against the nest interface.

  The row's reduction has loaded bounds and the reads go through position tables, as a sparse
  kernel's will."""
  a, x = node.args
  assert a.value is not None
  rows, cols = np.nonzero(a.value)
  rowptr = np.searchsorted(rows, np.arange(a.shape[0] + 1))
  lctx.alloc_tmp(node)
  ranges, (i,) = lctx.nest(node.shape, "i_csr")
  lctx.emit(ranges, [lctx.write(node, (i,), p.const_float(0.0))])
  ranges, (i,) = lctx.nest(node.shape, "i_csr")
  row, (q,) = lctx.reduce([(lctx.index_at(rowptr[:-1], i), lctx.index_at(rowptr[1:], i))], "q_csr")
  product = p.mul(lctx.read(a, (lctx.index_at(rows, q), lctx.index_at(cols, q))), lctx.read(x, (lctx.index_at(cols, q),)))
  lctx.emit([*ranges, *row], [lctx.write(node, (i,), p.add(lctx.read(node, (i,)), product))])


_SPARSE = np.array([[0.0, 2.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 4.0, 0.0], [0.0, 5.0, 6.0, 0.0, 7.0]])


def _sparse_matvec() -> sc.Function:
  @sc.function(sc.arg("x", 5), outputs=sc.arg("y"), name="nest_csr_matvec")
  def fn(x):
    return sc.const(_SPARSE) @ x

  return fn


def test_sparse_row_loop_has_loaded_bounds_and_table_reads(monkeypatch) -> None:
  monkeypatch.setitem(ctx._RULES, ExprOp.MATMUL, _csr_matvec)
  proc = _lowered(_sparse_matvec())
  (row,) = [r for r in walk_program(proc) if r.op == ProgramOp.RANGE and r.attrs["kind"] == RangeKind.REDUCE]
  outer = next(r.attrs["name"] for r in walk_program(proc) if r.op == ProgramOp.RANGE and r.attrs["kind"] == RangeKind.GLOBAL and r is not row)
  assert all(any(n.op == ProgramOp.VAR for n in walk_program(bound)) for bound in row.args[:2])
  assert row.args[1].op == ProgramOp.LOAD, "this row pointer has no affine structure, so its stops come from a table"
  assert outer
  tables = {b.attrs["name"] for b in walk_program(proc) if b.op == ProgramOp.BUFFER and b.dtype == sc.dtypes.int64}
  assert any(
    v.attrs["buffer"] == "x" and v.args[0].op == ProgramOp.LOAD and v.args[0].args[0].attrs["buffer"] in tables
    for v in walk_program(proc)
    if v.op == ProgramOp.VIEW
  )


@needs_cc
def test_sparse_row_loop_matches_numpy(monkeypatch, isolated_cache) -> None:
  monkeypatch.setitem(ctx._RULES, ExprOp.MATMUL, _csr_matvec)
  fn = _sparse_matvec()
  fn.recompile()
  xv = np.random.default_rng(2).standard_normal(5)
  np.testing.assert_array_equal(fn(xv), _SPARSE @ xv)


def _bare_ctx() -> ctx.LowerCtx:
  """A lowering context with only the state ``nest`` and ``emit`` use."""
  lctx = object.__new__(ctx.LowerCtx)
  lctx.names = NameScope()
  lctx.statements = []
  return lctx


def _unbound(stmt: ProgramNode, bound: frozenset[str] = frozenset()) -> set[str]:
  """Loop variables read below ``stmt`` that no enclosing range binds."""
  if stmt.op == ProgramOp.FOR:
    rng, *body = stmt.args
    inner = bound | {rng.attrs["name"]}
    return _unbound(rng, bound).union(*(_unbound(s, inner) for s in body))
  return {n.attrs["name"] for n in walk_program(stmt) if n.op == ProgramOp.VAR and n.attrs["name"] not in bound}


def test_unit_loop_not_from_zero_keeps_constant_indices() -> None:
  lctx = _bare_ctx()
  x, y = (p.buffer(n, sc.dtypes.float64, (5,)) for n in ("x", "y"))
  ranges, (q,) = lctx.reduce([(p.const_int(2), p.const_int(3))], "q")
  lctx.emit(ranges, [p.store(p.view(y, [p.const_int(0)]), p.load(p.view(x, [q])))])
  (stmt,) = lctx.statements
  assert [v.args[0] for v in walk_program(stmt) if v.op == ProgramOp.VIEW and v.attrs["buffer"] == "y"] == [p.const_int(0)]


def test_merge_rewrites_the_bounds_of_inner_loops() -> None:
  """Rows ``3 i + j`` of a 2 x 3 nest merge into one row loop, inner bounds included."""
  lctx = _bare_ctx()
  x, y = p.buffer("x", sc.dtypes.float64, (7,)), p.buffer("y", sc.dtypes.float64, (6,))
  rowptr = p.const_buffer("rowptr", sc.dtypes.int64, (7,), [0, 1, 3, 4, 4, 5, 7])
  ranges, (i, j) = lctx.nest((2, 3), "row")
  r = p.add(p.mul(i, p.const_int(3)), j)
  row, (q,) = lctx.reduce([(p.load(p.view(rowptr, [r])), p.load(p.view(rowptr, [p.add(r, p.const_int(1))])))], "q")
  target = p.view(y, [r])
  lctx.emit([*ranges, *row], [p.store(target, p.add(p.load(target), p.load(p.view(x, [q]))))])
  (stmt,) = lctx.statements
  assert not _unbound(stmt)
  (outer, _) = _loops(stmt)
  assert int(outer.args[1].attrs["value"]) == 6
  assert _strides(stmt, "rowptr") == [{outer.attrs["name"]: 1}] * 2


def test_inner_bounds_can_forbid_a_merge() -> None:
  """The body is contiguous across ``i`` and ``j``, but a bound read at ``rowptr[i]`` is not."""
  lctx = _bare_ctx()
  x, y = p.buffer("x", sc.dtypes.float64, (7,)), p.buffer("y", sc.dtypes.float64, (6,))
  rowptr = p.const_buffer("rowptr", sc.dtypes.int64, (3,), [0, 3, 7])
  ranges, (i, j) = lctx.nest((2, 3), "row")
  row, (q,) = lctx.reduce([(p.load(p.view(rowptr, [i])), p.load(p.view(rowptr, [p.add(i, p.const_int(1))])))], "q")
  target = p.view(y, [p.add(p.mul(i, p.const_int(3)), j)])
  lctx.emit([*ranges, *row], [p.store(target, p.add(p.load(target), p.load(p.view(x, [q]))))])
  (stmt,) = lctx.statements
  assert not _unbound(stmt)
  assert [int(r.args[1].attrs["value"]) for r in _loops(stmt)[:2]] == [2, 3]


def test_a_loaded_store_offset_is_not_a_producer() -> None:
  """``tmp[i + offset[0]] = x[i]`` is not ``tmp`` row-major over its loop, so fusion must not inline it."""
  tmp = p.buffer("tmp", sc.dtypes.float64, (4,), address_space="private")
  table = p.const_buffer("offset", sc.dtypes.int64, (1,), [2])
  i = p.var("i")
  store = p.store(p.view(tmp, [p.add(i, p.load(p.view(table, [p.const_int(0)])))]), p.load(p.view(p.buffer("x", sc.dtypes.float64, (4,)), [i])))
  assert inline_producer(p.for_(p.range_("i", 0, 2), [store])) is None
