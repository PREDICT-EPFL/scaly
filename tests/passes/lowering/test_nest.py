"""Loop nests: NumPy differential tests per op and shape class, the access maps the issue pins, and
loop merging under loaded bounds."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import jvp_many
from scaly.codegen.toolchain import find_c_compiler
from scaly.function.concrete import ConcreteFunction
from scaly.ir import program as p
from scaly.ir.expr import Expr
from scaly.ir.program import ProgramNode, ProgramOp, RangeKind, walk_program
from scaly.passes.lowering import ctx, lower_function, main_proc
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
  return [{k: c for k, c in (ctx._coefficients(v.args[0], names) or {}).items() if k} for v in views]


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


_BROADCASTS = sorted(case for case, (a, b) in _ELEMENTWISE.items() if np.prod(b) > 1 and a != b)


@pytest.mark.parametrize("case", [c for c in _BROADCASTS if not c.startswith("empty")])
def test_broadcasts_in_one_flat_loop(case) -> None:
  """Lanes over a broadcast's nest of rows store each row apart and the reads after it stall, so until
  lanes handle nests a broadcast is one flat loop: an operand of the output's shape reads at the loop
  variable, the broadcast one at quotients and remainders of it."""
  a_shape, _ = _ELEMENTWISE[case]
  (loop,) = _nests(_lowered(_compiled(f"nest_bflat_{case}", _ELEMENTWISE[case], lambda a, b: a * b)))
  (rng,) = _loops(loop)
  assert int(rng.args[1].attrs["value"]) == int(np.prod(np.broadcast_shapes(*_ELEMENTWISE[case])))
  assert _strides(loop, "y") == [{rng.attrs["name"]: 1}]
  if a_shape == np.broadcast_shapes(*_ELEMENTWISE[case]):
    assert _strides(loop, "x0") == [{rng.attrs["name"]: 1}]


def test_strided_slice_copies_in_one_flat_loop() -> None:
  """Lanes over a nest of short rows split each row's store across cache lines, so until lanes handle
  nests a strided slice copy is one flat loop whose source coordinates divide."""
  (copy,) = _nests(_lowered(_compiled("nest_flat_slice", ((7, 27),), lambda a: a[:, 9:18])))
  (rng,) = _loops(copy)
  assert int(rng.args[1].attrs["value"]) == 63
  assert _strides(copy, "y") == [{rng.attrs["name"]: 1}]
  assert _divides(copy)


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
  """Seeds are a batch axis and are read directly by the product's reduction."""
  proc = _lowered(_leading_seed_matvec())
  (product,) = (s for s in _nests(proc) if [r.attrs["kind"] for r in _loops(s)] == [RangeKind.GLOBAL] * 2)
  assert not _divides(product)
  batch, _ = (r.attrs["name"] for r in _loops(product))
  assert _strides(product, "S") == [{batch: 5}]
  assert any(n.op == ProgramOp.LOAD and n.args[0].attrs["buffer"] == "A" for n in walk_program(product))
  (tangent,) = (s for s in _nests(proc) if any(v.op == ProgramOp.VIEW and v.attrs["buffer"] == "t" for v in walk_program(s)))
  (rng,) = _loops(tangent)
  assert int(rng.args[1].attrs["value"]) == 12 and _strides(tangent, "t") == [{rng.attrs["name"]: 1}]


def test_reshape_reads_a_materialized_transpose_in_place() -> None:
  """A reshape after a transpose is not affine in the reshape's coordinates. Lowering materializes
  the transpose with strided reads, and the reshape aliases that copy, which the product reads at
  its own flat position."""
  fn = _compiled("nest_pin_transpose", ((2, 3, 4), (6,)), lambda a, b: a.transpose((2, 0, 1)).reshape((4, 6)) * b)
  copy, product = _nests(_lowered(fn))
  assert not _divides(copy)
  # Output axes 1 and 2 are source axes 0 and 1, contiguous in both, so the copy is a 4 x 6 nest.
  outer, inner = (r.attrs["name"] for r in _loops(copy))
  assert [int(r.args[1].attrs["value"]) for r in _loops(copy)] == [4, 6]
  assert _strides(copy, "x0") == [{outer: 1, inner: 4}]
  copied = next(v.attrs["buffer"] for v in walk_program(copy) if v.op == ProgramOp.VIEW and v.attrs["buffer"] != "x0")
  (rng,) = _loops(product)
  assert _strides(product, copied) == [{rng.attrs["name"]: 1}]


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


def test_single_seed_jvp_keeps_reduction_outermost():
  @sc.function(sc.arg("a", (3, 4)), sc.arg("b", (4, 5)), sc.arg("da", (3, 4)), outputs=sc.arg("t"))
  def f(a, b, da):
    return sc.jvp(a @ b, a, da).block()

  proc = _lowered(f)
  product = next(
    nest for nest in _nests(proc) if any(n.op == ProgramOp.FOR and n.args[0].attrs["kind"] == RangeKind.REDUCE for n in walk_program(nest))
  )
  assert _loops(product)[0].attrs["kind"] == RangeKind.REDUCE
