"""Multi-seed forward mode through ``scan`` and ``while_loop`` carries every seed through one tangent loop.

``jvp_many`` (behind ``jacobian`` and ``hessian``) builds one loop whose carry holds the primal and
all its tangents, differentiates a loop body with a joint multi-seed pass, inlines a call whose
callee holds a loop unless the call's seeds can be pruned, and reads a strided tangent trajectory in
place. Every derivative here is
checked against finite differences and against the single-seed ``jvp`` taken one column at a time,
with ``SCALY_STRICT_JVP_MANY`` set wherever the structural rule must be the one that ran.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import basis, jacobian
from scaly.ad.forward import _contains_loop, _jvp_many_joint, _strided_view, jvp, jvp_many
from scaly.codegen import render_c_module
from scaly.function.sugar import _scan_node, _while_node
from scaly.ir.expr import CALLEE_OPS, ExprOp, callees_of, topo

RNG = np.random.default_rng(93)


def _fn(name: str, inputs: Sequence[sc.Expr], outputs: Sequence[sc.Expr]) -> sc.Function:
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"out{k}" for k in range(len(outputs))])


def _run(name: str, inputs: Sequence[sc.Expr], outputs: Sequence[sc.Expr], point: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
  return _fn(name, inputs, outputs)._flat_numerical_call(*point)


def _columns(out: sc.Expr, wrt: sc.Expr) -> sc.Expr:
  """The Jacobian assembled from one single-seed ``jvp`` per column: the per-seed reference."""
  return sc.stack([jvp(out, wrt, basis(wrt.shape, j)).reshape((out.size,)) for j in range(wrt.size)], axis=1)


def _check_forward(
  name: str, inputs: Sequence[sc.Expr], outputs: Sequence[sc.Expr], point: Sequence[np.ndarray], *, rtol: float = 1e-6, atol: float = 1e-7
) -> None:
  """The multi-seed Jacobian of every output with respect to every input matches the per-seed
  columns to rounding and finite differences to truncation error."""
  primal = _fn(name, inputs, outputs)
  for k, wrt in enumerate(inputs):
    exprs = [e for out in outputs for e in (jacobian(out, wrt), _columns(out, wrt))]
    got = _run(f"{name}_d{k}", inputs, exprs, point)
    for i in range(len(outputs)):

      def at(v: np.ndarray, i: int = i, k: int = k) -> np.ndarray:
        return primal._flat_numerical_call(*point[:k], v, *point[k + 1 :])[i]

      many, single = got[2 * i], got[2 * i + 1]
      np.testing.assert_allclose(many, single, rtol=1e-10, atol=1e-12)
      np.testing.assert_allclose(many, finite_difference(at, point[k]).reshape(many.shape), rtol=rtol, atol=atol)


def _loop_nodes(exprs: Sequence[sc.Expr]) -> list[sc.Expr]:
  """Every ``scan`` and ``while_loop`` node in the graph of ``exprs``, callees included."""
  found, seen, todo = [], set(), list(exprs)
  while todo:
    for node in topo(todo):
      if node.id in seen:
        continue
      seen.add(node.id)
      if node.op in (ExprOp.SCAN, ExprOp.WHILE):
        found.append(node)
      if node.op in CALLEE_OPS:
        todo.extend(out for fn in callees_of(node) for out in fn.outputs)
    todo = [e for e in todo if e.id not in seen]
  return found


def _callee_names(exprs: Sequence[sc.Expr]) -> set[str]:
  """The names of every Function a node in the graph of ``exprs`` calls, maps or loops over."""
  names, seen, todo = set(), set(), list(exprs)
  while todo:
    for node in topo(todo):
      if node.id in seen:
        continue
      seen.add(node.id)
      if node.op in CALLEE_OPS:
        for fn in callees_of(node):
          names.add(fn.name)
          todo.extend(fn.outputs)
    todo = [e for e in todo if e.id not in seen]
  return names


def _mix_body(name: str) -> sc.ConcreteFunction:
  """A carry of two, two sliced inputs of sizes two and one, and two stacked outputs of different sizes."""
  c, a, b = sc.sym("c", 2), sc.sym("a", 2), sc.sym("b", 1)
  nxt = sc.stack([c[0] * a[0] + (c[1] * b[0]).sin(), c[1] - 0.3 * c[0] * a[1] + b[0] * b[0]])
  return sc.Function._from_exprs(name, [c, a, b], [nxt, sc.stack([c[0] * c[1] + a[0]]), (c * b[0]).cos()], ["c", "a", "b"], ["cn", "y", "z"])


def _fixed_point(name: str, tol: float = 1e-13) -> tuple[sc.Function, sc.Function]:
  """``x = 0.5 cos(x) + p`` for two independent pairs; the carry is ``[x, p]`` and the loop contracts."""
  c = sc.sym("c", 4)
  x, p = c[:2], c[2:]
  step = 0.5 * x.cos() + p
  body = sc.Function._from_exprs(f"{name}_step", [c], [sc.concat([step, p])], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"{name}_go", [c], [sc.norm_inf(step - x) > tol], ["c"], ["go"])
  return cond, body


# --- _strided_view, _jvp_many_joint and _contains_loop ------------------------------------------


def _view_reads(source: sc.Expr, composed: sc.Expr, idx: np.ndarray, rows: int, width: int, name: str) -> tuple[sc.Expr, int, int]:
  """Take the strided view and check numerically that step ``r`` of a loop slicing it reads
  ``composed.flat[idx[r]]``."""
  view, start, stride = _strided_view(composed, idx, rows, width)
  assert view.shape == (view.size,)
  value = RNG.standard_normal(source.shape)
  got_composed, got_view = _run(name, [source], [composed.reshape((composed.size,)), view], [value])
  for r, row in enumerate(idx.reshape(rows, width)):
    np.testing.assert_array_equal(got_view[start + r * stride : start + r * stride + width], got_composed[row])
  return view, start, stride


def _in_place_root(view: sc.Expr) -> sc.Expr:
  return view.args[0] if view.op == ExprOp.RESHAPE else view


@pytest.mark.parametrize(
  ("case", "rows", "width", "start", "stride"),
  [
    ("plain", 3, 2, 1, 4),
    ("reshape_chain", 3, 2, 1, 4),
    ("gather_chain", 3, 2, 7, -3),
    ("backwards", 4, 3, 21, -6),
    ("single_row", 1, 5, 2, 0),
    ("repeated_row", 3, 2, 4, 0),
  ],
)
def test_strided_view_reads_a_contiguous_composed_table_in_place(case: str, rows: int, width: int, start: int, stride: int) -> None:
  """When the index table, composed through the gathers and reshapes that produced the tensor,
  reads one contiguous block per row at a constant step, the view is the source itself."""
  y = sc.sym("y", 24)
  table = start + np.arange(rows)[:, None] * stride + np.arange(width)[None, :]
  if case == "reshape_chain":
    composed, idx = y.reshape((4, 6)).reshape((2, 3, 4)), table
  elif case == "gather_chain":
    # composed[k] = y[23 - k]: reversed, so the composed table reads backwards where idx reads forwards.
    composed, idx = sc.gather(y, np.arange(24)[::-1].copy()).reshape((6, 4)), 23 - table
  else:
    composed, idx = y, table
  view, got_start, got_stride = _view_reads(y, composed, idx, rows, width, f"sv_in_{case}")
  assert _in_place_root(view) is y
  assert (got_start, got_stride) == (start, stride)


@pytest.mark.parametrize(("case", "rows", "width"), [("gap_in_row", 3, 2), ("uneven_steps", 3, 2), ("single_row_gap", 1, 3), ("permuted", 2, 3)])
def test_strided_view_materializes_a_table_it_cannot_read_in_place(case: str, rows: int, width: int) -> None:
  """A table with a gap inside a row, rows at uneven steps, or a permutation is gathered into a
  fresh buffer that the loop reads with stride ``width`` (0 for a single row)."""
  y = sc.sym("y", 20)
  idx = {
    "gap_in_row": np.array([[0, 2], [4, 6], [8, 10]]),
    "uneven_steps": np.array([[0, 1], [3, 4], [9, 10]]),
    "single_row_gap": np.array([[3, 5, 6]]),
    "permuted": np.array([[2, 1, 0], [5, 4, 3]]),
  }[case]
  view, start, stride = _view_reads(y, y, idx, rows, width, f"sv_gather_{case}")
  assert view.op == ExprOp.GATHER
  assert (start, stride) == (0, 0 if rows == 1 else width)


def test_strided_view_of_an_empty_table() -> None:
  """A zero-width table, what a zero-size formal produces, still returns a readable view."""
  y = sc.sym("y", 6)
  view, start, stride = _strided_view(y, np.zeros((3, 0), dtype=np.int64), 3, 0)
  assert view.size == 0 and (start, stride) == (0, 0)


def test_jvp_many_joint_with_one_input_is_jvp_many() -> None:
  x = sc.sym("x", (2, 2))
  out = (x @ x).sin().reshape((4,))
  seeds = sc.const(RNG.standard_normal((3, 2, 2)))
  (joint,) = _jvp_many_joint([out], {x: seeds}, 3)
  point = RNG.standard_normal((2, 2))
  got, ref = _run("jj_one", [x], [joint, jvp_many(out, x, seeds)], [point])
  assert got.shape == (3, 4)
  np.testing.assert_array_equal(got, ref)


def test_jvp_many_joint_sums_the_tangents_of_several_inputs() -> None:
  """A scalar, a matrix and a vector viewed as one joint vector: each output's tangent is the sum of
  the per-input multi-seed tangents, shaped ``(nseed, *out.shape)``, and the joint symbol does not
  survive in the result."""
  nseed = 3
  s, m, v = sc.sym("s"), sc.sym("m", (2, 3)), sc.sym("v", 4)
  outs = [(m * s).sin() @ v[:3], sc.stack([s * v.sum(), (m.sum() * v[3]).exp()]), (s * s).reshape((1,))]
  seeds = {s: sc.sym("ds", (nseed,)), m: sc.sym("dm", (nseed, 2, 3)), v: sc.sym("dv", (nseed, 4))}
  joint = _jvp_many_joint(outs, seeds, nseed)
  assert [t.shape for t in joint] == [(nseed, *o.shape) for o in outs]
  free = {str(n.name) for n in topo(joint) if n.op == ExprOp.INPUT}
  assert free <= {"s", "m", "v", "ds", "dm", "dv"}
  ref = [sum((jvp_many(o, w, d) for w, d in seeds.items()), start=sc.const(np.zeros((nseed, *o.shape)))) for o in outs]
  inputs = [s, m, v, *seeds.values()]
  point = [RNG.standard_normal(x.shape) for x in inputs]
  got = _run("jj_many", inputs, [*joint, *ref], point)
  for a, b in zip(got[:3], got[3:], strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)


def test_jvp_many_joint_of_zero_seeds_is_zero() -> None:
  s, v = sc.sym("s"), sc.sym("v", 3)
  outs = [(v * s).sin(), s.cos()]
  joint = _jvp_many_joint(outs, {s: sc.const(np.zeros(2)), v: sc.const(np.zeros((2, 3)))}, 2)
  got = _run("jj_zero", [s, v], joint, [np.array(0.4), np.arange(3.0)])
  assert [g.shape for g in got] == [(2, 3), (2,)]
  assert all(not g.any() for g in got)


def test_contains_loop_looks_through_callees() -> None:
  x = sc.sym("x", 2)
  plain = sc.Function._from_exprs("cl_plain", [x], [x.sin()], ["x"], ["y"])
  step = sc.Function._from_exprs("cl_step", [x], [x * 0.5], ["x"], ["y"])
  (fin,) = sc.scan(step, x, [], length=3)
  scanned = sc.Function._from_exprs("cl_scan", [x], [fin], ["x"], ["y"])
  cond = sc.Function._from_exprs("cl_go", [x], [x[0] > 1.0], ["x"], ["go"])
  looped = sc.Function._from_exprs("cl_while", [x], [sc.while_loop(cond, step, x, max_iter=4)[0]], ["x"], ["y"])
  called = sc.Function._from_exprs("cl_call", [x], [scanned._flat_symbolic_call([plain._flat_symbolic_call([x])[0]])[0]], ["x"], ["y"])
  xs = sc.sym("xs", 4)
  mapped = sc.Function._from_exprs("cl_vmap", [xs], [sc.vmap(looped, 2, [(xs, 0, 2)])], ["xs"], ["y"])
  assert not _contains_loop(plain)
  assert all(_contains_loop(fn) for fn in (scanned, looped, called, mapped))


# --- scan -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("carry_shape", [(), (3,), (2, 2)])
def test_scan_carries_of_rank_0_1_2_with_a_matrix_formal(carry_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch) -> None:
  """The tangent carry is ``(nseed, *carry.shape)``, seed-major, for a scalar, vector or matrix
  carry, and a rank-2 formal sliced from a flat outer tensor gets ``(nseed, 2, 2)`` tangents."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, m = sc.sym("c", carry_shape), sc.sym("m", (2, 2))
  nxt = c * m[0, 0] + 0.1 * (c * c).sin() * m[1, 1] + m[0, 1]
  if carry_shape == (2, 2):
    nxt = nxt + 0.2 * (c @ m.transpose((1, 0))) + 0.1 * c.transpose((1, 0))
  tag = len(carry_shape)
  body = sc.Function._from_exprs(f"rk{tag}_step", [c, m], [nxt, sc.stack([(c * c).sum() * m[1, 0]])], ["c", "m"], ["cn", "y"])
  c0, ms = sc.sym("c0", carry_shape), sc.sym("ms", 16)
  fin, ys = sc.scan(body, c0, [(ms, 0, 4)], length=4)
  point = [0.3 + 0.1 * RNG.standard_normal(carry_shape), 0.8 + 0.2 * RNG.standard_normal(16)]
  _check_forward(f"rk{tag}", [c0, ms], [fin, ys], point)


def test_scan_with_inactive_sliced_inputs_names_its_active_set(monkeypatch: pytest.MonkeyPatch) -> None:
  """Only the sliced inputs whose tangent is nonzero enter the tangent body; the others stay
  primal. The tangent body's name records how many seeds it carries and which inputs are active."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("ia_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 10), sc.sym("b_all", 5)
  outs = sc.scan(body, c0, [(a_all, 0, 2), (b_all, 0, 1)], length=5)
  point = [np.array([0.4, -0.2]), np.linspace(0.5, 1.2, 10), np.linspace(-0.3, 0.6, 5)]
  _check_forward("ia", [c0, a_all, b_all], list(outs), point)
  for wrt, suffix in ((c0, "c"), (a_all, "0"), (b_all, "1")):
    names = {n.attrs["callee"].name for n in _loop_nodes([jacobian(outs[1], wrt)])}
    assert names == {f"ia_step_scanfwd{wrt.size}_{suffix}"}


def test_scan_with_every_input_active_through_a_shared_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
  """When one symbol drives the carry and both sliced inputs, all of them are active in one body."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("all_step")
  p = sc.sym("p", 3)
  a_all = sc.concat([p, p.sin(), p * p, p[:1]])
  b_all = (p * 2.0).cos()
  outs = sc.scan(body, p[:2] * 0.5, [(a_all, 1, 2), (b_all, 2, -1)], length=3)
  _check_forward("all", [p], list(outs), [np.array([0.3, -0.7, 1.1])])
  assert {n.attrs["callee"].name for n in _loop_nodes([jacobian(outs[0], p)])} == {"all_step_scanfwd3_0_1"}


def test_scan_with_one_outer_bound_to_two_formals(monkeypatch: pytest.MonkeyPatch) -> None:
  """Two formals read overlapping windows of one outer tensor; both windows' tangents count."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("tw_step")
  c0, u = sc.sym("c0", 2), sc.sym("u", 9)
  outs = sc.scan(body, c0, [(u, 0, 2), (u, 1, 2)], length=4)
  _check_forward("tw", [c0, u], list(outs), [np.array([0.2, 0.9]), np.linspace(-0.8, 0.8, 9)])


@pytest.mark.parametrize(
  ("a_spec", "b_spec"),
  [((0, 0), (8, -2)), ((3, 2), (0, 0)), ((10, -3), (2, 1)), ((1, 1), (5, -1)), ((10, 0), (8, 0))],
  ids=["broadcast_backward", "offset_broadcast", "backward_offset", "overlap_backward", "both_broadcast"],
)
def test_scan_slicing_starts_strides_and_broadcasts(a_spec: tuple[int, int], b_spec: tuple[int, int], monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("st_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 12), sc.sym("b_all", 9)
  outs = sc.scan(body, c0, [(a_all, *a_spec), (b_all, *b_spec)], length=4)
  point = [np.array([0.5, -0.4]), np.linspace(0.3, 1.4, 12), np.linspace(-0.9, 0.7, 9)]
  _check_forward("st", [c0, a_all, b_all], list(outs), point)


@pytest.mark.parametrize("length", [0, 1])
def test_scan_of_length_zero_and_one(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
  """Zero steps return the init's tangent and empty stacked outputs; one step is the body's tangent."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body(f"l{length}_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 4), sc.sym("b_all", 3)
  outs = sc.scan(body, c0, [(a_all, 2, 1), (b_all, 1, 1)], length=length)
  point = [np.array([0.5, -0.4]), np.linspace(0.3, 1.4, 4), np.linspace(-0.9, 0.7, 3)]
  _check_forward(f"l{length}", [c0, a_all, b_all], list(outs), point)
  if length == 0:
    got = _run("l0_j", [c0, a_all, b_all], [jacobian(outs[0], c0), jacobian(outs[0], a_all), jacobian(outs[1], c0)], point)
    np.testing.assert_array_equal(got[0], np.eye(2))
    assert not got[1].any() and got[2].shape == (0, 2)


def test_scan_with_a_single_seed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("ns1_step")
  c0, a_all, b = sc.sym("c0", 2), sc.sym("a_all", 8), sc.sym("b", 1)
  outs = sc.scan(body, c0, [(a_all, 0, 2), (b, 0, 0)], length=4)
  _check_forward("ns1", [c0, a_all, b], list(outs), [np.array([0.5, -0.4]), np.linspace(0.3, 1.4, 8), np.array([0.6])])
  assert {n.attrs["callee"].name for n in _loop_nodes([jacobian(outs[2], b)])} == {"ns1_step_scanfwd1_1"}


def test_scan_stored_carries_output(monkeypatch: pytest.MonkeyPatch) -> None:
  """Output ``-1``, the carry entering every step that reverse mode reads, has its own tangent:
  the stored tangent carries, seed-major."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("sc_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 10), sc.sym("b_all", 5)
  stored = _scan_node(body, c0, (a_all, b_all), (0, 4), (2, -1), 5, -1)
  point = [np.array([0.5, -0.4]), np.linspace(0.3, 1.4, 10), np.linspace(-0.9, 0.7, 5)]
  _check_forward("sc", [c0, a_all, b_all], [stored], point)


def test_scan_tangents_for_arbitrary_and_symbolic_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
  """Seeds need not be identity columns: a dense constant seed matrix and a seed matrix that is
  itself an input both match one single-seed ``jvp`` per row."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("sd_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 10), sc.sym("b_all", 5)
  outs = sc.scan(body, c0, [(a_all, 0, 2), (b_all, 4, -1)], length=5)
  const_seeds = RNG.standard_normal((4, 10))
  sym_seeds = sc.sym("S", (3, 10))
  many = [jvp_many(o, a_all, sc.const(const_seeds)) for o in outs] + [jvp_many(o, a_all, sym_seeds) for o in outs]
  single = [sc.stack([jvp(o, a_all, sc.const(row)) for row in const_seeds]) for o in outs]
  single += [sc.stack([jvp(o, a_all, sym_seeds[k]) for k in range(3)]) for o in outs]
  point = [np.array([0.5, -0.4]), np.linspace(0.3, 1.4, 10), np.linspace(-0.9, 0.7, 5), RNG.standard_normal((3, 10))]
  got = _run("sd", [c0, a_all, b_all, sym_seeds], [*many, *single], point)
  for a, b in zip(got[: len(many)], got[len(many) :], strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
  jac = _run("sd_jac", [c0, a_all, b_all], [jacobian(outs[0], a_all)], point[:3])[0]
  np.testing.assert_allclose(got[0], const_seeds @ jac.T, rtol=1e-12, atol=1e-13)


def test_scan_body_with_passthrough_and_boolean_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
  """A body that returns its carry unchanged, returns a sliced input as a stacked output, or stacks a
  boolean still gets one tangent loop; the boolean output has no tangent to carry."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, a = sc.sym("c", 2), sc.sym("a", 2)
  body = sc.Function._from_exprs("pb_step", [c, a], [c, a, c * a, sc.stack([c[0] > 0.1])], ["c", "a"], ["cn", "y", "z", "flag"])
  c0, a_all = sc.sym("c0", 2), sc.sym("a_all", 6)
  fin, ys, zs, _ = sc.scan(body, c0, [(a_all, 4, -2)], length=3)
  _check_forward("pb", [c0, a_all], [fin, ys, zs], [np.array([0.3, 0.2]), np.linspace(-1.0, 1.0, 6)])


def test_scan_whose_init_and_sliced_input_are_one_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
  """The same symbol seeds the carry and a broadcast sliced input, and the body's formal names are
  the caller's symbols in swapped roles: interned symbols must not be confused."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, a = sc.sym("c", 2), sc.sym("a", 2)
  step = sc.Function._from_exprs("so_step", [c, a], [c * a + a[::-1].sin(), c * 2.0], ["c", "a"], ["cn", "y"])
  outs = sc.scan(step, a, [(c, 0, 0)], length=3)
  point = [np.array([0.3, 0.7]), np.array([1.1, -0.4])]
  _check_forward("so", [c, a], list(outs), point)
  cost = sc.sumsqr(outs[0]) + outs[1].sum()
  for k in range(2):
    _check_hessian(f"so{k}", cost, [c, a], k, point)


@pytest.mark.parametrize(
  ("spec", "length", "in_place"), [(("ys", 4, -1), 5, True), (("ys", 0, 1), 5, True), (("zs", 1, 2), 4, False), (("zs", 8, -2), 5, True)]
)
def test_scan_reading_another_scans_tangent_trajectory(
  spec: tuple[str, int, int], length: int, in_place: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A second scan slices the stacked outputs of a first. Where every step's seeds sit in one
  contiguous block of the first scan's tangent output, the second reads it in place; otherwise the
  slices are gathered first. Both give the same derivatives."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body(f"tt_{spec[0]}{spec[1]}{spec[2]}_step")
  c, x = sc.sym("c", 2), sc.sym("x", 2 if spec == ("zs", 8, -2) else 1)
  second = sc.Function._from_exprs(
    f"tt_{spec[0]}{spec[1]}{spec[2]}_next", [c, x], [c * x[0] + 0.1 * c[::-1], sc.stack([c.sum()])], ["c", "x"], ["cn", "y"]
  )
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 10), sc.sym("b_all", 5)
  fin, ys, zs = sc.scan(body, c0, [(a_all, 0, 2), (b_all, 0, 1)], length=5)
  outer, start, stride = spec
  outs = sc.scan(second, fin, [({"ys": ys, "zs": zs}[outer], start, stride)], length=length)
  point = [np.array([0.3, 0.2]), np.linspace(0.3, 1.0, 10), np.linspace(-0.5, 0.5, 5)]
  _check_forward(f"tt_{outer}{start}{stride}", [c0, a_all, b_all], list(outs), point)
  (tangent_scan,) = [n for n in _loop_nodes([jacobian(outs[0], a_all)]) if n.attrs["callee"].name.endswith("_next_scanfwd10_0")]
  read = tangent_scan.args[-1]
  copied = False
  while read.op in (ExprOp.RESHAPE, ExprOp.GATHER):
    copied |= read.op == ExprOp.GATHER
    read = read.args[0]
  assert read.op == ExprOp.SCAN
  assert copied != in_place


def test_an_op_without_a_multi_seed_rule_in_a_body_falls_back_inside_one_loop(monkeypatch: pytest.MonkeyPatch) -> None:
  """``atan2`` has no multi-seed rule. Without strict mode the body's tangent falls back to one pass
  per seed inside the single tangent loop, so the loop count still does not grow with the seeds;
  strict mode names the op."""
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  body = sc.Function._from_exprs("fb_step", [c, u], [sc.stack([sc.atan2(c[1], c[0] + 2.0) + u[0], c[0] * u[0]])], ["c", "u"], ["cn"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 4)
  (fin,) = sc.scan(body, c0, [(us, 0, 1)], length=4)
  _check_forward("fb", [c0, us], [fin], [np.array([0.3, -0.2]), np.linspace(0.5, 1.0, 4)])
  counts = {nseed: len(_loop_nodes([jvp_many(fin, us, sc.const(RNG.standard_normal((nseed, 4))))])) for nseed in (2, 3)}
  assert counts[2] == counts[3] == 1
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  (fresh,) = sc.scan(sc.Function._from_exprs("fb_strict_step", list(body.inputs), list(body.outputs), ["c", "u"], ["cn"]), c0, [(us, 0, 1)], length=4)
  with pytest.raises(NotImplementedError, match="atan2"):
    jacobian(fresh, us)


# --- nested loops, calls and maps ---------------------------------------------------------------


def _inner_scan_body(name: str) -> sc.Function:
  ic, iu = sc.sym("ic", 2), sc.sym("iu", 1)
  return sc.Function._from_exprs(name, [ic, iu], [sc.stack([ic[0] + 0.3 * ic[1] * iu[0], ic[1].sin() + iu[0]])], ["ic", "iu"], ["icn"])


def test_scan_in_a_scan_body(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  inner = _inner_scan_body("ss_inner")
  c, w = sc.sym("c", 2), sc.sym("w", 3)
  (fin,) = sc.scan(inner, c, [(w, 0, 1)], length=3)
  outer = sc.Function._from_exprs("ss_outer", [c, w], [0.8 * fin + 0.1 * c.cos(), sc.stack([fin.sum()])], ["c", "w"], ["cn", "y"])
  c0, ws = sc.sym("c0", 2), sc.sym("ws", 12)
  outs = sc.scan(outer, c0, [(ws, 9, -3)], length=4)
  _check_forward("ss", [c0, ws], list(outs), [np.array([0.2, -0.3]), 0.4 * np.sin(np.arange(12.0))])


def _scan_callee(name: str) -> sc.Function:
  """A Function whose output is a scan: ``x`` evolved three steps under a broadcast parameter."""
  x, p = sc.sym("x", 2), sc.sym("p", 1)
  (fin,) = sc.scan(_inner_scan_body(f"{name}_step"), x, [(p, 0, 0)], length=3)
  return sc.Function._from_exprs(name, [x, p], [fin * p[0]], ["x", "p"], ["y"])


@pytest.mark.parametrize("pre", ["identity", "nonlinear"])
def test_vmap_of_a_function_holding_a_scan(pre: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """A ``vmap`` whose callee holds a scan gets a tangent helper with one multi-seed loop in it, for
  constant identity seeds (tiled or colored per formal) and for data-dependent seeds (generic)."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  g = _scan_callee(f"vs_{pre}_g")
  xs, ps = sc.sym("xs", 6), sc.sym("ps", 3)
  arg = xs if pre == "identity" else (xs * 0.7).sin() + xs
  mapped = sc.vmap(g, 3, [(arg, 0, 2), (ps, 0, 1)])
  _check_forward(f"vs_{pre}", [xs, ps], [mapped], [0.3 * np.cos(np.arange(6.0)), np.array([0.9, 1.1, 0.7])])
  names = _callee_names([jacobian(mapped, xs), jacobian(mapped, ps)])
  assert any(n.startswith(f"vs_{pre}_g_fwd") for n in names), names
  assert not any("scanfwd_" in n for n in names), "a per-seed tangent scan was built"


def test_vmap_in_a_scan_body(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  ci, ui = sc.sym("ci", 1), sc.sym("ui", 1)
  h = sc.Function._from_exprs("vm_h", [ci, ui], [ci * ui + (ci * 0.5).sin()], ["ci", "ui"], ["y"])
  c, u = sc.sym("c", 2), sc.sym("u", 2)
  mapped = sc.vmap(h, 2, [(c, 0, 1), (u, 0, 1)])
  body = sc.Function._from_exprs("vm_step", [c, u], [0.5 * c + mapped, sc.stack([mapped.sum()])], ["c", "u"], ["cn", "y"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 8)
  outs = sc.scan(body, c0, [(us, 0, 2)], length=4)
  _check_forward("vm", [c0, us], list(outs), [np.array([0.2, -0.3]), np.linspace(-1.0, 1.0, 8)])


def test_while_loop_in_a_scan_body(monkeypatch: pytest.MonkeyPatch) -> None:
  """Each step solves a fixed point to full precision, so the derivative through the solver's steps
  equals the smooth one that finite differences see."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  z = sc.sym("z", 2)
  fp = sc.Function._from_exprs("ws_fp", [z], [sc.stack([0.5 * z[0].cos() + z[1], z[1]])], ["z"], ["zn"])
  go = sc.Function._from_exprs("ws_go", [z], [(0.5 * z[0].cos() + z[1] - z[0]).abs() > 1e-14], ["z"], ["go"])
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  sol, _ = sc.while_loop(go, fp, sc.stack([c[0], c[0] * u[0] + 0.3 * c[1]]), max_iter=200)
  body = sc.Function._from_exprs("ws_step", [c, u], [sc.stack([sol[0], c[1] + 0.1 * sol[0]]), sol[:1]], ["c", "u"], ["cn", "x"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 3)
  outs = sc.scan(body, c0, [(us, 0, 1)], length=3)
  _check_forward("ws", [c0, us], list(outs), [np.array([0.1, 0.4]), np.array([0.5, -0.2, 0.8])], rtol=1e-5, atol=1e-6)


def test_scan_in_a_while_loop_body(monkeypatch: pytest.MonkeyPatch) -> None:
  """The while body runs a scan and counts its steps in the last carry entry; the count decides
  the number of iterations and has no tangent."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c = sc.sym("c", 3)
  (fin,) = sc.scan(_inner_scan_body("sw_inner"), c[:2], [(c[2:] * 0.0 + 0.4, 0, 0)], length=2)
  body = sc.Function._from_exprs("sw_step", [c], [sc.concat([0.5 * fin, c[2:] + 1.0])], ["c"], ["cn"])
  cond = sc.Function._from_exprs("sw_go", [c], [c[2] < 3.5], ["c"], ["go"])
  c0 = sc.sym("c0", 3)
  final, _ = sc.while_loop(cond, body, c0, max_iter=10)
  _check_forward("sw", [c0], [final[:2]], [np.array([0.3, -0.6, 0.0])])
  names = {n.attrs["callee"].name for n in _loop_nodes([jacobian(final, c0)]) if "fwd" in n.attrs["callee"].name}
  assert "sw_step_whilefwd3" in names and {n for n in names if n.startswith("sw_inner_scanfwd")} <= {"sw_inner_scanfwd3_c", "sw_inner_scanfwd3_0"}


@pytest.mark.parametrize("loop", ["scan", "while"])
def test_a_call_around_a_loop_is_differentiated_in_place(loop: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """A CALL whose callee holds a loop is inlined, so no per-callee tangent helper appears and the
  tangent loop sits in the caller's graph."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x, p = sc.sym("x", 2), sc.sym("p", 1)
  if loop == "scan":
    (fin,) = sc.scan(_inner_scan_body("ca_scan_step"), x, [(p, 0, 0)], length=4)
  else:
    cond, body = _fixed_point("ca_fp")
    fin = sc.while_loop(cond, body, sc.concat([x, x * p[0]]), max_iter=200)[0][:2]
  callee = sc.Function._from_exprs(f"ca_{loop}", [x, p], [fin * p[0], fin.sum().reshape((1,))], ["x", "p"], ["y", "s"])
  q = sc.sym("q", 3)
  y, s = callee._flat_symbolic_call([q[:2].sin(), q[2:]])
  host = [y + s[0], s * q[0]]
  _check_forward(f"ca_{loop}_host", [q], host, [np.array([0.4, -0.1, 0.8])], rtol=1e-5, atol=1e-6)
  jac = jacobian(host[0], q)
  names = _callee_names([jac])
  assert not any(n.startswith(f"ca_{loop}_fwd") for n in names), names
  assert any("fwd3" in n.attrs["callee"].name for n in _loop_nodes([jac]))


def test_a_call_inlined_with_its_formals_swapped(monkeypatch: pytest.MonkeyPatch) -> None:
  """The caller passes its ``y`` as the callee's ``x`` and its ``x`` as the callee's ``y``. Symbols
  are interned by name, so the inlining substitution must replace all formals at once."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x, y = sc.sym("x", 2), sc.sym("y", 2)
  c, a = sc.sym("c", 2), sc.sym("a", 1)
  step = sc.Function._from_exprs("cs_step", [c, a], [c * a[0] + c[::-1] * 0.3], ["c", "a"], ["cn"])
  (fin,) = sc.scan(step, x, [(y, 0, 1)], length=2)
  f = sc.Function._from_exprs("cs_f", [x, y], [fin + x * 0.1], ["x", "y"], ["o"])
  (swapped,) = f._flat_symbolic_call([y, x])
  (mixed,) = f._flat_symbolic_call([y * 2.0, x.sin()])
  _check_forward("cs", [x, y], [swapped, mixed], [np.array([0.3, 0.7]), np.array([1.1, -0.4])])


def test_vmap_of_a_loop_with_seed_tiles_that_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
  """Every other iteration of the map sees no seed. The zero tile's helper carries no seeds at all
  and the other tile's helper carries only the nonzero ones."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  g = _scan_callee("zt_g")
  q, ps = sc.sym("q", 2), sc.sym("ps", 4)
  zero = sc.const(np.zeros(2))
  mapped = sc.vmap(g, 4, [(sc.concat([q, zero, q, zero]), 0, 2), (ps, 0, 1)])
  _check_forward("zt", [q, ps], [mapped], [np.array([0.3, -0.2]), np.array([0.9, 1.1, 0.7, 1.2])])


# --- while_loop -----------------------------------------------------------------------------------


def test_while_loop_jacobian_carries_every_seed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  cond, body = _fixed_point("wj")
  c0 = sc.sym("c0", 4)
  final, count = sc.while_loop(cond, body, c0, max_iter=200)
  _check_forward("wj", [c0], [final, count.reshape((1,))], [np.array([0.0, 1.0, 0.3, -0.2])], rtol=1e-5, atol=1e-6)
  assert {n.attrs["callee"].name for n in _loop_nodes([jacobian(final, c0)]) if "fwd" in n.attrs["callee"].name} == {"wj_step_whilefwd4"}


@pytest.mark.parametrize("carry_shape", [(), (2, 2)])
def test_while_loop_carries_of_rank_0_and_2(carry_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  tag = len(carry_shape)
  c = sc.sym("c", carry_shape)
  nxt = 0.5 * c.cos() + 0.1 if tag == 0 else 0.5 * (c @ c.transpose((1, 0))).cos() * 0.5 + 0.1
  body = sc.Function._from_exprs(f"wr{tag}_step", [c], [nxt], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"wr{tag}_go", [c], [sc.norm_inf((nxt - c).reshape((c.size,))) > 1e-14], ["c"], ["go"])
  c0 = sc.sym("c0", carry_shape)
  final, _ = sc.while_loop(cond, body, c0, max_iter=100)
  _check_forward(f"wr{tag}", [c0], [final], [np.full(carry_shape, 0.3) + (0.1 * np.eye(2) if tag else 0.0)], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
  ("max_iter", "runs"), [(0, True), (1, True), (4, True), (6, False)], ids=["max_iter_0", "max_iter_1", "max_iter_4", "stops_at_0"]
)
def test_while_loop_edge_step_counts(max_iter: int, runs: bool, monkeypatch: pytest.MonkeyPatch) -> None:
  """No step (``max_iter`` 0, or a condition false at the start) gives the identity; one or more
  steps give the chain of body Jacobians."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c = sc.sym("c", 3)
  body = sc.Function._from_exprs(f"we{max_iter}{runs}_step", [c], [0.5 * c * c[::-1] + c.sin()], ["c"], ["cn"])
  cond = sc.Function._from_exprs(f"we{max_iter}{runs}_go", [c], [c[0] < 1e9 if runs else c[0] > 1e9], ["c"], ["go"])
  c0 = sc.sym("c0", 3)
  final, count = sc.while_loop(cond, body, c0, max_iter=max_iter)
  start = np.array([0.3, -0.2, 0.5])
  _check_forward(f"we{max_iter}{runs}", [c0], [final, count.reshape((1,))], [start])
  jac, dcount = _run(f"we{max_iter}{runs}_j", [c0], [jacobian(final, c0), jacobian(count, c0)], [start])
  if max_iter == 0 or not runs:
    np.testing.assert_array_equal(jac, np.eye(3))
  assert not dcount.any()


def test_while_loop_stored_carries_output(monkeypatch: pytest.MonkeyPatch) -> None:
  """Output ``-1`` holds the carry entering each of the ``max_iter`` slots, with the final carry
  repeated past the last step; its tangent follows the same layout."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c = sc.sym("c", 2)
  body = sc.Function._from_exprs("wsc_step", [c], [sc.stack([c[0] * c[1] + 0.2, (c[0] * 0.5).sin()])], ["c"], ["cn"])
  cond = sc.Function._from_exprs("wsc_go", [c], [c[0] < 0.75], ["c"], ["go"])
  c0 = sc.sym("c0", 2)
  stored = _while_node(cond, body, c0, 6, -1)
  _check_forward("wsc", [c0], [stored], [np.array([0.7, 0.8])])
  _check_forward("wsc_all", [c0], [stored], [np.array([0.1, 0.8])])


# --- Hessians (forward over reverse) ------------------------------------------------------------


def _check_hessian(name: str, cost: sc.Expr, inputs: Sequence[sc.Expr], k: int, point: Sequence[np.ndarray], *, rtol: float = 1e-6) -> np.ndarray:
  """The dense Hessian is symmetric, matches finite differences of the gradient, matches the
  per-seed Hessian and agrees with ``sparse_hessian`` in full and in one triangle."""
  wrt = inputs[k]
  grad = sc.gradient(cost, wrt).reshape((wrt.size,))
  hess = sc.hessian(cost, wrt)
  full, lower = sc.sparse_hessian(cost, wrt), sc.sparse_hessian(cost, wrt, triangle="lower")
  got = _run(name, inputs, [hess, _columns(grad, wrt), full.to_dense(), lower.values, grad], point)
  h, per_seed, dense_sparse, lower_values = got[:4]
  np.testing.assert_allclose(h, h.T, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(h, per_seed, rtol=1e-10, atol=1e-12)
  np.testing.assert_allclose(dense_sparse, h, rtol=1e-10, atol=1e-12)
  np.testing.assert_allclose(lower_values, h[list(lower.sparsity.rows), list(lower.sparsity.cols)], rtol=1e-10, atol=1e-12)
  grad_fn = _fn(f"{name}_g", inputs, [grad])

  def g_at(v: np.ndarray) -> np.ndarray:
    return grad_fn._flat_numerical_call(*point[:k], v, *point[k + 1 :])[0]

  np.testing.assert_allclose(h, finite_difference(g_at, point[k]), rtol=rtol, atol=1e-6)
  return h


@pytest.mark.parametrize("wrt", ["c0", "a_all", "b_all"])
def test_hessian_through_a_scan_with_broadcast_and_backward_slices(wrt: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """The adjoint scans read the tangent trajectory backwards; with a negative-stride and a broadcast
  input the Hessian still matches the gradient's finite differences."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body(f"hs_{wrt}_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 2), sc.sym("b_all", 9)
  fin, ys, zs = sc.scan(body, c0, [(a_all, 0, 0), (b_all, 8, -2)], length=4)
  cost = sc.sumsqr(fin) + (ys * ys).sum() + zs.sum()
  inputs = [c0, a_all, b_all]
  point = [np.array([0.5, -0.4]), np.array([0.9, 1.2]), np.linspace(-0.9, 0.7, 9)]
  _check_hessian(f"hs_{wrt}", cost, inputs, ["c0", "a_all", "b_all"].index(wrt), point)


def test_function_level_wrappers_through_a_scan(monkeypatch: pytest.MonkeyPatch) -> None:
  """``sc.jacobian``, ``sc.hessian`` and ``sc.sparse_hessian`` on a named Function output agree with
  the Expr-level derivatives, the sparse Hessian in each triangle."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body("fl_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 8), sc.sym("b_all", 4)
  fin, ys, zs = sc.scan(body, c0, [(a_all, 6, -2), (b_all, 0, 1)], length=4)
  cost = sc.sumsqr(fin) + (ys * zs[1::2]).sum()
  fn = sc.Function._from_exprs("fl", [c0, a_all, b_all], [cost, ys], ["c0", "a", "b"], ["f", "ys"])
  point = (np.array([0.4, -0.3]), np.linspace(0.2, 1.0, 8), np.linspace(-0.5, 0.5, 4))
  jac, hess = sc.jacobian(fn, "ys", "a")(point), sc.hessian(fn, "f", "a")(point)
  ref_jac, ref_hess = _run("fl_ref", [c0, a_all, b_all], [jacobian(ys, a_all), sc.hessian(cost, a_all)], point)
  np.testing.assert_allclose(jac, ref_jac, rtol=1e-12, atol=1e-14)
  np.testing.assert_allclose(hess, ref_hess, rtol=1e-12, atol=1e-14)
  for triangle in ("full", "lower", "upper"):
    pattern = sc.sparse_hessian(cost, a_all, triangle=triangle).sparsity
    values = sc.sparse_hessian(fn, "f", "a", triangle=triangle)(point)
    np.testing.assert_allclose(values, hess[list(pattern.rows), list(pattern.cols)], rtol=1e-12, atol=1e-14)


def test_hessian_through_a_nested_scan_inside_a_call(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  g = _scan_callee("hn_g")
  c, u = sc.sym("c", 2), sc.sym("u", 1)
  (y,) = g._flat_symbolic_call([c, u])
  body = sc.Function._from_exprs("hn_step", [c, u], [0.7 * y + 0.2 * c, sc.stack([sc.sumsqr(y)])], ["c", "u"], ["cn", "s"])
  c0, us = sc.sym("c0", 2), sc.sym("us", 4)
  fin, ss = sc.scan(body, c0, [(us, 3, -1)], length=4)
  cost = sc.sumsqr(fin) + ss.sum()
  point = [np.array([0.3, -0.5]), np.array([0.8, 1.1, 0.9, 1.3])]
  for k in range(2):
    _check_hessian(f"hn{k}", cost, [c0, us], k, point)


def test_hessian_through_a_while_loop(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  cond, body = _fixed_point("hw")
  c0 = sc.sym("c0", 4)
  final, count = sc.while_loop(cond, body, c0, max_iter=200)
  cost = sc.sumsqr(final * final) + final[0] * final[3] + 0.0 * count
  _check_hessian("hw", cost, [c0], 0, [np.array([0.0, 1.0, 0.3, -0.2])], rtol=1e-5)
  # A loop that stops after one of its six slots: the adjoint reads stored carries past the last step.
  c = sc.sym("c", 2)
  body = sc.Function._from_exprs("hwe_step", [c], [sc.stack([c[0] * c[1] + 0.2, (c[0] * 0.5).sin()])], ["c"], ["cn"])
  early = sc.Function._from_exprs("hwe_go", [c], [c[0] < 0.75], ["c"], ["go"])
  e0 = sc.sym("e0", 2)
  fin, _ = sc.while_loop(early, body, e0, max_iter=6)
  for k, start in enumerate((np.array([0.7, 0.8]), np.array([0.1, 0.8]))):
    _check_hessian(f"hwe{k}", sc.sumsqr(fin * fin) + fin[0], [e0], 0, [start])


@pytest.mark.parametrize("bound", [0, 1])
def test_hessian_through_loops_that_take_no_or_one_step(bound: int, monkeypatch: pytest.MonkeyPatch) -> None:
  """A zero-length scan and a zero-bound while loop give empty stored trajectories, whose tangents
  must still have the right (empty) shape."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  body = _mix_body(f"h{bound}_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 2), sc.sym("b_all", 1)
  fin, ys, _ = sc.scan(body, c0, [(a_all, 0, 0), (b_all, 0, 0)], length=bound)
  c = sc.sym("c", 2)
  step = sc.Function._from_exprs(f"h{bound}_wstep", [c], [c * c[::-1] + 0.1], ["c"], ["cn"])
  go = sc.Function._from_exprs(f"h{bound}_go", [c], [c[0] < 1e9], ["c"], ["go"])
  wfin, _ = sc.while_loop(go, step, fin, max_iter=bound)
  cost = sc.sumsqr(wfin) + ys.sum() * a_all[1]
  point = [np.array([0.5, -0.4]), np.array([0.9, 1.2]), np.array([0.3])]
  for k in range(3):
    _check_hessian(f"h{bound}_{k}", cost, [c0, a_all, b_all], k, point)


def _higher_order_loop(loop: str) -> tuple[sc.Expr, list[sc.Expr], list[np.ndarray]]:
  """A loop output and its inputs: a scan with a sliced input, or a while loop over a carry."""
  if loop == "scan":
    body = _mix_body("ho_scan_step")
    c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 2), sc.sym("b_all", 4)
    fin, ys, _ = sc.scan(body, c0, [(a_all, 0, 0), (b_all, 3, -1)], length=4)
    return sc.sumsqr(fin) + (ys * ys).sum(), [c0, a_all, b_all], [np.array([0.5, -0.4]), np.array([0.9, 1.2]), np.linspace(-0.9, 0.7, 4)]
  c = sc.sym("c", 3)
  step = sc.Function._from_exprs("ho_while_step", [c], [sc.stack([c[0] * c[1] * 0.5 + c[2].sin(), c[1] * 0.9, c[2] + c[0] * 0.1])], ["c"], ["cn"])
  go = sc.Function._from_exprs("ho_while_go", [c], [c[0] < 1e9], ["c"], ["go"])
  c0 = sc.sym("c0", 3)
  fin, _ = sc.while_loop(go, step, c0, max_iter=3)
  return sc.sumsqr(fin) + fin[0] * fin[2], [c0], [np.array([0.4, 0.8, -0.3])]


@pytest.mark.parametrize("order", ["jacobian_of_jacobian", "jacobian_of_hessian"])
@pytest.mark.parametrize("loop", ["scan", "while"])
def test_forward_over_forward_through_a_loop(loop: str, order: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """Differentiating a derivative again differentiates the tangent loop's own body. For a scan that
  body already has an input named ``fwd:<formal>`` for each active sliced input, and the second pass
  must not reuse that name for the new tangents."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  cost, inputs, point = _higher_order_loop(loop)
  wrt = inputs[-1]
  first = jacobian(cost.reshape((1,)), wrt) if order == "jacobian_of_jacobian" else sc.hessian(cost, wrt)
  first = first.reshape((first.size,))
  second = jacobian(first, wrt)
  got = _run(f"ho_{loop}_{order}", inputs, [second, _columns(first, wrt)], point)
  first_fn = _fn(f"ho_{loop}_{order}_first", inputs, [first])

  def at(v: np.ndarray) -> np.ndarray:
    return first_fn._flat_numerical_call(*point[:-1], v)[0]

  np.testing.assert_allclose(got[0], got[1], rtol=1e-10, atol=1e-12)
  np.testing.assert_allclose(got[0], finite_difference(at, point[-1]), rtol=1e-5, atol=1e-6)


# --- custom derivatives, determinism and size -----------------------------------------------------


def _custom_scan(name: str) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  """A scan whose body has a deliberately wrong forward rule, ``5 dc + du`` for ``c*c + u``, so a
  result that honors the rule is told apart from one that differentiates the body."""
  c, u = sc.sym("c", 1), sc.sym("u", 1)
  step = sc.Function._from_exprs(f"{name}_step", [c, u], [c * c + u, c * 2.0], ["c", "u"], ["cn", "y"])
  dc, du = sc.sym("dc", 1), sc.sym("du", 1)
  rule = sc.Function._from_exprs(f"{name}_jvp", [c, u, dc, du], [5.0 * dc + du, 7.0 * dc], ["c", "u", "dc", "du"], ["dcn", "dy"])
  stepped = sc.custom_derivative(step, jvp=rule)
  c0, us = sc.sym("c0", 1), sc.sym("us", 3)
  fin, ys = sc.scan(stepped, c0, [(us, 0, 1)], length=3)
  return fin, ys, c0, us


def test_custom_jvp_on_a_scan_body_is_honored_with_many_seeds() -> None:
  """Without strict mode the multi-seed pass falls back to per-seed tangents, which use the rule."""
  fin, ys, c0, us = _custom_scan("cj")
  jf, jy = _run("cj", [c0, us], [jacobian(fin, us), jacobian(ys, us)], [np.array([0.5]), np.zeros(3)])
  np.testing.assert_array_equal(jf, [[25.0, 5.0, 1.0]])
  np.testing.assert_array_equal(jy, [[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [35.0, 7.0, 0.0]])


def test_custom_jvp_on_a_loop_body_raises_in_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  fin, _, _, us = _custom_scan("cjs")
  with pytest.raises(NotImplementedError, match="custom jvp"):
    jacobian(fin, us)
  cond, body = _fixed_point("cjw")
  x, dx = sc.sym("x", 4), sc.sym("dx", 4)
  custom = sc.custom_derivative(body, jvp=sc.Function._from_exprs("cjw_jvp", [x, dx], [3.0 * dx], ["x", "dx"], ["d"]))
  final, _ = sc.while_loop(cond, custom, x, max_iter=5)
  with pytest.raises(NotImplementedError, match="custom jvp"):
    jacobian(final, x)
  monkeypatch.delenv("SCALY_STRICT_JVP_MANY")
  never = sc.Function._from_exprs("cjw_never", [x], [x[0] > 1e9], ["x"], ["go"])
  always = sc.Function._from_exprs("cjw_always", [x], [x[0] < 1e9], ["x"], ["go"])
  two = sc.while_loop(always, custom, x, max_iter=2)[0]
  jac_two, jac_none = _run("cjw", [x], [jacobian(two, x), jacobian(sc.while_loop(never, custom, x, max_iter=2)[0], x)], [np.zeros(4)])
  np.testing.assert_array_equal(jac_two, 9.0 * np.eye(4))
  np.testing.assert_array_equal(jac_none, np.eye(4))


def _determinism_hessian() -> sc.Function:
  body = _mix_body("det_step")
  c0, a_all, b_all = sc.sym("c0", 2), sc.sym("a_all", 8), sc.sym("b_all", 5)
  fin, ys, zs = sc.scan(body, c0, [(a_all, 6, -2), (b_all, 0, 1)], length=4)
  g = _scan_callee("det_g")
  (y,) = g._flat_symbolic_call([fin, b_all[:1]])
  cost = sc.Function._from_exprs("det_cost", [c0, a_all, b_all], [sc.sumsqr(y) + (ys * zs[::2]).sum()], ["c0", "a", "b"], ["f"])
  return sc.hessian(cost, "f", "a")


def test_building_the_same_hessian_twice_gives_identical_c() -> None:
  """Two independently built graphs with the same names render byte-identical C, whatever the
  process-wide counters and caches hold from the first build."""
  first, second = _determinism_hessian(), _determinism_hessian()
  assert first is not second
  assert render_c_module(first).body == render_c_module(second).body
  point = (np.array([0.4, -0.3]), np.linspace(0.2, 1.0, 8), np.linspace(-0.5, 0.5, 5))
  np.testing.assert_array_equal(first(point), second(point))


def _size_graphs(a_all: sc.Expr) -> dict[str, sc.Expr]:
  """One graph per loop kind, all functions of ``a_all`` alone."""
  body = _mix_body("sz_step")
  c0, b_all = a_all[:2] * 0.5, (a_all[4:] * 0.3).cos()
  fin, ys, zs = sc.scan(body, c0, [(a_all, 0, 2), (b_all, 3, -1)], length=4)
  cost = sc.sumsqr(fin) + (ys * ys).sum() + zs.sum()
  cond, wbody = _fixed_point("sz_w")
  wfin, _ = sc.while_loop(cond, wbody, a_all[:4] * 0.5, max_iter=50)
  g = _scan_callee("sz_g")
  (called,) = g._flat_symbolic_call([a_all[:2], a_all[2:3]])
  mapped = sc.vmap(g, 4, [(a_all, 0, 2), (b_all, 0, 1)])
  return {"scan": sc.concat([fin, ys, zs]), "while": wfin, "call": called, "vmap": mapped, "gradient": sc.gradient(cost, a_all).reshape((8,))}


@pytest.mark.parametrize("graph", ["scan", "while", "call", "vmap", "gradient"])
def test_loop_count_is_independent_of_the_number_of_seeds(graph: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """Two, four or eight dense seeds give the same number of loop nodes: the seeds ride in the carry.
  (One seed is left out because a ``vmap`` then takes its joint path instead of coloring each formal.)"""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  a_all = sc.sym("a_all", 8)
  out = _size_graphs(a_all)[graph]
  counts = []
  for nseed in (2, 4, 8):
    seeds = sc.const(RNG.standard_normal((nseed, 8)))
    tangent = jvp_many(out, a_all, seeds)
    counts.append(len(_loop_nodes([tangent])))
    per_seed = sc.stack([jvp(out, a_all, seeds[k]) for k in range(nseed)])
    got, ref = _run(f"sz_{graph}_{nseed}", [a_all], [tangent, per_seed], [np.linspace(0.2, 1.0, 8)])
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
  assert counts[0] == counts[1] == counts[2] > 0, counts


# --- Regressions from the C-93 review ------------------------------------------------------------


def test_a_body_input_named_like_a_tangent_is_not_aliased_by_it(monkeypatch: pytest.MonkeyPatch) -> None:
  """The body's carry is named ``fwd:u`` and its sliced input ``u``. The tangent input for ``u``
  with three seeds would be named ``fwd:u`` with the carry's size, and symbols are interned by name
  and type, so it would be the carry itself and the Jacobian silently wrong."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c, u = sc.sym("fwd:u", 3), sc.sym("u", 1)
  body = sc.Function._from_exprs("nc_step", [c, u], [c * u[0] + c.sin(), (c * c).sum().reshape((1,))], ["fwd:u", "u"], ["cn", "y"])
  c0, us = sc.sym("nc_c0", 3), sc.sym("nc_us", 3)
  fin, ys = sc.scan(body, c0, [(us, 0, 1)], length=3)
  _check_forward("nc", [c0, us], [fin, ys], [np.array([0.3, -0.2, 0.5]), np.array([0.9, 1.1, -0.4])])


def _shooting_intervals(k: int, layers: int) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  """``k`` calls of one Function holding a scan, each on its own slice of the variables, with some
  elementwise work around the scan: unrolled multiple shooting."""
  z, u = sc.sym("z", 2), sc.sym("u", 1)
  step = sc.Function._from_exprs("ms_step", [z, u], [sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * (u[0] - z[0].sin())])], ["z", "u"], ["zn"])
  x0, us = sc.sym("x0", 2), sc.sym("us", 4)
  pre = x0
  for _ in range(layers):
    pre = (pre * 1.01 + 0.1).sin() + pre.cos() * 0.5
  (fin,) = sc.scan(step, pre, [(us, 0, 1)], length=4)
  shoot = sc.Function._from_exprs("ms_shoot", [x0, us], [fin * fin + fin], ["x0", "us"], ["xf"])
  xs, big_u = sc.sym(f"ms_X{k}", 2 * k), sc.sym(f"ms_U{k}", 4 * k)
  y = sc.concat([shoot._flat_symbolic_call([xs[2 * i : 2 * i + 2], big_u[4 * i : 4 * i + 4]])[0] for i in range(k)])
  return y, xs, big_u


def test_calls_around_a_loop_carry_only_the_seeds_they_depend_on(monkeypatch: pytest.MonkeyPatch) -> None:
  """Each of K calls depends on its own two variables. Inlining every call would carry all 2K seeds
  through each one, so the code would grow like K²; the call helper keeps only the seeds that are
  nonzero at that call, so each call adds about the same amount of code."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  y, xs, big_u = _shooting_intervals(2, 3)
  _check_forward("ms2", [xs, big_u], [y], [np.array([0.2, -0.3, 0.5, 0.1]), np.linspace(-1.0, 1.0, 8)], rtol=1e-5, atol=1e-6)

  def lines(k: int) -> int:
    y, xs, big_u = _shooting_intervals(k, 10)
    return len(render_c_module(_fn(f"ms_jac{k}", [xs, big_u], [jacobian(y, xs)])).body.splitlines())

  n4, n8 = lines(4), lines(8)
  assert n8 < 2.8 * n4  # about 2.3 when each call adds its own share; 4 or more when it grows like K²


def test_a_body_built_with_a_fallback_does_not_satisfy_a_later_strict_build(monkeypatch: pytest.MonkeyPatch) -> None:
  """``atan2`` has no multi-seed rule, so without strict mode the tangent body falls back to one
  pass per seed inside the loop. A later strict build must not reuse that body and skip its check."""
  c, u = sc.sym("c", 1), sc.sym("u", 1)
  body = sc.Function._from_exprs("sc_step", [c, u], [sc.atan2(c, u + 2.0)], ["c", "u"], ["cn"])
  c0, us = sc.sym("sc_c0", 1), sc.sym("sc_us", 3)
  (fin,) = sc.scan(body, c0, [(us, 0, 1)], length=3)
  monkeypatch.delenv("SCALY_STRICT_JVP_MANY", raising=False)
  jacobian(fin, us)
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  with pytest.raises(NotImplementedError, match="atan2"):
    jacobian(fin, us)


def test_two_inlined_calls_keep_their_own_tangents(monkeypatch: pytest.MonkeyPatch) -> None:
  """Two calls of one Function holding a scan, both differentiated in place (the seed pruning that
  would otherwise keep these calls is switched off). The inlined graphs are temporary and tangents
  are memoized by node id, so the first graph must stay alive, or the second reuses its ids and
  picks up its tangents: the second call's rows then depended on the first call's variables."""
  import scaly.ad.forward as forward

  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  monkeypatch.setattr(forward, "_prunable_call", lambda *_: False)
  z, u = sc.sym("z", 2), sc.sym("u", 1)
  step = sc.Function._from_exprs("ti_step", [z, u], [sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * (u[0] - z[0].sin())])], ["z", "u"], ["zn"])
  x0, us = sc.sym("x0", 2), sc.sym("us", 4)
  (fin,) = sc.scan(step, (x0 * 1.01 + 0.1).sin(), [(us, 0, 1)], length=1)
  shoot = sc.Function._from_exprs("ti_shoot", [x0, us], [fin], ["x0", "us"], ["xf"])
  xs, big_u = sc.sym("ti_X", 4), sc.sym("ti_U", 8)
  y = sc.concat([shoot._flat_symbolic_call([xs[:2], big_u[:4]])[0], shoot._flat_symbolic_call([xs[2:], big_u[4:]])[0]])
  _check_forward("ti", [xs, big_u], [y], [np.array([0.2, -0.3, 0.5, 0.1]), np.linspace(-1.0, 1.0, 8)], rtol=1e-5, atol=1e-6)


# --- Per-step Jacobian compression (few body inputs, many seeds) --------------------------------


def _pendulum_step(name: str) -> sc.Function:
  z, u = sc.sym("z", 2), sc.sym("u", 1)
  nxt = sc.stack([z[0] + 0.1 * z[1], z[1] + 0.1 * (u[0] * z[0].cos() - 9.81 * z[0].sin())])
  return sc.Function._from_exprs(name, [z, u], [nxt, sc.stack([sc.sumsqr(z) * u[0] + u[0] ** 3])], ["z", "u"], ["zn", "c"])


@pytest.mark.parametrize("length", [2, 12, 60])
def test_many_seeds_through_a_small_body_use_its_step_jacobian(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
  """With 3 body inputs and ``length`` seeds, the tangents are ``seeds @ J`` with the body's 3-column
  Jacobian formed once per step: fewer seeds than inputs push the seeds through the body (length 2),
  a moderate count forms the products in the body (12), many seeds map a small product callee over
  the seeds next to a callee for the step's Jacobian (60). All three agree with the per-seed columns
  and finite differences."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  step = _pendulum_step(f"cp{length}_step")
  x0, us = sc.sym("cp_x0", 2), sc.sym(f"cp_us{length}", length)
  fin, costs = sc.scan(step, x0, [(us, 0, 1)], length=length)
  _check_forward(f"cp{length}", [x0, us], [fin, costs], [np.array([0.3, -0.2]), np.linspace(-0.5, 0.5, length)], rtol=1e-5, atol=1e-6)
  names = _callee_names([jacobian(fin, us), jacobian(costs, us)])
  split = length == 60
  assert any(n.startswith(f"cp{length}_step_stepjac") for n in names) == split
  assert any("_seed" in n for n in names) == split


def test_a_hessian_with_many_seeds_matches_the_per_seed_columns(monkeypatch: pytest.MonkeyPatch) -> None:
  """Forward over reverse with 40 seeds: the adjoint bodies have more inputs than the forward body
  (cotangent, carry, slice), and a nonlinear adjoint makes the step Jacobian vary per step."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  step = _pendulum_step("ch_step")
  x0, us = sc.sym("ch_x0", 2), sc.sym("ch_us", 40)
  fin, costs = sc.scan(step, x0, [(us, 0, 1)], length=40)
  cost = costs.sum() + 5.0 * sc.sumsqr(fin)
  _check_hessian("ch", cost, [x0, us], 1, [np.array([0.3, -0.2]), np.linspace(-0.5, 0.5, 40)], rtol=1e-4)


def test_a_while_loop_with_many_seeds_uses_its_step_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  c = sc.sym("c", 2)
  body = sc.Function._from_exprs("cw_step", [c], [sc.stack([0.5 * c[0].cos() + 0.1 * c[1], 0.9 * c[1]])], ["c"], ["cn"])
  cond = sc.Function._from_exprs("cw_go", [c], [c[1].abs() > 1e-3], ["c"], ["go"])
  p = sc.sym("cw_p", 300)
  final, _ = sc.while_loop(cond, body, sc.stack([p[:150].sum() * 0.01, p[150:].sum() * 0.01]), max_iter=100)
  jac = jacobian(final, p)
  assert any(n.startswith("cw_step_stepjac") for n in _callee_names([jac]))
  point = np.linspace(-0.2, 0.4, 300)
  got = _run("cw_jac", [p], [jac, _columns(final, p)], [point])
  np.testing.assert_allclose(got[0], got[1], rtol=1e-10, atol=1e-12)


# --- One backward scan per scan (C-91) ------------------------------------------------------------


def _adjoint_scans(exprs: Sequence[sc.Expr]) -> set[str]:
  return {n.attrs["callee"].name for n in _loop_nodes(exprs) if "scanadj" in n.attrs["callee"].name}


@pytest.mark.parametrize("through_call", [False, True])
def test_every_used_output_of_a_scan_shares_one_backward_scan(through_call: bool) -> None:
  """The final carry and both stacked outputs all carry cotangent, directly or through a call that
  returns them separately: reverse mode builds one backward scan that takes all three, not one per
  output."""
  step = _pendulum_step("ob_step")
  z, u = step.inputs
  zn, c = step.outputs
  body = sc.Function._from_exprs("ob_body", [z, u], [zn, c, zn * zn], ["z", "u"], ["zn", "c", "q"])
  x0, us = sc.sym("ob_x0", 2), sc.sym("ob_us", 6)
  fin, costs, squares = sc.scan(body, x0, [(us, 0, 1)], length=6)
  if through_call:
    roll = sc.Function._from_exprs("ob_roll", [x0, us], [fin, costs, squares], ["x0", "us"], ["fin", "costs", "q"])
    fin, costs, squares = roll._flat_symbolic_call([x0, us])
  cost = costs.sum() + sc.sumsqr(fin) + 0.5 * squares.sum()
  (grad,) = sc.vjp((cost,), (us,), (sc.const(1.0),))
  assert len(_adjoint_scans([grad])) == 1
  point = [np.array([0.3, -0.2]), np.linspace(-0.5, 0.5, 6)]
  got = _run(f"ob_grad{through_call}", [x0, us], [grad], point)[0]
  value = _fn(f"ob_cost{through_call}", [x0, us], [cost])
  np.testing.assert_allclose(got, finite_difference(lambda v: value._flat_numerical_call(point[0], v)[0], point[1]).reshape(-1), rtol=1e-6, atol=1e-8)


def test_reverse_over_reverse_merges_the_stored_carries_cotangent() -> None:
  """The gradient reads the scan's stored carries (output -1); differentiating the gradient again in
  reverse gives that node a cotangent too, and it joins the other outputs' backward scan. The result
  is the Hessian-vector product."""
  step = _pendulum_step("rr_step")
  x0, us = sc.sym("rr_x0", 2), sc.sym("rr_us", 5)
  fin, costs = sc.scan(step, x0, [(us, 0, 1)], length=5)
  cost = costs.sum() + sc.sumsqr(fin)
  (grad,) = sc.vjp((cost,), (us,), (sc.const(1.0),))
  v = np.linspace(1.0, -1.0, 5)
  (hv,) = sc.vjp(((grad * sc.const(v)).sum(),), (us,), (sc.const(1.0),))
  point = [np.array([0.3, -0.2]), np.linspace(-0.5, 0.5, 5)]
  got_hv, hess = _run("rr", [x0, us], [hv, jacobian(grad, us)], point)
  np.testing.assert_allclose(got_hv, hess @ v, rtol=1e-10, atol=1e-12)
