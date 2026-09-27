"""Forward-mode AD: single-seed ``jvp`` and multi-seed ``jvp_many``.

``jvp_many`` has structural rules that share work across seeds; when an op has none it falls back
to unrolling ``jvp`` per seed unless ``SCALY_STRICT_JVP_MANY`` forbids it.
"""

from __future__ import annotations

import hashlib
import itertools
import weakref
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..function import ConcreteFunction
from ..function.sugar import _scan_node, _while_node, vmap, while_parts
from ..ir.expr import (
  CALLEE_OPS,
  PREDICATE_OPS,
  Expr,
  ExprOp,
  callees_of,
  cast,
  concat,
  copysign,
  equal,
  not_equal,
  gather,
  put,
  put_add,
  ragged_add,
  sparse_ldl_solve,
  ragged_dot,
  reduce_min,
  scatter,
  segment_min,
  segment_sum,
  solve_triangular,
  stack,
  independent,
  substitute,
  take,
  topo,
  where,
  zeros_like,
)
from ..passes.expr import simplify_cse_fixpoint
from ..utils.env import env_bool
from ..utils.options import get_options
from .sparsity import _depends_on, _jac_mask, _mask_sparsity, column_coloring


# Cache derivative helper Functions per live callee object. Do not key by ``id(callee)``:
# CPython may reuse ids after a short-lived Function is collected, which can splice a stale
# call-JVP helper into a different graph under xdist/CI-sized test runs.
_CALL_JVP_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, tuple[int, ...]], tuple[Any, tuple[int, ...], tuple[int, ...]]]] = (
  weakref.WeakKeyDictionary()
)
_CALL_JVP_MANY_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]] = (
  weakref.WeakKeyDictionary()
)
_CALL_JVP_PACK_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], Any]] = weakref.WeakKeyDictionary()


class _JVPManyUnsupported(Exception):
  def __init__(self, op: str):
    super().__init__(op)
    self.op = op


SPARSE_LDL_NO_DERIVATIVE = (
  "the derivative of a looped sparse LDL^T factorization is not implemented: SparseLDL.solve differentiates "
  "implicitly without it, and SparseLDL(..., schedule='scan') differentiates the factorization through its loops"
)
"""Why a ``sparse_ldl_factor`` node refuses a nonzero tangent or cotangent."""

LU_NO_DERIVATIVE = (
  "the dense LU factorization has no derivative: linalg.solve(a, b, assume='gen') solves with it and "
  "differentiates implicitly, so its derivative never reaches the factorization"
)
"""Why an ``lu`` node refuses a nonzero tangent or cotangent."""


def _is_zero_const(expr: Expr) -> bool:
  return expr.op == ExprOp.CONST and expr.value is not None and bool(np.all(expr.value == 0))


def claim_name(base: str, taken: set[str]) -> str:
  """``base``, primed until it is not in ``taken``, then added to it.

  Derivative helpers name their new inputs after the callee's (``fwd:u`` for the tangent of ``u``).
  Symbols are interned by name and type, so a new input named like one the callee already has, with
  the same shape, would be that very input: a tangent would silently alias a primal. That happens
  whenever the callee is itself a derivative helper (a Jacobian of a Jacobian through a scan) or a
  user names an input ``fwd:...``. Claiming every new name against the callee's names prevents it."""
  name = base
  while name in taken:
    name += "'"
  taken.add(name)
  return name


def jvp(expr: Expr, wrt: Expr, seed: Expr) -> Expr:
  """Forward-mode derivative: ``J(expr, wrt) @ seed``, with ``seed`` shaped like ``wrt``.

  One pass per seed. For many seeds at once use ``jvp_many``, which shares the expensive work.
  """
  (expr,), (at,), back = independent((expr,), (wrt,))
  ret = _jvp(expr, {at: seed}, {}, {})
  return substitute(ret, back) if back else ret


def _jvp(expr: Expr, seeds: dict[Expr, Expr], memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  if expr.id in memo:
    return memo[expr.id]
  if not any(not _is_zero_const(seed) and _depends_on(expr, wrt, dep_memo) for wrt, seed in seeds.items()):
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seeds.get(expr, zeros_like(expr))
    return ret
  if expr.op == ExprOp.CONST or expr.op in PREDICATE_OPS:
    memo[expr.id] = ret = zeros_like(expr)
    return ret
  if expr.op in (ExprOp.CALL, ExprOp.VMAP):
    # Only the tangents the derivative reads are formed: a custom rule may ignore an argument's
    # tangent (an implicit solve rule ignores its factor's), and forming it could differentiate
    # what the rule exists to avoid.
    callee, output = expr.attrs["callee"], expr.attrs["output"]
    seeded = [wrt for wrt, seed in seeds.items() if not _is_zero_const(seed)]
    candidates = tuple(i for i, arg in enumerate(expr.args) if any(_depends_on(arg, wrt, dep_memo) for wrt in seeded))
    try:
      _, _, read = _call_jvp_function(callee, output, candidates) if candidates else (None, (), ())
    except NotImplementedError:
      # The callee cannot be differentiated in every argument that depends on the seeds (one enters
      # through a comparison, say): form every tangent and keep the nonzero ones, as it is done
      # without the probe.
      read = tuple(range(len(expr.args)))
      candidates = tuple(i for i in candidates if not _is_zero_const(_jvp(expr.args[i], seeds, memo, dep_memo)))
    tangents = [_jvp(arg, seeds, memo, dep_memo) if i in read else zeros_like(arg) for i, arg in enumerate(expr.args)]
    active = tuple(i for i in candidates if i not in read or not _is_zero_const(tangents[i]))
    if not active:
      memo[expr.id] = ret = zeros_like(expr)
      return ret
    fn, arg_indices, seed_indices = _call_jvp_function(callee, output, active)
    if not seed_indices:  # the derivative reads no tangent: it is zero
      memo[expr.id] = ret = zeros_like(expr)
      return ret
    if expr.op == ExprOp.CALL:
      call_args = [expr.args[i] for i in arg_indices] + [tangents[i] for i in seed_indices]
      memo[expr.id] = ret = fn._flat_symbolic_call(call_args)[0]
    else:
      starts, strides = expr.attrs["starts"], expr.attrs["strides"]
      specs = [(expr.args[i], starts[i], strides[i]) for i in arg_indices] + [(tangents[i], starts[i], strides[i]) for i in seed_indices]
      memo[expr.id] = ret = vmap(fn, expr.attrs["length"], specs)
    return ret
  if expr.op == ExprOp.SCAN:
    memo[expr.id] = ret = _scan_jvp(expr, [_jvp(arg, seeds, memo, dep_memo) for arg in expr.args])
    return ret
  if expr.op == ExprOp.WHILE:
    memo[expr.id] = ret = _while_jvp(expr, [_jvp(arg, seeds, memo, dep_memo) for arg in expr.args])
    return ret
  if expr.op == ExprOp.SOLVER_CALL:
    # Solver outputs are treated as non-differentiable today. Implicit
    # function theorem AD (e.g. cyipopt-style adjoint through KKT residuals)
    # is future work; for now any JVP through a solver returns zero.
    memo[expr.id] = ret = zeros_like(expr)
    return ret

  def save(ret: Expr) -> Expr:
    memo[expr.id] = ret
    return ret

  args = expr.args
  d = [_jvp(a, seeds, memo, dep_memo) for a in args]
  if expr.op == ExprOp.NEG:
    return save(-d[0])
  if expr.op == ExprOp.ADD:
    return save(d[0] + d[1])
  if expr.op == ExprOp.SUB:
    return save(d[0] - d[1])
  if expr.op == ExprOp.MUL:
    return save((2 * args[0]) * d[0] if args[0] is args[1] else d[0] * args[1] + args[0] * d[1])
  if expr.op == ExprOp.DIV:
    return save((d[0] - expr * d[1]) * (1.0 / args[1]))
  if expr.op == ExprOp.POW:
    if args[1].op == ExprOp.CONST:
      return save(args[1] * (args[0] ** _minus_one(args[1])) * d[0])
    return save(expr * (d[1] * args[0].log() + args[1] * d[0] / args[0]))
  if expr.op == ExprOp.SIN:
    return save(args[0].cos() * d[0])
  if expr.op == ExprOp.COS:
    return save(-args[0].sin() * d[0])
  if expr.op == ExprOp.TAN:
    return save(d[0] / (args[0].cos() ** 2))
  if expr.op == ExprOp.ASIN:
    return save(d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == ExprOp.ACOS:
    return save(-d[0] / (1 - args[0] ** 2).sqrt())
  if expr.op == ExprOp.ATAN:
    return save(d[0] / (1 + args[0] ** 2))
  if expr.op == ExprOp.ATAN2:
    y, x = args
    dy, dx = d
    return save((x * dy - y * dx) / (x * x + y * y))
  if expr.op == ExprOp.SINH:
    return save(args[0].cosh() * d[0])
  if expr.op == ExprOp.COSH:
    return save(args[0].sinh() * d[0])
  if expr.op == ExprOp.TANH:
    return save(d[0] * (1 - expr * expr))
  if expr.op == ExprOp.ERF:
    return save((2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp() * d[0])
  if expr.op == ExprOp.EXP:
    return save(expr * d[0])
  if expr.op == ExprOp.LOG:
    return save(d[0] / args[0])
  if expr.op == ExprOp.SQRT:
    return save(d[0] * (0.5 / expr))
  if expr.op == ExprOp.ABS:
    return save(sign(args[0]) * d[0])
  if expr.op in {ExprOp.FLOOR, ExprOp.CEIL}:
    raise NotImplementedError(f"JVP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op in {ExprOp.MINIMUM, ExprOp.MAXIMUM}:
    w = extremum_weight(expr)
    return save(w * d[0] + (1.0 - w) * d[1])
  if expr.op in {ExprOp.MAX, ExprOp.MIN}:
    return save((reduce_weights(expr) * d[0]).sum())
  if expr.op in {ExprOp.SEGMENT_MAX, ExprOp.SEGMENT_MIN}:
    return save(segment_sum(segment_weights(expr) * d[0], expr.attrs["indices"], expr.size))
  if expr.op in {ExprOp.INDEX_ADD, ExprOp.INDEX_SET}:
    return save(Expr(expr.op, (d[0], d[1]), expr.type, attrs=dict(expr.attrs), lowering=expr.lowering))
  if expr.op == ExprOp.TAKE:  # linear in x; a fill is a constant
    return save(take(d[0], args[1], in_range=bool(expr.attrs.get("in_range"))))
  if expr.op in {ExprOp.RAGGED_ADD, ExprOp.RAGGED_DOT}:
    return save(ragged_tangent(expr, [None if _is_zero_const(t) else t for t in d]))
  if expr.op in {ExprOp.CHOLESKY, ExprOp.LDL}:
    return save(zeros_like(expr) if _is_zero_const(d[0]) else factor_tangent(expr, d[0]))
  if expr.op == ExprOp.SPARSE_LDL:  # reached only with a tangent: without one, the dependence check gave zero
    raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
  if expr.op == ExprOp.LU:
    raise NotImplementedError(LU_NO_DERIVATIVE)
  if expr.op == ExprOp.SPARSE_LDL_SOLVE:  # linear in b; in the factor, not implemented
    if not _is_zero_const(d[0]):
      raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
    return save(sparse_ldl_solve(args[0], d[1], dict(expr.attrs)))
  if expr.op == ExprOp.TRISOLVE:
    dt, db = (None if _is_zero_const(t) else t for t in d)
    return save(zeros_like(expr) if dt is None and db is None else trisolve_tangent(expr, dt, db))
  if expr.op in {ExprOp.PUT_ADD, ExprOp.PUT}:
    return save((put_add if expr.op == ExprOp.PUT_ADD else put)(d[0], args[1], d[2], in_range=bool(expr.attrs.get("in_range"))))
  if expr.op == ExprOp.SELECT:
    return save(where(args[0], d[1], d[2]))
  if expr.op == ExprOp.COPYSIGN:
    return save(_copysign_slope(args[0], args[1]) * d[0])
  if expr.op == ExprOp.CAST:
    return save(cast(d[0], expr.type.dtype) if expr.type.diff else zeros_like(expr))
  if expr.op == ExprOp.SUM:
    return save(d[0].sum())
  if expr.op == ExprOp.RESHAPE:
    return save(d[0].reshape(expr.shape))
  if expr.op == ExprOp.TRANSPOSE:
    return save(d[0].transpose(expr.attrs["axes"]))
  if expr.op == ExprOp.SLICE:
    return save(Expr.const(d[0].value[expr.attrs["index"]]) if d[0].op == ExprOp.CONST and d[0].value is not None else d[0][expr.attrs["index"]])
  if expr.op == ExprOp.GATHER:
    return save(
      Expr.const(np.take(d[0].value.reshape(-1), expr.attrs["indices"]).reshape(expr.attrs["indices"].shape))
      if d[0].op == ExprOp.CONST and d[0].value is not None
      else gather(d[0], expr.attrs["indices"])
    )
  if expr.op == ExprOp.SCATTER:
    return save(scatter(d[0], expr.attrs["indices"], expr.shape))
  if expr.op == ExprOp.STACK:
    return save(stack(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == ExprOp.CONCAT:
    return save(concat(d, axis=expr.attrs.get("axis", 0)))
  if expr.op == ExprOp.MATMUL:
    if args[0] is args[1] and len(args[0].shape) == 1:
      return save(2 * (args[0] @ d[0]))
    return save(d[0] @ args[1] + args[0] @ d[1])
  raise NotImplementedError(f"JVP for op {expr.op!r} is not implemented")


_SCAN_JVP_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, ...], ConcreteFunction]] = weakref.WeakKeyDictionary()


def _scan_jvp_body(callee: ConcreteFunction, active: tuple[int, ...]) -> ConcreteFunction:
  """The body of the tangent scan: the carry is ``[c, dc]`` flat, the sliced inputs are the primal
  ones then the tangents of the ``active`` ones, and the outputs are ``[c', dc']``, the primal
  stacked outputs, then their tangents."""
  cache = _SCAN_JVP_CACHE.setdefault(callee, {})
  if active not in cache:
    carry, xs = callee.inputs[0], callee.inputs[1:]
    cs = carry.size
    taken = {*callee.input_names, *callee.output_names}
    aug = Expr.sym(claim_name(f"fwd:{callee.input_names[0]}", taken), (2 * cs,))
    dcarry = Expr.sym(claim_name(f"fwd:{callee.input_names[0]}:dc", taken), carry.shape)
    dxs = {i: Expr.sym(claim_name(f"fwd:{callee.input_names[i + 1]}", taken), xs[i].shape) for i in active}
    tangents = body_tangents(callee, {0: dcarry, **{i + 1: dx for i, dx in dxs.items()}})
    split = {carry: aug[:cs].reshape(carry.shape), dcarry: aug[cs:].reshape(carry.shape)}
    nxt = concat([callee.outputs[0].reshape((cs,)), tangents[0].reshape((cs,))])
    outputs = [substitute(e, split) for e in (nxt, *callee.outputs[1:], *tangents[1:])]
    inputs = [aug, *xs, *dxs.values()]
    names = [str(aug.name), *callee.input_names[1:], *(str(dx.name) for dx in dxs.values())]
    out_names = [claim_name("fwd:carry", taken), *callee.output_names[1:], *(claim_name(f"fwd:{n}", taken) for n in callee.output_names[1:])]
    suffix = "_".join(str(i) for i in active) or "c"
    cache[active] = ConcreteFunction._from_exprs(
      f"{callee.name}_scanfwd_{suffix}", inputs, [callee._inherit_lowering(simplify_cse_fixpoint(o)) for o in outputs], names, out_names
    )
  return cache[active]


def _scan_jvp(expr: Expr, tangents: list[Expr]) -> Expr:
  """A scan's tangent is another scan whose carry also carries the tangent."""
  callee, length, output = expr.attrs["callee"], expr.attrs["length"], expr.attrs["output"]
  if all(_is_zero_const(t) for t in tangents):
    return zeros_like(expr)
  init, outers = expr.args[0], expr.args[1:]
  cs, n_ys = init.size, len(callee.outputs) - 1
  active = tuple(i for i, t in enumerate(tangents[1:]) if not _is_zero_const(t))
  fn = _scan_jvp_body(callee, active)
  starts, strides = expr.attrs["starts"], expr.attrs["strides"]
  aug_init = concat([init.reshape((cs,)), tangents[0].reshape((cs,))])
  aug_outers = (*outers, *(tangents[i + 1] for i in active))
  aug_starts = (*starts, *(starts[i] for i in active))
  aug_strides = (*strides, *(strides[i] for i in active))

  def node(k: int) -> Expr:
    return _scan_node(fn, aug_init, aug_outers, aug_starts, aug_strides, length, k)

  if output == 0:
    return node(0)[cs:].reshape(expr.shape)
  if output == -1:
    picks = (np.arange(length)[:, None] * 2 * cs + cs + np.arange(cs)[None, :]).reshape(-1)
    return gather(node(-1), picks)
  return node(1 + n_ys + output - 1)


# The tangent body per body (whatever the condition), the tangent condition per (body, condition).
_WHILE_JVP_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], tuple[ConcreteFunction, Expr, list[Expr]]]] = weakref.WeakKeyDictionary()
_WHILE_JVP_COND_CACHE: weakref.WeakKeyDictionary[Any, weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], ConcreteFunction]]] = (
  weakref.WeakKeyDictionary()
)


def _loop_suffix(index: bool, active: tuple[int, ...]) -> str:
  """What sets a while loop's derivative Functions apart for one body: whether it takes the step
  number, and which of its params are differentiated. Each combination is its own procedure."""
  return ("_k" if index else "") + "".join(f"_p{i}" for i in active)


def _tangent_params(body: ConcreteFunction, index: bool, active: tuple[int, ...], taken: set[str], shape: Any) -> list[Expr]:
  """Symbols for the tangents of the loop's params ``active`` (by position among the params), shaped
  by ``shape(param)``; the tangent loop takes them as params after the primal ones."""
  first = 1 + int(index)
  return [Expr.sym(claim_name(f"fwd:{body.input_names[first + i]}", taken), shape(body.inputs[first + i])) for i in active]


def _tangent_cond(cond: ConcreteFunction, body: ConcreteFunction, aug: Expr, cs: int, dparams: Sequence[Expr], suffix: str) -> ConcreteFunction:
  """The tangent loop's condition: the primal one on ``c`` and the params, ignoring the tangents.
  It takes the tangent body's inputs, so it is named after the body too: one condition shared by
  two bodies gives two of them."""
  go = substitute(cond.outputs[0], {cond.inputs[0]: aug[:cs].reshape(cond.inputs[0].shape)})
  inputs = [aug, *cond.inputs[1:], *dparams]
  taken: set[str] = set()
  names = [claim_name(n, taken) for n in (str(aug.name), *cond.input_names[1:], *(str(d.name) for d in dparams))]
  return ConcreteFunction._from_exprs(f"{cond.name}_{body.name}_whilefwd{suffix}", inputs, [go], names, ["go"])


def _while_jvp_functions(
  cond: ConcreteFunction, body: ConcreteFunction, index: bool, active: tuple[int, ...]
) -> tuple[ConcreteFunction, ConcreteFunction]:
  """The condition and body of the tangent loop, over the carry ``[c, dc]`` flat and with the
  tangents of the ``active`` params as extra params: the condition reads ``c`` and the params only,
  so the tangent loop takes exactly the primal's steps."""
  cache = _WHILE_JVP_CACHE.setdefault(body, {})
  key = (index, active)
  suffix = _loop_suffix(index, active)
  cs = body.inputs[0].size
  if key not in cache:
    carry = body.inputs[0]
    first = 1 + int(index)
    taken = {*body.input_names, *body.output_names}
    aug = Expr.sym(claim_name(f"fwd:{body.input_names[0]}", taken), (2 * cs,))
    dcarry = Expr.sym(claim_name(f"fwd:{body.input_names[0]}:dc", taken), carry.shape)
    dparams = _tangent_params(body, index, active, taken, lambda formal: formal.shape)
    (tangent,) = body_tangents(body, {0: dcarry, **{first + i: d for i, d in zip(active, dparams, strict=True)}})
    split = {carry: aug[:cs].reshape(carry.shape), dcarry: aug[cs:].reshape(carry.shape)}
    nxt = substitute(concat([body.outputs[0].reshape((cs,)), tangent.reshape((cs,))]), split)
    # The step number and the params stay inputs of the tangent body, the params' tangents follow.
    inputs = [aug, *body.inputs[1:], *dparams]
    aug_body = ConcreteFunction._from_exprs(
      f"{body.name}_whilefwd{suffix}", inputs, [body._inherit_lowering(simplify_cse_fixpoint(nxt))], [str(e.name) for e in inputs], ["fwd:carry"]
    )
    cache[key] = (aug_body, aug, dparams)
  aug_body, aug, dparams = cache[key]
  conds = _WHILE_JVP_COND_CACHE.setdefault(body, weakref.WeakKeyDictionary()).setdefault(cond, {})
  if key not in conds:
    conds[key] = _tangent_cond(cond, body, aug, cs, dparams, suffix)
  return conds[key], aug_body


def _while_jvp(expr: Expr, tangents: Sequence[Expr]) -> Expr:
  """A while loop's tangent is a while loop whose carry also carries the tangent, with the params'
  tangents as further params. The step count is piecewise constant in the input, so its derivative
  is zero."""
  output = int(expr.attrs["output"])
  cond, body, init, params, max_iter, index = while_parts(expr)
  active = tuple(i for i, t in enumerate(tangents[1:]) if not _is_zero_const(t))
  if output == 1 or (_is_zero_const(tangents[0]) and not active):
    return zeros_like(expr)
  cs = init.size
  aug_cond, aug_body = _while_jvp_functions(cond, body, index, active)
  aug_init = concat([init.reshape((cs,)), tangents[0].reshape((cs,))])
  aug_params = (*params, *(tangents[1 + i] for i in active))
  if output == 0:
    return _while_node(aug_cond, aug_body, aug_init, max_iter, 0, aug_params, index)[cs:].reshape(expr.shape)
  picks = (np.arange(max_iter)[:, None] * 2 * cs + cs + np.arange(cs)[None, :]).reshape(-1)
  return gather(_while_node(aug_cond, aug_body, aug_init, max_iter, -1, aug_params, index), picks)


def _tri_mask(n: int, lower: bool, unit: bool) -> Expr:
  """Ones on the triangle a triangular solve reads (its diagonal too unless unit), zeros elsewhere."""
  mask = np.tril(np.ones((n, n))) if lower else np.triu(np.ones((n, n)))
  if unit:
    np.fill_diagonal(mask, 0.0)
  return Expr.const(mask)


def _lower_as_symmetric(d: Expr) -> Expr:
  """The symmetric matrix whose lower triangle is that of ``d``: what a factorization reads."""
  n, nd = d.shape[-1], len(d.shape)
  strict = d * Expr.const(np.tril(np.ones((n, n)), -1))
  return d * Expr.const(np.tril(np.ones((n, n)))) + strict.transpose((*range(nd - 2), nd - 1, nd - 2))


def _sandwich(t: Expr, s: Expr, *, unit: bool) -> Expr:
  """``L^{-1} S L^{-T}`` for a symmetric ``S`` and the lower triangle of ``t``."""
  z = solve_triangular(t, s, lower=True, unit_diagonal=unit)
  return solve_triangular(t, z.T, lower=True, unit_diagonal=unit).T


def ragged_tangent(expr: Expr, d: list[Expr | None]) -> Expr:
  """``ragged_add`` and ``ragged_dot`` are linear in each floating operand: the tangent is the same op
  with one operand replaced by its tangent at a time (``None`` for a zero tangent)."""
  if expr.op == ExprOp.RAGGED_ADD:
    base, src, lo, hi, scale = expr.args
    maps = {"dst_map": expr.attrs["dst_map"], "src_map": expr.attrs["src_map"]}
    out = d[0] if d[0] is not None else zeros_like(expr)
    if d[1] is not None:
      out = ragged_add(out, d[1], lo, hi, scale, **maps)
    if d[4] is not None:
      out = ragged_add(out, src, lo, hi, d[4], **maps)
    return out
  a, b, lo, hi = expr.args
  maps = {"a_map": expr.attrs["a_map"], "b_map": expr.attrs["b_map"]}
  terms = [ragged_dot(d[0], b, lo, hi, **maps)] if d[0] is not None else []
  if d[1] is not None:
    terms.append(ragged_dot(a, d[1], lo, hi, **maps))
  return (terms[0] + terms[1] if len(terms) == 2 else terms[0]) if terms else zeros_like(expr)


def factor_tangent(expr: Expr, d: Expr) -> Expr:
  """The tangent of ``cholesky`` or ``ldl`` along a tangent ``d`` of the matrix.

  ``A = L L^T``: with ``X = L^{-1} S L^{-T}``, ``dL = L (tril(X) - diag(X) / 2)``.
  ``A = L D L^T`` (packed ``F``): ``dD = diag(X)`` and ``dL = L stril(X) D^{-1}`` for the unit ``L``.
  ``S`` is the symmetric matrix whose lower triangle is that of ``d``."""
  n = expr.shape[0]
  s = _lower_as_symmetric(d)
  if expr.op == ExprOp.CHOLESKY:
    phi = np.tril(np.ones((n, n)))
    np.fill_diagonal(phi, 0.5)
    return expr @ (_sandwich(expr, s, unit=False) * Expr.const(phi))
  x = _sandwich(expr, s, unit=True)
  unit_l = expr * Expr.const(np.tril(np.ones((n, n)), -1)) + Expr.const(np.eye(n))
  inv_d = 1.0 / gather(expr.reshape((n * n,)), np.arange(n) * (n + 1))
  return (unit_l @ (x * Expr.const(np.tril(np.ones((n, n)), -1)))) * inv_d.reshape((1, n)) + x * Expr.const(np.eye(n))


def trisolve_tangent(expr: Expr, dt: Expr | None, db: Expr | None) -> Expr:
  """``op(T) dX = dB - op(dT) X`` with ``dT`` restricted to the triangle the solve reads."""
  t, _ = expr.args
  lower, trans, unit = (bool(expr.attrs[k]) for k in ("lower", "trans", "unit"))
  rhs = db
  if dt is not None:
    masked = dt * _tri_mask(t.shape[0], lower, unit)
    term = (masked.T if trans else masked) @ expr
    rhs = -term if rhs is None else rhs - term
  assert rhs is not None
  return solve_triangular(t, rhs, lower=lower, trans=trans, unit_diagonal=unit)


def _seeds_as_columns(x: Expr) -> tuple[Expr, tuple[int, ...]]:
  """A seeded matrix operand ``(nseed, n[, m])`` as one matrix ``(n, nseed * m)``, with what undoes it."""
  nseed, n = x.shape[0], x.shape[1]
  m = 1 if len(x.shape) == 2 else x.shape[2]
  cols = x.reshape((nseed, n, m)).transpose((1, 0, 2)).reshape((n, nseed * m))
  return cols, (nseed, n, m)


def _columns_as_seeds(cols: Expr, layout: tuple[int, ...], shape: tuple[int, ...]) -> Expr:
  nseed, n, m = layout
  return cols.reshape((n, nseed, m)).transpose((1, 0, 2)).reshape(shape)


def _seed_solve(t: Expr, rhs: Expr, **flags: bool) -> Expr:
  """One triangular solve for every seed: the seeds become right-hand-side columns."""
  cols, layout = _seeds_as_columns(rhs)
  return _columns_as_seeds(solve_triangular(t, cols, **flags), layout, rhs.shape)


def _seed_left(m: Expr, x: Expr) -> Expr:
  """``m @ x_s`` for every seed of a seeded matrix ``x``."""
  cols, layout = _seeds_as_columns(x)
  return _columns_as_seeds(m @ cols, layout, x.shape)


def _seed_transpose(x: Expr) -> Expr:
  return x.transpose((0, 2, 1))


def _jvp_many_dense(expr: Expr, d: list[Expr], nseed: int) -> Expr:
  """Multi-seed tangents of ``cholesky``, ``ldl`` and ``solve_triangular``: the single-seed rules
  with every seed a column of one solve or product."""
  n = expr.shape[0]
  if expr.op == ExprOp.TRISOLVE:
    t, _ = expr.args
    lower, trans, unit = (bool(expr.attrs[k]) for k in ("lower", "trans", "unit"))
    rhs = None if _is_zero_const(d[1]) else d[1]
    if not _is_zero_const(d[0]):
      masked = d[0] * _tri_mask(n, lower, unit)
      op = _seed_transpose(masked) if trans else masked
      x = expr if len(expr.shape) == 2 else expr.reshape((n, 1))
      term = (op.reshape((nseed * n, n)) @ x).reshape((nseed, *expr.shape))
      rhs = -term if rhs is None else rhs - term
    if rhs is None:
      return Expr.const(np.zeros((nseed, *expr.shape)))
    return _seed_solve(t, rhs, lower=lower, trans=trans, unit_diagonal=unit)
  if _is_zero_const(d[0]):
    return Expr.const(np.zeros((nseed, *expr.shape)))
  s = _lower_as_symmetric(d[0])
  unit = expr.op == ExprOp.LDL
  z = _seed_solve(expr, s, lower=True, unit_diagonal=unit)
  x = _seed_transpose(_seed_solve(expr, _seed_transpose(z), lower=True, unit_diagonal=unit))
  if expr.op == ExprOp.CHOLESKY:
    phi = np.tril(np.ones((n, n)))
    np.fill_diagonal(phi, 0.5)
    return _seed_left(expr, x * Expr.const(phi))
  unit_l = expr * Expr.const(np.tril(np.ones((n, n)), -1)) + Expr.const(np.eye(n))
  inv_d = 1.0 / gather(expr.reshape((n * n,)), np.arange(n) * (n + 1))
  return _seed_left(unit_l, x * Expr.const(np.tril(np.ones((n, n)), -1))) * inv_d.reshape((1, 1, n)) + x * Expr.const(np.eye(n))


def _minus_one(exponent: Expr) -> Expr:
  """``exponent - 1``, folded when the exponent is a constant: the derivative of ``x ** p`` is then
  again a power with a constant exponent, whose own derivative needs no ``log(x)`` (NaN at 0)."""
  if exponent.op == ExprOp.CONST and exponent.value is not None:
    return Expr.const(np.asarray(exponent.value) - 1, dtype=exponent.type.dtype)
  return exponent - 1


def sign(x: Expr) -> Expr:
  """``-1``, ``0`` or ``1``: the derivative of ``abs``, zero at zero whatever the options say."""
  one, zero = Expr.const(1.0, dtype=x.type.dtype), Expr.const(0.0, dtype=x.type.dtype)
  return where(x > 0.0, one, where(x < 0.0, -one, zero))


def _nonsmooth_mode(op: ExprOp | str) -> str:
  mode = get_options().nonsmooth
  if mode == "error":
    raise NotImplementedError(f"derivative of nonsmooth op {ExprOp(op).value!r} refused under sc.options(nonsmooth='error')")
  return mode


def extremum_weight(expr: Expr) -> Expr:
  """The share of the derivative of ``maximum(a, b)`` or ``minimum(a, b)`` that goes to ``a``, shaped
  like the result; ``b`` gets the rest. C's ``fmax`` and ``fmin`` return the other operand when one
  is NaN, and the share follows the operand returned."""
  a, b = expr.args
  mode = _nonsmooth_mode(expr.op)
  wins = (a > b if expr.op == ExprOp.MAXIMUM else a < b) | not_equal(b, b)
  one, half, zero = (Expr.const(v, dtype=expr.type.dtype) for v in (1.0, 0.5, 0.0))
  if mode == "first":
    return where(wins | equal(a, b), one, zero)
  return where(wins, one, where(equal(a, b), half, zero))


def reduce_weights(expr: Expr) -> Expr:
  """How the derivative of ``reduce_max(x)`` or ``reduce_min(x)`` spreads over ``x``: equally over the
  tied entries under ``"split"``, all to the lowest tied index under ``"first"``."""
  x = expr.args[0]
  mode = _nonsmooth_mode(expr.op)
  hit = equal(x, expr)
  if mode == "first":
    index = Expr.const(np.arange(x.size, dtype=np.float64).reshape(x.shape))
    return cast(equal(index, reduce_min(where(hit, index, float(x.size)))), x.type.dtype)
  count = cast(hit, x.type.dtype)
  return count / count.sum()


def segment_weights(expr: Expr) -> Expr:
  """``reduce_weights`` bin by bin for ``segment_max`` and ``segment_min``: the share of each bin's
  derivative that each value receives."""
  x, ids, n = expr.args[0], expr.attrs["indices"], expr.size
  mode = _nonsmooth_mode(expr.op)
  hit = equal(x, gather(expr, ids))
  if mode == "first":
    index = Expr.const(np.arange(x.size, dtype=np.float64))
    first = segment_min(where(hit, index, float(x.size)), ids, n, fill=float(x.size))
    return cast(equal(index, gather(first, ids)), x.type.dtype)
  count = cast(hit, x.type.dtype)
  return count / gather(segment_sum(count, ids, n), ids)


def body_tangents(fn: Any, seeds: dict[int, Expr]) -> list[Expr]:
  """The tangent of every output of ``fn``, in terms of its inputs and the ``seeds`` (by input index):
  from its forward rule when it has one, else by differentiating its body. Every derivative that
  looks inside a Function (a call, a map, a loop body) goes through here, so a rule is never skipped."""
  if fn.custom_jvp is not None:
    rule, n = fn.custom_jvp, len(fn.inputs)
    # A tangent the rule never reads (an implicit rule ignoring a precomputed factor) is passed as
    # a zero, so the call does not depend on it and nothing forms it.
    dep: dict[tuple[int, int], bool] = {}
    read = [any(_depends_on(out, rule.inputs[n + i], dep) for out in rule.outputs) for i in range(n)]
    tangents = [seeds[i] if i in seeds and read[i] else zeros_like(inp) for i, inp in enumerate(fn.inputs)]
    return list(rule._flat_symbolic_call([*fn.inputs, *tangents]))
  memo: dict[int, Expr] = {}
  dep: dict[tuple[int, int], bool] = {}
  by_input = {fn.inputs[i]: seed for i, seed in seeds.items()}
  return [_jvp(out, by_input, memo, dep) for out in fn.outputs]


def custom_vjp_call(callee: Any, args: Sequence[Expr], cots: dict[int, Expr]) -> tuple[Expr, ...]:
  """Input cotangents from a callee's own reverse rule, given cotangents of some of its outputs (by
  index; the rest are zero). The primal outputs it receives are the call's own outputs, so the rule
  reuses the solution. Lives here, beside the other flat-call synthesis, for reverse mode to use."""
  outputs = callee._flat_symbolic_call(list(args))
  full = [cots[j] if j in cots else zeros_like(out) for j, out in enumerate(outputs)]
  grads = callee.custom_vjp._flat_symbolic_call([*args, *outputs, *full])
  # A rule that returns a constant zero for an input says that input receives nothing: hand back the
  # constant itself, so reverse mode can see it and does not differentiate what produced the input.
  return tuple(
    g if not _is_zero_const(r) else Expr.const(np.zeros(r.shape), dtype=r.type.dtype) for g, r in zip(grads, callee.custom_vjp.outputs, strict=True)
  )


def _copysign_slope(x: Expr, s: Expr) -> Expr:
  """``d copysign(x, s) / dx = sign(x) * sign(s)``, taking the sign bit of zero as C does."""
  return copysign(1.0, x) * copysign(1.0, s)


# --- Multi-seed forward mode through loops ------------------------------------------------------
# One loop whose carry holds the primal and all ``nseed`` tangents, instead of one tangent loop per
# seed. The loop body's own tangent is again a multi-seed pass, so work is shared across seeds inside
# the step as well (C-93).

_SCAN_JVP_MANY_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], ConcreteFunction]] = weakref.WeakKeyDictionary()
_WHILE_JVP_MANY_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], tuple[ConcreteFunction, Expr, list[Expr]]]] = weakref.WeakKeyDictionary()
_WHILE_JVP_MANY_COND_CACHE: weakref.WeakKeyDictionary[Any, weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], ConcreteFunction]]] = (
  weakref.WeakKeyDictionary()
)
_CONTAINS_LOOP: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
# ``Expr.sym`` interns by name and type, so each joint vector needs its own name: two nested joints
# of one size with a shared name would be one node.
_JOINT_IDS = itertools.count()


def _contains_loop(fn: Any) -> bool:
  """Whether a Function's graph holds a ``scan`` or ``while_loop``, directly or through a callee."""
  if fn not in _CONTAINS_LOOP:
    _CONTAINS_LOOP[fn] = any(
      node.op in (ExprOp.SCAN, ExprOp.WHILE) or (node.op in CALLEE_OPS and any(_contains_loop(c) for c in callees_of(node)))
      for node in topo(fn.outputs)
    )
  return _CONTAINS_LOOP[fn]


def _jvp_many_joint(outs: Sequence[Expr], seeds: dict[Expr, Expr], nseed: int) -> list[Expr]:
  """Multi-seed tangents of ``outs`` with respect to several inputs at once. ``seeds[w]`` has shape
  ``(nseed, *w.shape)``. The inputs are viewed as slices of one joint vector, so a single
  ``jvp_many`` pass covers all of them."""
  wrts = list(seeds)
  if len(wrts) == 1:
    (w,) = wrts
    return [jvp_many(out, w, seeds[w]) for out in outs]
  total = sum(w.size for w in wrts)
  joint = Expr.sym(f"fwd:joint{next(_JOINT_IDS)}", (total,))
  view: dict[Expr, Expr] = {}
  offset = 0
  for w in wrts:
    view[w] = joint[offset : offset + w.size].reshape(w.shape)
    offset += w.size
  seed = concat([seeds[w].reshape((nseed, w.size)) for w in wrts], axis=1)
  back = {joint: concat([w.reshape((w.size,)) for w in wrts])}
  return [substitute(jvp_many(substitute(out, view), joint, seed), back) for out in outs]


# Above this many multiply-adds in a step's seed products, the step's Jacobian becomes a callee of its
# own (straight-line code) and the products stay loops: fully unrolled, they cost seconds of C compile
# time for no faster code.
SEED_PRODUCT_UNROLL_LIMIT = 256


def _jvp_many_compressed(
  outs: Sequence[Expr], seeds: dict[Expr, Expr], nseed: int, *, owner: ConcreteFunction | None = None, name: str = ""
) -> tuple[list[Expr], list[Expr]]:
  """Multi-seed tangents of a loop body's ``outs``, formed once per step as a small Jacobian.

  When the seeded inputs hold fewer entries ``m`` than there are seeds, the body's Jacobian with
  respect to them (``m`` columns, from a pass with constant unit seeds that folds to the nonzero
  partials) costs less than pushing ``nseed`` tangents through the body, and the tangents are one
  product ``seeds @ J`` per input: the work that depends on the seeds is a dense matrix product
  instead of the body's operations repeated per seed. A Hessian through a scan has ``nseed`` equal to
  the horizon and ``m`` equal to a few states and inputs.

  Returns the primal outputs to use (``outs`` itself unless split) and the tangents. With ``owner``
  (the Function whose outputs these are) and products larger than ``SEED_PRODUCT_UNROLL_LIMIT``, the
  primal outputs and every Jacobian that is not a constant are computed by a callee named ``name``
  (straight-line code), and each tangent is a map over the seeds of a small callee that multiplies one
  seed row by the Jacobian: one loop over the seeds whose body keeps its accumulators in registers,
  rather than the products fully unrolled (seconds of C compile time) or one loop per operation
  (memory traffic on every term)."""
  sizes = {w: w.size for w in seeds}
  m = sum(sizes.values())
  if m >= nseed:
    return list(outs), _jvp_many_joint(outs, seeds, nseed)
  unit = np.eye(m, dtype=np.float64)
  offsets: dict[Expr, int] = {}
  offset = 0
  for w, size in sizes.items():
    offsets[w] = offset
    offset += size
  columns = {w: Expr.const(unit[:, offsets[w] : offsets[w] + size].reshape((m, *w.shape))) for w, size in sizes.items()}
  jacobians = [simplify_cse_fixpoint(j.reshape((m, out.size))) for j, out in zip(_jvp_many_joint(outs, columns, m), outs, strict=True)]
  primals = list(outs)
  products = nseed * m * sum(out.size for out in outs)
  split = owner is not None and products > SEED_PRODUCT_UNROLL_LIMIT
  if split:
    assert owner is not None
    varying = [k for k, j in enumerate(jacobians) if j.op != ExprOp.CONST]
    exprs = [*(o.reshape((o.size,)) for o in outs), *(jacobians[k].reshape((jacobians[k].size,)) for k in varying)]
    step = ConcreteFunction._from_exprs(
      name,
      list(owner.inputs),
      [owner._inherit_lowering(simplify_cse_fixpoint(e)) for e in exprs],
      list(owner.input_names),
      [f"out{k}" for k in range(len(outs))] + [f"jac{k}" for k in varying],
    )
    called = step._flat_symbolic_call(list(owner.inputs))
    primals = [c.reshape(o.shape) for c, o in zip(called, outs, strict=False)]
    for k, c in zip(varying, called[len(outs) :], strict=True):
      jacobians[k] = c.reshape(jacobians[k].shape)
  tangents = []
  for k, (out, flat) in enumerate(zip(outs, jacobians, strict=True)):
    if split:
      tangents.append(_seed_product_map(f"{name}_seed{k}", flat, seeds, sizes, offsets, nseed).reshape((nseed, *out.shape)))
      continue
    total: Expr | None = None
    for w, size in sizes.items():
      rows = flat[offsets[w] : offsets[w] + size]
      if _is_zero_const(simplify_cse_fixpoint(rows)):
        continue
      term = seeds[w].reshape((nseed, size)) @ rows
      total = term if total is None else total + term
    tangents.append(Expr.const(np.zeros((nseed, *out.shape), dtype=np.float64)) if total is None else total.reshape((nseed, *out.shape)))
  return primals, tangents


def _seed_product_map(name: str, jac: Expr, seeds: dict[Expr, Expr], sizes: dict[Expr, int], offsets: dict[Expr, int], nseed: int) -> Expr:
  """``sum_w seeds[w] @ jac[rows of w]`` for every seed, as a map over the seeds. A constant
  Jacobian is folded into the mapped callee; otherwise it is read by every seed (stride 0)."""
  m, width = jac.shape
  constant = jac.op == ExprOp.CONST
  jsym = Expr.sym("jac", (m * width,))
  matrix = jac if constant else jsym.reshape((m, width))
  rows = {w: Expr.sym(f"seed{i}", (size,)) for i, (w, size) in enumerate(sizes.items())}
  total: Expr | None = None
  for w, size in sizes.items():
    block = matrix[offsets[w] : offsets[w] + size]
    if constant and _is_zero_const(simplify_cse_fixpoint(block)):
      continue
    term = rows[w] @ block
    total = term if total is None else total + term
  if total is None:
    return Expr.const(np.zeros(nseed * width, dtype=np.float64))
  inputs = [*rows.values(), *([] if constant else [jsym])]
  product = ConcreteFunction._from_exprs(name, inputs, [simplify_cse_fixpoint(total)], [str(e.name) for e in inputs], ["tangent"])
  specs = [(seeds[w].reshape((nseed * size,)), 0, size) for w, size in sizes.items()]
  if not constant:
    specs.append((jac.reshape((m * width,)), 0, 0))
  # ``block``: the loop body stays a loop over the seeds instead of being unrolled seed by seed.
  return vmap(product, nseed, specs).with_lowering("block")


def _strided_view(x: Expr, idx: np.ndarray, rows: int, width: int) -> tuple[Expr, int, int]:
  """Arrange ``x.flat[idx]`` as ``rows`` consecutive blocks of ``width`` for a loop to slice.

  When the index table, composed through any gathers and reshapes that produced ``x``, reads one
  contiguous block per row at a constant stride, the loop reads the source in place and this returns
  ``(source, start, stride)``. Otherwise the gather is materialized and read with stride ``width``
  (or 0 for a single row, which every step reads)."""
  src, table = x, idx.reshape(-1)
  while src.op in (ExprOp.RESHAPE, ExprOp.GATHER):
    if src.op == ExprOp.GATHER:
      table = np.asarray(src.attrs["indices"]).reshape(-1)[table]
    src = src.args[0]
  if table.size:
    blocks = table.reshape(rows, width)
    firsts = blocks[:, 0]
    step = int(firsts[1] - firsts[0]) if rows > 1 else 0
    if np.array_equal(blocks, firsts[:, None] + np.arange(width)[None, :]) and np.array_equal(firsts, firsts[0] + step * np.arange(rows)):
      return src.reshape((src.size,)), int(firsts[0]), step
  return gather(x.reshape((x.size,)), idx.reshape(-1)), 0, (0 if rows == 1 else width)


def _scan_jvp_many_body(callee: ConcreteFunction, active: tuple[int, ...], nseed: int) -> ConcreteFunction:
  """The body of the multi-seed tangent scan. The carry is ``[c, Dc]`` flat, with ``Dc`` the
  ``(nseed, *c.shape)`` tangents seed-major. The sliced inputs are the primal ones, then for each
  ``active`` one its ``(nseed, *x.shape)`` tangents. The outputs are ``[c', Dc']``, the primal stacked
  outputs, then their tangents, each flat and seed-major."""
  # Strict mode is part of the key: a body built with a per-seed fallback inside must not satisfy a
  # later strict build, which would then skip the check it asked for.
  key = (active, nseed, env_bool("SCALY_STRICT_JVP_MANY", False))
  cache = _SCAN_JVP_MANY_CACHE.setdefault(callee, {})
  if key not in cache:
    carry, xs = callee.inputs[0], callee.inputs[1:]
    cs = carry.size
    taken = {*callee.input_names, *callee.output_names}
    aug = Expr.sym(claim_name(f"fwd:{callee.input_names[0]}", taken), (cs * (1 + nseed),))
    dxs = {i: Expr.sym(claim_name(f"fwd:{callee.input_names[i + 1]}", taken), (nseed * xs[i].size,)) for i in active}
    seeds = {carry: aug[cs:].reshape((nseed, *carry.shape)), **{xs[i]: dx.reshape((nseed, *xs[i].shape)) for i, dx in dxs.items()}}
    primals, tangents = _jvp_many_compressed(
      callee.outputs, seeds, nseed, owner=callee, name=f"{callee.name}_stepjac{nseed}_" + ("_".join(str(i) for i in active) or "c")
    )
    nxt = concat([primals[0].reshape((cs,)), tangents[0].reshape((nseed * cs,))])
    flat = [t.reshape((t.size,)) for t in tangents[1:]]
    split = {carry: aug[:cs].reshape(carry.shape)}
    outputs = [substitute(e, split) for e in (nxt, *primals[1:], *flat)]
    inputs = [aug, *xs, *dxs.values()]
    names = [str(aug.name), *callee.input_names[1:], *(str(dx.name) for dx in dxs.values())]
    out_names = [claim_name("fwd:carry", taken), *callee.output_names[1:], *(claim_name(f"fwd:{n}", taken) for n in callee.output_names[1:])]
    suffix = "_".join(str(i) for i in active) or "c"
    cache[key] = ConcreteFunction._from_exprs(
      f"{callee.name}_scanfwd{nseed}_{suffix}", inputs, [callee._inherit_lowering(simplify_cse_fixpoint(o)) for o in outputs], names, out_names
    )
  return cache[key]


def _scan_jvp_many(expr: Expr, tangents: list[Expr], nseed: int) -> Expr:
  """A scan's tangents for ``nseed`` seeds: one scan whose carry also carries all of them."""
  callee, length, output = expr.attrs["callee"], int(expr.attrs["length"]), int(expr.attrs["output"])
  init, outers = expr.args[0], expr.args[1:]
  cs, n_ys = init.size, len(callee.outputs) - 1
  if length == 0:
    return tangents[0].reshape((nseed, *expr.shape)) if output == 0 else Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
  starts, strides = expr.attrs["starts"], expr.attrs["strides"]
  active = tuple(i for i, t in enumerate(tangents[1:]) if not _is_zero_const(t))
  fn = _scan_jvp_many_body(callee, active, nseed)
  aug_init = concat([init.reshape((cs,)), tangents[0].reshape((nseed * cs,))])
  views = []
  for i in active:
    size, outer_size = callee.inputs[i + 1].size, outers[i].size
    rows = 1 if strides[i] == 0 else length
    idx = starts[i] + np.arange(rows)[:, None, None] * strides[i] + np.arange(nseed)[None, :, None] * outer_size + np.arange(size)[None, None, :]
    views.append(_strided_view(tangents[i + 1], idx, rows, nseed * size))
  aug_outers = (*outers, *(v[0] for v in views))
  aug_starts = (*starts, *(v[1] for v in views))
  aug_strides = (*strides, *(v[2] for v in views))

  def node(k: int) -> Expr:
    return _scan_node(fn, aug_init, aug_outers, aug_starts, aug_strides, length, k)

  if output == 0:
    return node(0)[cs:].reshape((nseed, *expr.shape))
  if output == -1:
    width = cs * (1 + nseed)
    idx = np.arange(length)[None, :, None] * width + cs + np.arange(nseed)[:, None, None] * cs + np.arange(cs)[None, None, :]
    return gather(node(-1), idx.reshape(-1)).reshape((nseed, length * cs))
  size = callee.outputs[output].size
  idx = np.arange(length)[None, :, None] * (nseed * size) + np.arange(nseed)[:, None, None] * size + np.arange(size)[None, None, :]
  return gather(node(n_ys + output), idx.reshape(-1)).reshape((nseed, length * size))


def _while_jvp_many_functions(
  cond: ConcreteFunction, body: ConcreteFunction, nseed: int, index: bool, active: tuple[int, ...]
) -> tuple[ConcreteFunction, ConcreteFunction]:
  """Condition and body of the multi-seed tangent loop over the carry ``[c, Dc]``, with the ``active``
  params' seeded tangents (``(nseed, *shape)``) as extra params; the condition reads ``c`` and the
  params only, so the loop takes exactly the primal's steps."""
  cache = _WHILE_JVP_MANY_CACHE.setdefault(body, {})
  key = (nseed, env_bool("SCALY_STRICT_JVP_MANY", False), index, active)
  suffix = f"{nseed}" + _loop_suffix(index, active)
  cs = body.inputs[0].size
  if key not in cache:
    carry = body.inputs[0]
    first = 1 + int(index)
    taken = {*body.input_names, *body.output_names}
    aug = Expr.sym(claim_name(f"fwd:{body.input_names[0]}", taken), (cs * (1 + nseed),))
    dparams = _tangent_params(body, index, active, taken, lambda formal: (nseed, *formal.shape))
    seeds = {carry: aug[cs:].reshape((nseed, *carry.shape))}
    seeds.update({body.inputs[first + i]: d for i, d in zip(active, dparams, strict=True)})
    (primal,), (tangent,) = _jvp_many_compressed(body.outputs, seeds, nseed, owner=body, name=f"{body.name}_stepjac{suffix}")
    split = {carry: aug[:cs].reshape(carry.shape)}
    nxt = substitute(concat([primal.reshape((cs,)), tangent.reshape((nseed * cs,))]), split)
    inputs = [aug, *body.inputs[1:], *dparams]
    aug_body = ConcreteFunction._from_exprs(
      f"{body.name}_whilefwd{suffix}", inputs, [body._inherit_lowering(simplify_cse_fixpoint(nxt))], [str(e.name) for e in inputs], ["fwd:carry"]
    )
    cache[key] = (aug_body, aug, dparams)
  aug_body, aug, dparams = cache[key]
  conds = _WHILE_JVP_MANY_COND_CACHE.setdefault(body, weakref.WeakKeyDictionary()).setdefault(cond, {})
  if key not in conds:
    conds[key] = _tangent_cond(cond, body, aug, cs, dparams, suffix)
  return conds[key], aug_body


def _while_jvp_many(expr: Expr, tangents: Sequence[Expr], nseed: int) -> Expr:
  """A while loop's tangents for ``nseed`` seeds: one loop carrying all of them, the params' seeded
  tangents as further params. The step count's derivative is zero."""
  output = int(expr.attrs["output"])
  cond, body, init, params, max_iter, index = while_parts(expr)
  active = tuple(i for i, t in enumerate(tangents[1:]) if not _is_zero_const(t))
  if output == 1 or (_is_zero_const(tangents[0]) and not active):
    return Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
  cs = init.size
  aug_cond, aug_body = _while_jvp_many_functions(cond, body, nseed, index, active)
  aug_init = concat([init.reshape((cs,)), tangents[0].reshape((nseed * cs,))])
  aug_params = (*params, *(tangents[1 + i] for i in active))
  if output == 0:
    return _while_node(aug_cond, aug_body, aug_init, max_iter, 0, aug_params, index)[cs:].reshape((nseed, *expr.shape))
  width = cs * (1 + nseed)
  idx = np.arange(max_iter)[None, :, None] * width + cs + np.arange(nseed)[:, None, None] * cs + np.arange(cs)[None, None, :]
  return gather(_while_node(aug_cond, aug_body, aug_init, max_iter, -1, aug_params, index), idx.reshape(-1)).reshape((nseed, max_iter * cs))


def _call_jvp_many_function(
  callee: Any, output_index: int, formal_indices: tuple[int, ...], nseed: int, constants: tuple[np.ndarray | None, ...]
) -> tuple[Any, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
  key = (output_index, formal_indices, nseed, tuple(None if value is None else value.tobytes() for value in constants))
  cache = _CALL_JVP_MANY_CACHE.setdefault(callee, {})
  if key not in cache:
    active = tuple(range(nseed))
    if all(value is not None for value in constants):
      active = tuple(row for row in range(nseed) if any(value is not None and np.any(value[row] != 0) for value in constants))
    taken = {*callee.input_names, *callee.output_names}
    seed_names = {i: claim_name(f"fwd:{callee.input_names[i]}", taken) for i in formal_indices}
    seeds = {
      i: Expr.sym(seed_names[i], (nseed, *callee.inputs[i].shape)) if value is None else Expr.const(value[list(active)])
      for i, value in zip(formal_indices, constants, strict=True)
    }
    out = callee.outputs[output_index]
    single_constant = len(formal_indices) == 1 and constants[0] is not None
    if _contains_loop(callee):
      # Per-seed tangents would put one copy of every loop in the helper per seed; one multi-seed
      # pass keeps a single loop carrying all of them.
      deriv = _jvp_many_joint([out], {callee.inputs[i]: seeds[i] for i in formal_indices}, len(active))[0]
    elif single_constant:
      formal_index = formal_indices[0]
      deriv = _jvp_many_unrolled(out, callee.inputs[formal_index], seeds[formal_index])
    else:
      deriv = stack([_jvp(out, {callee.inputs[i]: seed[row] for i, seed in seeds.items()}, {}, {}) for row in range(len(active))], axis=0)
    deriv = callee._inherit_lowering(simplify_cse_fixpoint(deriv))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    seed_indices = tuple(i for i, value in zip(formal_indices, constants, strict=True) if value is None and _depends_on(deriv, seeds[i], dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[i] for i in seed_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(seed_names[i] for i in seed_indices)
    if single_constant:
      formal_index = formal_indices[0]
      seed_hash = hashlib.sha1(constants[0].tobytes()).hexdigest()[:10]  # type: ignore[union-attr]
      name = f"{callee.name}_fwd{nseed}c{seed_hash}_{callee.output_names[output_index]}_{callee.input_names[formal_index]}"
      output_name = claim_name(f"fwd:{callee.output_names[output_index]}:{callee.input_names[formal_index]}", taken)
    else:
      seed_hash = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
      name = f"{callee.name}_fwd{nseed}j{seed_hash}_{output_index}_" + "_".join(str(i) for i in formal_indices)
      output_name = claim_name(f"fwd:{callee.output_names[output_index]}", taken)
    fn = ConcreteFunction._from_exprs(name, inputs, [deriv], input_names, [output_name])
    cache[key] = (fn, arg_indices, seed_indices, active)
  return cache[key]


def _call_jvp_many_const_function(
  callee: Any, output_index: int, formal_index: int, seed_value: np.ndarray
) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  """Specialize the joint JVP helper for one formal with a constant seed."""
  fn, arg_indices, seed_indices, active = _call_jvp_many_function(callee, output_index, (formal_index,), seed_value.shape[0], (seed_value,))
  assert not seed_indices
  return fn, arg_indices, active


def _call_jvp_function(callee: Any, output_index: int, formal_indices: tuple[int, ...]) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
  key = (output_index, formal_indices)
  cache = _CALL_JVP_CACHE.setdefault(callee, {})
  if key not in cache:
    taken = {*callee.input_names, *callee.output_names}
    seeds = {i: Expr.sym(claim_name(f"fwd:{callee.input_names[i]}", taken), callee.inputs[i].shape) for i in formal_indices}
    deriv = callee._inherit_lowering(body_tangents(callee, seeds)[output_index])
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(deriv, inp, dep_memo))
    seed_indices = tuple(i for i, seed in seeds.items() if _depends_on(deriv, seed, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + tuple(seeds[i] for i in seed_indices)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + tuple(seeds[i].name for i in seed_indices)
    name = f"{callee.name}_fwd{output_index}_" + "_".join(str(i) for i in formal_indices)
    fn = ConcreteFunction._from_exprs(name, inputs, [deriv], input_names, [claim_name(f"fwd:{callee.output_names[output_index]}", taken)])
    cache[key] = (fn, arg_indices, seed_indices)
  return cache[key]


def _pack_jvp_maps(callee: Any, result: Expr, maps: list[Expr]) -> Expr:
  """Share specialized tangent bodies through one mapped result, preserving each seed layout."""
  groups: list[tuple[int, dict[Expr, tuple[Expr, int, int]], list[Expr]]] = []
  for mapped in maps:
    fn = mapped.attrs["callee"]
    specs = dict(zip(fn.inputs, zip(mapped.args, mapped.attrs["starts"], mapped.attrs["strides"], strict=True), strict=True))
    for length, bindings, members in groups:
      if length == mapped.attrs["length"] and all(formal not in bindings or bindings[formal] == spec for formal, spec in specs.items()):
        bindings.update(specs)
        members.append(mapped)
        break
    else:
      groups.append((mapped.attrs["length"], specs, [mapped]))
  replacements = {}
  cache = _CALL_JVP_PACK_CACHE.setdefault(callee, {})
  for length, bindings, members in groups:
    if len(members) == 1:
      continue
    functions = tuple(mapped.attrs["callee"] for mapped in members)
    inputs = tuple(bindings)
    key = (*functions, inputs)
    if key not in cache:
      outputs = [fn.outputs[0].reshape((fn.outputs[0].size,)) for fn in functions]
      packed = callee._inherit_lowering(simplify_cse_fixpoint(concat(outputs)))
      name_hash = hashlib.sha1(";".join(fn.name for fn in functions).encode()).hexdigest()[:10]
      names = {inp: name for fn in functions for inp, name in zip(fn.inputs, fn.input_names, strict=True)}
      cache[key] = ConcreteFunction._from_exprs(f"{callee.name}_fwd_pack_{name_hash}", inputs, [packed], [names[inp] for inp in inputs], ["fwd"])
    fn = cache[key]
    mapped = vmap(fn, length, list(bindings.values()))
    width = fn.outputs[0].size
    offset = 0
    for member, function in zip(members, functions, strict=True):
      size = function.outputs[0].size
      indices = (np.arange(length)[:, None] * width + offset + np.arange(size)).reshape(-1)
      replacements[member] = gather(mapped, indices)
      offset += size
  return substitute(result, replacements) if replacements else result


def _periodic_seed_tiles(
  tangent: Expr, nseed: int, outer_size: int, start: int, stride: int, formal_size: int, length: int
) -> tuple[np.ndarray, int] | None:
  if not length or tangent.op != ExprOp.CONST or tangent.value is None:
    return None
  flat = np.asarray(tangent.value, dtype=np.float64).reshape(nseed, outer_size)
  tiles = np.stack([flat[:, start + it * stride : start + it * stride + formal_size] for it in range(length)])
  period = next(
    (k for k in range(1, min(8, length // 2) + 1) if length % k == 0 and np.array_equal(tiles, np.tile(tiles[:k], (length // k, 1, 1)))),
    None,
  )
  return None if period is None else (tiles, period)


def _local_seed_colors(callee_out: Expr, formal: Expr, nseed: int) -> tuple[Any, tuple[int, ...]] | None:
  mask = _jac_mask(callee_out, formal, {})
  colors = column_coloring(_mask_sparsity(mask)) if mask.nnz else ()
  return (mask, colors) if colors and max(colors) + 1 < nseed else None


def jvp_many(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  """Forward mode over several seeds in one pass. ``seeds`` has shape ``(n, *wrt.shape)``.

  Structural rules share work across seeds — one ``cos`` serves every column of a ``sin``'s
  derivative — so this is much cheaper than ``n`` separate ``jvp`` calls. Ops without a
  multi-seed rule fall back to per-seed evaluation; set ``SCALY_STRICT_JVP_MANY=1`` to raise
  instead of falling back. Returns shape ``(n, *expr.shape)``.
  """
  if len(seeds.shape) < 1 or seeds.shape[1:] != wrt.shape:
    raise ValueError(f"multi-seed JVP expects seeds shape (nseed, *{wrt.shape}), got {seeds.shape}")
  if wrt.op != ExprOp.INPUT:
    (expr,), (at,), back = independent((expr,), (wrt,))
    return substitute(jvp_many(expr, at, seeds), back)
  if seeds.shape[0] == 0:
    return Expr.const(np.zeros((0, *expr.shape), dtype=np.float64))
  strict = env_bool("SCALY_STRICT_JVP_MANY", False)
  try:
    ret = _jvp_many_structural(expr, wrt, seeds, {}, {})
  except _JVPManyUnsupported as unsupported:
    if strict:
      raise NotImplementedError(
        f"structural jvp_many does not support {unsupported.op!r}; SCALY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
      ) from unsupported
    return _jvp_many_unrolled(expr, wrt, seeds)
  expected = (seeds.shape[0], *expr.shape)
  if ret.shape == expected:
    return ret
  if strict:
    raise NotImplementedError(
      f"structural jvp_many returned shape {ret.shape} for {expr.op!r}, expected {expected}; SCALY_STRICT_JVP_MANY=1 forbids the unrolled fallback"
    )
  return _jvp_many_unrolled(expr, wrt, seeds)


def _jvp_many_unrolled(expr: Expr, wrt: Expr, seeds: Expr) -> Expr:
  return stack([jvp(expr, wrt, seeds[i]) for i in range(seeds.shape[0])], axis=0)


def _formable_tangent(expr: Expr, i: int, wrt: Expr, seeds: Expr, memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  """The tangents of argument ``i`` of a call, or zeros if they cannot be formed (a looped sparse
  factor's) and the callee's derivative never reads them, as an implicit solve rule never reads its
  factor's; one that is read and cannot be formed raises."""
  try:
    return simplify_cse_fixpoint(_jvp_many_structural(expr.args[i], wrt, seeds, memo, dep_memo))
  except NotImplementedError:
    active = tuple(k for k, a in enumerate(expr.args) if _depends_on(a, wrt, dep_memo))
    if i in _call_jvp_function(expr.attrs["callee"], expr.attrs["output"], active)[2]:
      raise
    return Expr.const(np.zeros((seeds.shape[0], *expr.args[i].shape), dtype=np.float64))


def _prunable_call(expr: Expr, wrt: Expr, seeds: Expr, memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> bool:
  """Whether every argument tangent of a call is a constant and some seed is zero in all of them.

  Stops at the first tangent that is not a constant. One that cannot be formed (a looped sparse
  factor's) is not a constant either: the body, differentiated in place, forms only the tangents
  it reads, and an implicit rule inside it does not read that one."""
  tangents = []
  for arg in expr.args:
    try:
      t = simplify_cse_fixpoint(_jvp_many_structural(arg, wrt, seeds, memo, dep_memo))
    except NotImplementedError:
      return False
    if t.op != ExprOp.CONST or t.value is None:
      return False
    tangents.append(t)
  nseed = seeds.shape[0]
  live = np.zeros(nseed, dtype=bool)
  for t in tangents:
    live |= np.any(np.asarray(t.value).reshape(nseed, -1) != 0, axis=1)
  return not bool(live.all())


def _jvp_many_structural(expr: Expr, wrt: Expr, seeds: Expr, memo: dict[int, Expr], dep_memo: dict[tuple[int, int], bool]) -> Expr:
  if expr.id in memo:
    return memo[expr.id]
  nseed = seeds.shape[0]
  if _is_zero_const(seeds) or not _depends_on(expr, wrt, dep_memo):
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.INPUT:
    memo[expr.id] = ret = seeds if expr.id == wrt.id else Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.CONST or expr.op in PREDICATE_OPS:
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    return ret
  if expr.op == ExprOp.SELECT:
    cond, a, b = expr.args
    da = _broadcast_tangent(_jvp_many_structural(a, wrt, seeds, memo, dep_memo), a, expr, nseed)
    db = _broadcast_tangent(_jvp_many_structural(b, wrt, seeds, memo, dep_memo), b, expr, nseed)
    memo[expr.id] = ret = where(cond, da, db)
    return ret
  if expr.op == ExprOp.SLICE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)[(slice(None), *expr.attrs["index"])]
    return ret
  if expr.op == ExprOp.RESHAPE:
    memo[expr.id] = ret = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.TRANSPOSE:
    if len(expr.shape) > 3:
      raise _JVPManyUnsupported(str(expr.op))  # seed axis would make a rank-5 TRANSPOSE, beyond the rank-4 lowering limit
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    memo[expr.id] = ret = d0.transpose((0, *(axis + 1 for axis in expr.attrs["axes"])))
    return ret
  if expr.op == ExprOp.ADD:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    if _is_zero_const(d0) and expr.args[1].shape == expr.shape:
      memo[expr.id] = ret = d1
    elif _is_zero_const(d1) and expr.args[0].shape == expr.shape:
      memo[expr.id] = ret = d0
    else:
      memo[expr.id] = ret = _broadcast_tangent(d0, expr.args[0], expr, nseed) + _broadcast_tangent(d1, expr.args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.SUB:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    d1 = _jvp_many_structural(expr.args[1], wrt, seeds, memo, dep_memo)
    if _is_zero_const(d0) and expr.args[1].shape == expr.shape:
      memo[expr.id] = ret = -d1
    elif _is_zero_const(d1) and expr.args[0].shape == expr.shape:
      memo[expr.id] = ret = d0
    else:
      memo[expr.id] = ret = _broadcast_tangent(d0, expr.args[0], expr, nseed) - _broadcast_tangent(d1, expr.args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.NEG:
    memo[expr.id] = ret = -_jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    return ret
  if expr.op == ExprOp.CONCAT:
    memo[expr.id] = ret = concat([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == ExprOp.STACK:
    memo[expr.id] = ret = stack([_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in expr.args], axis=expr.attrs.get("axis", 0) + 1)
    return ret
  if expr.op == ExprOp.GATHER:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.args[0].size + indices[None, :]).reshape(-1)
    memo[expr.id] = ret = gather(d0.reshape((nseed * expr.args[0].size,)), full).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.SCATTER:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    indices = expr.attrs["indices"].reshape(-1)
    full = (np.arange(nseed, dtype=np.int64)[:, None] * expr.size + indices[None, :]).reshape(-1)
    memo[expr.id] = ret = scatter(d0.reshape((nseed * expr.args[0].size,)), full, (nseed * expr.size,)).reshape((nseed, *expr.shape))
    return ret
  if expr.op == ExprOp.CAST:
    if not expr.type.diff:
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    else:
      memo[expr.id] = ret = cast(_jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo), expr.type.dtype)
    return ret
  if expr.op == ExprOp.SUM:
    d0 = _jvp_many_structural(expr.args[0], wrt, seeds, memo, dep_memo)
    ones = Expr.const(np.ones(expr.args[0].size), dtype=d0.type.dtype)
    memo[expr.id] = ret = d0.reshape((nseed, expr.args[0].size)) @ ones
    return ret
  if expr.op == ExprOp.CALL and expr.attrs["callee"].custom_jvp is not None:
    # The rule once per seed, as one map: primal arguments broadcast, tangents sliced by seed.
    rule = expr.attrs["callee"].custom_jvp
    # A tangent the rule never reads (an implicit rule ignoring a precomputed factor) is not formed.
    rule_dep: dict[tuple[int, int], bool] = {}
    out = rule.outputs[int(expr.attrs["output"])]
    read = [_depends_on(out, rule.inputs[len(expr.args) + i], rule_dep) for i in range(len(expr.args))]
    tangents = [_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) if r else None for arg, r in zip(expr.args, read, strict=True)]
    if all(t is None or _is_zero_const(t) for t in tangents):
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
      return ret
    specs = [(arg.reshape((arg.size,)), 0, 0) for arg in expr.args]
    specs += [
      (Expr.const(np.zeros(arg.size)), 0, 0) if t is None else (t.reshape((nseed * arg.size,)), 0, arg.size)
      for t, arg in zip(tangents, expr.args, strict=True)
    ]
    memo[expr.id] = ret = vmap(rule, nseed, specs, output=int(expr.attrs["output"])).reshape((nseed, *expr.shape))
    return ret
  if expr.op in (ExprOp.VMAP, ExprOp.SCAN, ExprOp.WHILE) and expr.attrs["callee"].custom_jvp is not None:
    raise _JVPManyUnsupported("custom jvp")  # the per-seed fallback honors the rule through ``_call_jvp_function``
  if expr.op == ExprOp.SCAN:
    tangents = [simplify_cse_fixpoint(_jvp_many_structural(arg, wrt, seeds, memo, dep_memo)) for arg in expr.args]
    if all(_is_zero_const(t) for t in tangents):
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    else:
      memo[expr.id] = ret = _scan_jvp_many(expr, tangents, nseed)
    return ret
  if expr.op == ExprOp.WHILE:
    if int(expr.attrs["output"]) == 1:  # the step count is piecewise constant
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
    else:
      tangents = [simplify_cse_fixpoint(_jvp_many_structural(arg, wrt, seeds, memo, dep_memo)) for arg in expr.args]
      memo[expr.id] = ret = _while_jvp_many(expr, tangents, nseed)
    return ret
  if expr.op == ExprOp.VMAP:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    length = expr.attrs["length"]
    starts = expr.attrs["starts"]
    strides = expr.attrs["strides"]
    slice_size = expr.attrs["slice_size"]
    callee_out = callee.outputs[output_idx]

    maps: list[Expr] = []

    def mapped_call(fn: Any, count: int, specs: list[tuple[Expr, int, int]]) -> Expr:
      mapped = vmap(fn, count, specs)
      maps.append(mapped)
      return mapped

    generic_seeds: dict[int, Expr] = {}
    ret: Expr | None = None
    for formal_idx, actual_outer in enumerate(expr.args):
      actual_tan = _formable_tangent(expr, formal_idx, wrt, seeds, memo, dep_memo)
      if _is_zero_const(actual_tan):
        continue
      formal = callee.inputs[formal_idx]
      formal_size = formal.size
      outer_size = actual_outer.size
      start = starts[formal_idx]
      stride = strides[formal_idx]

      # A constant tangent whose per-iteration tiles repeat with a short period (at most 8, and at
      # least twice, so a short horizon of distinct tiles is not unrolled) is baked into one
      # const-seed callee per tile: the 0/1 products fold inside the body and no seed table or
      # gather is emitted. Iteration ``it`` uses tile ``it % period``; residue class ``r`` is mapped
      # with start ``start + r * stride`` and stride ``stride * period``.
      periodic = _periodic_seed_tiles(actual_tan, nseed, outer_size, start, stride, formal_size, length)
      if periodic is not None:
        tiles, period = periodic
        n = length // period
        zero_rows = Expr.const(np.zeros((n, slice_size), dtype=np.float64))
        parts: list[Expr] = []
        for r in range(period):
          inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(
            callee, output_idx, formal_idx, tiles[r].reshape((nseed, *formal.shape))
          )
          if not active:
            parts.append(Expr.const(np.zeros((nseed, n, slice_size), dtype=np.float64)))
            continue
          primal_specs = [(expr.args[i], starts[i] + r * strides[i], strides[i] * period) for i in primal_arg_indices]
          mapped_3d = mapped_call(inner_fn, n, primal_specs).reshape((n, len(active), slice_size))
          if len(active) == nseed:
            parts.append(mapped_3d.transpose((1, 0, 2)))
          else:
            active_pos = {row: i for i, row in enumerate(active)}
            parts.append(stack([mapped_3d[:, active_pos[row], :] if row in active_pos else zero_rows for row in range(nseed)], axis=0))
        if all(_is_zero_const(part) for part in parts):
          continue
        term = (parts[0] if period == 1 else stack(parts, axis=2)).reshape((nseed, length * slice_size))
        ret = term if ret is None else ret + term
        continue

      # Local coloring of the callee's Jacobian tile w.r.t. this formal. When this gives c_f < nseed
      # local colors, baking those local seeds into the per-iteration JVP callee yields a body of
      # size O(c_f) instead of O(nseed). The global compressed JVP is then assembled back by, for
      # each local color `c_local`, gathering the unique nonzero column of `actual_tan` for each
      # output row and multiplying with the per-iter compressed-JVP slice.
      local = _local_seed_colors(callee_out, formal, nseed)
      if local is not None:
        local_mask, local_colors = local
        c_f = max(local_colors) + 1
        seed_f = np.zeros((c_f, formal_size), dtype=np.float64)
        for j, c in enumerate(local_colors):
          seed_f[c, j] = 1.0
        seed_f_shaped = seed_f.reshape((c_f, *formal.shape)) if formal.shape != (formal.size,) else seed_f
        inner_fn, primal_arg_indices, active = _call_jvp_many_const_function(callee, output_idx, formal_idx, seed_f_shaped)
        active_count = len(active)
        if active_count == 0:
          continue
        # For each local color, pick a representative formal column j whose Jacobian row k has a
        # nonzero. By the coloring property, that nonzero j is unique within the color group, so the
        # column's value in actual_tan is the full contribution at row k. If no j in the color group
        # has a nonzero at row k, the per-iter JVP entry is zero and the choice of j is irrelevant.
        local_colors_arr = np.asarray(local_colors, dtype=np.int64)
        unique_j = np.zeros((c_f, slice_size), dtype=np.int64)
        for c in range(c_f):
          cols = np.flatnonzero(local_colors_arr == c)
          for k in range(slice_size):
            for j in cols:
              if local_mask[k, j]:
                unique_j[c, k] = int(j)
                break
        primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
        mapped_flat = mapped_call(inner_fn, length, primal_specs)
        mapped_3d = mapped_flat.reshape((length, active_count, slice_size))
        c_arr = np.arange(nseed, dtype=np.int64).reshape(nseed, 1, 1)
        it_arr = np.arange(length, dtype=np.int64).reshape(1, length, 1)
        for pos, c_local in enumerate(active):
          uj_arr = unique_j[c_local].reshape(1, 1, slice_size)
          flat_idx = (c_arr * outer_size + start + it_arr * stride + uj_arr).reshape(-1)
          gathered = gather(actual_tan, flat_idx).reshape((nseed, length, slice_size))
          mapped_slice = mapped_3d[:, pos, :].reshape((1, length, slice_size))
          contribution = (gathered * mapped_slice).reshape((nseed, length * slice_size))
          ret = contribution if ret is None else ret + contribution
        continue

      it_arr = np.arange(length, dtype=np.int64)[:, None, None]
      c_arr = np.arange(nseed, dtype=np.int64)[None, :, None]
      j_arr = np.arange(formal_size, dtype=np.int64)[None, None, :]
      tile_indices = (c_arr * outer_size + start + it_arr * stride + j_arr).reshape(-1)
      seed_buffer = gather(actual_tan, tile_indices)
      generic_seeds[formal_idx] = seed_buffer
    if generic_seeds:
      formals = tuple(generic_seeds)
      inner_fn, primal_arg_indices, seed_indices, _ = _call_jvp_many_function(callee, output_idx, formals, nseed, (None,) * len(formals))
      primal_specs = [(expr.args[i], starts[i], strides[i]) for i in primal_arg_indices]
      seed_specs = [(generic_seeds[i], 0, nseed * callee.inputs[i].size) for i in seed_indices]
      mapped_flat = mapped_call(inner_fn, length, [*primal_specs, *seed_specs])
      term = mapped_flat.reshape((length, nseed, slice_size)).transpose((1, 0, 2)).reshape((nseed, length * slice_size))
      ret = term if ret is None else ret + term
    ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    memo[expr.id] = ret = _pack_jvp_maps(callee, ret, maps)
    return ret
  if expr.op == ExprOp.CALL and _contains_loop(expr.attrs["callee"]) and not _prunable_call(expr, wrt, seeds, memo, dep_memo):
    # Differentiate the call's body in place. Reverse mode inlines such a body too, so in a Hessian
    # the tangent loops built here are the same nodes as the ones built for the gradient's stored
    # carries, and every output of the call shares them: they are computed once. A call whose
    # arguments have a constant tangent that is zero for some seeds (one of K shooting intervals,
    # say) keeps its helper below instead, which drops those seeds; inlined, each of the K calls
    # would carry every seed.
    callee = expr.attrs["callee"]
    inlined = substitute(callee.outputs[expr.attrs["output"]], dict(zip(callee.inputs, expr.args, strict=True)))
    # ``memo`` is keyed by node id, and interned nodes are freed once unreferenced, so a later
    # inlined graph could reuse an id and pick up this one's tangents. Keep the graph alive for as
    # long as ``memo`` is (a negative key never collides with an id).
    memo[~inlined.id] = inlined
    memo[expr.id] = ret = _jvp_many_structural(inlined, wrt, seeds, memo, dep_memo)
    return ret
  if expr.op == ExprOp.CALL:
    tangents = [_formable_tangent(expr, i, wrt, seeds, memo, dep_memo) for i in range(len(expr.args))]
    formals = tuple(i for i, tangent in enumerate(tangents) if not _is_zero_const(tangent))
    if not formals:
      memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64))
      return ret
    constants = tuple(tangents[i].value if tangents[i].op == ExprOp.CONST else None for i in formals)
    fn, arg_indices, seed_indices, active = _call_jvp_many_function(expr.attrs["callee"], expr.attrs["output"], formals, nseed, constants)
    call_args = [expr.args[i] for i in arg_indices] + [tangents[i] for i in seed_indices]
    ret = fn._flat_symbolic_call(call_args)[0]
    if len(active) != nseed:
      active_pos = {row: i for i, row in enumerate(active)}
      ret = stack([ret[active_pos[row]] if row in active_pos else zeros_like(expr) for row in range(nseed)], axis=0)
    memo[expr.id] = ret
    return ret

  args = expr.args
  d = [_jvp_many_structural(arg, wrt, seeds, memo, dep_memo) for arg in args]
  if expr.op == ExprOp.MUL:
    if args[0] is args[1]:
      memo[expr.id] = ret = _seed_axis(2 * args[0], nseed) * d[0]
      return ret
    memo[expr.id] = ret = _broadcast_tangent(d[0], args[0], expr, nseed) * _seed_axis(args[1], nseed, expr) + _seed_axis(
      args[0], nseed, expr
    ) * _broadcast_tangent(d[1], args[1], expr, nseed)
    return ret
  if expr.op == ExprOp.DIV:
    inv = _seed_axis(1.0 / args[1], nseed, expr)
    memo[expr.id] = ret = (
      _broadcast_tangent(d[0], args[0], expr, nseed) - _seed_axis(expr, nseed) * _broadcast_tangent(d[1], args[1], expr, nseed)
    ) * inv
    return ret
  if expr.op == ExprOp.POW:
    if args[1].op == ExprOp.CONST:
      memo[expr.id] = ret = _seed_axis(args[1] * (args[0] ** _minus_one(args[1])), nseed, expr) * _broadcast_tangent(d[0], args[0], expr, nseed)
    else:
      x = _seed_axis(args[0], nseed, expr)
      y = _seed_axis(args[1], nseed, expr)
      dx = _broadcast_tangent(d[0], args[0], expr, nseed)
      dy = _broadcast_tangent(d[1], args[1], expr, nseed)
      memo[expr.id] = ret = _seed_axis(expr, nseed) * (dy * x.log() + y * dx / x)
    return ret
  if expr.op == ExprOp.SIN:
    memo[expr.id] = ret = _seed_axis(args[0].cos(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.COS:
    memo[expr.id] = ret = -_seed_axis(args[0].sin(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.TAN:
    memo[expr.id] = ret = d[0] / (_seed_axis(args[0].cos(), nseed) ** 2)
    return ret
  if expr.op == ExprOp.EXP:
    memo[expr.id] = ret = _seed_axis(expr, nseed) * d[0]
    return ret
  if expr.op == ExprOp.LOG:
    memo[expr.id] = ret = d[0] / _seed_axis(args[0], nseed)
    return ret
  if expr.op == ExprOp.SQRT:
    memo[expr.id] = ret = d[0] * _seed_axis(0.5 / expr, nseed)
    return ret
  if expr.op == ExprOp.TANH:
    memo[expr.id] = ret = d[0] * (1 - _seed_axis(expr * expr, nseed))
    return ret
  if expr.op == ExprOp.ERF:
    memo[expr.id] = ret = _seed_axis((2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.COSH:
    memo[expr.id] = ret = _seed_axis(args[0].sinh(), nseed) * d[0]
    return ret
  if expr.op == ExprOp.SINH:
    memo[expr.id] = ret = _seed_axis(args[0].cosh(), nseed) * d[0]
    return ret
  if expr.op in {ExprOp.CHOLESKY, ExprOp.LDL, ExprOp.TRISOLVE}:
    memo[expr.id] = ret = _jvp_many_dense(expr, d, nseed)
    return ret
  if expr.op == ExprOp.SPARSE_LDL:
    raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
  if expr.op == ExprOp.LU:
    raise NotImplementedError(LU_NO_DERIVATIVE)
  if expr.op == ExprOp.SPARSE_LDL_SOLVE:
    if not _is_zero_const(d[0]):
      raise NotImplementedError(SPARSE_LDL_NO_DERIVATIVE)
    if nseed > 64:
      raise _JVPManyUnsupported(str(expr.op))  # one solve per seed would outgrow the per-seed fallback
    memo[expr.id] = ret = stack([sparse_ldl_solve(args[0], d[1][k], dict(expr.attrs)) for k in range(nseed)], axis=0)
    return ret
  if expr.op in {ExprOp.RAGGED_ADD, ExprOp.RAGGED_DOT}:
    if nseed > 64:
      raise _JVPManyUnsupported(str(expr.op))  # one tangent op per seed would outgrow the per-seed fallback
    rows = [ragged_tangent(expr, [None if _is_zero_const(t) else t[k] for t in d]) for k in range(nseed)]
    memo[expr.id] = ret = stack(rows, axis=0)
    return ret
  if expr.op == ExprOp.TAKE:  # the seed axis leads and ``take`` indexes the last one
    memo[expr.id] = ret = take(d[0], args[1], in_range=bool(expr.attrs.get("in_range")))
    return ret
  if expr.op in {ExprOp.PUT_ADD, ExprOp.PUT}:
    memo[expr.id] = ret = (put_add if expr.op == ExprOp.PUT_ADD else put)(d[0], args[1], d[2], in_range=bool(expr.attrs.get("in_range")))
    return ret
  if expr.op == ExprOp.MATMUL:
    if args[0] is args[1] and len(args[0].shape) == 1:
      memo[expr.id] = ret = 2 * (d[0] @ args[0])
      return ret
    ret: Expr | None = None
    if not _is_zero_const(d[0]):
      ret = _jvp_many_matmul_left(args[0], args[1], d[0], nseed)
    if not _is_zero_const(d[1]):
      term = _jvp_many_matmul_right(args[0], args[1], d[1], nseed)
      ret = term if ret is None else ret + term
    memo[expr.id] = ret = Expr.const(np.zeros((nseed, *expr.shape), dtype=np.float64)) if ret is None else ret
    return ret
  raise _JVPManyUnsupported(str(expr.op))


def _seed_axis(expr: Expr, nseed: int, output: Expr | None = None) -> Expr:
  # tangents carry a leading seed axis, so a stacked primal must also be rank-aligned against (nseed, *output.shape) when its rank is lower
  if expr.shape == ():
    return expr
  missing = 0 if output is None else len(output.shape) - len(expr.shape)
  t = stack([expr] * nseed, axis=0)
  return t if missing == 0 else t.reshape((nseed, *(1,) * missing, *expr.shape))


def _broadcast_tangent(tangent: Expr, operand: Expr, output: Expr, nseed: int) -> Expr:
  missing = len(output.shape) - len(operand.shape)
  return tangent if missing == 0 else tangent.reshape((nseed, *(1,) * missing, *operand.shape))


def _jvp_many_matmul_left(x: Expr, y: Expr, dx: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return stack([dx[i] @ y for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 2:
    return dx @ y
  if len(x.shape) == 1 and len(y.shape) == 1:
    return dx @ y  # (nseed, n) @ (n,) -> (nseed,); a plain .sum() would also contract the seed axis
  return stack([dx[i] @ y for i in range(nseed)], axis=0)


def _jvp_many_matmul_right(x: Expr, y: Expr, dy: Expr, nseed: int) -> Expr:
  if len(x.shape) == 2 and len(y.shape) == 1:
    return dy @ x.T
  if len(x.shape) == 1 and len(y.shape) == 2:
    return stack([x @ dy[i] for i in range(nseed)], axis=0)
  if len(x.shape) == 1 and len(y.shape) == 1:
    return dy @ x  # (nseed, n) @ (n,) -> (nseed,); a plain .sum() would also contract the seed axis
  return stack([x @ dy[i] for i in range(nseed)], axis=0)
