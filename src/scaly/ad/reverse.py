"""Reverse-mode AD: ``vjp`` and the per-op local adjoint rules."""

from __future__ import annotations

import weakref
from typing import Any, Iterable, Sequence

import numpy as np

from ..function.concrete import ConcreteFunction
from ..function.sugar import _mapped_call
from ..ir.expr import Expr, ExprOp, as_expr, concat, gather, scatter, stack, topo, zeros_like
from ..passes.expr import simplify_cse_fixpoint
from .sparsity import _depends_on


_VMAP_ADJ_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[int, tuple[int, ...]], tuple[Any, tuple[int, ...]]]] = weakref.WeakKeyDictionary()


def _substitute(expr: Expr, replacements: dict[int, Expr]) -> Expr:
  memo: dict[int, Expr] = {}
  for node in topo((expr,)):
    if node.id in replacements:
      memo[node.id] = replacements[node.id]
      continue
    args = tuple(memo[arg.id] for arg in node.args)
    memo[node.id] = (
      node
      if all(a is b for a, b in zip(args, node.args, strict=True))
      else Expr(node.op, args, node.type, node.name, node.value, dict(node.attrs), node.lowering)
    )
  return memo[expr.id]


def _vmap_adj_function(callee: Any, output_index: int, active_formals: tuple[int, ...]) -> tuple[Any, tuple[int, ...]]:
  key = (output_index, active_formals)
  cache = _VMAP_ADJ_CACHE.setdefault(callee, {})
  if key not in cache:
    out = callee.outputs[output_index]
    lam_name = f"lam:{callee.output_names[output_index]}"
    lam = Expr.sym(lam_name, out.shape)
    grads = vjp((out,), tuple(callee.inputs[i] for i in active_formals), (lam,))
    adj = callee._inherit_lowering(simplify_cse_fixpoint(concat([grad.reshape((grad.size,)) for grad in grads])))
    dep_memo: dict[tuple[int, int], bool] = {}
    arg_indices = tuple(i for i, inp in enumerate(callee.inputs) if _depends_on(adj, inp, dep_memo))
    inputs = tuple(callee.inputs[i] for i in arg_indices) + (lam,)
    input_names = tuple(callee.input_names[i] for i in arg_indices) + (lam_name,)
    # Suffix by formal index, not name: joined names are not injective ({a_b} vs {a, b}) and
    # lowering dedupes callees by name, so a collision would silently reuse the wrong proc body.
    name = f"{callee.name}_adj{output_index}_" + "_".join(str(i) for i in active_formals)
    fn = ConcreteFunction._from_exprs(name, inputs, [adj], input_names, [f"adj:{callee.output_names[output_index]}"], role="adjoint")
    cache[key] = (fn, arg_indices)
  return cache[key]


def _vmap_vjp(vmap_expr: Expr, cot: Expr, wrts: Sequence[Expr], dep_memo: dict[tuple[int, int], bool]) -> list[tuple[Expr, Expr]]:
  callee = vmap_expr.attrs["callee"]
  output_idx = vmap_expr.attrs["output"]
  length = vmap_expr.attrs["length"]
  if length == 0:
    return []
  starts = vmap_expr.attrs["starts"]
  strides = vmap_expr.attrs["strides"]
  slice_size = vmap_expr.attrs["slice_size"]
  active_formals = tuple(k for k, arg in enumerate(vmap_expr.args) if any(_depends_on(arg, wrt, dep_memo) for wrt in wrts))
  if not active_formals:
    return []

  adj_fn, arg_indices = _vmap_adj_function(callee, output_idx, active_formals)
  primal_specs = [(vmap_expr.args[i], starts[i], strides[i]) for i in arg_indices]
  mapped = _mapped_call(adj_fn, length, [*primal_specs, (cot, 0, slice_size)])
  adj_size = sum(callee.inputs[k].size for k in active_formals)
  ret: list[tuple[Expr, Expr]] = []
  offset = 0
  for k in active_formals:
    arg, start, stride = vmap_expr.args[k], starts[k], strides[k]
    formal_size = callee.inputs[k].size
    if stride == 0:
      indices = np.asarray([it * adj_size + offset + j for it in range(length) for j in range(formal_size)], dtype=np.int64)
      segments = gather(mapped, indices).reshape((length, formal_size))
      vbar = Expr.const(np.ones(length, dtype=np.float64)) @ segments
      if start != 0 or formal_size != arg.size:
        vbar = scatter(vbar, start + np.arange(formal_size, dtype=np.int64), arg.shape)
    else:
      pieces: list[Expr] = []
      groups = -(-formal_size // stride)
      for group in range(groups):
        iterations = range(group, length, groups)
        indices = np.asarray([it * adj_size + offset + j for it in iterations for j in range(formal_size)], dtype=np.int64)
        destinations = np.asarray([start + it * stride + j for it in iterations for j in range(formal_size)], dtype=np.int64)
        if indices.size:
          pieces.append(scatter(gather(mapped, indices), destinations, arg.shape))
      vbar = _sum_exprs(pieces)
    ret.append((arg, vbar))
    offset += formal_size
  return ret


def vjp(outputs: Sequence[Expr], wrts: Sequence[Expr], cotangents: Sequence[Expr]) -> tuple[Expr, ...]:
  """Reverse-mode derivative: one adjoint per entry of ``wrts``, seeded by ``cotangents``.

  Each cotangent is shaped like its output. One sweep computes the derivative of a scalar with
  respect to every input at once, which is why gradients go through here.
  """
  if len(outputs) != len(cotangents):
    raise ValueError(f"expected {len(outputs)} cotangents, got {len(cotangents)}")
  adjoints: dict[int, Expr] = {}
  nodes = topo(outputs)
  expr_ids = {e.id for e in nodes}
  dep_memo: dict[tuple[int, int], bool] = {}

  def needed(expr: Expr) -> bool:
    return any(_depends_on(expr, wrt, dep_memo) for wrt in wrts)

  for out, cot in zip(outputs, cotangents, strict=True):
    if out.shape != cot.shape:
      raise ValueError(f"cotangent for output shape {out.shape} has shape {cot.shape}")
    adjoints[out.id] = cot if out.id not in adjoints else adjoints[out.id] + cot

  for expr in reversed(nodes):
    cot = adjoints.get(expr.id)
    if cot is None or expr.op in {ExprOp.INPUT, ExprOp.CONST} or not needed(expr):
      continue
    if expr.op == ExprOp.VMAP:
      for arg, arg_cot in _vmap_vjp(expr, cot, wrts, dep_memo):
        if arg.id in expr_ids:
          adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot
      continue
    active = tuple(i for i, arg in enumerate(expr.args) if needed(arg)) if expr.op == ExprOp.CALL else None
    for arg, arg_cot in zip(expr.args, _local_vjp(expr, cot, active), strict=True):
      if arg.id in expr_ids:
        adjoints[arg.id] = arg_cot if arg.id not in adjoints else adjoints[arg.id] + arg_cot

  return tuple(adjoints.get(wrt.id, zeros_like(wrt)) for wrt in wrts)


def _local_vjp(expr: Expr, cot: Expr, active: tuple[int, ...] | None = None) -> tuple[Expr, ...]:
  args = expr.args
  if expr.op == ExprOp.NEG:
    return (-cot,)
  if expr.op == ExprOp.ADD:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(cot, args[1].shape, expr.shape))
  if expr.op == ExprOp.SUB:
    return (_unbroadcast(cot, args[0].shape, expr.shape), _unbroadcast(-cot, args[1].shape, expr.shape))
  if expr.op == ExprOp.MUL:
    return (_unbroadcast(cot * args[1], args[0].shape, expr.shape), _unbroadcast(cot * args[0], args[1].shape, expr.shape))
  if expr.op == ExprOp.DIV:
    return (
      _unbroadcast(cot / args[1], args[0].shape, expr.shape),
      _unbroadcast(-((cot / args[1]) * expr), args[1].shape, expr.shape),
    )
  if expr.op == ExprOp.POW:
    return (
      _unbroadcast(cot * args[1] * (args[0] ** (args[1] - 1)), args[0].shape, expr.shape),
      _unbroadcast(cot * expr * args[0].log(), args[1].shape, expr.shape),
    )
  if expr.op == ExprOp.SIN:
    return (cot * args[0].cos(),)
  if expr.op == ExprOp.COS:
    return (-cot * args[0].sin(),)
  if expr.op == ExprOp.TAN:
    return (cot / (args[0].cos() ** 2),)
  if expr.op == ExprOp.ASIN:
    return (cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == ExprOp.ACOS:
    return (-cot / (1 - args[0] ** 2).sqrt(),)
  if expr.op == ExprOp.ATAN:
    return (cot / (1 + args[0] ** 2),)
  if expr.op == ExprOp.ATAN2:
    y, x = args
    denom = x * x + y * y
    return (_unbroadcast(cot * x / denom, y.shape, expr.shape), _unbroadcast(-cot * y / denom, x.shape, expr.shape))
  if expr.op == ExprOp.SINH:
    return (cot * args[0].cosh(),)
  if expr.op == ExprOp.COSH:
    return (cot * args[0].sinh(),)
  if expr.op == ExprOp.TANH:
    return (cot * (1 - expr * expr),)
  if expr.op == ExprOp.ERF:
    return (cot * (2 / np.sqrt(np.pi)) * (-(args[0] ** 2)).exp(),)
  if expr.op == ExprOp.EXP:
    return (cot * expr,)
  if expr.op == ExprOp.LOG:
    return (cot / args[0],)
  if expr.op == ExprOp.SQRT:
    return (cot / (2 * expr),)
  if expr.op == ExprOp.ABS:
    return (cot * args[0] / args[0].abs(),)
  if expr.op in {ExprOp.FLOOR, ExprOp.CEIL, ExprOp.MINIMUM, ExprOp.MAXIMUM}:
    raise NotImplementedError(f"VJP for nonsmooth op {expr.op!r} is not implemented")
  if expr.op == ExprOp.SUM:
    return (cot * _ones_like(args[0]),)
  if expr.op == ExprOp.RESHAPE:
    return (cot.reshape(args[0].shape),)
  if expr.op == ExprOp.TRANSPOSE:
    axes = expr.attrs["axes"]
    inv = tuple(int(np.argsort(axes)[i]) for i in range(len(axes)))
    return (cot.transpose(inv),)
  if expr.op == ExprOp.SLICE:
    indices = np.arange(args[0].size).reshape(args[0].shape)[expr.attrs["index"]]
    return (scatter(cot, indices, args[0].shape),)
  if expr.op == ExprOp.GATHER:
    return (_gather_vjp(cot, expr.attrs["indices"], args[0].shape),)
  if expr.op == ExprOp.SCATTER:
    return (gather(cot, expr.attrs["indices"]),)
  if expr.op == ExprOp.STACK:
    return _stack_vjp(cot, len(args), expr.attrs.get("axis", 0))
  if expr.op == ExprOp.CONCAT:
    return _concat_vjp(cot, args, expr.attrs.get("axis", 0))
  if expr.op == ExprOp.MATMUL:
    return _matmul_vjp(args[0], args[1], cot)
  if expr.op == ExprOp.CALL:
    callee = expr.attrs["callee"]
    output_idx = expr.attrs["output"]
    callee_out = callee.outputs[output_idx]
    # Differentiate the callee body against a fresh cotangent symbol, then graft the real ``cot``
    # in via the same substitution that maps formals to actuals. Passing ``cot`` directly into the
    # inner vjp would make it part of the substituted graph: if the caller reuses a callee formal
    # symbol (the usual construction pattern), occurrences of that symbol *inside the cotangent*
    # would be rewritten to this call's actuals, corrupting the adjoint.
    lam = Expr.sym(f"lam:{callee.output_names[output_idx]}", callee_out.shape)
    replacements = dict(zip((inp.id for inp in callee.inputs), args, strict=True))
    replacements[lam.id] = cot
    indices = tuple(range(len(args))) if active is None else active
    grads = vjp((callee_out,), tuple(callee.inputs[i] for i in indices), (lam,))
    adjoints = dict(zip(indices, grads, strict=True))
    return tuple(_substitute(adjoints[i], replacements) if i in adjoints else zeros_like(arg) for i, arg in enumerate(args))
  if expr.op == ExprOp.SOLVER_CALL:
    if cot.op == ExprOp.CONST and cot.value is not None and not np.any(cot.value):
      return tuple(zeros_like(arg) for arg in args)
    raise NotImplementedError("active derivative through SOLVER_CALL is not implemented")
  raise NotImplementedError(f"VJP for op {expr.op!r} is not implemented")


def _ones_like(expr: Expr) -> Expr:
  return Expr.const(np.ones(expr.shape, dtype=np.float64), lowering=expr.lowering)


def _sum_exprs(exprs: Iterable[Expr]) -> Expr:
  ret: Expr | None = None
  for expr in exprs:
    ret = expr if ret is None else ret + expr
  return as_expr(0.0) if ret is None else ret


def _unbroadcast(cot: Expr, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> Expr:
  if in_shape == out_shape:
    return cot
  if not in_shape:
    return cot.sum()
  source = np.arange(int(np.prod(in_shape, dtype=int))).reshape(in_shape)
  source = np.broadcast_to(source, out_shape).reshape(-1)
  vals = []
  for i in range(int(np.prod(in_shape, dtype=int))):
    vals.append(gather(cot, np.nonzero(source == i)[0]).sum())
  return stack(vals).reshape(in_shape)


def _gather_vjp(cot: Expr, indices: np.ndarray, shape: tuple[int, ...]) -> Expr:
  flat = indices.reshape(-1)
  vals = []
  for i in range(int(np.prod(shape, dtype=int))):
    positions = np.nonzero(flat == i)[0]
    vals.append(gather(cot, positions).sum() if positions.size else as_expr(0.0))
  return stack(vals).reshape(shape)


def _stack_vjp(cot: Expr, nargs: int, axis: int) -> tuple[Expr, ...]:
  ret: list[Expr] = []
  for i in range(nargs):
    index = tuple(i if dim == axis else slice(None) for dim in range(len(cot.shape)))
    ret.append(cot[index])
  return tuple(ret)


def _concat_vjp(cot: Expr, args: Sequence[Expr], axis: int) -> tuple[Expr, ...]:
  ret: list[Expr] = []
  start = 0
  for arg in args:
    index: list[Any] = [slice(None)] * len(cot.shape)
    index[axis] = slice(start, start + arg.shape[axis])
    ret.append(cot[tuple(index)])
    start += arg.shape[axis]
  return tuple(ret)


def _matmul_vjp(x: Expr, y: Expr, cot: Expr) -> tuple[Expr, Expr]:
  if len(x.shape) == 1 and len(y.shape) == 1:
    return cot * y, cot * x
  if len(x.shape) == 2 and len(y.shape) == 1:
    return cot.reshape((x.shape[0], 1)) @ y.reshape((1, x.shape[1])), x.T @ cot
  if len(x.shape) == 1 and len(y.shape) == 2:
    return y @ cot, x.reshape((x.shape[0], 1)) @ cot.reshape((1, y.shape[1]))
  if len(x.shape) == 2 and len(y.shape) == 2:
    return cot @ y.T, x.T @ cot
  raise NotImplementedError(f"matmul VJP for {x.shape} @ {y.shape} is not implemented")
