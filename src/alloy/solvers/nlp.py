"""``al.nlp(...)`` — build an opaque solver Function wrapping IPOPT.

NLP shape (per ``docs/roadmap.md``)::

    min   f(x, p)
    s.t.  h_eq(x, p) = 0
          l_ineq ≤ g_ineq(x, p) ≤ u_ineq
          x_lb  ≤ x ≤ x_ub

The IPOPT backend stacks ``[h_eq; g_ineq]`` into IPOPT's ``g(x)`` with bounds
``[0; l_ineq] ≤ g ≤ [0; u_ineq]``. Derivatives come from Alloy's factory:
sparse Jacobians for the constraints, sparse Lagrangian Hessian via the
``gamma`` aux. The Hessian is filtered to its lower triangle inside the
backend (IPOPT's convention).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..expr import Expr, as_expr, concat
from ..function import Function
from ..ops import Ops
from ..types import SparsityType
from ._ipopt import IPOPT_INF, solve_ipopt
from ._oracle import collect_free_inputs
from .solver_function import SolverDescriptor, SolverFunction, SolverStatus

_SUPPORTED_SOLVERS = {"ipopt"}


def _ensure_sym(name: str, value: Any) -> Expr:
  if not isinstance(value, Expr) or value.op != Ops.INPUT:
    raise TypeError(f"NLP input {name!r} must be a symbolic Expr.sym, got {type(value).__name__}")
  if value.name is None:
    raise ValueError(f"NLP input {name!r} must have a non-empty name")
  return value


def _sym_name(e: Expr) -> str:
  if e.name is None:
    raise ValueError(f"symbolic input id={e.id} has no name")
  return e.name


def _as_expr_optional(value: Any) -> Expr | None:
  if value is None:
    return None
  return as_expr(value)


def _normalize_params(p: Any) -> tuple[Expr, ...]:
  if p is None:
    return ()
  if isinstance(p, Expr):
    return (_ensure_sym("p", p),)
  if isinstance(p, Sequence):
    return tuple(_ensure_sym("p", e) for e in p)
  raise TypeError(f"p must be None, an Expr, or a sequence of Expr, got {type(p).__name__}")


def nlp(
  *,
  x: Any,
  f: Any,
  p: Any = None,
  h_eq: Any = None,
  g_ineq: Any = None,
  l_ineq: Any = None,
  u_ineq: Any = None,
  x_lb: Any = None,
  x_ub: Any = None,
  solver: str = "ipopt",
  name: str | None = None,
  options: dict[str, str | int | float] | None = None,
) -> SolverFunction:
  if solver not in _SUPPORTED_SOLVERS:
    raise ValueError(f"unsupported NLP solver {solver!r}; supported: {sorted(_SUPPORTED_SOLVERS)}")

  x_sym = _ensure_sym("x", x)
  if len(x_sym.shape) != 1:
    raise ValueError(f"x must be a rank-1 symbol, got shape {x_sym.shape}")
  n = x_sym.shape[0]

  f_expr = as_expr(f)
  if f_expr.size != 1:
    raise ValueError(f"f must be scalar, got shape {f_expr.shape}")
  if f_expr.shape != ():
    f_expr = f_expr.reshape(())

  h_e = _as_expr_optional(h_eq)
  g_e = _as_expr_optional(g_ineq)
  if h_e is not None and len(h_e.shape) != 1:
    raise ValueError(f"h_eq must be rank-1, got shape {h_e.shape}")
  if g_e is not None and len(g_e.shape) != 1:
    raise ValueError(f"g_ineq must be rank-1, got shape {g_e.shape}")
  n_h = h_e.shape[0] if h_e is not None else 0
  n_g = g_e.shape[0] if g_e is not None else 0

  if g_e is not None:
    if l_ineq is None and u_ineq is None:
      raise ValueError("g_ineq requires at least one of l_ineq / u_ineq")
    l_e = as_expr(l_ineq) if l_ineq is not None else as_expr(np.full(n_g, -IPOPT_INF))
    u_e = as_expr(u_ineq) if u_ineq is not None else as_expr(np.full(n_g, IPOPT_INF))
    if l_e.shape != (n_g,) or u_e.shape != (n_g,):
      raise ValueError(f"l_ineq/u_ineq must have shape ({n_g},), got {l_e.shape} / {u_e.shape}")
  else:
    l_e = u_e = None

  xl_e = as_expr(x_lb) if x_lb is not None else as_expr(np.full(n, -IPOPT_INF))
  xu_e = as_expr(x_ub) if x_ub is not None else as_expr(np.full(n, IPOPT_INF))
  if xl_e.shape != (n,) or xu_e.shape != (n,):
    raise ValueError(f"x_lb/x_ub must have shape ({n},), got {xl_e.shape} / {xu_e.shape}")

  x_name = _sym_name(x_sym)

  g_all: Expr | None
  if n_h and n_g:
    assert h_e is not None and g_e is not None
    g_all = concat([h_e, g_e], axis=0)
  elif n_h:
    g_all = h_e
  elif n_g:
    g_all = g_e
  else:
    g_all = None

  declared_params = _normalize_params(p)
  if declared_params:
    params = declared_params
  else:
    targets: list[Expr] = [f_expr]
    if g_all is not None:
      targets.append(g_all)
    targets.extend([xl_e, xu_e])
    if n_g and l_e is not None and u_e is not None:
      targets.extend([l_e, u_e])
    params = tuple(pe for pe in collect_free_inputs(targets) if pe.id != x_sym.id)

  param_names: tuple[str, ...] = tuple(_sym_name(pe) for pe in params)
  base_inputs = (x_sym, *params)
  base_input_names: tuple[str, ...] = (x_name, *param_names)
  if g_all is not None:
    base_outputs: tuple[Expr, ...] = (f_expr, g_all)
    base_output_names: tuple[str, ...] = ("f", "g")
  else:
    base_outputs = (f_expr,)
    base_output_names = ("f",)

  base_name = (name or "nlp") + "_base"
  base_fn = Function(base_name, base_inputs, base_outputs, list(base_input_names), list(base_output_names))

  bound_outs: list[Expr] = [xl_e, xu_e]
  bound_names = ["x_lb", "x_ub"]
  if g_all is not None and l_e is not None and u_e is not None:
    bound_outs.extend([l_e, u_e])
    bound_names.extend(["l_ineq", "u_ineq"])
  bound_fn = Function(
    (name or "nlp") + "_bounds",
    list(params),
    bound_outs,
    list(param_names),
    bound_names,
  )

  grad_fn = base_fn.factory(
    (name or "nlp") + "_grad",
    list(base_input_names),
    ["grad:f:" + x_name],
  )

  jac_fn: Function | None
  if g_all is not None:
    jac_fn = base_fn.factory(
      (name or "nlp") + "_jac",
      list(base_input_names),
      ["spjac:g:" + x_name],
    )
    jac_sparsity = jac_fn.output_sparsities[0]
    assert jac_sparsity is not None
  else:
    jac_fn = None
    jac_sparsity = SparsityType.empty((0, n))

  if g_all is not None:
    hess_fn = base_fn.factory(
      (name or "nlp") + "_hess",
      [x_name, "lam:f", "lam:g", *param_names],
      ["sphess:gamma:" + x_name + ":" + x_name],
      aux={"gamma": ["f", "g"]},
    )
  else:
    hess_fn = base_fn.factory(
      (name or "nlp") + "_hess",
      [x_name, "lam:f", *param_names],
      ["sphess:gamma:" + x_name + ":" + x_name],
      aux={"gamma": ["f"]},
    )
  hess_sparsity = hess_fn.output_sparsities[0]
  assert hess_sparsity is not None

  hess_rows_full = np.asarray(hess_sparsity.rows, dtype=np.int32)
  hess_cols_full = np.asarray(hess_sparsity.cols, dtype=np.int32)
  lower_mask = hess_rows_full >= hess_cols_full

  resolved_options: dict[str, str | int | float] = {"print_level": 0, "sb": "yes"}
  if options:
    resolved_options.update(options)

  input_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x0", (n,)),
    ("lam_eq0", (n_h,)),
    ("lam_ineq0", (n_g,)),
    *((pname, pe.shape) for pname, pe in zip(param_names, params, strict=True)),
  )
  output_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x", (n,)),
    ("f", ()),
    ("h_eq", (n_h,)),
    ("g_ineq", (n_g,)),
    ("lam_eq", (n_h,)),
    ("lam_ineq", (n_g,)),
    ("lam_box", (n,)),
  )

  descriptor = SolverDescriptor(
    name=name or "nlp_ipopt",
    backend="ipopt",
    n=n,
    n_eq=n_h,
    n_ineq=n_g,
    input_signature=input_signature,
    output_signature=output_signature,
    param_names=param_names,
    base=base_fn,
    grad=grad_fn,
    jac=jac_fn,
    hess=hess_fn,
    bounds=bound_fn,
    jac_sparsity=jac_sparsity,
    hess_sparsity=hess_sparsity,
    hess_lower_mask=tuple(bool(v) for v in lower_mask.tolist()),
    options=tuple(sorted(resolved_options.items())),
  )
  return SolverFunction(descriptor)


def _nlp_backend(descriptor: SolverDescriptor, inputs: Sequence[np.ndarray]) -> tuple[list[np.ndarray], SolverStatus]:
  """Pure-numeric IPOPT runner used by direct ``SolverFunction`` calls."""
  n, n_h, n_g = descriptor.n, descriptor.n_eq, descriptor.n_ineq
  m = n_h + n_g

  base_fn = descriptor.base
  grad_fn = descriptor.grad
  jac_fn = descriptor.jac
  hess_fn = descriptor.hess
  bound_fn = descriptor.bounds
  assert base_fn is not None and grad_fn is not None and hess_fn is not None and bound_fn is not None

  x0 = inputs[0]
  param_args = list(inputs[3:])

  bounds = dict(zip(bound_fn.output_names, bound_fn.eval_list(*param_args), strict=True))
  x_lb = bounds["x_lb"]
  x_ub = bounds["x_ub"]
  if m:
    l_in = bounds["l_ineq"] if "l_ineq" in bounds else np.full(n_g, -IPOPT_INF)
    u_in = bounds["u_ineq"] if "u_ineq" in bounds else np.full(n_g, IPOPT_INF)
  else:
    l_in = np.zeros(0)
    u_in = np.zeros(0)
  g_L = np.concatenate([np.zeros(n_h), l_in])
  g_U = np.concatenate([np.zeros(n_h), u_in])

  jac_sp = descriptor.jac_sparsity
  hess_sp = descriptor.hess_sparsity
  assert hess_sp is not None
  hess_rows_full = np.asarray(hess_sp.rows, dtype=np.int32)
  hess_cols_full = np.asarray(hess_sp.cols, dtype=np.int32)
  lower_mask = np.asarray(descriptor.hess_lower_mask, dtype=bool)
  hess_rows = hess_rows_full[lower_mask]
  hess_cols = hess_cols_full[lower_mask]
  jac_rows = np.asarray(jac_sp.rows, dtype=np.int32) if jac_sp is not None else np.zeros(0, dtype=np.int32)
  jac_cols = np.asarray(jac_sp.cols, dtype=np.int32) if jac_sp is not None else np.zeros(0, dtype=np.int32)

  def eval_f(x_val: np.ndarray) -> float:
    out = base_fn.eval_list(x_val, *param_args)
    return float(out[0])

  def eval_g(x_val: np.ndarray) -> np.ndarray:
    out = base_fn.eval_list(x_val, *param_args)
    return np.asarray(out[1], dtype=np.float64).reshape(-1)

  def eval_grad_f(x_val: np.ndarray) -> np.ndarray:
    out = grad_fn.eval_list(x_val, *param_args)
    return np.asarray(out[0], dtype=np.float64).reshape(-1)

  def eval_jac_g(x_val: np.ndarray) -> np.ndarray:
    assert jac_fn is not None
    out = jac_fn.eval_list(x_val, *param_args)
    return np.asarray(out[0], dtype=np.float64).reshape(-1)

  def eval_h(x_val: np.ndarray, obj_factor: float, lam_all: np.ndarray) -> np.ndarray:
    if m:
      out = hess_fn.eval_list(x_val, np.asarray(obj_factor, dtype=np.float64), lam_all, *param_args)
    else:
      out = hess_fn.eval_list(x_val, np.asarray(obj_factor, dtype=np.float64), *param_args)
    vals = np.asarray(out[0], dtype=np.float64).reshape(-1)
    return vals[lower_mask]

  sol = solve_ipopt(
    n=n,
    m=m,
    x0=np.asarray(x0, dtype=np.float64).reshape(-1),
    x_L=x_lb,
    x_U=x_ub,
    g_L=g_L,
    g_U=g_U,
    jac_rows=jac_rows,
    jac_cols=jac_cols,
    hess_rows=hess_rows,
    hess_cols=hess_cols,
    eval_f=eval_f,
    eval_grad_f=eval_grad_f,
    eval_g=eval_g if m else (lambda _x: np.zeros(0)),
    eval_jac_g=eval_jac_g if m else (lambda _x: np.zeros(0)),
    eval_h=eval_h,
    options={k: v for k, v in descriptor.options},
  )

  h_out = sol.g[:n_h] if n_h else np.zeros(0)
  g_out = sol.g[n_h:] if n_g else np.zeros(0)
  lam_h = sol.mult_g[:n_h] if n_h else np.zeros(0)
  lam_g = sol.mult_g[n_h:] if n_g else np.zeros(0)
  lam_box = sol.mult_x_U - sol.mult_x_L
  outs = [
    sol.x,
    np.asarray(sol.obj, dtype=np.float64),
    h_out,
    g_out,
    lam_h,
    lam_g,
    lam_box,
  ]
  return outs, SolverStatus(code=sol.status, name=sol.status_name)
