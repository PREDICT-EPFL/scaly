"""``al.qp(...)`` — build an opaque solver Function wrapping PIQP.

The QP shape (see ``docs/guide/solvers.md``) is::

    min   0.5 xᵀ P x + cᵀ x
    s.t.  A_eq x = b_eq
          l_ineq ≤ G_ineq x ≤ u_ineq
          x_lb  ≤ x ≤ x_ub

Each symbolic input may be an Alloy ``Expr`` over a set of free parameters ``p``,
or a plain numpy/python value (which becomes a constant). At call time the
parameter values are passed through to evaluate the QP data, and the QP is
solved through PIQP's dense interface — or its sparse interface with
``sparse=True``, where the structural CSC patterns of ``P`` (upper triangle),
``A_eq``, and ``G_ineq`` are computed here at build time and baked into the
generated wrapper (generated C backend only).

QP inputs are (in order): ``x0``, ``lam_eq0``, ``lam_ineq0`` plus every free
parameter in deterministic name/id order. Initial dual values are accepted for
API symmetry but PIQP does not yet consume warm starts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..ir.expr import Expr, as_expr
from ..function import Function
from ..ir.types import SparsityType
from ._oracle import collect_free_inputs
from .registry import require_backend
from .solver_function import SolverDescriptor, SolverFunction

PIQP_INF = 1e30


def _as_expr_optional(value: Any) -> Expr | None:
  if value is None:
    return None
  return as_expr(value)


def _check_shape(name: str, expr: Expr | None, expected: tuple[int, ...]) -> None:
  if expr is None:
    return
  if expr.shape != expected:
    raise ValueError(f"QP input {name!r} has shape {expr.shape}, expected {expected}")


def _qp_matrix_sparsity(mat: Expr, params: Sequence[Expr], probe: np.ndarray, *, triu: bool = False) -> SparsityType:
  """Structural pattern of a matrix-valued Expr, in CSC order.

  An entry is structurally nonzero if it depends on any parameter (dependency
  mask — kept regardless of its probed value) or if its constant value is
  nonzero (``probe`` is the matrix evaluated at an arbitrary parameter draw,
  which is exact for constant entries). ``triu`` keeps only the upper
  triangle (PIQP's convention for P). A structurally zero matrix keeps a
  single ``(0, 0)`` entry so the generated CSC handle stays valid — the
  gathered value is the structural zero itself.

  Matrices computed from a nested ``SOLVER_CALL`` are rejected in ``qp()``
  before the probe runs: ``_jac_mask`` treats solver outputs as opaque zeros,
  so their entries would be classified solely by the probed value — silently
  wrong whenever the inner solve is zero at the probe but nonzero at runtime.
  """
  from ..ad.sparsity import _jac_mask

  nrow, ncol = mat.shape
  vec = mat.vec()
  keep = np.asarray(probe, dtype=np.float64).reshape(-1) != 0.0
  for param in params:
    keep |= np.asarray(_jac_mask(vec, param, {}).sum(axis=1)).reshape(-1) != 0
  rows, cols = np.divmod(np.flatnonzero(keep), ncol)
  if triu:
    upper = rows <= cols
    rows, cols = rows[upper], cols[upper]
  if rows.size == 0:
    rows, cols = np.array([0]), np.array([0])
  order = np.lexsort((rows, cols))  # CSC order (by column, then row): the compact value buffer needs no runtime permutation
  return SparsityType((nrow, ncol), tuple(int(r) for r in rows[order]), tuple(int(c) for c in cols[order]))


def _gathered(mat: Expr, sp: SparsityType) -> Expr:
  """Compact CSC-ordered value vector of ``mat`` (row-major flat gather)."""
  flat = np.asarray(sp.rows, dtype=np.int64) * mat.shape[1] + np.asarray(sp.cols, dtype=np.int64)
  return mat.vec().gather(flat)


def _reaches_solver_call(exprs: Sequence[Expr]) -> bool:
  """True if any expr reaches a ``SOLVER_CALL``, recursing through CALL/VMAP callees."""
  from ..ir.expr import topo
  from ..ir.expr import ExprOp

  seen: set[int] = set()

  def visit(targets: Sequence[Expr]) -> bool:
    for node in topo(list(targets)):
      if node.op == ExprOp.SOLVER_CALL:
        return True
      if node.op in {ExprOp.CALL, ExprOp.VMAP}:
        callee = node.attrs["callee"]
        if id(callee) not in seen:
          seen.add(id(callee))
          if visit(callee.outputs):
            return True
    return False

  return visit(exprs)


def qp(
  *,
  P: Any,
  c: Any,
  A_eq: Any = None,
  b_eq: Any = None,
  G_ineq: Any = None,
  l_ineq: Any = None,
  u_ineq: Any = None,
  x_lb: Any = None,
  x_ub: Any = None,
  solver: str = "piqp",
  name: str | None = None,
  options: dict[str, float | int] | None = None,
  sparse: bool = False,
) -> SolverFunction:
  """Build a quadratic-program solver as a callable ``Function``.

  Solves ``min 0.5 x' P x + c' x`` subject to ``A_eq x = b_eq``,
  ``l_ineq <= G_ineq x <= u_ineq`` and ``x_lb <= x <= x_ub``.

  Every argument may be an alloy ``Expr`` over free parameters — which is what makes the solver
  reusable across states — a NumPy array or scalar, or ``None`` for the optional blocks. The
  returned ``SolverFunction`` takes ``x0``, ``lam_eq0``, ``lam_ineq0`` and then every free
  parameter found, and returns ``x``, ``cost``, ``lam_eq``, ``lam_ineq`` and ``lam_box``.

  Because it is a real ``Function``, ``solver.call([...])`` nests it inside a larger graph and the
  whole thing compiles to one shared library. See ``docs/guide/solvers.md``.

  Args:
    solver: the backend plugin to use; ``piqp`` today.
    options: backend settings, passed through to ``piqp_settings`` field names.
    sparse: route through PIQP's sparse interface, baking the CSC patterns at build time.
  """
  require_backend(solver, "qp")

  P_e = as_expr(P)
  c_e = as_expr(c)
  if len(P_e.shape) != 2 or P_e.shape[0] != P_e.shape[1]:
    raise ValueError(f"P must be square 2D, got shape {P_e.shape}")
  n = P_e.shape[0]
  if c_e.shape != (n,):
    raise ValueError(f"c must have shape ({n},), got {c_e.shape}")

  A_e = _as_expr_optional(A_eq)
  b_e = _as_expr_optional(b_eq)
  G_e = _as_expr_optional(G_ineq)
  l_e = _as_expr_optional(l_ineq)
  u_e = _as_expr_optional(u_ineq)
  xl_e = _as_expr_optional(x_lb)
  xu_e = _as_expr_optional(x_ub)

  if (A_e is None) != (b_e is None):
    raise ValueError("A_eq and b_eq must be provided together")
  if (G_e is None) and (l_e is not None or u_e is not None):
    raise ValueError("G_ineq is required when l_ineq/u_ineq are provided")

  p_dim = 0 if A_e is None else A_e.shape[0]
  m_dim = 0 if G_e is None else G_e.shape[0]

  _check_shape("A_eq", A_e, (p_dim, n))
  _check_shape("b_eq", b_e, (p_dim,))
  _check_shape("G_ineq", G_e, (m_dim, n))
  _check_shape("l_ineq", l_e, (m_dim,))
  _check_shape("u_ineq", u_e, (m_dim,))
  _check_shape("x_lb", xl_e, (n,))
  _check_shape("x_ub", xu_e, (n,))

  if l_e is None and m_dim:
    l_e = as_expr(np.full(m_dim, -PIQP_INF))
  if u_e is None and m_dim:
    u_e = as_expr(np.full(m_dim, PIQP_INF))
  if xl_e is None:
    xl_e = as_expr(np.full(n, -PIQP_INF))
  if xu_e is None:
    xu_e = as_expr(np.full(n, PIQP_INF))

  assert xl_e is not None and xu_e is not None
  dense_targets: list[Expr] = [P_e.vec(), c_e]
  if p_dim:
    assert A_e is not None and b_e is not None
    dense_targets.extend([A_e.vec(), b_e])
  if m_dim:
    assert G_e is not None and l_e is not None and u_e is not None
    dense_targets.extend([G_e.vec(), l_e, u_e])
  dense_targets.extend([xl_e, xu_e])
  # Params are collected from the dense expressions so the call signature does
  # not depend on the sparse pattern.
  params = collect_free_inputs(dense_targets)
  param_names: tuple[str, ...] = tuple(p.name or f"p{i}" for i, p in enumerate(params))

  P_sp = A_sp = G_sp = None
  if sparse:
    # Structural patterns, baked at codegen time. Constant entries are probed
    # exactly at one arbitrary parameter draw; parameter-dependent entries are
    # kept by the dependency mask regardless of the draw, so the pattern is
    # deterministic.
    mats: list[Expr] = [P_e, *([A_e] if p_dim else []), *([G_e] if m_dim else [])]  # ty: ignore[invalid-assignment]
    if _reaches_solver_call(mats):
      raise NotImplementedError(
        "sparse=True cannot derive the structural pattern of QP data computed from a nested solver output; use the dense interface"
      )
    probe_fn = Function(
      (name or "qp") + "_pattern_probe", list(params), [mat.vec() for mat in mats], list(param_names), [f"m{i}" for i in range(len(mats))]
    )
    rng = np.random.default_rng(0)
    probes = probe_fn.eval_list(*[rng.standard_normal(p.shape) for p in params])
    P_sp = _qp_matrix_sparsity(P_e, params, probes[0], triu=True)
    if p_dim:
      assert A_e is not None
      A_sp = _qp_matrix_sparsity(A_e, params, probes[1])
    if m_dim:
      assert G_e is not None
      G_sp = _qp_matrix_sparsity(G_e, params, probes[-1])

  oracle_outs: list[Expr] = [_gathered(P_e, P_sp) if P_sp is not None else P_e.vec(), c_e]
  oracle_names = ["P", "c"]
  if p_dim:
    assert A_e is not None and b_e is not None
    oracle_outs.extend([_gathered(A_e, A_sp) if A_sp is not None else A_e.vec(), b_e])
    oracle_names.extend(["A_eq", "b_eq"])
  if m_dim:
    assert G_e is not None and l_e is not None and u_e is not None
    oracle_outs.extend([_gathered(G_e, G_sp) if G_sp is not None else G_e.vec(), l_e, u_e])
    oracle_names.extend(["G_ineq", "l_ineq", "u_ineq"])
  oracle_outs.extend([xl_e, xu_e])
  oracle_names.extend(["x_lb", "x_ub"])

  oracle_name = (name or "qp") + "_oracle"
  oracle = Function(
    oracle_name,
    list(params),
    oracle_outs,
    list(param_names),
    oracle_names,
  )

  input_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x0", (n,)),
    ("lam_eq0", (p_dim,)),
    ("lam_ineq0", (m_dim,)),
    *((pname, pe.shape) for pname, pe in zip(param_names, params, strict=True)),
  )
  output_signature: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("x", (n,)),
    ("cost", ()),
    ("lam_eq", (p_dim,)),
    ("lam_ineq", (m_dim,)),
    ("lam_box", (n,)),
  )

  resolved_options = {"verbose": 0, **(options or {})}
  descriptor = SolverDescriptor(
    name=name or f"qp_{solver}",
    backend=solver,
    n=n,
    n_eq=p_dim,
    n_ineq=m_dim,
    input_signature=input_signature,
    output_signature=output_signature,
    param_names=param_names,
    oracle=oracle,
    options=tuple(sorted(resolved_options.items())),
    oracle_output_names=tuple(oracle_names),
    sparse=sparse,
    P_sparsity=P_sp,
    A_sparsity=A_sp,
    G_sparsity=G_sp,
  )
  return SolverFunction(descriptor)
