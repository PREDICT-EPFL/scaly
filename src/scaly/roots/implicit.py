"""``custom_root``: the identity on a solution found by any iteration, whose derivatives are the implicit function theorem's at ``F(z, p) = 0``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from ..ad.derivatives import jacobian as dense_jacobian
from ..ad.forward import jvp
from ..ad.reverse import vjp
from ..function.model import ConcreteFunction
from ..function.sugar import custom_derivative
from ..ir.expr import Expr, substitute
from ..linalg.dense import solve

type Residual = Callable[[Expr, Sequence[Expr]], Expr]
"""``(z, params) -> F``: a residual at the unknowns ``z``, every other quantity read from ``params``."""


class Linear(Protocol):
  """How to solve with a matrix at a point, a Newton step's or the implicit derivative's:
  ``factor(z, params)`` gives the factors at ``z`` and ``solve(factors, r)`` applies the matrix's
  inverse to ``r``. Simplified Newton with a tolerance passes the factors into its loop, so they must
  then be a list of expressions."""

  def factor(self, z: Expr, params: Sequence[Expr]) -> Any: ...

  def solve(self, factors: Any, r: Expr) -> Expr: ...


def jacobian_at(residual: Residual, z: Expr, params: Sequence[Expr], name: str = "z") -> Expr:
  """``F_z`` at ``z``, which may be any expression: the dense Jacobian at a free vector, put back."""
  free = Expr.sym(f"{name}free", z.shape)
  return substitute(dense_jacobian(residual(free, params), free), {free: z})


def custom_root(
  residual: Residual,
  z: Expr,
  params: Sequence[Expr],
  *,
  name: str,
  names: Sequence[str] | None = None,
  z_name: str = "z",
  jacobian: Residual | None = None,
  linear: Linear | None = None,
) -> ConcreteFunction[Any, Any, Any, Any]:
  """``(params..., zstar) -> z`` (one group, as ``from_exprs`` makes it), the identity on ``zstar``,
  with the derivatives of the root ``z(p)`` of ``residual(z, params) = 0`` at ``zstar``: ``dz = -F_z^{-1} F_p dp`` forward and one transposed
  solve in reverse. Call it on the solution whatever iteration found it, and no derivative ever goes
  through that iteration: the rules never read ``zstar``'s tangent and give it a zero cotangent.

  ``z`` and ``params`` give the shapes and dtypes (``z`` a vector, as many residuals as its
  entries); ``names`` labels the parameters (``p0``, ``p1``, ... by default) and ``z_name`` the
  solution. ``jacobian(z, params)`` gives ``F_z`` when the dense Jacobian of ``residual`` is not the
  cheapest way to it, and every solve is then ``solve(..., assume="gen")``, so ``F_z`` need only be
  nonsingular at the root. ``linear`` solves with ``F_z`` instead (a Cholesky or sparse ``L D L^T``
  factorization, say); the reverse rule solves with it too, so ``F_z`` must then be symmetric.

  Two levels, as ``SparseLDL.solve``: the level-2 rules find the root through the level-1 Function,
  so a second derivative applies the level-1 rules and is exact; only a third would reach
  ``zstar``'s iteration."""
  if len(z.shape) != 1:
    raise ValueError(f"custom_root needs a vector of unknowns, got shape {z.shape}")
  m = z.shape[0]
  labels_in = list(names) if names is not None else [f"p{i}" for i in range(len(params))]
  if len(labels_in) != len(params):
    raise ValueError(f"custom_root got {len(labels_in)} names for {len(params)} parameters")
  kinds = [(p.shape, p.type.dtype) for p in params]
  star = f"{z_name}star"

  def matrix(zz: Expr, held: Sequence[Expr]) -> Expr:
    return jacobian(zz, held) if jacobian is not None else jacobian_at(residual, zz, held, z_name)

  def symbols(prefix: str = "") -> list[Expr]:
    return [Expr.sym(f"{prefix}{nm}", shape, dtype=dtype) for nm, (shape, dtype) in zip(labels_in, kinds, strict=True)]

  held, zstar = symbols(), Expr.sym(star, (m,))
  base = ConcreteFunction.from_exprs(name, [*held, zstar], [zstar + 0.0], [*labels_in, star], [z_name])

  def jvp_rule(inner: ConcreteFunction[Any, Any, Any, Any], level: int) -> ConcreteFunction[Any, Any, Any, Any]:
    held, zstar, tangents, dzstar = symbols(), Expr.sym(star, (m,)), symbols("d"), Expr.sym(f"d{star}", (m,))
    k = inner.symbolic_call(tuple([*held, zstar]) if held else zstar)
    k = k[0] if isinstance(k, tuple) else k
    # F's partial derivatives hold z fixed: differentiate at a free z, then put the root back. At
    # level 2, k depends on the inputs through the level-1 rules, and differentiating through it
    # would give the total derivative of F(z(p), p), which is zero.
    free = Expr.sym(f"{z_name}free", (m,))
    g = residual(free, held)
    push = substitute(sum((jvp(g, p, t) for p, t in zip(held, tangents, strict=True)), Expr.const(np.zeros(m))), {free: k})
    dk = -solve(matrix(k, held), push, assume="gen") if linear is None else -linear.solve(linear.factor(k, held), push)
    labels = [*labels_in, star, *(f"d{nm}" for nm in labels_in), f"d{star}"]
    return ConcreteFunction.from_exprs(f"{name}_jvp{level}", [*held, zstar, *tangents, dzstar], [dk], labels, [f"d{z_name}"])

  # The reverse rule reads the root from the Function's output, whose derivative is the Function's
  # own, so one rule serves both levels.
  held, zstar, k, kbar = symbols(), Expr.sym(star, (m,)), Expr.sym(f"{z_name}o", (m,)), Expr.sym(f"{z_name}bar", (m,))
  lam = solve(matrix(k, held).T, kbar, assume="gen") if linear is None else linear.solve(linear.factor(k, held), kbar)
  grads = [-g for g in vjp([residual(k, held)], held, [lam])]
  labels, outs = [*labels_in, star, f"{z_name}o", f"{z_name}bar"], [*grads, Expr.const(np.zeros(m))]
  vjp_rule = ConcreteFunction.from_exprs(f"{name}_vjp", [*held, zstar, k, kbar], outs, labels, [f"{nm}bar" for nm in (*labels_in, star)])
  inner = base
  for level in (1, 2):
    inner = custom_derivative(base, jvp=jvp_rule(inner, level), vjp=vjp_rule)
  return inner


__all__ = ["Linear", "Residual", "custom_root", "jacobian_at"]
