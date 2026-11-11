"""Typed derivative requests consumed by Function.factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..ad.derivatives import gradient, hessian, jacobian
from ..ad.forward import jvp
from ..ad.reverse import vjp
from ..ad.sparse import Triangle, _sparse_hessian, _sparse_jacobian, _validate_triangle
from ..ir.expr import Expr
from ..ir.layout import Layout
from .concrete import DerivSpec


@dataclass(frozen=True, slots=True)
class Jac(DerivSpec):
  """Dense Jacobian ``d of / d wrt``."""

  kind = "jac"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    return jacobian(self._out(outputs, self.of), self._in(inputs, self.wrt)), None


@dataclass(frozen=True, slots=True)
class Grad(DerivSpec):
  """Gradient of a scalar output."""

  kind = "grad"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    return gradient(self._out(outputs, self.of), self._in(inputs, self.wrt)), None


@dataclass(frozen=True, slots=True)
class Fwd(DerivSpec):
  """Seeded forward mode: ``J(of, wrt) @ fwd:wrt``."""

  kind = "fwd"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    return jvp(self._out(outputs, self.of), self._in(inputs, self.wrt), self._in(inputs, f"fwd:{self.wrt}")), None


@dataclass(frozen=True, slots=True)
class Adj(DerivSpec):
  """Seeded reverse mode: ``J(of, wrt).T @ lam:of``."""

  kind = "adj"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    return vjp((self._out(outputs, self.of),), (self._in(inputs, self.wrt),), (self._in(inputs, f"lam:{self.of}"),))[0], None


@dataclass(frozen=True, slots=True)
class SpJac(DerivSpec):
  """Sparse Jacobian, stored in ``layout``."""

  kind = "spjac"
  layout: Layout = field(default="csc", kw_only=True)

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    sj = _sparse_jacobian(self._out(outputs, self.of), self._in(inputs, self.wrt))
    return sj.leaf(), sj.coloring_width


@dataclass(frozen=True, slots=True)
class Hess(DerivSpec):
  """Dense Hessian of a scalar output."""

  kind = "hess"

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}_{self.wrt}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    y = self._out(outputs, self.of)
    x0 = self._in(inputs, self.wrt)
    return hessian(y, x0), None


@dataclass(frozen=True, slots=True)
class SpHess(DerivSpec):
  """Sparse Hessian of a scalar output, one ``triangle`` of it stored in ``layout``."""

  kind = "sphess"
  triangle: Triangle = field(default="full", kw_only=True)
  layout: Layout = field(default="csc", kw_only=True)

  def __post_init__(self) -> None:
    _validate_triangle(self.triangle)

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}_{self.wrt}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, int | None]:
    y = self._out(outputs, self.of)
    x0 = self._in(inputs, self.wrt)
    sh = _sparse_hessian(y, x0, triangle=self.triangle)
    return sh.leaf(), sh.coloring_width


__all__ = ["Jac", "Grad", "Hess", "SpJac", "SpHess", "Fwd", "Adj", "DerivSpec"]
