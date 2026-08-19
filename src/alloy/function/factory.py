"""The derivative kinds ``Function.factory`` understands, and the AD each dispatches to.

One frozen dataclass per kind, constructed as ``al.jac("eq", "z")``; ``build`` turns the request
into the derived expression plus, for the compact kinds, its sparsity. The derived output is named
``{kind}_{of}_{wrt}``, which is what the generated C symbols are keyed on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..ad.derivatives import gradient, hessian, jacobian
from ..ad.forward import jvp
from ..ad.reverse import vjp
from ..ad.sparse import sparse_hessian, sparse_jacobian
from ..ir.expr import Expr
from ..ir.types import SparsityType
from ..passes.expr import simplify
from .model import DerivSpec


@dataclass(frozen=True, slots=True)
class jac(DerivSpec):
  """Dense Jacobian ``d of / d wrt``."""

  kind = "jac"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    return jacobian(self._out(outputs, self.of), self._in(inputs, self.wrt)), None


@dataclass(frozen=True, slots=True)
class grad(DerivSpec):
  """Gradient of a scalar output."""

  kind = "grad"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    return gradient(self._out(outputs, self.of), self._in(inputs, self.wrt)), None


@dataclass(frozen=True, slots=True)
class fwd(DerivSpec):
  """Seeded forward mode: ``J(of, wrt) @ fwd:wrt``."""

  kind = "fwd"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    return jvp(self._out(outputs, self.of), self._in(inputs, self.wrt), self._in(inputs, f"fwd:{self.wrt}")), None


@dataclass(frozen=True, slots=True)
class adj(DerivSpec):
  """Seeded reverse mode: ``J(of, wrt).T @ lam:of``."""

  kind = "adj"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    return vjp((self._out(outputs, self.of),), (self._in(inputs, self.wrt),), (self._in(inputs, f"lam:{self.of}"),))[0], None


@dataclass(frozen=True, slots=True)
class spjac(DerivSpec):
  """Compact nonzero Jacobian values, with the sparsity that indexes them."""

  kind = "spjac"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    sj = sparse_jacobian(self._out(outputs, self.of), self._in(inputs, self.wrt))
    return sj.values, sj.sparsity


@dataclass(frozen=True, slots=True)
class hess(DerivSpec):
  """Dense Hessian of a scalar output; ``wrt2`` defaults to ``wrt``."""

  kind = "hess"
  wrt2: str | None = None

  @property
  def _second(self) -> str:
    return self.wrt if self.wrt2 is None else self.wrt2

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}_{self._second}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    y = self._out(outputs, self.of)
    x0 = self._in(inputs, self.wrt)
    x1 = self._in(inputs, self._second)
    if self.wrt == self._second:
      return hessian(y, x0), None
    return jacobian(gradient(y, x0).reshape((x0.size,)), x1), None


@dataclass(frozen=True, slots=True)
class sphess(DerivSpec):
  """Compact nonzero Hessian values, with the sparsity that indexes them; ``wrt2`` defaults to ``wrt``."""

  kind = "sphess"
  wrt2: str | None = None

  @property
  def _second(self) -> str:
    return self.wrt if self.wrt2 is None else self.wrt2

  @property
  def output_name(self) -> str:
    return f"{self.kind}_{self.of}_{self.wrt}_{self._second}"

  def build(self, inputs: Mapping[str, Expr], outputs: Mapping[str, Expr]) -> tuple[Expr, SparsityType | None]:
    y = self._out(outputs, self.of)
    x0 = self._in(inputs, self.wrt)
    x1 = self._in(inputs, self._second)
    if self.wrt == self._second:
      sh = sparse_hessian(y, x0)
    else:
      sh = sparse_jacobian(simplify(gradient(y, x0).reshape((x0.size,))), x1)
    return sh.values, sh.sparsity
