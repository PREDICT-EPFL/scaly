"""The primal, dual and slack variables of PIQP 0.7.0's ``piqp/variables.hpp``, as Scaly values."""

from __future__ import annotations

from typing import NamedTuple

import scaly as sc

from .utils import zeros


class Variables(NamedTuple):
  """One value per variable of the QP, its constraints and their slacks.

  PIQP stores the box duals and slacks of the finite bounds first. Here they have one entry per
  variable, zero where the bound is infinite, so that one mask selects them.
  """

  x: sc.Expr
  y: sc.Expr
  z_l: sc.Expr
  z_u: sc.Expr
  z_bl: sc.Expr
  z_bu: sc.Expr
  
  s_l: sc.Expr
  s_u: sc.Expr
  s_bl: sc.Expr
  s_bu: sc.Expr

  @classmethod
  def zeros(cls, n: int, p: int, m: int) -> Variables:
    return cls(zeros(n), zeros(p), zeros(m), zeros(m), zeros(n), zeros(n), zeros(m), zeros(m), zeros(n), zeros(n))

  def named(self, prefix: str) -> dict[str, sc.Expr]:
    return {f"{prefix}{field}": value for field, value in zip(self._fields, self, strict=True)}

  @classmethod
  def from_named(cls, values: dict[str, sc.Expr], prefix: str) -> Variables:
    return cls(*(values[f"{prefix}{field}"] for field in cls._fields))
