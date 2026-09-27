"""Butcher tableaus: the named Runge-Kutta families and the order conditions that check them."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache

import numpy as np

__all__ = ["TABLEAUS", "Tableau", "order_conditions", "tableau"]


@dataclass(frozen=True, eq=False)
class Tableau:
  """A Runge-Kutta method ``(A, b, c)`` with ``s`` stages: ``k_i = f(x + h sum_j a_ij k_j)`` at time
  ``t + c_i h``, and ``x_next = x + h sum_i b_i k_i``.

  Attributes:
    a: the ``(s, s)`` stage coefficients; strictly lower triangular for an explicit method.
    b: the ``(s,)`` weights of the step.
    c: the ``(s,)`` nodes, the row sums of ``a``.
    order: the classical order of the step, checked against the order conditions at construction.
    name: how the method is spelled in ``tableau``, and in the names of the Functions built from it.
    b_err: for an embedded pair, the weights of the lower-order solution, whose difference from the
      step estimates its local error; ``None`` otherwise.
  """

  a: np.ndarray
  b: np.ndarray
  c: np.ndarray
  order: int
  name: str = "custom"
  b_err: np.ndarray | None = field(default=None)

  def __post_init__(self) -> None:
    a, b, c = (np.array(v, dtype=np.float64) for v in (self.a, self.b, self.c))
    s = b.size
    if a.shape != (s, s) or b.shape != (s,) or c.shape != (s,) or s == 0:
      raise ValueError(f"tableau {self.name}: a must be (s, s), b and c (s,), got {a.shape}, {b.shape}, {c.shape}")
    if not np.allclose(a.sum(axis=1), c, rtol=0, atol=1e-13):
      raise ValueError(f"tableau {self.name}: the nodes c must be the row sums of a")
    reached = order_conditions(a, b, self.order)
    if reached < self.order:
      raise ValueError(f"tableau {self.name}: declared order {self.order}, but the order conditions hold only to order {reached}")
    object.__setattr__(self, "a", a)
    object.__setattr__(self, "b", b)
    object.__setattr__(self, "c", c)
    if self.b_err is not None:
      b_err = np.array(self.b_err, dtype=np.float64)
      if b_err.shape != (s,):
        raise ValueError(f"tableau {self.name}: b_err must be ({s},), got {b_err.shape}")
      object.__setattr__(self, "b_err", b_err)

  @property
  def stages(self) -> int:
    """The number of stages ``s``."""
    return self.b.size

  @property
  def explicit(self) -> bool:
    """Whether every stage reads only earlier ones: ``a`` strictly lower triangular."""
    return not np.triu(self.a).any()

  def __repr__(self) -> str:
    return f"Tableau({self.name!r}, stages={self.stages}, order={self.order})"


# A rooted tree is the sorted tuple of its root's subtrees; the leaf is ().
type _Tree = tuple[_Tree, ...]


@cache
def _trees(order: int) -> tuple[_Tree, ...]:
  """Every rooted tree with ``order`` vertices, each once."""
  if order == 1:
    return ((),)
  found: set[_Tree] = set()

  def forests(budget: int, largest: int) -> list[tuple[_Tree, ...]]:
    """Multisets of trees with ``budget`` vertices in total, none larger than ``largest``, as sorted tuples."""
    if budget == 0:
      return [()]
    out = []
    for size in range(min(budget, largest), 0, -1):
      for sub in _trees(size):
        for rest in forests(budget - size, size):
          out.append(tuple(sorted((sub, *rest))))
    return out

  for children in forests(order - 1, order - 1):
    found.add(children)
  return tuple(sorted(found))


def _density(tree: _Tree) -> int:
  """``gamma(t)``: the tree's order times its subtrees' densities."""
  return _size(tree) * int(np.prod([_density(sub) for sub in tree], dtype=np.int64))


def _size(tree: _Tree) -> int:
  return 1 + sum(_size(sub) for sub in tree)


def order_conditions(a: np.ndarray, b: np.ndarray, up_to: int, *, tol: float = 1e-12) -> int:
  """The largest order ``p <= up_to`` whose conditions ``b . phi(t) = 1 / gamma(t)`` hold for every
  rooted tree ``t`` with at most ``p`` vertices (Butcher's theory of B-series)."""
  a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
  memo: dict[_Tree, np.ndarray] = {}

  def phi(tree: _Tree) -> np.ndarray:
    """The stage vector of the tree's elementary weight: the product over subtrees of ``A phi(sub)``."""
    if tree not in memo:
      memo[tree] = np.prod([a @ phi(sub) for sub in tree], axis=0) if tree else np.ones(b.size)
    return memo[tree]

  for p in range(1, up_to + 1):
    for tree in _trees(p):
      if abs(b @ phi(tree) - 1.0 / _density(tree)) > tol * max(1.0, 1.0 / _density(tree)):
        return p - 1
  return up_to


def _explicit(name: str, order: int, rows: list[list[Fraction | int]], b: list[Fraction | int], b_err: list[Fraction | int] | None = None) -> Tableau:
  """An explicit tableau from the rows below the diagonal: row ``i`` holds ``a_i0 .. a_i,i-1``."""
  s = len(b)
  a = np.zeros((s, s))
  for i, row in enumerate(rows, start=1):
    a[i, : len(row)] = [float(v) for v in row]
  return Tableau(a, np.array([float(v) for v in b]), a.sum(axis=1), order, name, None if b_err is None else np.array([float(v) for v in b_err]))


F = Fraction
TABLEAUS: dict[str, Tableau] = {
  t.name: t
  for t in (
    _explicit("euler", 1, [], [1]),
    _explicit("heun", 2, [[1]], [F(1, 2), F(1, 2)]),
    _explicit("midpoint", 2, [[F(1, 2)]], [0, 1]),
    _explicit("ralston", 2, [[F(2, 3)]], [F(1, 4), F(3, 4)]),
    _explicit("rk3", 3, [[F(1, 2)], [-1, 2]], [F(1, 6), F(2, 3), F(1, 6)]),
    _explicit("ssprk3", 3, [[1], [F(1, 4), F(1, 4)]], [F(1, 6), F(1, 6), F(2, 3)]),
    _explicit("rk4", 4, [[F(1, 2)], [0, F(1, 2)], [0, 0, 1]], [F(1, 6), F(1, 3), F(1, 3), F(1, 6)]),
    _explicit("rk38", 4, [[F(1, 3)], [F(-1, 3), 1], [1, -1, 1]], [F(1, 8), F(3, 8), F(3, 8), F(1, 8)]),
    # Bogacki-Shampine 3(2): the fourth stage is the next step's first (first same as last).
    _explicit(
      "bs32",
      3,
      [[F(1, 2)], [0, F(3, 4)], [F(2, 9), F(1, 3), F(4, 9)]],
      [F(2, 9), F(1, 3), F(4, 9), 0],
      [F(7, 24), F(1, 4), F(1, 3), F(1, 8)],
    ),
    # Dormand-Prince 5(4), the pair behind ode45 and solve_ivp's RK45; first same as last.
    _explicit(
      "dopri5",
      5,
      [
        [F(1, 5)],
        [F(3, 40), F(9, 40)],
        [F(44, 45), F(-56, 15), F(32, 9)],
        [F(19372, 6561), F(-25360, 2187), F(64448, 6561), F(-212, 729)],
        [F(9017, 3168), F(-355, 33), F(46732, 5247), F(49, 176), F(-5103, 18656)],
        [F(35, 384), 0, F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84)],
      ],
      [F(35, 384), 0, F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84), 0],
      [F(5179, 57600), 0, F(7571, 16695), F(393, 640), F(-92097, 339200), F(187, 2100), F(1, 40)],
    ),
  )
}
"""The named explicit methods, by the name ``tableau`` and the integrators take."""


def tableau(method: str | Tableau) -> Tableau:
  """The tableau a method name spells, or ``method`` itself when it is already one."""
  if isinstance(method, Tableau):
    return method
  found = TABLEAUS.get(method)
  if found is None:
    raise ValueError(f"unknown Runge-Kutta method {method!r}; known methods: {sorted(TABLEAUS)}")
  return found
