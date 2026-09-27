"""Butcher tableaus: the named Runge-Kutta families and the order conditions that check them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache

import numpy as np

from .polynomial import gauss_nodes, lagrange_integrals, lobatto_nodes, radau_nodes

__all__ = ["FAMILIES", "TABLEAUS", "Tableau", "gauss_legendre", "lobatto_iiia", "lobatto_iiic", "order_conditions", "radau_iia", "tableau"]


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

  @property
  def diagonally_implicit(self) -> bool:
    """Whether each stage is implicit in itself alone: ``a`` lower triangular with a nonzero
    diagonal, so the stages solve one after another, each a system of the state's size."""
    return not np.triu(self.a, 1).any() and bool(np.all(np.diag(self.a) != 0))

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
"""The named methods, explicit and implicit, by the name ``tableau`` and the integrators take."""


def _collocation(name: str, nodes: np.ndarray, order: int) -> Tableau:
  """The collocation method at ``nodes``: ``a_ij`` and ``b_j`` integrate the Lagrange basis from 0 to
  ``c_i`` and to 1, so the stages are the derivative of the polynomial through them."""
  integrals = lagrange_integrals(nodes, np.append(nodes, 1.0))
  return Tableau(integrals[:-1], integrals[-1], nodes, order, name)


def _stages(family: str, s: int, least: int = 1) -> int:
  if int(s) != s or s < least:
    raise ValueError(f"{family} needs at least {least} stage{'s' if least > 1 else ''}, got {s}")
  return int(s)


def gauss_legendre(s: int) -> Tableau:
  """The ``s``-stage Gauss-Legendre method: collocation at the Gauss nodes, order ``2s``, A-stable
  and symplectic. One stage is the implicit midpoint rule."""
  s = _stages("gauss_legendre", s)
  return _collocation(f"gauss_legendre{s}", gauss_nodes(s), 2 * s)


def radau_iia(s: int) -> Tableau:
  """The ``s``-stage Radau IIA method: collocation at the right Radau nodes, order ``2s - 1``,
  L-stable and stiffly accurate (the step is the last stage). One stage is backward Euler."""
  s = _stages("radau_iia", s)
  return _collocation(f"radau_iia{s}", radau_nodes(s), 2 * s - 1)


def lobatto_iiia(s: int) -> Tableau:
  """The ``s``-stage Lobatto IIIA method: collocation at the Lobatto nodes, order ``2s - 2``,
  A-stable and stiffly accurate, its first stage explicit. Two stages are the trapezoidal rule."""
  s = _stages("lobatto_iiia", s, 2)
  return _collocation(f"lobatto_iiia{s}", lobatto_nodes(s), 2 * s - 2)


def lobatto_iiic(s: int) -> Tableau:
  """The ``s``-stage Lobatto IIIC method: order ``2s - 2``, L-stable and stiffly accurate. At the
  Lobatto nodes, each row of ``a`` has ``a_i1 = b_1`` and integrates polynomials of degree up to
  ``s - 2`` exactly (Chipman's construction)."""
  s = _stages("lobatto_iiic", s, 2)
  c = lobatto_nodes(s)
  b = lagrange_integrals(c, np.ones(1))[0]
  powers = np.arange(1, s)  # the conditions sum_j a_ij c_j^(k-1) = c_i^k / k, k = 1 .. s - 1
  system = np.vstack([np.eye(s)[0], c[None, :] ** (powers[:, None] - 1)])
  a = np.stack([np.linalg.solve(system, np.concatenate([[b[0]], c[i] ** powers / powers])) for i in range(s)])
  return Tableau(a, b, c, 2 * s - 2, f"lobatto_iiic{s}")


def _sdirk3() -> Tableau:
  """Alexander's three-stage SDIRK: order 3, L-stable, stiffly accurate, ``gamma`` the root of
  ``6 g^3 - 18 g^2 + 9 g - 1`` in ``(1/6, 1/2)``."""
  g = next(r.real for r in np.roots([6.0, -18.0, 9.0, -1.0]) if abs(r.imag) < 1e-12 and 1 / 6 < r.real < 1 / 2)
  tau = (1 + g) / 2
  b1, b2 = -(6 * g * g - 16 * g + 1) / 4, (6 * g * g - 20 * g + 5) / 4
  a = np.array([[g, 0, 0], [tau - g, g, 0], [b1, b2, g]])
  return Tableau(a, a[-1], a.sum(axis=1), 3, "sdirk3")


def _sdirk2() -> Tableau:
  """Alexander's two-stage SDIRK: order 2, L-stable, stiffly accurate, ``gamma = 1 - 1/sqrt(2)``."""
  g = 1 - 1 / np.sqrt(2)
  a = np.array([[g, 0], [1 - g, g]])
  return Tableau(a, a[-1], a.sum(axis=1), 2, "sdirk2")


FAMILIES: dict[str, Callable[[int], Tableau]] = {
  "gauss_legendre": gauss_legendre,
  "radau_iia": radau_iia,
  "lobatto_iiia": lobatto_iiia,
  "lobatto_iiic": lobatto_iiic,
}
"""The implicit families whose member ``tableau`` builds from a number of stages."""

TABLEAUS.update(
  {
    "backward_euler": Tableau(np.ones((1, 1)), np.ones(1), np.ones(1), 1, "backward_euler"),
    "implicit_midpoint": Tableau(np.full((1, 1), 0.5), np.ones(1), np.full(1, 0.5), 2, "implicit_midpoint"),
    "trapezoidal": Tableau(np.array([[0.0, 0.0], [0.5, 0.5]]), np.full(2, 0.5), np.array([0.0, 1.0]), 2, "trapezoidal"),
    "sdirk2": _sdirk2(),
    "sdirk3": _sdirk3(),
  }
)


def tableau(method: str | Tableau, stages: int | None = None) -> Tableau:
  """The tableau a method name spells, a member of a family (``FAMILIES``) given its number of
  ``stages``, or ``method`` itself when it is already one."""
  if isinstance(method, Tableau):
    if stages is not None:
      raise ValueError("stages goes with a family name, not with a Tableau")
    return method
  if method in FAMILIES:
    if stages is None:
      raise ValueError(f"{method} is a family: say how many stages, as in stages=3")
    return FAMILIES[method](stages)
  found = TABLEAUS.get(method)
  if found is None:
    raise ValueError(f"unknown Runge-Kutta method {method!r}; known methods: {sorted(TABLEAUS)}, and the families {sorted(FAMILIES)}")
  if stages is not None:
    raise ValueError(f"{method} has a fixed number of stages; stages goes with a family name, one of {sorted(FAMILIES)}")
  return found
