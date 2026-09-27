"""Butcher tableaus: the order conditions themselves, and every named method against them."""

from __future__ import annotations

import numpy as np
import pytest

from scaly import integrators as si
from scaly.integrators.tableau import _trees


def test_rooted_trees_are_counted_as_in_oeis_a000081() -> None:
  assert [len(_trees(n)) for n in range(1, 9)] == [1, 1, 2, 4, 9, 20, 48, 115]


@pytest.mark.parametrize("name", sorted(si.TABLEAUS))
def test_named_methods_have_exactly_their_order(name: str) -> None:
  tab = si.TABLEAUS[name]
  assert tab.name == name
  assert si.order_conditions(tab.a, tab.b, tab.order + 2) == tab.order


FAMILY_ORDER = {
  "gauss_legendre": lambda s: 2 * s,
  "radau_iia": lambda s: 2 * s - 1,
  "lobatto_iiia": lambda s: 2 * s - 2,
  "lobatto_iiic": lambda s: 2 * s - 2,
}


@pytest.mark.parametrize("family", sorted(si.FAMILIES))
@pytest.mark.parametrize("s", range(1, 6))
def test_families_have_exactly_their_order(family: str, s: int) -> None:
  if family.startswith("lobatto") and s < 2:
    with pytest.raises(ValueError, match="at least 2 stages"):
      si.tableau(family, s)
    return
  tab = si.tableau(family, s)
  assert tab.name == f"{family}{s}" and tab.stages == s and not tab.explicit
  assert si.order_conditions(tab.a, tab.b, tab.order + 1) == tab.order == FAMILY_ORDER[family](s)


def _stability(tab, z: complex) -> complex:
  """``R(z) = 1 + z b^T (I - z A)^{-1} 1``: one step of the method on ``x' = lambda x``, ``z = h lambda``."""
  s = tab.stages
  return 1 + z * tab.b @ np.linalg.solve(np.eye(s) - z * tab.a, np.ones(s))


@pytest.mark.parametrize(
  ("method", "stages", "l_stable"),
  [
    ("gauss_legendre", 3, False),
    ("radau_iia", 3, True),
    ("lobatto_iiia", 3, False),
    ("lobatto_iiic", 3, True),
    ("sdirk2", None, True),
    ("sdirk3", None, True),
  ],
)
def test_implicit_methods_are_a_stable_and_the_l_stable_ones_damp_infinity(method: str, stages: int | None, l_stable: bool) -> None:
  tab = si.tableau(method, stages)
  for y in np.geomspace(1e-3, 1e4, 60):
    assert abs(_stability(tab, 1j * y)) <= 1 + 1e-12
  for z in -np.geomspace(1e-3, 1e4, 60):
    assert abs(_stability(tab, complex(z))) <= 1 + 1e-12
  at_infinity = abs(_stability(tab, -1e8))  # far enough out, and short of where the cancellation in 1 + z b.(...) loses the digits
  assert at_infinity < 1e-6 if l_stable else abs(at_infinity - 1) < 1e-6


def test_structure_of_the_families() -> None:
  radau, lobatto_a, lobatto_c, gauss = si.radau_iia(3), si.lobatto_iiia(3), si.lobatto_iiic(3), si.gauss_legendre(3)
  for stiffly_accurate in (radau, lobatto_a, lobatto_c, si.TABLEAUS["sdirk3"]):
    np.testing.assert_allclose(stiffly_accurate.a[-1], stiffly_accurate.b, atol=1e-15)
  assert not lobatto_a.a[0].any()  # the first stage is explicit
  np.testing.assert_allclose(lobatto_c.a[:, 0], lobatto_c.b[0], atol=1e-15)
  symplectic = gauss.b[:, None] * gauss.a + (gauss.b[:, None] * gauss.a).T - np.outer(gauss.b, gauss.b)
  np.testing.assert_allclose(symplectic, 0, atol=1e-15)
  np.testing.assert_allclose(si.radau_iia(2).a, [[5 / 12, -1 / 12], [3 / 4, 1 / 4]], atol=1e-15)
  np.testing.assert_allclose(si.lobatto_iiic(2).a, [[1 / 2, -1 / 2], [1 / 2, 1 / 2]], atol=1e-15)
  for alias, member in (("backward_euler", si.radau_iia(1)), ("implicit_midpoint", si.gauss_legendre(1)), ("trapezoidal", si.lobatto_iiia(2))):
    np.testing.assert_allclose(si.TABLEAUS[alias].a, member.a, atol=1e-15)
  assert [si.TABLEAUS[m].diagonally_implicit for m in ("backward_euler", "sdirk2", "sdirk3", "trapezoidal", "rk4")] == [
    True,
    True,
    True,
    False,
    False,
  ]
  assert not radau.diagonally_implicit


@pytest.mark.parametrize(("name", "order"), [("bs32", 2), ("dopri5", 4)])
def test_embedded_weights_have_the_lower_order(name: str, order: int) -> None:
  tab = si.TABLEAUS[name]
  assert tab.b_err is not None
  assert si.order_conditions(tab.a, tab.b_err, order + 2) == order


def test_a_perturbed_coefficient_loses_order() -> None:
  tab = si.TABLEAUS["rk4"]
  a = tab.a.copy()
  a[3, 2] += 1e-6
  assert si.order_conditions(a, tab.b, 4) == 1  # the row sum moves, so even b . c = 1/2 fails


def test_construction_checks_shapes_nodes_and_order() -> None:
  rk4 = si.TABLEAUS["rk4"]
  with pytest.raises(ValueError, match="must be"):
    si.Tableau(rk4.a[:3], rk4.b, rk4.c, 4)
  with pytest.raises(ValueError, match="row sums"):
    si.Tableau(rk4.a, rk4.b, rk4.c + 0.1, 4)
  with pytest.raises(ValueError, match="only to order 4"):
    si.Tableau(rk4.a, rk4.b, rk4.c, 5, "rk4?")
  with pytest.raises(ValueError, match="b_err"):
    si.Tableau(rk4.a, rk4.b, rk4.c, 4, b_err=np.ones(3))
  custom = si.Tableau(rk4.a, rk4.b, rk4.c, 3, "mine")
  assert custom.stages == 4 and custom.explicit and repr(custom) == "Tableau('mine', stages=4, order=3)"


def test_lookup() -> None:
  assert si.tableau("rk4") is si.TABLEAUS["rk4"]
  tab = si.TABLEAUS["heun"]
  assert si.tableau(tab) is tab
  with pytest.raises(ValueError, match="unknown Runge-Kutta method 'rk5'"):
    si.tableau("rk5")
  with pytest.raises(ValueError, match="radau_iia is a family: say how many stages"):
    si.tableau("radau_iia")
  with pytest.raises(ValueError, match="rk4 has a fixed number of stages"):
    si.tableau("rk4", 3)
  with pytest.raises(ValueError, match="stages goes with a family name, not with a Tableau"):
    si.tableau(tab, 2)
