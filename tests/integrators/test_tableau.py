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
  assert tab.explicit and tab.name == name
  assert si.order_conditions(tab.a, tab.b, tab.order + 2) == tab.order


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
