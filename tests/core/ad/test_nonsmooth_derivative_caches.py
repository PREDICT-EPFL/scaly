"""A derivative through a map or a loop follows the ``nonsmooth`` setting in force when it is built,
whatever was built from the same callee before: the derivative Functions built from a callee's body
are cached, and named, per setting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import options_tag
from scaly.ir.expr import callees_of, topo

type Body = Callable[[sc.Expr], sc.Expr]


def _mapped(name: str, body: Body) -> tuple[sc.Expr, sc.Expr]:
  x = sc.sym("x")
  fn = sc.Function.from_exprs(name, [x], [body(x)], ["x"], ["y"])
  xs = sc.sym("xs", 2)
  return xs, sc.vmap(fn, 2, [(xs, 0, 1)])


def _scanned(name: str, body: Body) -> tuple[sc.Expr, sc.Expr]:
  c, x = sc.sym("c"), sc.sym("x")
  step = sc.Function.from_exprs(name, [c, x], [c + body(x)], ["c", "x"], ["c_next"])
  xs = sc.sym("xs", 2)
  return xs, sc.scan(step, sc.const(0.0), [(xs, 0, 1)], length=2)[0].reshape((1,))


def _looped(name: str, body: Body) -> tuple[sc.Expr, sc.Expr]:
  """A while loop differentiated in its param; it takes one step at the points used here."""
  c, x = sc.sym("c", 2), sc.sym("x", 2)
  step = sc.Function.from_exprs(name, [c, x], [c + body(x)], ["c", "x"], ["c_next"])
  go = sc.Function.from_exprs(f"{name}_go", [c, x], [c.sum() < 0.5], ["c", "x"], ["go"])
  xs = sc.sym("xs", 2)
  return xs, sc.while_loop(go, step, sc.const(np.zeros(2)), max_iter=3, params=[xs])[0]


WRAPS: dict[str, Callable[[str, Body], tuple[sc.Expr, sc.Expr]]] = {"vmap": _mapped, "scan": _scanned, "while": _looped}
KINDS = pytest.mark.parametrize("kind", list(WRAPS))


def _tie(x: sc.Expr) -> sc.Expr:
  return sc.maximum(x, 1.0 - x)


def _derivatives(xs: sc.Expr, y: sc.Expr) -> list[sc.Expr]:
  """The gradient of ``y.sum()``, its derivative along ones and the Jacobian of ``y``: one through
  each derivative cache, reverse, forward and multi-seed forward."""
  cost = y.sum()
  return [sc.gradient(cost, xs), sc.jvp(cost, xs, sc.const(np.ones(2))), sc.jacobian(y, xs)]


def _functions(*exprs: sc.Expr) -> dict[int, Any]:
  """Every Function the graphs of ``exprs`` run, callees of callees included, by identity."""
  seen: dict[int, Any] = {}
  todo = [c for node in topo(exprs) for c in callees_of(node)]
  while todo:
    fn = todo.pop()
    if id(fn) not in seen:
      seen[id(fn)] = fn
      todo.extend(c for node in topo(fn.outputs) for c in callees_of(node))
  return seen


# d/dx maximum(x, 1 - x) at the tie x = 0.5: 1/2 - 1/2 under "split", all of the left operand's 1
# under "first".
TIE_SLOPE = {"split": 0.0, "first": 1.0}


@KINDS
@pytest.mark.parametrize("order", [("split", "first"), ("first", "split")], ids=["split-first", "first-split"])
def test_a_derivative_follows_the_setting_it_is_built_under(kind: str, order: tuple[str, str]) -> None:
  name = f"ns_cache_tie_{kind}_{order[0]}"
  xs, y = WRAPS[kind](name, _tie)
  built: list[sc.Expr] = []
  for mode in order:
    with sc.options(nonsmooth=mode):
      built.extend(_derivatives(xs, y))
  # Both settings' derivatives in one Function: lowering refuses two Functions with one name.
  fn = sc.Function.from_exprs(f"{name}_both", [xs], built, ["xs"], [f"{out}{i}" for i in range(2) for out in "gtj"])
  values = fn(np.full(2, 0.5))
  for i, mode in enumerate(order):
    gradient, tangent, jacobian = values[3 * i : 3 * i + 3]
    slope = TIE_SLOPE[mode]
    np.testing.assert_array_equal(gradient, [slope, slope], err_msg=mode)
    np.testing.assert_array_equal(tangent, 2 * slope, err_msg=mode)
    np.testing.assert_array_equal(np.asarray(jacobian).sum(axis=0), [slope, slope], err_msg=mode)


@KINDS
def test_error_refuses_a_derivative_built_before_under_the_default(kind: str) -> None:
  name = f"ns_cache_floor_{kind}"
  xs, y = WRAPS[kind](name, lambda x: (x - x.floor()).sin())
  built = _derivatives(xs, y)
  cost = y.sum()
  with sc.options(nonsmooth="error"):
    for build in (lambda: sc.gradient(cost, xs), lambda: sc.jvp(cost, xs, sc.const(np.ones(2))), lambda: sc.jacobian(y, xs)):
      with pytest.raises(NotImplementedError, match="nonsmooth='error'"):
        build()
  point = np.array([0.3, 1.7])
  slope = np.cos(point - np.floor(point))
  gradient, tangent, jacobian = sc.Function.from_exprs(f"{name}_default", [xs], built, ["xs"], ["g", "t", "j"])(point)
  np.testing.assert_allclose(gradient, slope, rtol=1e-14)
  np.testing.assert_allclose(tangent, slope.sum(), rtol=1e-14)
  np.testing.assert_allclose(np.asarray(jacobian).sum(axis=0), slope, rtol=1e-14)


def _built_under_both(kind: str, label: str, body: Body) -> tuple[dict[int, Any], dict[int, Any]]:
  """The Functions of the three derivatives built under the default and then under ``"first"``."""
  xs, y = WRAPS[kind](f"ns_cache_{label}_{kind}", body)
  split = _functions(*_derivatives(xs, y))
  with sc.options(nonsmooth="first"):
    return split, _functions(*_derivatives(xs, y))


@KINDS
def test_a_derivative_built_again_under_another_setting_gets_its_own_name(kind: str) -> None:
  split, _ = _built_under_both(kind, "smooth", lambda x: x.sin())
  split_tie, first_tie = _built_under_both(kind, "tie", _tie)
  # Under the default, a nonsmooth callee's derivatives are named as a smooth one's: they are C symbols.
  assert sorted(fn.name.replace("_tie_", "_smooth_") for fn in split_tie.values()) == sorted(fn.name for fn in split.values())
  with sc.options(nonsmooth="first"):
    tag = options_tag()
  rebuilt = [first_tie[k].name for k in first_tie.keys() - split_tie.keys()]
  assert tag and rebuilt and all(name.endswith(tag) for name in rebuilt), rebuilt


@KINDS
@pytest.mark.xfail(strict=True, reason="C-195: helper caches are keyed on every option, not the ones the callee reads")
def test_a_smooth_derivative_is_shared_across_settings(kind: str) -> None:
  split, first = _built_under_both(kind, "smooth", lambda x: x.sin())
  assert first.keys() == split.keys()  # the same derivative under every setting: the same Functions
