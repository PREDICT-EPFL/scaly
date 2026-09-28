"""``sc.options`` scopes, validates and isolates conventions, and a convention lives in the graph it shaped."""

from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import options_tag
from scaly.codegen import render_c_source
from scaly.utils.options import register_option_namespace


def _count(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclasses.dataclass(frozen=True)
class _Shaping:
  level: int = 0


@dataclasses.dataclass(frozen=True)
class _Layout:
  width: int = 8


# Namespaces as packages declare them, once per process: one a derivative depends on, one it does not.
register_option_namespace("test_shaping", _Shaping(), affects_derivatives=True, checks={"level": _count})
register_option_namespace("test_layout", _Layout(), affects_derivatives=False)


@pytest.fixture(autouse=True)
def _restore_default():
  yield
  defaults = sc.Options()
  sc.set_options(
    **{f.name: getattr(defaults, f.name) for f in dataclasses.fields(sc.Options) if f.name != "changed"},
    **{name: dataclasses.asdict(defaults.namespace(name)) for name, _ in sc.get_options().changed},
  )


def test_default_nesting_and_restoration_after_an_exception() -> None:
  assert sc.get_options() == sc.Options(nonsmooth="split")
  with sc.options(nonsmooth="first") as outer:
    assert outer.nonsmooth == sc.get_options().nonsmooth == "first"
    with pytest.raises(RuntimeError):
      with sc.options(nonsmooth="error"):
        assert sc.get_options().nonsmooth == "error"
        raise RuntimeError
    assert sc.get_options().nonsmooth == "first"
  assert sc.get_options().nonsmooth == "split"
  sc.set_options(nonsmooth="first")
  assert sc.get_options().nonsmooth == "first"
  with sc.options(nonsmooth="split"):
    assert sc.get_options().nonsmooth == "split"
  assert sc.get_options().nonsmooth == "first"


def test_unknown_names_and_values_raise_at_the_call() -> None:
  with pytest.raises(TypeError, match="unknown scaly option 'tie'"):
    with sc.options(tie="split"):
      pass
  with pytest.raises(ValueError, match="is not one of"):
    sc.set_options(nonsmooth="average")
  assert sc.get_options().nonsmooth == "split"
  for bad in (-1, 2.5, True, "8"):
    with pytest.raises(ValueError, match="non-negative integer"):
      sc.set_options(max_trajectory=bad)
    with pytest.raises(ValueError, match="non-negative integer"):
      sc.set_options(test_shaping={"level": bad})
  with sc.options(max_trajectory=0) as inside:
    assert inside.max_trajectory == 0
  with sc.options(test_shaping={"level": 0}, test_layout={"width": 3}) as inside:
    assert inside.namespace("test_shaping").level == 0 and inside.namespace("test_layout").width == 3
  with pytest.raises(TypeError, match=r"unknown option test_shaping.'unroll'"):
    sc.set_options(test_shaping={"unroll": 3})
  with pytest.raises(TypeError, match="takes a dict"):
    sc.set_options(test_shaping=3)
  # A name is looked up as a package that declares it; one that declares none is still unknown.
  with pytest.raises(TypeError, match="unknown scaly option 'ext'"):
    sc.set_options(ext={})
  with pytest.raises(KeyError, match="unknown option namespace 'tie'"):
    sc.get_options().namespace("tie")
  assert sc.get_options() == sc.Options()


def test_a_block_is_local_to_its_thread() -> None:
  seen: list[str] = []
  entered, checked = threading.Event(), threading.Event()

  def other() -> None:
    entered.wait()
    seen.append(sc.get_options().nonsmooth)
    checked.set()

  thread = threading.Thread(target=other)
  thread.start()
  with sc.options(nonsmooth="error"):
    entered.set()
    checked.wait()
  thread.join()
  assert seen == ["split"]


def test_the_convention_is_part_of_the_graph_and_the_generated_code() -> None:
  x = sc.sym("x", 3)
  cost = sc.maximum(x, 1.0).sum() + x.max()

  def grad(mode: str) -> sc.Function:
    with sc.options(nonsmooth=mode):
      (g,) = sc.vjp((cost,), (x,), (sc.const(1.0),))
    return sc.Function.from_exprs(f"opt_grad_{mode}", [x], [g], ["x"], ["g"])

  split, first = grad("split"), grad("first")
  tie = np.array([1.0, 1.0, 0.0])
  # Leaving the block does not change what was built inside it.
  np.testing.assert_allclose(split(tie), [0.5 + 0.5, 0.5 + 0.5, 0.0])
  np.testing.assert_allclose(first(tie), [1.0 + 1.0, 1.0, 0.0])
  assert render_c_source(split).replace("split", "_") != render_c_source(first).replace("first", "_")
  with sc.options(nonsmooth="error"), pytest.raises(NotImplementedError, match="nonsmooth='error'"):
    sc.vjp((cost,), (x,), (sc.const(1.0),))


def test_max_trajectory_refuses_a_reverse_pass_that_stores_too_much() -> None:
  c = sc.sym("c", 50)
  body = sc.Function.from_exprs("traj_step", [c], [c.sin()], ["c"], ["cn"])
  cond = sc.Function.from_exprs("traj_go", [c], [c[0] < 10.0], ["c"], ["go"])
  x = sc.sym("x", 50)
  (scanned,) = sc.scan(body, x, length=100)
  walked, _ = sc.while_loop(cond, body, x, max_iter=100)
  for out in (scanned, walked):
    with sc.options(max_trajectory=4999):
      with pytest.raises(ValueError, match=r"traj_step.*100 carries of 50 values .*max_trajectory=4999.*custom_derivative"):
        sc.vjp((out.sum(),), (x,), (sc.const(1.0),))
    with sc.options(max_trajectory=5000):
      sc.vjp((out.sum(),), (x,), (sc.const(1.0),))


def test_only_namespaces_that_affect_derivatives_name_helpers_apart() -> None:
  assert options_tag() == ""
  with sc.options(test_layout=dict(width=3)):
    assert options_tag() == ""
  with sc.options(test_shaping=dict(level=2)) as inside:
    assert inside.namespace("test_shaping") == _Shaping(2)
    assert options_tag().startswith("_o")
  with sc.options(test_shaping=dict(level=0)) as inside:
    assert inside.changed == () and options_tag() == ""
  with pytest.raises(ValueError, match="already declared"):
    register_option_namespace("test_shaping", _Shaping(), affects_derivatives=False)
