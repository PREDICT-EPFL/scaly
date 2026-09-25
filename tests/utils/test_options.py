"""``sc.options`` scopes, validates and isolates conventions, and a convention lives in the graph it shaped."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source


@pytest.fixture(autouse=True)
def _restore_default():
  yield
  sc.set_options(nonsmooth="split")


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
    return sc.Function._from_exprs(f"opt_grad_{mode}", [x], [g], ["x"], ["g"])

  split, first = grad("split"), grad("first")
  tie = np.array([1.0, 1.0, 0.0])
  # Leaving the block does not change what was built inside it.
  np.testing.assert_allclose(split(tie), [0.5 + 0.5, 0.5 + 0.5, 0.0])
  np.testing.assert_allclose(first(tie), [1.0 + 1.0, 1.0, 0.0])
  assert render_c_source(split).replace("split", "_") != render_c_source(first).replace("first", "_")
  with sc.options(nonsmooth="error"), pytest.raises(NotImplementedError, match="nonsmooth='error'"):
    sc.vjp((cost,), (x,), (sc.const(1.0),))
