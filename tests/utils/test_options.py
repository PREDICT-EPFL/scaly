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


@pytest.fixture(autouse=True)
def _restore_default():
  yield
  defaults = sc.Options()
  sc.set_options(
    **{f.name: getattr(defaults, f.name) for f in dataclasses.fields(sc.Options) if f.name != "changed"},
    linalg=dataclasses.asdict(defaults.namespace("linalg")),
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
    for name in ("dense_unroll", "sparse_unroll"):
      with pytest.raises(ValueError, match="non-negative integer"):
        sc.set_options(linalg={name: bad})
  with sc.options(max_trajectory=0) as inside:
    assert inside.max_trajectory == 0
  for name in ("dense_unroll", "sparse_unroll"):
    with sc.options(linalg={name: 0}) as inside:
      assert getattr(inside.namespace("linalg"), name) == 0
  with pytest.raises(TypeError, match=r"unknown option linalg.'unroll'"):
    sc.set_options(linalg={"unroll": 3})
  with pytest.raises(TypeError, match="takes a dict"):
    sc.set_options(linalg=3)
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
    return sc.Function._from_exprs(f"opt_grad_{mode}", [x], [g], ["x"], ["g"])

  split, first = grad("split"), grad("first")
  tie = np.array([1.0, 1.0, 0.0])
  # Leaving the block does not change what was built inside it.
  np.testing.assert_allclose(split(tie), [0.5 + 0.5, 0.5 + 0.5, 0.0])
  np.testing.assert_allclose(first(tie), [1.0 + 1.0, 1.0, 0.0])
  assert render_c_source(split).replace("split", "_") != render_c_source(first).replace("first", "_")
  with sc.options(nonsmooth="error"), pytest.raises(NotImplementedError, match="nonsmooth='error'"):
    sc.vjp((cost,), (x,), (sc.const(1.0),))


def test_dense_unroll_is_decided_when_the_node_is_built() -> None:
  a, b = sc.sym("a", (4, 4)), sc.sym("b", 4)
  with sc.options(linalg=dict(dense_unroll=0)):
    looped = [sc.linalg.cholesky(a), sc.linalg.ldl(a), sc.linalg.solve_triangular(a, b)]
  unrolled = [sc.linalg.cholesky(a), sc.linalg.ldl(a), sc.linalg.solve_triangular(a, b)]
  assert all(not e.attrs["unroll"] for e in looped) and all(e.attrs["unroll"] for e in unrolled)
  fns = {tag: sc.Function._from_exprs(f"du_{tag}", [a, b], outs, ["a", "b"], ["l", "d", "x"]) for tag, outs in (("loop", looped), ("flat", unrolled))}
  for k, (loop_op, flat_op) in enumerate(zip(looped, unrolled, strict=True)):
    for tag, op, has_loop in (("loop", loop_op, True), ("flat", flat_op, False)):
      src = render_c_source(sc.Function._from_exprs(f"du_{tag}{k}", [a, b], [op], ["a", "b"], ["o"]))
      assert ("for (" in src.split(f"int du_{tag}{k}")[1]) == has_loop
  m = np.random.default_rng(3).standard_normal((4, 4))
  av, bv = m @ m.T + 4 * np.eye(4), np.arange(4.0)
  for got, ref in zip(fns["loop"]._flat_numerical_call(av, bv), fns["flat"]._flat_numerical_call(av, bv), strict=True):
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-14)
  with sc.options(linalg=dict(dense_unroll=16)):
    assert sc.linalg.cholesky(sc.sym("big", (12, 12))).attrs["unroll"]


def test_max_trajectory_refuses_a_reverse_pass_that_stores_too_much() -> None:
  c = sc.sym("c", 50)
  body = sc.Function._from_exprs("traj_step", [c], [c.sin()], ["c"], ["cn"])
  cond = sc.Function._from_exprs("traj_go", [c], [c[0] < 10.0], ["c"], ["go"])
  x = sc.sym("x", 50)
  (scanned,) = sc.scan(body, x, length=100)
  walked, _ = sc.while_loop(cond, body, x, max_iter=100)
  for out in (scanned, walked):
    with sc.options(max_trajectory=4999):
      with pytest.raises(ValueError, match=r"traj_step.*100 carries of 50 values .*max_trajectory=4999.*custom_derivative"):
        sc.vjp((out.sum(),), (x,), (sc.const(1.0),))
    with sc.options(max_trajectory=5000):
      sc.vjp((out.sum(),), (x,), (sc.const(1.0),))


@dataclasses.dataclass(frozen=True)
class _Shaping:
  level: int = 0


# A namespace a derivative depends on, declared once per process as a package would.
register_option_namespace("test_shaping", _Shaping(), affects_derivatives=True)


def test_only_namespaces_that_affect_derivatives_name_helpers_apart() -> None:
  assert options_tag() == ""
  with sc.options(linalg=dict(dense_unroll=0, sparse_unroll=3)):
    assert options_tag() == ""
  with sc.options(test_shaping=dict(level=2)) as inside:
    assert inside.namespace("test_shaping") == _Shaping(2)
    assert options_tag().startswith("_o")
  with sc.options(test_shaping=dict(level=0)) as inside:
    assert inside.changed == () and options_tag() == ""
  with pytest.raises(ValueError, match="already declared"):
    register_option_namespace("linalg", _Shaping(), affects_derivatives=False)
