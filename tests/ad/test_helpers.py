"""Derived-callee keys and names: one digest-named helper per key, and clashes renamed at lowering."""

from __future__ import annotations

import re

import numpy as np

import scaly as sc
from scaly.codegen.aot import render_c_source
from scaly.passes.lowering import lower_function


def _stage(name: str, scale: float) -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("z"), name=name)
  def stage(inputs):
    x, y = inputs
    return scale * (x * y).sin()

  return stage


def test_helper_names_are_stable_across_fresh_builds() -> None:
  def render() -> str:
    stage = _stage("named_stage", 1.0)

    @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.arg("j"), name="named_jacobian")
    def jacobian(inputs):
      x, y = inputs
      return sc.jacobian(stage((x, y)), x)

    return render_c_source(jacobian, lanes=1)

  first = render()
  assert re.search(r"\bnamed_stage_fwd_[0-9a-f]{10}", first)
  assert render() == first


def test_same_named_callees_get_distinct_helper_procedures() -> None:
  first, second = _stage("twin", 1.0), _stage("twin", 3.0)

  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.group(sc.arg("a"), sc.arg("b")), name="twins")
  def twins(inputs):
    x, y = inputs
    return first((x, y)), second((x, y))

  jac = twins.factory("twins_jac", ["x", "y"], [sc.factory.Jac("a", "x"), sc.factory.Jac("b", "x")])
  procs = [str(proc.attrs["name"]) for proc in lower_function(jac).args[:-1]]
  helpers = [name for name in procs if "_fwd_" in name]
  assert len(helpers) == 2 and helpers[0] != helpers[1]
  x, y = np.array([0.3, -0.7]), np.array([1.1, 0.4])
  ja, jb = jac(x, y)
  np.testing.assert_allclose(ja, np.diag(np.cos(x * y) * y), rtol=1e-14, atol=1e-14)
  np.testing.assert_allclose(jb, np.diag(3.0 * np.cos(x * y) * y), rtol=1e-14, atol=1e-14)
