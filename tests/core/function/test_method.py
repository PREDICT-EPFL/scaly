"""The method interface (``function/method.py``): a registry over fake methods declared as entry points,
``Status``, ``Support``, and ``Info`` as a Function's output tree."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint
from typing import Any, ClassVar

import numpy as np
import pytest

import scaly as sc
from scaly.function import method as m
from scaly.function.method import Info, MethodError, MethodHint, MethodRegistry, Status, Support


@dataclass(frozen=True)
class Toy:
  """A fake problem class: a size, which the fake methods read."""

  method_api: ClassVar[int] = 2
  size: int


@dataclass(frozen=True)
class Other:
  method_api: ClassVar[int] = 1


@dataclass(frozen=True)
class Alpha:
  name: ClassVar[str] = "fake.alpha"
  problem: ClassVar[type] = Toy
  api: ClassVar[int] = 2
  limit: int = 10

  def supports(self, problem: Toy) -> Support:
    return Support.unless(too_large=problem.size > self.limit)

  def build(self, problem: Toy, *, name: str) -> sc.ConcreteFunction:
    x = sc.sym("x", problem.size)
    return sc.Function.from_exprs(name, [x], [2.0 * x], ["x"], ["y"])


@dataclass(frozen=True)
class Beta(Alpha):
  name: ClassVar[str] = "fake.beta"
  limit: int = 100


class Misnamed(Alpha):
  name: ClassVar[str] = "fake.something_else"


class Old(Alpha):
  name: ClassVar[str] = "fake.old"
  api: ClassVar[int] = 1


@dataclass(frozen=True)
class Elsewhere(Alpha):
  name: ClassVar[str] = "fake.elsewhere"
  problem: ClassVar[type] = Other
  api: ClassVar[int] = 1


ENTRY_POINTS = [
  EntryPoint(f"fake.{name}", f"tests.core.function.test_method:{cls}", m.METHOD_ENTRY_POINTS)
  for name, cls in (("alpha", "Alpha"), ("beta", "Beta"), ("misnamed", "Misnamed"), ("old", "Old"), ("elsewhere", "Elsewhere"))
] + [EntryPoint("other.gamma", "tests.core.function.test_method:Alpha", m.METHOD_ENTRY_POINTS), EntryPoint("x", "y:z", "scaly.adapters")]


@pytest.fixture(autouse=True)
def _fake_entry_points(monkeypatch) -> None:
  monkeypatch.setattr(m, "entry_points", lambda group: [ep for ep in ENTRY_POINTS if ep.group == group])


def _registry(**kw: Any) -> MethodRegistry:
  return MethodRegistry("fake", **kw)


def test_status_codes_are_stable() -> None:
  """They are the ``status`` of the solver statistics ABI: never renumber one."""
  assert [(s.name, int(s)) for s in Status] == [
    ("OK", 0),
    ("ACCEPTABLE", 1),
    ("MAX_ITER", 2),
    ("PRIMAL_INFEASIBLE", 3),
    ("DUAL_INFEASIBLE", 4),
    ("NUMERICS", 5),
    ("USER_STOP", 6),
    ("ERROR", 7),
  ]
  assert Status.OK.ok and Status.ACCEPTABLE.ok and not Status.MAX_ITER.ok
  assert sc.Status is Status


def test_methods_are_found_by_domain_and_loaded_on_first_use() -> None:
  reg = _registry()
  assert sorted(reg.installed()) == ["alpha", "beta", "elsewhere", "misnamed", "old"]  # not other.gamma, not an adapter
  assert not reg._loaded
  assert reg.get("alpha") is Alpha and reg.get("fake.alpha") is Alpha
  assert list(reg._loaded) == ["alpha"]
  assert m.method_names() == {"fake": ["alpha", "beta", "elsewhere", "misnamed", "old"], "other": ["gamma"]}


def test_a_missing_method_names_the_distribution_to_install() -> None:
  reg = _registry(hints=(MethodHint("gamma", "Gamma", "scaly-gamma"),))
  with pytest.raises(MethodError, match=r"fake\.gamma is not installed; it comes with scaly-gamma \(uv add scaly-gamma\)"):
    reg.get("gamma")
  with pytest.raises(MethodError, match=r"no method fake\.delta; installed: \['alpha'"):
    reg.get("delta")


def test_a_method_that_does_not_fit_its_entry_point_or_api_is_refused() -> None:
  reg = _registry()
  with pytest.raises(MethodError, match="declares name 'fake.something_else'"):
    reg.get("misnamed")
  with pytest.raises(MethodError, match="implements API 1 of Toy; this scaly has API 2"):
    reg.get("old")


def test_auto_takes_the_first_method_that_supports_the_problem() -> None:
  """A method that fails to load or check is skipped with a warning; one for another problem class too."""
  reg = _registry(preference=("alpha", "beta"))
  assert type(reg.auto(Toy(5))) is Alpha
  assert type(reg.auto(Toy(50))) is Beta  # too large for alpha
  with pytest.warns(RuntimeWarning) as caught, pytest.raises(MethodError, match=r"fake\.alpha: too large; fake\.beta: too large"):
    reg.auto(Toy(500))
  assert sorted(str(w.message).split()[1] for w in caught) == ["fake.misnamed", "fake.old"]


def test_resolve_takes_an_instance_a_name_or_auto() -> None:
  reg = _registry(preference=("alpha", "beta"))
  assert reg.resolve(Beta(limit=3), Toy(2)) == Beta(limit=3)
  assert reg.resolve("beta", Toy(2)) == Beta()
  assert reg.resolve("auto", Toy(50)) == Beta()
  with pytest.raises(ValueError, match="cannot solve this problem: too large"):
    reg.resolve("alpha", Toy(50))
  with pytest.raises(TypeError, match="solves Toy, not Other"):
    reg.resolve(Alpha(), Other())
  fn = reg.resolve("alpha", Toy(3)).build(Toy(3), name="fake_alpha_built")
  np.testing.assert_array_equal(fn(np.ones(3)), [2.0, 2.0, 2.0])


def test_a_namespace_resolves_method_classes_lazily() -> None:
  reg = _registry(hints=(MethodHint("gamma", "Gamma", "scaly-gamma"),))
  getattr_ = reg.attribute("scaly.fake")
  assert getattr_("Beta") is Beta
  with pytest.raises(AttributeError, match=r"scaly\.fake\.Gamma needs scaly-gamma"):
    getattr_("Gamma")
  with pytest.raises(AttributeError, match="has no attribute 'Nope'"):
    getattr_("Nope")


def test_a_domain_has_one_registry() -> None:
  first = m.registry("fake_domain_once", hints=(MethodHint("a", "A", "scaly-a"),))
  assert m.registry("fake_domain_once") is first and first.hints[0].distribution == "scaly-a"


@dataclass(frozen=True)
class ToyInfo(Info):
  residual: Any


def test_info_is_a_record_of_named_outputs() -> None:
  """A Function returns an ``Info``: fields ``Expr`` symbolically, arrays numerically, ``status``
  and ``iter`` as ``int64``."""

  @sc.function(3, output=sc.G("y", ToyInfo.tree("info_")))
  def solve(x):
    return 2.0 * x, ToyInfo(sc.const(np.int64(Status.MAX_ITER), dtype="int64"), sc.const(np.int64(7), dtype="int64"), sc.norm_inf(x))

  assert solve.output_names == ("y", "info_status", "info_iter", "info_residual")
  y, info = solve(np.array([1.0, -3.0, 2.0]))
  assert isinstance(info, ToyInfo) and Status(int(info.status)) is Status.MAX_ITER and int(info.iter) == 7
  assert float(info.residual) == 3.0
  np.testing.assert_array_equal(y, [2.0, -6.0, 4.0])
  z, symbolic = solve(sc.sym("x", 3))
  assert isinstance(symbolic, ToyInfo) and isinstance(symbolic.residual, sc.Expr) and symbolic.status.type.dtype.name == "int64"
  with pytest.raises(TypeError, match="needs one tree per field"):
    m.Record(ToyInfo, status="s", iter="i")
