"""The output-adapter registry: a registered adapter's hooks reach the header, the source, the entry
and the workspace, by name, without the renderer knowing it; names resolve through the entry
points, and conflicting or unknown adapters are refused. The installed adapters (``cpp``,
``casadi``) are ``scaly.export``'s, tested in ``tests/export/``; these use throwaway ones."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import adapter as adapters_module
from scaly.codegen import render_c_api_header, render_c_module, render_c_source, workspace_size
from scaly.codegen.adapter import EntryHook, HeaderSpec, available_adapters, get_adapter, register_adapter


def _fun() -> sc.ConcreteFunction:
  x = sc.sym("x", 3)
  return sc.Function.from_exprs("adapter_probe", [x], [x * 2.0], ["x"], ["y"])


@pytest.fixture
def scratch_registry() -> Iterator[None]:
  """Adapters registered during a test are removed after it. The installed ones are loaded first:
  a module registers once, when first imported, so the registry itself is never swapped out."""
  for name in available_adapters():
    with contextlib.suppress(ModuleNotFoundError):  # an entry point whose distribution is missing
      get_adapter(name)
  before = set(adapters_module._ADAPTERS)
  yield
  for name in set(adapters_module._ADAPTERS) - before:
    del adapters_module._ADAPTERS[name]


@pytest.fixture
def probe(scratch_registry: None) -> str:
  """A throwaway adapter using every additive hook."""
  register_adapter(
    "probe",
    defines=("#define PROBE_DEFINE 1",),
    declarations=lambda symbol: (f"int {symbol}_probe(void);",),
    extra_workspace=lambda fun: 5,
    entry_prologue=lambda fun, base: EntryHook({"y": "y_probe"}, (f"  double* y_probe = w + {base};",), ("  res[0][0] = y_probe[0];",)),
    extra_source=lambda fun, sz_w: (f"int adapter_probe_probe(void) {{ return {sz_w}; }}",),
  )
  return "probe"


def test_every_hook_reaches_its_place(probe: str) -> None:
  fun = _fun()
  header = render_c_api_header(fun, adapters=(probe,))
  assert "#define PROBE_DEFINE 1" in header and "int adapter_probe_probe(void);" in header
  assert "#define adapter_probe_SZ_W 5" in header
  source = render_c_source(fun, adapters=(probe,))
  assert source.count("#define PROBE_DEFINE 1") == 1
  assert "double* y_probe = w + 0;" in source and "res[0][0] = y_probe[0];" in source
  assert "int adapter_probe_probe(void) { return 5; }" in source
  assert workspace_size(fun, adapters=(probe,)) == workspace_size(fun) + 5
  assert render_c_module(fun, adapters=(probe,)).adapters == ("probe",)


def test_a_header_adapter_replaces_the_c_header(scratch_registry: None) -> None:
  seen: list[HeaderSpec] = []

  def header(spec: HeaderSpec) -> str:
    seen.append(spec)
    return "// a header\n"

  register_adapter("probe_header", header=header, header_suffix="hh", source_includes_header=False)
  module = render_c_module(_fun(), adapters=("probe_header",))
  assert module.header == "// a header\n" and module.header_name == "adapter_probe.hh"
  assert module.source == module.body
  assert seen[0].fun.name == "adapter_probe" and seen[0].sparsities == (None,)
  register_adapter("probe_header_too", header=header, header_suffix="hpp")
  with pytest.raises(ValueError, match="probe_header, probe_header_too each set the header"):
    render_c_module(_fun(), adapters=("probe_header", "probe_header_too"))


def test_names_resolve_through_entry_points_and_are_checked(probe: str, monkeypatch: pytest.MonkeyPatch) -> None:
  """An installed adapter is listed before it loads, and loads on the first ask for it."""
  loads: list[str] = []

  def load() -> None:
    loads.append("probe_installed")
    register_adapter("probe_installed", header_suffix="hx", header=lambda spec: "// installed\n")

  installed = SimpleNamespace(name="probe_installed", load=load)
  monkeypatch.setattr(adapters_module, "entry_points", lambda group, name=None: [installed] if name in (None, installed.name) else [])
  assert "probe_installed" in available_adapters() and not loads
  assert get_adapter("probe_installed").header_suffix == "hx" and loads == ["probe_installed"]
  get_adapter("probe_installed")
  assert loads == ["probe_installed"]
  with pytest.raises(ValueError, match="already registered"):
    register_adapter(probe)
  with pytest.raises(ValueError, match="unknown output adapter 'nope'"):
    get_adapter("nope")


def test_a_header_adapter_leaves_the_kernel_alone(scratch_registry: None) -> None:
  register_adapter("probe_header", header=lambda spec: "// a header\n", header_suffix="hh", source_includes_header=False)
  fun = _fun()
  np.testing.assert_allclose(fun(np.array([1.0, 2.0, 3.0])), [2.0, 4.0, 6.0])
  assert render_c_module(fun, adapters=("probe_header",)).body == render_c_module(fun).body
