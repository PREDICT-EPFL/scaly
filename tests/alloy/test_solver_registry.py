"""Structural tests for the solver plugin protocol (docs/solver_plugins.md).

These exercise the registry gates and the core/plugin codegen handoff with a
fake in-test backend — no C toolchain or vendored solver library involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.codegen import solver_c
from alloy.solvers import registry
from alloy.solvers.registry import SOLVER_PLUGIN_PROTOCOL_VERSION, SolverPluginError
from alloy.solvers.solver_function import SolverDescriptor, SolverFunction


class _FakeEntryPoint:
  def __init__(self, backend: object) -> None:
    self._backend = backend

  def load(self) -> object:
    return self._backend


class _FakeBackend:
  name = "fake"
  kind = "qp"
  protocol_version = SOLVER_PLUGIN_PROTOCOL_VERSION
  lib_stem = "fake"
  link_flags = ("-lfake",)
  header = "fake/fake.h"

  def include_dir(self) -> Path:
    return Path("/nonexistent/include")

  def lib_dir(self) -> Path:
    return Path("/nonexistent/lib")

  def render_wrapper(self, fun, ctx):  # noqa: ANN001, ANN201 - protocol mirror
    inputs = ", ".join(f"const double* in{i}" for i in range(len(fun.descriptor.input_signature)))
    outputs = ", ".join(f"double* out{i}" for i in range(len(fun.descriptor.output_signature)))
    return [
      f"static void {ctx.raw_symbol}({inputs}, {outputs}, double* w) {{",
      f"  {ctx.stats_symbol}.version = ALLOY_SOLVER_STATS_VERSION;",
      "}",
    ]


def _fake_solver_function(backend: str = "fake") -> SolverFunction:
  desc = SolverDescriptor(
    name="fake_qp",
    backend=backend,
    n=1,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
    output_signature=(("x", (1,)),),
    param_names=(),
  )
  return SolverFunction(desc)


def test_protocol_version_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  class _Stale(_FakeBackend):
    protocol_version = SOLVER_PLUGIN_PROTOCOL_VERSION - 1

  monkeypatch.setattr(registry, "available_backends", lambda: {"stale": _FakeEntryPoint(_Stale())})
  with pytest.raises(SolverPluginError, match="protocol version"):
    registry.get_backend("stale")


def test_entry_point_name_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "available_backends", lambda: {"other": _FakeEntryPoint(_FakeBackend())})
  with pytest.raises(SolverPluginError, match="must equal the entry-point name"):
    registry.get_backend("other")


def test_missing_render_wrapper_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  class _NoRender(_FakeBackend):
    render_wrapper = None

  monkeypatch.setattr(registry, "available_backends", lambda: {"fake": _FakeEntryPoint(_NoRender())})
  with pytest.raises(SolverPluginError, match="render_wrapper"):
    registry.get_backend("fake")


def test_broken_path_provider_does_not_hide_other_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
  class _BrokenPaths(_FakeBackend):
    def include_dir(self):  # noqa: ANN201
      raise OSError("boom")

    def lib_dir(self):  # noqa: ANN201
      raise OSError("boom")

  monkeypatch.setattr(registry, "loaded_backends", lambda: {"fake": _FakeBackend(), "broken": _BrokenPaths()})
  with pytest.warns(RuntimeWarning, match="could not inspect solver plugin 'broken'"):
    paths = registry.installed_backend_paths()
  assert len(paths) == 1


def test_loaded_backends_skips_version_mismatch_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
  class _Stale(_FakeBackend):
    protocol_version = SOLVER_PLUGIN_PROTOCOL_VERSION - 1

  monkeypatch.setattr(registry, "available_backends", lambda: {"fake": _FakeEntryPoint(_FakeBackend()), "stale": _FakeEntryPoint(_Stale())})
  with pytest.warns(RuntimeWarning, match="could not load solver plugin 'stale'"):
    backends = registry.loaded_backends()
  assert list(backends) == ["fake"]


def test_generic_lib_env_var_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
  from alloy import toolchain

  lib = tmp_path / f"libfake{toolchain.shared_lib_ext()}"
  lib.write_text("")
  monkeypatch.setattr(toolchain, "_backends", lambda: {"fake": _FakeBackend()})
  monkeypatch.setattr(toolchain, "_plugin_solver_paths", lambda: [])
  monkeypatch.setenv("ALLOY_FAKE_LIB", str(lib))
  paths = toolchain.solver_paths()
  assert paths.loads["fake"] == str(lib)
  assert paths.source == "ALLOY_FAKE_LIB"


def test_kind_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  with pytest.raises(SolverPluginError, match="solves qp problems, not nlp"):
    registry.require_backend("fake", "nlp")


def test_missing_backend_error_lists_installed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "available_backends", lambda: {})
  with pytest.raises(SolverPluginError, match="no solver plugin 'nope'"):
    registry.get_backend("nope")


def test_render_solver_raw_dispatches_to_plugin_and_frames_stats(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  fun = _fake_solver_function()
  lines = solver_c.render_solver_raw(fun)
  # Core-owned framing: stats storage before the plugin body, accessor after.
  assert lines[0] == "static alloy_solver_stats fake_qp_stats_data;"
  assert "static void fake_qp_raw(const double* in0, const double* in1, const double* in2, double* out0, double* w) {" in lines
  assert "int fake_qp_stats(alloy_solver_stats* out) {" in lines
  # The plugin body was told about the same stats symbol core declared.
  assert "  fake_qp_stats_data.version = ALLOY_SOLVER_STATS_VERSION;" in lines


def test_solver_backends_used_and_includes(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  fun = _fake_solver_function()
  assert solver_c.solver_backends_used(fun) == ("fake",)
  assert solver_c.solver_includes(fun) == ['#include "fake/fake.h"']
