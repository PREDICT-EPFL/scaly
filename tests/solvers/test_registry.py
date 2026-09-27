"""Structural tests for the solver plugin protocol (docs/dev/solver_plugins.md).

These exercise the registry gates and the core/plugin codegen handoff with a
fake in-test backend — no C toolchain or vendored solver library involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import solver
from scaly.solvers import graph as solver_graph
from scaly.solvers import registry
from scaly.solvers.registry import SOLVER_PLUGIN_PROTOCOL_VERSION, SolverPluginError
from scaly.solvers.model import ExternalOracle, SolverDescriptor, descriptor_function


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
      f"  {ctx.stats_symbol}.version = SCALY_SOLVER_STATS_VERSION;",
      "}",
    ]


def _fake_solver_function(backend: str = "fake") -> sc.ConcreteFunction:
  desc = SolverDescriptor(
    name="fake_qp",
    backend=backend,
    n=1,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
    output_signature=(("x", (1,)),),
    param_names=(),
    n_var_blocks=0,
  )
  return descriptor_function(desc)


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
  from scaly.solvers import paths as solver_paths_module
  from scaly.utils import env

  lib = tmp_path / f"libfake{env.shared_lib_ext()}"
  lib.write_text("")
  monkeypatch.setattr(solver_paths_module, "_backends", lambda: {"fake": _FakeBackend()})
  monkeypatch.setattr(solver_paths_module, "_plugin_solver_paths", lambda: [])
  monkeypatch.setenv("SCALY_FAKE_LIB", str(lib))
  paths = solver_paths_module.solver_paths()
  assert paths.loads["fake"] == str(lib)
  assert paths.source == "SCALY_FAKE_LIB"


def test_kind_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  with pytest.raises(SolverPluginError, match="solves qp problems, not nlp"):
    registry.require_backend("fake", "nlp")


@pytest.mark.parametrize("triangle", ["lower", "upper"])
def test_nlp_descriptor_uses_backend_hessian_triangle(monkeypatch: pytest.MonkeyPatch, triangle: str) -> None:
  class _FakeNlpBackend(_FakeBackend):
    kind = "nlp"
    hess_triangle = triangle

  fake_backend = lambda name: _FakeNlpBackend()  # noqa: E731
  monkeypatch.setattr(registry, "get_backend", fake_backend)
  monkeypatch.setattr(sys.modules["scaly.solvers.solver"], "get_backend", fake_backend)

  @sc.problem(vars=sc.L(f"layout_x_{triangle}", 2), name=f"layout_{triangle}")
  def problem(x):
    return sc.ProblemSpec(minimize=x[0] * x[1])

  nlp = sc.solver(problem, "fake", name=f"layout_{triangle}")
  sparsity = nlp.descriptor.hess_sparsity
  assert sparsity is not None
  assert all(row >= col if triangle == "lower" else row <= col for row, col in zip(sparsity.rows, sparsity.cols, strict=True))
  assert not hasattr(nlp.descriptor, "hess_lower_mask")


def test_nlp_backend_must_declare_a_hessian_triangle(monkeypatch: pytest.MonkeyPatch) -> None:
  class _MissingTriangle(_FakeBackend):
    kind = "nlp"
    hess_triangle = None

  monkeypatch.setattr(registry, "get_backend", lambda name: _MissingTriangle())
  with pytest.raises(SolverPluginError, match="must declare hess_triangle"):
    registry.require_backend("fake", "nlp")


def test_missing_backend_error_lists_installed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "available_backends", lambda: {})
  with pytest.raises(SolverPluginError, match="no solver plugin 'nope'"):
    registry.get_backend("nope")


def test_render_solver_raw_dispatches_to_plugin_and_frames_stats(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  fun = _fake_solver_function()
  lines = solver.render_solver_raw(fun)
  # Core-owned framing: stats storage before the plugin body, accessor after.
  assert lines[0] == "static scaly_solver_stats fake_qp_stats_data;"
  assert "static void fake_qp_raw(const double* in0, const double* in1, const double* in2, double* out0, double* w) {" in lines
  assert "int fake_qp_stats(scaly_solver_stats* out) {" in lines
  # The plugin body was told about the same stats symbol core declared.
  assert "  fake_qp_stats_data.version = SCALY_SOLVER_STATS_VERSION;" in lines


def test_external_oracle_source_and_symbol_cross_the_plugin_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
  oracle = ExternalOracle(
    name="foreign_base",
    raw_symbol="foreign_base_raw",
    source="static void foreign_base_raw(const double* x, double* y, double* w) { y[0] = x[0]; (void)w; }",
    input_signature=(("x", (1,)),),
    output_signature=(("f", ()),),
  )
  desc = SolverDescriptor(
    name="external_qp",
    backend="fake",
    n=1,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
    output_signature=(("x", (1,)),),
    param_names=(),
    n_var_blocks=0,
    base=oracle,
  )

  class _ExternalBackend(_FakeBackend):
    def render_wrapper(self, fun, ctx):  # noqa: ANN001, ANN201
      assert ctx.raw_symbol_of(fun.descriptor.base) == "foreign_base_raw"
      return super().render_wrapper(fun, ctx)

  monkeypatch.setattr(registry, "get_backend", lambda name: _ExternalBackend())
  source = "\n".join(solver.render_solver_raw(descriptor_function(desc)))
  assert oracle.source in source
  assert source.index(oracle.source) < source.index("static scaly_solver_stats external_qp_stats_data;")


def test_external_oracle_workspace_is_part_of_solver_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
  oracle = ExternalOracle(
    name="foreign_base",
    raw_symbol="foreign_base_raw",
    source="static void foreign_base_raw(const double* x, double* y, double* w) { w[6] = x[0]; y[0] = w[6]; }",
    input_signature=(("x", (1,)),),
    output_signature=(("f", ()),),
    workspace_size=7,
  )
  desc = SolverDescriptor(
    name="external_workspace",
    backend="fake",
    n=1,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
    output_signature=(("x", (1,)),),
    param_names=(),
    n_var_blocks=0,
    base=oracle,
  )
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  from scaly.codegen.aot import render_c_module

  header = render_c_module(descriptor_function(desc)).header
  assert "#define external_workspace_SZ_W 7" in header


def _external_solver(name: str, oracle: ExternalOracle) -> sc.ConcreteFunction:
  return descriptor_function(
    SolverDescriptor(
      name=name,
      backend="fake",
      n=1,
      n_eq=0,
      n_ineq=0,
      input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
      output_signature=(("x", (1,)),),
      param_names=(),
      n_var_blocks=0,
      base=oracle,
    )
  )


def test_external_oracle_source_is_deduplicated_across_solver_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
  source = "static void shared_raw(const double* x, double* y, double* w) { y[0] = x[0]; (void)w; }"
  oracle = ExternalOracle("shared", "shared_raw", source, (("x", (1,)),), (("f", ()),))
  left, right = _external_solver("left_solver", oracle), _external_solver("right_solver", oracle)
  args = [sc.const(np.zeros(1)), sc.const(np.zeros(0)), sc.const(np.zeros(0))]
  host = sc.Function._from_exprs("two_external_solvers", [], [left(tuple(args)) + right(tuple(args))], [], ["x"])
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())

  from scaly.codegen.aot import render_c_module

  assert render_c_module(host).source.count(source) == 1


def test_conflicting_external_oracle_symbol_definitions_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  first = ExternalOracle("first", "shared_raw", "static void shared_raw(void) {}", (), ())
  second = ExternalOracle("second", "shared_raw", "static void shared_raw(int x) { (void)x; }", (), ())
  left, right = _external_solver("left_conflict", first), _external_solver("right_conflict", second)
  args = [sc.const(np.zeros(1)), sc.const(np.zeros(0)), sc.const(np.zeros(0))]
  host = sc.Function._from_exprs("conflicting_external_solvers", [], [left(tuple(args)) + right(tuple(args))], [], ["x"])
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())

  from scaly.codegen.aot import render_c_module

  with pytest.raises(ValueError, match="conflicting source definitions"):
    render_c_module(host)


def test_stats_abi_v3_layout_is_additive() -> None:
  """The v3 diagnostics tail appends after the v2 fields — never reorders them —
  and keeps the struct 8-aligned (four doubles at offset 96, two int32)."""
  import ctypes

  from scaly.solvers.stats import SCALY_SOLVER_STATS_VERSION, STATS_FIELDS, CSolverStats

  assert SCALY_SOLVER_STATS_VERSION == 3
  names = [name for name, _ in STATS_FIELDS]
  assert names[:17] == [
    "version", "status", "native_status", "iter",
    "obj", "t_total", "t_fe", "t_solver", "t_qp", "t_globalization", "t_glue",
    "n_eval_f", "n_eval_grad_f", "n_eval_g", "n_eval_jac_g", "n_eval_h", "_pad0",
  ]  # fmt: skip
  assert names[17:] == ["primal_viol", "step_inf", "alpha", "merit_penalty", "backtracks", "qp_iter"]
  assert CSolverStats.primal_viol.offset == 96
  assert ctypes.sizeof(CSolverStats) == 136


def test_solver_backends_used_and_includes(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(registry, "get_backend", lambda name: _FakeBackend())
  fun = _fake_solver_function()
  assert solver_graph.solver_backends_used(fun) == ("fake",)
  assert solver.solver_includes(fun) == ['#include "fake/fake.h"']
