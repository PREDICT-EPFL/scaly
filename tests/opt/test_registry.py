"""Structural tests for external optimization methods (docs/dev/solver_plugins.md).

These exercise the registry gates and the core/plugin codegen handoff with a fake in-test method
(an ``External``) — no C toolchain or vendored solver library involved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly.function import method as core_method
from scaly.function.extern import ExternRenderCtx
from scaly.function.method import MethodError, MethodRegistry
from scaly.opt import method as opt_method
from scaly.opt.external import External, wrapper
from scaly.opt.external import graph as solver_graph
from scaly.opt.external.graph import solver_descriptor
from scaly.opt.external.model import ExternalOracle, SolverDescriptor, descriptor_function
from scaly.opt.method import METHOD_API


@dataclass(frozen=True)
class FakeMethod(External):
  name = "opt.fake"
  kind = "qp"
  lib_stem = "fake"
  link_flags = ("-lfake",)
  header = "fake/fake.h"

  @staticmethod
  def include_dir() -> Path:
    return Path("/nonexistent/include")

  @staticmethod
  def lib_dir() -> Path:
    return Path("/nonexistent/lib")

  def render_wrapper(self, fun, ctx):  # noqa: ANN001, ANN201 - the hook's signature
    inputs = ", ".join(f"const double* in{i}" for i in range(len(solver_descriptor(fun).input_signature)))
    outputs = ", ".join(f"double* out{i}" for i in range(len(solver_descriptor(fun).output_signature)))
    return [
      f"static void {ctx.raw_symbol}({inputs}, {outputs}, double* w) {{",
      f"  {ctx.stats_symbol}.version = SCALY_SOLVER_STATS_VERSION;",
      "}",
    ]


@dataclass(frozen=True)
class StaleMethod(FakeMethod):
  name = "opt.stale"
  api = METHOD_API - 1


def _fake_solver_function(method: External | None = None) -> sc.ConcreteFunction:
  desc = SolverDescriptor(
    name="fake_qp",
    backend="fake",
    n=1,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (1,)), ("lam_eq0", (0,)), ("lam_ineq0", (0,))),
    output_signature=(("x", (1,)),),
    param_names=(),
    n_var_blocks=0,
    method=method or FakeMethod(),
  )
  return descriptor_function(desc)


FAKES = [
  EntryPoint("opt.fake", "tests.opt.test_registry:FakeMethod", core_method.METHOD_ENTRY_POINTS),
  EntryPoint("opt.stale", "tests.opt.test_registry:StaleMethod", core_method.METHOD_ENTRY_POINTS),
]


@pytest.fixture
def fake_opt_registry(monkeypatch: pytest.MonkeyPatch) -> MethodRegistry:
  """The opt registry seeing only the fake methods, with nothing loaded yet."""
  monkeypatch.setattr(core_method, "entry_points", lambda group: [ep for ep in FAKES if ep.group == group])
  monkeypatch.setattr(opt_method.REGISTRY, "_loaded", {})
  monkeypatch.setattr(opt_method, "_EXTERNAL", {})
  return opt_method.REGISTRY


def test_a_method_api_mismatch_is_rejected(fake_opt_registry: MethodRegistry) -> None:
  with pytest.raises(MethodError, match=f"implements API {METHOD_API - 1} of NLP; this scaly has API {METHOD_API}"):
    fake_opt_registry.get("stale")


def test_external_methods_skip_a_broken_install_with_a_warning(fake_opt_registry: MethodRegistry) -> None:
  with pytest.warns(RuntimeWarning, match="could not load opt.stale"):
    methods = opt_method.external_methods()
  assert list(methods) == ["fake"] and isinstance(methods["fake"], FakeMethod)


def test_a_missing_method_names_its_distribution(fake_opt_registry: MethodRegistry) -> None:
  with pytest.raises(MethodError, match=r"opt\.ipopt is not installed; it comes with scaly-ipopt"):
    fake_opt_registry.get("ipopt")
  with pytest.raises(MethodError, match=r"no method opt\.nope; installed: \['fake', 'stale'\]"):
    fake_opt_registry.get("nope")


def test_broken_path_provider_does_not_hide_other_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.opt.external import paths

  @dataclass(frozen=True)
  class BrokenPaths(FakeMethod):
    @staticmethod
    def include_dir() -> Path:
      raise OSError("boom")

  monkeypatch.setattr(paths, "_backends", lambda: {"fake": FakeMethod(), "broken": BrokenPaths()})
  with pytest.warns(RuntimeWarning, match="could not inspect opt.broken"):
    found = paths._plugin_solver_paths()
  assert len(found) == 1


def test_generic_lib_env_var_override(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.opt.external import paths as solver_paths_module
  from scaly.utils import env

  lib = tmp_path / f"libfake{env.shared_lib_ext()}"
  lib.write_text("")
  monkeypatch.setattr(solver_paths_module, "_backends", lambda: {"fake": FakeMethod()})
  monkeypatch.setattr(solver_paths_module, "_plugin_solver_paths", lambda: [])
  monkeypatch.setenv("SCALY_FAKE_LIB", str(lib))
  paths = solver_paths_module.solver_paths()
  assert paths.loads["fake"] == str(lib)
  assert paths.source == "SCALY_FAKE_LIB"


def test_a_method_declares_what_it_solves() -> None:
  @dataclass(frozen=True)
  class NoKind(FakeMethod):
    kind = "lp"

  @dataclass(frozen=True)
  class NoTriangle(FakeMethod):
    kind = "nlp"
    hess_triangle = None

  with pytest.raises(TypeError, match="must declare kind 'qp' or 'nlp'"):
    NoKind()
  with pytest.raises(TypeError, match="must declare hess_triangle"):
    NoTriangle()


@pytest.mark.parametrize("triangle", ["lower", "upper"])
def test_nlp_descriptor_uses_the_methods_hessian_triangle(triangle: str) -> None:
  @dataclass(frozen=True)
  class FakeNlp(FakeMethod):
    kind = "nlp"
    hess_triangle = triangle

  @sc.opt.problem(vars=sc.L(f"layout_x_{triangle}", 2), name=f"layout_{triangle}")
  def problem(x):
    return sc.opt.ProblemSpec(minimize=x[0] * x[1])

  nlp = sc.opt.solver(problem, FakeNlp(), name=f"layout_{triangle}")
  sparsity = solver_descriptor(nlp).hess_sparsity
  assert sparsity is not None
  assert all(row >= col if triangle == "lower" else row <= col for row, col in zip(sparsity.rows, sparsity.cols, strict=True))
  assert not hasattr(solver_descriptor(nlp), "hess_lower_mask")


def test_a_qp_method_refuses_a_problem_that_is_not_quadratic() -> None:
  @sc.opt.problem(vars=sc.L("nq_x", 2), name="not_quadratic")
  def problem(x):
    return sc.opt.ProblemSpec(minimize=(x**4).sum())

  support = FakeMethod().supports(problem)
  assert not support and "not quadratic" in support.reasons[0]
  with pytest.raises(sc.opt.NotQuadratic, match="cost is not quadratic"):
    sc.opt.solver(problem, FakeMethod())


def test_render_solver_dispatches_to_plugin_and_frames_stats(monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fake_solver_function()
  lines = wrapper.render_solver(fun, solver_descriptor(fun), ExternRenderCtx("fake_qp", "fake_qp_raw"))
  # Core-owned framing: stats storage before the plugin body, which defines ``<raw>_solve`` over the
  # solution; ``<raw>`` calls it and writes the Info outputs from the stats; the accessor comes last.
  assert lines[0] == "static scaly_solver_stats fake_qp_stats_data;"
  assert "static void fake_qp_raw_solve(const double* in0, const double* in1, const double* in2, double* out0, double* w) {" in lines
  frame = lines.index(
    "static void fake_qp_raw(const double* in0, const double* in1, const double* in2, double* out0, double* out1, double* out2, double* out3, double* out4, double* w) {"
  )
  assert lines[frame + 1 : frame + 6] == [
    "  fake_qp_raw_solve(in0, in1, in2, out0, w);",
    "  out1[0] = (double)fake_qp_stats_data.status;",
    "  out2[0] = (double)fake_qp_stats_data.iter;",
    "  out3[0] = (double)fake_qp_stats_data.obj;",
    "  out4[0] = (double)fake_qp_stats_data.primal_viol;",
  ]
  assert "int fake_qp_stats(scaly_solver_stats* out) {" in lines
  # The plugin body was told about the same stats symbol core declared.
  assert "  fake_qp_stats_data.version = SCALY_SOLVER_STATS_VERSION;" in lines


def test_external_oracle_source_and_symbol_cross_the_plugin_boundary() -> None:
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
    method=FakeMethod(),
  )

  @dataclass(frozen=True)
  class ExternalBackend(FakeMethod):
    def render_wrapper(self, fun, ctx):  # noqa: ANN001, ANN201
      assert ctx.raw_symbol_of(solver_descriptor(fun).base) == "foreign_base_raw"
      return super().render_wrapper(fun, ctx)

  desc = replace(desc, method=ExternalBackend())
  source = sc.codegen.render_c_source(descriptor_function(desc))
  assert source.count(oracle.source) == 1
  assert source.index(oracle.source) < source.index("static scaly_solver_stats external_qp_stats_data;")


def test_external_oracle_workspace_is_part_of_solver_workspace() -> None:
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
    method=FakeMethod(),
  )
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
      method=FakeMethod(),
    )
  )


def test_external_oracle_source_is_deduplicated_across_solver_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
  source = "static void shared_raw(const double* x, double* y, double* w) { y[0] = x[0]; (void)w; }"
  oracle = ExternalOracle("shared", "shared_raw", source, (("x", (1,)),), (("f", ()),))
  left, right = _external_solver("left_solver", oracle), _external_solver("right_solver", oracle)
  args = [sc.const(np.zeros(1)), sc.const(np.zeros(0)), sc.const(np.zeros(0))]
  host = sc.Function.from_exprs("two_external_solvers", [], [left(tuple(args))[0] + right(tuple(args))[0]], [], ["x"])

  from scaly.codegen.aot import render_c_module

  assert render_c_module(host).source.count(source) == 1


def test_conflicting_external_oracle_symbol_definitions_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  first = ExternalOracle("first", "shared_raw", "static void shared_raw(void) {}", (), ())
  second = ExternalOracle("second", "shared_raw", "static void shared_raw(int x) { (void)x; }", (), ())
  left, right = _external_solver("left_conflict", first), _external_solver("right_conflict", second)
  args = [sc.const(np.zeros(1)), sc.const(np.zeros(0)), sc.const(np.zeros(0))]
  host = sc.Function.from_exprs("conflicting_external_solvers", [], [left(tuple(args))[0] + right(tuple(args))[0]], [], ["x"])

  from scaly.codegen.aot import render_c_module

  with pytest.raises(ValueError, match="conflicting source definitions"):
    render_c_module(host)


def test_stats_abi_v3_layout_is_additive() -> None:
  """The v3 diagnostics tail appends after the v2 fields — never reorders them —
  and keeps the struct 8-aligned (four doubles at offset 96, two int32)."""
  import ctypes

  from scaly.opt.external.stats import SCALY_SOLVER_STATS_VERSION, STATS_FIELDS, CSolverStats

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
  fun = _fake_solver_function()
  assert solver_graph.solver_backends_used(fun) == ("fake",)
  assert wrapper.solver_requirements("fake_qp", solver_descriptor(fun)).includes == ("#include <time.h>", '#include "fake/fake.h"')
