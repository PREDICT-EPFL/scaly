"""Structural tests for solver function input validation and the shared solver ABI.

These cover Scaly's own machinery — argument checking, the universal C entry, the
`scaly_solver_stats` handshake, identifier mangling across one translation unit, and
JIT cache keying. PIQP is only the vehicle: a second QP backend would exercise the
same code. Backend-specific behaviour (PIQP's CSC baking, its status mapping) stays
in `plugins/scaly-piqp/tests/`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

import scaly as sc
from tests.solvers.problem_helpers import build_qp
from scaly.codegen import render_c_api_header, render_c_source
from scaly.codegen.jit import CompiledFunction, JitError
from scaly.solvers.registry import available_backends
from scaly.solvers.stats import CSolverStats

pytestmark = pytest.mark.skipif(
  "piqp" not in available_backends(), reason="structural tests build the private QP differential fixture and need the scaly-piqp plugin installed"
)


def test_solver_function_signature_errors() -> None:
  """Mis-shaped or missing inputs raise ``TypeError`` / ``ValueError``."""
  P = np.eye(2)
  c = np.zeros(2)
  qp = build_qp(P=P, c=c)
  with pytest.raises(ValueError, match="declared structure"):
    qp.function.numerical_call(*(np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0)))  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match=r"expected shape \(2,\).*got \(3,\)"):
    qp((), x0=np.zeros(3))


def test_standalone_qp_renders_universal_entry_and_stats_query() -> None:
  qp = build_qp(P=np.eye(2), c=np.zeros(2), name="standalone_qp")
  source = render_c_source(qp.function)
  header = render_c_api_header(qp.function)
  assert (
    "int standalone_qp_with_options(const double** arg, double** res, int* iw, double* w, int mem, const scaly_solver_option* const* solver_options)"
    in source
  )
  assert "int standalone_qp_stats(scaly_solver_stats* out);" in header
  assert "SCALY_SOLVER_STATS_VERSION 3" in header


@pytest.mark.parametrize(
  "backend,options",
  [
    pytest.param("piqp", ({"eps_abs": 1e-8}, {"eps_abs": 1e-5, "max_iter": 1, "sparse": True}), marks=pytest.mark.solver("piqp")),
    pytest.param(
      "sqp",
      ({"max_iter": 30}, {"max_iter": 1, "globalization": "l1", "watchdog": 5, "hessian": "objective", "qp": "dense", "trace": True}),
      marks=pytest.mark.solver("sqp"),
    ),
    pytest.param(
      "ipopt", ({"tol": 1e-8}, {"tol": 1e-5, "max_iter": 1, "hessian_approximation": "limited-memory"}), marks=pytest.mark.solver("ipopt")
    ),
  ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_solver_options_share_one_build(
  backend: str, options: tuple[dict[str, Any], dict[str, Any]], nested: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  from scaly.codegen import jit

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("solver_options", 2), name="shared_options")
  def tracking(x: sc.Expr, solver_options: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - solver_options), eq=(x.sum() - 1.0,), lb=sc.const(0.0))

  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  monkeypatch.setattr(jit, "_artifact_cache", {})
  compiles = []
  run = jit.subprocess.run

  entry_name = f"shared_options_{backend}" + ("_host" if nested else "")

  def counted_run(command, *args, **kwargs):
    if any(str(arg).endswith(f"/{entry_name}.c") for arg in command):
      compiles.append(command)
    return run(command, *args, **kwargs)

  monkeypatch.setattr(jit.subprocess, "run", counted_run)
  solvers = [sc.solver(tracking, backend, options=opts) for opts in options]
  functions = [solver.function for solver in solvers]
  if nested:
    functions = []
    for solve in solvers:

      @sc.function(sc.arg("target", 2), outputs=sc.arg("x"), name=entry_name)
      def host(target: sc.Expr) -> sc.Expr:
        return solve(target, x0=sc.const(np.array([0.5, 0.5])))[0]

      functions.append(host)
  first, second = [CompiledFunction(function) for function in functions]
  assert first.cache_key == second.cache_key
  assert first.lib_path == second.lib_path
  assert len(compiles) == 1
  target = np.array([0.2, 0.8])
  args = [target] if nested else [np.array([0.5, 0.5]), np.zeros(2), np.zeros(1), np.zeros(0), target]
  limited = {
    "piqp": [0.21795554224819894, 0.7820949997913061],
    "sqp": [0.2000001895631653, 0.7999998118984609],
    "ipopt": [0.29999999733333327, 0.7000000026666667],
  }
  for compiled, expected, status in (
    (first, [0.2, 0.8], 0),
    (second, limited[backend], 0 if backend == "sqp" else 2),
    (first, [0.2, 0.8], 0),
    (second, limited[backend], 0 if backend == "sqp" else 2),
  ):
    np.testing.assert_allclose(compiled.run(args)[0], expected, atol=3e-7, rtol=0)
    assert int(compiled.solver_stats().status) == status


@pytest.mark.parametrize("backend", [pytest.param("piqp", marks=pytest.mark.solver("piqp")), pytest.param("sqp", marks=pytest.mark.solver("sqp"))])
def test_sparse_solver_needs_no_dense_staging_memory(backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  from scaly.codegen import aot

  @sc.problem(vars=sc.arg("x", 2), name="staging_memory")
  def tracking(x: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - 1.0))

  render = aot._render_solver_bearing_source
  monkeypatch.setattr(
    aot,
    "_render_solver_bearing_source",
    lambda *args, **kwargs: render(*args, **kwargs).replace("#include <stdlib.h>", "#include <stdlib.h>\n#define malloc(size) NULL"),
  )
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  sparse = {"sparse": True} if backend == "piqp" else {"qp": "sparse"}
  dense = {"sparse": False} if backend == "piqp" else {"qp": "dense"}
  solve = sc.solver(tracking, backend, options=sparse)
  np.testing.assert_allclose(solve(())[0], np.ones(2), atol=1e-5, rtol=0)
  assert solve.stats().status == sc.ScalySolveStatus.OK
  solve = sc.solver(tracking, backend, options=dense)
  np.testing.assert_array_equal(solve(())[0], np.zeros(2))
  assert solve.stats().status == sc.ScalySolveStatus.ERROR


@pytest.mark.solver("piqp")
def test_solver_stats_reject_uninitialized_and_mismatched_versions() -> None:
  """The `scaly_solver_stats` handshake is Scaly's contract with every backend."""
  qp = build_qp(P=np.eye(2), c=np.zeros(2), name="stats_version_qp")
  compiled = CompiledFunction(qp.function)
  with pytest.raises(JitError, match="has not run yet"):
    compiled.solver_stats()

  def mismatched_stats(out: ctypes.c_void_p) -> int:
    ctypes.cast(out, ctypes.POINTER(CSolverStats)).contents.version = sc.SCALY_SOLVER_STATS_VERSION + 1
    return 0

  compiled._stats_entries["stats_version_qp"] = mismatched_stats
  with pytest.raises(JitError, match="ABI mismatch.*artifact version 4, expected 3"):
    compiled.solver_stats()


def test_sparse_qp_rejects_nested_solver_data() -> None:
  """QP data computed from a nested solver output cannot be pattern-analyzed
  so the structural QP proof must reject it before the sparse-pattern probe
  would execute the inner solve."""
  inner = build_qp(P=np.eye(2), c=np.array([-1.0, 0.0]), x_lb=np.zeros(2), x_ub=np.ones(2), name="inner_for_pattern")
  x_inner = inner.function.symbolic_call(*(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), ()))[0]
  P = sc.stack([sc.stack([2.0 + x_inner[0], sc.const(0.0)]), sc.stack([sc.const(0.0), sc.const(2.0)])], axis=0)
  with pytest.raises(sc.NotQuadratic, match="cannot prove QP structure through a nested solver"):
    build_qp(P=P, c=np.zeros(2), sparse=True, name="outer_sparse_over_solver")


@pytest.mark.solver("piqp")
def test_two_solver_wrappers_in_one_translation_unit() -> None:
  """Two distinct solvers (one sparse, one dense) called from one host
  Function share a single generated TU; their static state must not collide."""

  @sc.function(sc.arg("t", (2,)), outputs=sc.arg("x_sum"), name="two_qp_host")
  def host(t: sc.Expr) -> sc.Expr:
    qp_a = build_qp(P=np.diag([2.0, 4.0]), c=sc.stack([t[0], t[1]]), sparse=True, name="tu_qp_a")
    qp_b = build_qp(P=np.diag([1.0, 1.0]), c=sc.stack([t[1], -t[0]]), name="tu_qp_b")
    xa, *_ = qp_a(t)
    xb, *_ = qp_b(t)
    return xa + xb

  tv = np.array([1.0, -2.0])
  # qp_a: x = -c / diag(P) = [-0.5, 0.5]; qp_b: x = [-t1, t0] = [2, 1].
  np.testing.assert_allclose(host(tv), [1.5, 1.5], atol=1e-7)


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_exported_solver_accepts_runtime_options(lang: str, tmp_path: Path) -> None:
  from scaly.codegen import render_c_module

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("solver_options", 2), name="export_options")
  def tracking(x: sc.Expr, solver_options: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=0.5 * sc.sumsqr(x) - sc.dot(x, solver_options))

  qp = sc.solver(tracking, "piqp", name="export_options", options={"max_iter": 1})
  module = render_c_module(qp.function, lang=lang)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  driver = tmp_path / ("driver.cpp" if lang == "cpp" else "driver.c")
  driver.write_text(
    f"""#include "{module.header_name}"
int main(void) {{
  double zero[2] = {{0, 0}}, x[2], lam[2], empty[1] = {{0}};
  double target[2] = {{1, 2}};
  const double* arg[] = {{zero, zero, empty, empty, target}};
  double* res[] = {{x, lam, empty, empty}};
  scaly_solver_option values[] = {{{{ "max_iter", 0, 1, 0.0, 0 }}, {{ "verbose", 0, 0, 0.0, 0 }}, {{ "sparse", 0, 0, 0.0, 0 }}, {{ 0, 0, 0, 0.0, 0 }}}};
  const scaly_solver_option* options[] = {{values}};
  double w[export_options_SZ_W > 0 ? export_options_SZ_W : 1];
  scaly_solver_stats stats;
  if ({"export_options::" if lang == "cpp" else ""}export_options_with_options(arg, res, 0, w, 0, options)) return 1;
  {"export_options::" if lang == "cpp" else ""}export_options_stats(&stats);
  if (stats.status != SCALY_SOLVE_MAX_ITER) return 2;
  if ({"export_options::" if lang == "cpp" else ""}export_options(arg, res, 0, w, 0)) return 3;
  {"export_options::" if lang == "cpp" else ""}export_options_stats(&stats);
  return stats.status != SCALY_SOLVE_OK || fabs(x[0] - 1.0) > 1e-6 || fabs(x[1] - 2.0) > 1e-6;
}}
""".replace('#include "', '#include <math.h>\n#include "', 1)
  )
  executable = tmp_path / "driver"
  subprocess.run(
    ["c++" if lang == "cpp" else "cc", "-O2", str(source), str(driver), *module.link_flags, "-lm", "-o", str(executable)],
    check=True,
    capture_output=True,
  )
  subprocess.run([str(executable)], check=True, capture_output=True)
