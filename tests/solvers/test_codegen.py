"""Structural tests for solver function input validation and the shared solver ABI.

These cover Scaly's own machinery — argument checking, the universal C entry, the
`scaly_solver_stats` handshake, identifier mangling across one translation unit, and
JIT cache keying. PIQP is only the vehicle: a second QP backend would exercise the
same code. Backend-specific behaviour (PIQP's CSC baking, its status mapping) stays
in `plugins/scaly-piqp/tests/`.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

import scaly as sc
from tests.solvers.problem_helpers import build_qp
from tests.solvers.problem_helpers import render_solver_source as render_c_source
from scaly.codegen.jit import CompiledFunction, JitError
from scaly.solvers.registry import available_backends
from scaly.solvers.stats import CSolverStats
from scaly.solvers.paths import solver_paths
from tests.compilers import c_compiler, cxx_compiler

pytestmark = pytest.mark.skipif(
  "piqp" not in available_backends(), reason="structural tests build the private QP differential fixture and need the scaly-piqp plugin installed"
)


@pytest.mark.parametrize(
  "backend,options",
  [
    *(
      pytest.param(
        "piqp",
        ({"sparse": sparse, "eps_abs": 1e-8}, {"sparse": sparse, "eps_abs": 1e-5, "max_iter": 1, "kkt_solver": int(sparse)}),
        marks=pytest.mark.solver("piqp"),
      )
      for sparse in (False, True)
    ),
    pytest.param(
      "sqp",
      ({"max_iter": 30}, {"max_iter": 1, "globalization": "l1", "watchdog": 5, "hessian": "objective", "trace": True}),
      marks=pytest.mark.solver("sqp"),
    ),
    pytest.param(
      "sqp",
      (
        {"qp": "dense", "max_iter": 30},
        {"qp": "dense", "max_iter": 1, "globalization": "l1", "watchdog": 5, "hessian": "objective", "trace": True},
      ),
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
def test_solver_interface_is_static(backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  @sc.problem(vars=sc.arg("x", 2), name="static_interface")
  def tracking(x: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - 1.0))

  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  solvers = [
    sc.solver(tracking, backend, options={"sparse": sparse} if backend == "piqp" else {"qp": "sparse" if sparse else "dense"})
    for sparse in (False, True)
  ]
  compiled = [CompiledFunction(solve.function) for solve in solvers]
  assert compiled[0].cache_key != compiled[1].cache_key
  for sparse, solve in zip((False, True), solvers, strict=True):
    source = render_c_source(solve.function)
    assert f"piqp_setup_{'sparse' if sparse else 'dense'}(" in source
    assert f"piqp_setup_{'dense' if sparse else 'sparse'}(" not in source
    descriptor = solve.function.instantiate().descriptor
    assert not any(option.name in (b"sparse", b"qp") for option in descriptor.runtime_options)
    if backend == "piqp":
      assert (descriptor.P_sparsity is not None) == sparse
      assert descriptor.oracle.outputs[0].size == (2 if sparse else 4)
    np.testing.assert_allclose(solve(())[0], np.ones(2), atol=1e-5, rtol=0)
    assert solve.stats().status == sc.ScalySolveStatus.OK


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

  @sc.function(sc.arg("t", (2,)), outputs=sc.arg("x_sum", 2), name="two_qp_host")
  def host(t: sc.Expr) -> sc.Expr:
    qp_a = build_qp(P=np.diag([2.0, 4.0]), c=sc.stack([t[0], t[1]]), sparse=True, name="tu_qp_a")
    qp_b = build_qp(P=np.diag([1.0, 1.0]), c=sc.stack([t[1], -t[0]]), name="tu_qp_b")
    xa, *_ = qp_a(t)
    xb, *_ = qp_b(t)
    return xa + xb

  tv = np.array([1.0, -2.0])
  # qp_a: x = -c / diag(P) = [-0.5, 0.5]; qp_b: x = [-t1, t0] = [2, 1].
  np.testing.assert_allclose(host(tv), [1.5, 1.5], atol=1e-7)


@pytest.mark.solver("sqp")
def test_two_sqp_wrappers_in_one_translation_unit() -> None:
  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2), name="tu_tracking")
  def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - target))

  first = sc.solver(tracking, "sqp", name="tu_sqp_a")
  second = sc.solver(tracking, "sqp", name="tu_sqp_b", options={"qp": "dense"})

  @sc.function(sc.arg("target", 2), outputs=sc.arg("sum", 2), name="two_sqp_host")
  def host(target: sc.Expr) -> sc.Expr:
    return first(target)[0] + second(2.0 * target)[0]

  target = np.array([1.0, -2.0])
  np.testing.assert_allclose(host(target), 3.0 * target, atol=2e-5, rtol=0)


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
  scaly_solver_option values[] = {{{{ "max_iter", 0, 1, 0.0, 0 }}, {{ "verbose", 0, 0, 0.0, 0 }}, {{ 0, 0, 0, 0.0, 0 }}}};
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
  from scaly.codegen.external import compile_library

  objects = [obj for build in module._library_builds for obj in compile_library(build, c_compiler())]
  obj = source.with_suffix(".o")
  includes = [flag for flag in module.link_flags if flag.startswith("-I")]
  cc = c_compiler()
  subprocess.run([*cc, "-O2", "-Wall", "-Werror", "-c", str(source), *includes, "-o", str(obj)], check=True, capture_output=True)
  subprocess.run(
    [
      *(cxx_compiler() if lang == "cpp" else cc),
      "-std=c++17" if lang == "cpp" else "-std=c11",
      "-O2",
      "-Wall",
      "-Werror",
      str(obj),
      *objects,
      str(driver),
      *module.link_flags,
      "-lm",
      "-o",
      str(executable),
    ],
    check=True,
    capture_output=True,
  )
  # Windows has no rpath, so the executable finds the solver DLLs on PATH.
  path = os.pathsep.join([*map(str, solver_paths().lib_dirs), os.environ["PATH"]])
  subprocess.run([str(executable)], check=True, capture_output=True, env={**os.environ, "PATH": path})


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize(
  "input_name,output_name",
  [("target", "scaly_solver_option"), ("scaly_solver_option", "scaly_solver_option_2")],
)
def test_typed_wrapper_parameters_do_not_hide_solver_option_type(lang: str, input_name: str, output_name: str, tmp_path: Path) -> None:
  from scaly.codegen import render_c_module

  @sc.function(sc.arg(input_name, 2), outputs=sc.arg(output_name), name="option_names")
  def host(target: sc.Expr) -> sc.Expr:
    qp = build_qp(P=sc.const(np.eye(2)), c=-target, name="typed_option_qp")
    return qp(target)[0]

  module = render_c_module(host, lang=lang, lanes=1)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  prefix = "option_names::" if lang == "cpp" else "option_names_"
  args = "x, y, scratch" if lang == "cpp" else "&x, &y, &scratch"
  options_call = "option_names::call" if lang == "cpp" else "option_names_call_with_options"
  default_call = "option_names::call" if lang == "cpp" else "option_names_call"
  driver = tmp_path / ("driver.cpp" if lang == "cpp" else "driver.c")
  driver.write_text(
    f"""#include <math.h>
#include "{module.header_name}"
int main(void) {{
  {prefix}{input_name}_t x = {{{{1, -2}}}};
  {prefix}{output_name}_t y = {{{{0}}}};
  {prefix}workspace_t scratch;
  scaly_solver_option values[] = {{{{"max_iter", 0, 100, 0.0, 0}}, {{"verbose", 0, 0, 0.0, 0}}, {{0, 0, 0, 0.0, 0}}}};
  const scaly_solver_option* options[] = {{values}};
  int err = {options_call}({args}, options);
  if (err) return err;
  if (fabs(y.data[0] - 1.0) > 1e-6 || fabs(y.data[1] + 2.0) > 1e-6) return 10;
  return {default_call}({args});
}}
"""
  )
  from scaly.codegen.external import compile_library

  objects = [obj for build in module._library_builds for obj in compile_library(build, c_compiler())]
  obj = source.with_suffix(".o")
  includes = [flag for flag in module.link_flags if flag.startswith("-I")]
  cc = c_compiler()
  subprocess.run([*cc, "-O2", "-Wall", "-Werror", "-c", str(source), *includes, "-o", str(obj)], check=True, capture_output=True)
  executable = tmp_path / "driver"
  subprocess.run(
    [
      *(cxx_compiler() if lang == "cpp" else cc),
      "-std=c++17" if lang == "cpp" else "-std=c11",
      "-O2",
      "-Wall",
      "-Werror",
      str(obj),
      *objects,
      str(driver),
      *module.link_flags,
      "-lm",
      "-o",
      str(executable),
    ],
    check=True,
    capture_output=True,
  )
  path = os.pathsep.join([*map(str, solver_paths().lib_dirs), os.environ["PATH"]])
  subprocess.run([str(executable)], check=True, capture_output=True, env={**os.environ, "PATH": path})


@pytest.mark.parametrize(
  "backend", [pytest.param("piqp", marks=pytest.mark.solver("piqp")), pytest.param("ipopt", marks=pytest.mark.solver("ipopt"))]
)
def test_solver_manifest_builds_without_backend_flags(backend: str, tmp_path: Path) -> None:
  import json
  import sys
  from scaly.codegen import write_module

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2), name="manifest_tracking")
  def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - target))

  solve = sc.solver(tracking, backend, name="manifest_solver")
  module = write_module(solve.function, tmp_path)
  manifest = json.loads((tmp_path / "manifest_solver.build.json").read_text())
  assert manifest["libraries"][0]["link"] == ["piqpc" if backend == "piqp" else "ipopt"]
  assert manifest["libraries"][0]["library_dirs"]
  cc = c_compiler()
  objects, flags, dll_dirs = [], [], []
  for index, library in enumerate(manifest["libraries"]):
    includes = ["-I" + str(tmp_path / path) for path in library["include_dirs"]]
    defines = [f"-D{name}" + (f"={value}" if value is not None else "") for name, value in library["defines"].items()]
    for i, source in enumerate(library["sources"]):
      obj = tmp_path / f"library_{index}_{i}.o"
      subprocess.run(
        [*cc, "-c", str(tmp_path / source), *includes, *defines, *library["compile_flags"], "-o", str(obj)], check=True, capture_output=True
      )
      objects.append(str(obj))
    dll_dirs.extend(library["library_dirs"])
    flags.extend("-L" + directory for directory in library["library_dirs"])
    if sys.platform != "win32":
      flags.extend("-Wl,-rpath," + directory for directory in library["library_dirs"])
    flags.extend("-l" + name for name in library["link"])
  driver = tmp_path / "main.c"
  driver.write_text(f"""#include <math.h>
#include "{module.header_name}"
int main(void) {{
  double zero[2] = {{0,0}}, empty[1] = {{0}}, target[2] = {{0.3,-0.7}}, x[2], lam[2];
  const double* arg[] = {{zero, zero, empty, empty, target}};
  double* res[] = {{x, lam, empty, empty}};
  double w[manifest_solver_SZ_W > 0 ? manifest_solver_SZ_W : 1];
  scaly_solver_stats stats;
  if (manifest_solver(arg, res, 0, w, 0)) return 1;
  if (manifest_solver_stats(&stats) || stats.status != SCALY_SOLVE_OK) return 2;
  return fabs(x[0]-target[0]) > 1e-6 || fabs(x[1]-target[1]) > 1e-6;
}}
""")
  executable = tmp_path / ("manifest.exe" if sys.platform == "win32" else "manifest")
  subprocess.run(
    [*cc, str(tmp_path / manifest["source"]), str(driver), *objects, *flags, *("-l" + name for name in manifest["link"]), "-o", str(executable)],
    check=True,
    capture_output=True,
  )
  path = os.pathsep.join([*dll_dirs, os.environ["PATH"]])
  subprocess.run([str(executable)], env={**os.environ, "PATH": path}, check=True, capture_output=True)


@pytest.mark.solver("piqp")
def test_missing_solver_library_keeps_the_solver_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.codegen import external, render_c_module

  solve = build_qp(P=np.eye(2), c=np.array([1.0, 2.0]), name="missing_solver")
  monkeypatch.setattr(external, "solver_discoverable", lambda name: False)
  for action in (lambda: render_c_module(solve.function).link_flags, lambda: solve(())):
    with pytest.raises(sc.SolverLibraryError, match="SCALY_BUILD_SOLVERS=required uv sync"):
      action()


def test_solver_system_soname_does_not_add_the_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
  from dataclasses import replace
  from scaly.solvers import paths
  from scaly.function.model import as_concrete

  discovered = paths.solver_paths()
  monkeypatch.setattr(paths, "solver_paths", lambda **kwargs: replace(discovered, loads={**discovered.loads, "piqp": "libpiqpc.so.0"}))
  solve = build_qp(P=np.eye(2), c=np.zeros(2))
  library = as_concrete(solve.function).outputs[0].attrs["callee"].library
  assert library.library_dirs == ()


@pytest.mark.solver("ipopt")
def test_solver_oracles_follow_the_module_recipe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  from scaly.codegen import render_c_module, write_module
  from scaly.codegen.external import prepare_library
  from scaly.codegen.toolchain import find_c_compiler, no_gather_flags, vector_libm_flags
  import json
  import platform

  @sc.problem(vars=sc.arg("x", 64), params=sc.arg("target", 64), name="recipe_solver")
  def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=(x.sin() * target + x * x).sum() + sc.sumsqr(x.exp() / x.exp().sum()))

  solve = sc.solver(tracking, "ipopt")
  portable = render_c_module(solve.function, lanes=1, dialect="c", frame_budget=0)
  source = next(iter(dict(portable._libraries[0].sources).values()))
  assert "__attribute__((vector_size" not in source
  assert portable.workspace_size > 0
  monkeypatch.setenv("SCALY_CC_OPT", "-O0")
  vector_libm = "glibc" if platform.machine().lower() == "x86_64" and platform.libc_ver()[0] == "glibc" else "none"
  module = render_c_module(solve.function, cpu="native", lanes=4, vector_libm=vector_libm, reciprocal=True)
  library = module._libraries[0]
  assert library.sources != portable._libraries[0].sources
  if vector_libm == "glibc":
    assert "_ZGV" in next(iter(dict(library.sources).values()))
    assert "mvec" in library.link
  generic = render_c_module(module.fun, lanes=1, dialect="c")
  assert not set(module.recipe.cpu_flags) & set(generic._libraries[0].compile_flags)
  assert "mvec" not in generic._libraries[0].link
  compiler = find_c_compiler()
  assert compiler is not None
  flags = prepare_library(library, compiler.command).flags
  for flag in (
    "-O0",
    *module.recipe.cpu_flags,
    "-fno-math-errno",
    *no_gather_flags(compiler.command),
    *vector_libm_flags(compiler.command, vector_libm),
  ):
    assert flag in flags
  write_module(solve.function, tmp_path, cpu="native", lanes=4, vector_libm=vector_libm)
  manifest = json.loads((tmp_path / "recipe_solver_ipopt.build.json").read_text())
  assert "-O0" in manifest["libraries"][0]["compile_flags"]
