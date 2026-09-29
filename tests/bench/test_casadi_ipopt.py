from __future__ import annotations

import subprocess
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import pytest

from bench.harness.casadi_ipopt import CompiledCasadiIpopt, _transformed_nlpsol
from bench.harness.provenance import collect
from scaly_ipopt import BUILD_CONFIG

ca = pytest.importorskip("casadi")

pytestmark = pytest.mark.method("opt.ipopt")


def test_compiled_nlpsol_uses_scaly_ipopt_and_refreshes_outputs() -> None:
  x = ca.MX.sym("x", 2)
  p = ca.MX.sym("p")
  nlp = ca.Function("compiled_nlpsol_test", [x, p], [ca.sumsqr(x - p), ca.sum1(x)])
  solver = CompiledCasadiIpopt(
    "compiled_nlpsol_test_solver",
    nlp,
    {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes"},
  )
  lower, upper = np.full(2, -10.0), np.full(2, 10.0)
  zeros_x, zeros_g = np.zeros(2), np.zeros(1)

  first = solver(zeros_x, np.array([2.0]), lower, upper, np.array([1.0]), np.array([1.0]), zeros_x, zeros_g)
  second = solver(zeros_x, np.array([-1.0]), lower, upper, np.array([-1.0]), np.array([-1.0]), zeros_x, zeros_g)

  np.testing.assert_allclose(first[0], [0.5, 0.5], rtol=0.0, atol=1e-8)
  np.testing.assert_allclose(second[0], [-0.5, -0.5], rtol=0.0, atol=1e-8)
  assert second[5].size == 0
  assert solver.last_stats is not None
  assert solver.last_stats.status.value <= 1
  assert solver.last_stats.iter > 0
  assert solver.last_stats.t_total >= solver.last_stats.t_fe > 0.0
  assert solver.last_stats.t_glue > 0.0
  assert solver.last_stats.n_eval_h > 0
  assert solver.last_stats.n_eval_h == solver.last_stats.iter
  np.testing.assert_allclose(solver.last_stats.t_total, solver.last_stats.t_fe + solver.last_stats.t_solver + solver.last_stats.t_glue)
  assert min(solver.last_stats.n_eval_f, solver.last_stats.n_eval_grad_f, solver.last_stats.n_eval_g, solver.last_stats.n_eval_jac_g) > 0
  generated = (solver.library.parent / "compiled_nlpsol_test_solver.c").read_text()
  assert "/* compiled_nlpsol_test:" not in generated
  dependency_command = ["otool", "-L", str(solver.library)] if sys.platform == "darwin" else ["ldd", str(solver.library)]
  dependencies = subprocess.run(dependency_command, check=True, text=True, capture_output=True).stdout
  assert solver.ipopt_library.name in dependencies

  with pytest.raises(ValueError, match="input 0 has 1 values, expected 2"):
    solver(np.zeros(1), np.array([2.0]), lower, upper, np.array([1.0]), np.array([1.0]), zeros_x, zeros_g)


def test_compiled_nlpsol_supports_limited_memory() -> None:
  x = ca.MX.sym("x", 2)
  p = ca.MX.sym("p")
  nlp = ca.Function("limited_nlpsol_test", [x, p], [ca.sumsqr(x - p), ca.sum1(x)])
  solver = CompiledCasadiIpopt(
    "limited_nlpsol_test_solver",
    nlp,
    {"print_time": False, "ipopt.print_level": 0, "ipopt.hessian_approximation": "limited-memory"},
  )
  zeros_x, zeros_g = np.zeros(2), np.zeros(1)
  solver(zeros_x, np.array([2.0]), np.full(2, -10.0), np.full(2, 10.0), np.array([1.0]), np.array([1.0]), zeros_x, zeros_g)
  assert solver.last_stats is not None and solver.last_stats.status.value <= 1
  assert solver.last_stats.n_eval_h == 0


def test_provenance_records_the_ipopt_stack() -> None:
  provenance = collect([], compiler="cc")
  native_solvers = cast(dict[str, Any], provenance["native_solvers"])
  ipopt = cast(dict[str, Any], native_solvers["ipopt"])
  assert Path(ipopt["library"]).resolve().name.startswith("libipopt")
  assert Path(ipopt["link_library"]).resolve().name.startswith("libipopt")
  assert ipopt["build"] == BUILD_CONFIG


@pytest.mark.parametrize("order", ["wheel-first", "compiled-first"])
def test_compiled_nlpsol_coexists_with_wheel_ipopt(order: str) -> None:
  script = """
import casadi as ca
import numpy as np
import sys
from bench.harness.casadi_ipopt import CompiledCasadiIpopt
x = ca.MX.sym("x")
p = ca.MX.sym("p")
nlp = ca.Function("pin_check", [x, p], [(x - p) ** 2, ca.MX.zeros(0)])
def build_wheel():
  return ca.nlpsol("wheel_solver", "ipopt", {"x": x, "f": x * x}, {"print_time": False, "ipopt.print_level": 0})
def build_compiled():
  return CompiledCasadiIpopt("pin_check_solver", nlp, {"print_time": False, "ipopt.print_level": 0})
if sys.argv[1] == "wheel-first":
  wheel, compiled = build_wheel(), build_compiled()
else:
  compiled, wheel = build_compiled(), build_wheel()
wheel(x0=1.0)
empty = np.empty(0)
compiled(np.zeros(1), np.ones(1), np.full(1, -10.0), np.full(1, 10.0), empty, empty, np.zeros(1), empty)
"""
  subprocess.run([sys.executable, "-c", script, order], check=True, text=True, capture_output=True)


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("limited_memory", [False, True])
def test_compiled_oracles_use_transformed_values_and_derivatives(expand: bool, limited_memory: bool) -> None:
  x, p = ca.MX.sym("x", 2), ca.MX.sym("p")
  a, b = ca.sin(x[0] + p), ca.sin(x[0] + p)
  problem = {"x": x, "p": p, "f": (a + b + x[1]) ** 2, "g": ca.vertcat(a * b, x[1] ** 2)}
  options = {"print_time": False, "ipopt.print_level": 0, "expand": expand}
  if limited_memory:
    options["ipopt.hessian_approximation"] = "limited-memory"
  original = ca.nlpsol("original", "ipopt", problem, options)
  transformed = _transformed_nlpsol("transformed", problem, options)
  names = original.get_function()
  assert ("nlp_hess_l" in names) != limited_memory
  for name in names:
    before, after = original.get_function(name), transformed.get_function(name)
    assert after.serialize() == before.transform({}).serialize()
    values = [np.full(before.size_in(i), 0.4 + i) for i in range(before.n_in())]
    for expected, actual in zip(before.call(values), after.call(values), strict=True):
      np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
  assert transformed.get_function("nlp_f").n_instructions() < original.get_function("nlp_f").n_instructions()
