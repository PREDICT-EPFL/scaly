"""Call a Function and say whether this process lowered it to find its library.

Run it twice with one cache: the first run compiles, the second finds the library through the JIT
index without lowering anything, whatever its ``PYTHONHASHSEED``::

  SCALY_CACHE_DIR=/tmp/warm PYTHONHASHSEED=1 uv run tests/codegen/jit_warm_start.py
  SCALY_CACHE_DIR=/tmp/warm PYTHONHASHSEED=2 uv run tests/codegen/jit_warm_start.py

Under ``SCALY_JIT_KEY=verify`` the second run lowers too, and checks the C it renders is the C the
indexed library was built from.
"""

import json
import os

import numpy as np

import scaly as sc
import scaly.codegen.aot as aot
import scaly.codegen.jit as jit
import scaly.passes.lowering as lowering

lowered = []
lower = aot.lower_function


def counted(fun, *args, **kwargs):
  lowered.append(fun.name)
  return lower(fun, *args, **kwargs)


setattr(aot, "lower_function", counted)
setattr(lowering, "lower_function", counted)


@sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="warm_start")
def inner(x):
  return x.sin() * x


@sc.function(sc.arg("x", 3), sc.arg("p", 2), outputs=sc.arg("y"), name="warm_start_outer")
def outer(x, p):
  return (inner(x) * p[0] + p[1] * x.exp()).sum()


if os.environ.get("TEST_SOLVER"):

  @sc.problem(vars=sc.arg("x", 3), params=sc.arg("target", 3), name="warm_solver")
  def tracking(x, target):
    return sc.ProblemSpec(minimize=sc.sumsqr(x - target))

  solve = sc.solver(tracking, "piqp")
  construction = len(lowered)
  result = solve(np.array([0.5, -1.0, 2.0]))[0]
  lowered_at_call = len(lowered)
  compiled = jit.get_compiled(solve.function)
else:
  construction = len(lowered)
  result = outer(np.array([0.5, -1.0, 2.0]), np.array([2.0, 0.25]))
  lowered_at_call = len(lowered)
  compiled = jit.get_compiled(outer)
print(
  json.dumps({"lowered": lowered_at_call, "construction": construction, "result": np.asarray(result).tolist(), "library": str(compiled.lib_path)})
)
