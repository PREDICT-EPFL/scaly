"""An extern callee that is not a solver: the compiler lowers its dependency, adds its C source and
its includes, renders its body, and the JIT compiles and runs it, all through the protocol.

The callee computes ``y = 2 g(x) + 1`` with ``g(x) = x * x`` a generated Function and the doubling
in a hand-written C source, so every hook of ``scaly.function.extern`` is on the path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import scaly as sc
from scaly.function.extern import BuildRequirements, ExternRenderCtx, ExternSource, ExternState, extern_function, extern_functions

N = 3

_DOUBLE = """static void extern_test_double_raw(const double* in0, double* out0, double* w) {
  (void)w;
  for (int i = 0; i < 3; ++i) out0[i] = 2.0 * in0[i];
}"""


@sc.function(N, output="y", name="extern_test_square")
def _square(x):
  return x * x


@dataclass(frozen=True)
class _Callee:
  """Implements the protocol structurally, as a plugin would."""

  def dependencies(self) -> tuple[sc.ConcreteFunction, ...]:
    return (_square.concrete,)

  def extern_sources(self) -> tuple[ExternSource, ...]:
    return (ExternSource("extern_test_double_raw", _DOUBLE),)

  def render(self, fun: sc.ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    return [
      f"static void {ctx.raw_symbol}(const double* in0, double* out0, double* w) {{",
      "  double tmp[3];",
      f"  {ctx.raw_symbol_of(_square)}(in0, tmp, w);",
      "  extern_test_double_raw(tmp, out0, w);",
      "  for (int i = 0; i < 3; ++i) out0[i] += EXTERN_TEST_OFFSET;",
      "}",
    ]

  def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
    return BuildRequirements(includes=("#include <string.h>",), source_blocks=(("#define EXTERN_TEST_OFFSET 1.0",),))

  def state(self, fun: sc.ConcreteFunction) -> ExternState | None:
    return None


def _extern() -> sc.ConcreteFunction:
  return extern_function("extern_test_callee", _Callee(), [("x", (N,))], [("y", (N,))])


def test_an_extern_callee_compiles_links_its_source_and_runs() -> None:
  fn = _extern()
  x = np.array([0.5, -1.0, 2.0])
  np.testing.assert_allclose(fn(x), 2.0 * x * x + 1.0)
  source = sc.codegen.render_c_source(fn)
  assert source.count("static void extern_test_double_raw(") == 1
  assert "#include <string.h>" in source and "#define EXTERN_TEST_OFFSET 1.0" in source
  assert "void extern_test_square_raw(" in source  # the dependency, lowered


def test_an_extern_callee_nests_in_a_generated_function() -> None:
  inner = _extern()

  @sc.function(N, output="z", name="extern_test_host")
  def host(x):
    return inner(x.sin()).sum() + x

  x = np.array([0.1, 0.2, 0.3])
  np.testing.assert_allclose(host(x), (2.0 * np.sin(x) ** 2 + 1.0).sum() + x)
  assert extern_functions(host.concrete) == (inner,)


def test_a_derivative_through_an_extern_callee_raises() -> None:
  inner = _extern()
  x = sc.sym("x", N)
  with pytest.raises(NotImplementedError, match="extern_test_callee.*custom_derivative"):
    sc.jacobian(inner(x), x)
