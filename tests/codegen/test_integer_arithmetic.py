"""Signed arithmetic wraps at each dtype's limits in both lowerings, including sanitized C."""

from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import CModule, render_c_module
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.lowering import ctx
from tests.compilers import c_compiler, named_compiler


@pytest.mark.parametrize("dtype", ["int64", "int32"])
@pytest.mark.parametrize("variant", ["block", "scalar"])
@pytest.mark.parametrize("sanitize", [False, True])
def test_signed_arithmetic_matches_numpy(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, variant: Literal["block", "scalar"], sanitize: bool
) -> None:
  command = named_compiler("gcc") if sanitize else c_compiler()
  # #55 owns enabling narrower scalar stores generally. Exercise this renderer path now.
  if dtype == "int32" and variant == "scalar":
    monkeypatch.setattr(ctx, "_EXACT_STORES", (*ctx._EXACT_STORES, sc.dtypes.int32))
  info = np.iinfo(dtype)
  edge = np.array([info.min, info.min + 1, -(1 << (info.bits // 2)), -2, -1, 0, 1, 2, 1 << (info.bits // 2), info.max - 1, info.max], dtype=dtype)
  a, b = (v.reshape(-1) for v in np.meshgrid(edge, edge, indexing="ij"))
  # Split before converting to double, so INT64_MAX reaches the cast without rounding up to 2**63.
  inputs = np.array([a // 1024, a % 1024, b // 1024, b % 1024], dtype=np.float64)
  base = 1 << (info.bits // 2)

  @sc.function(sc.arg("x", inputs.shape), outputs=sc.arg("out"), name=f"signed_{dtype}_{variant}")
  def arithmetic(x: sc.Expr) -> sc.Expr:
    i = sc.cast(x[0], dtype) * 1024 + sc.cast(x[1], dtype)
    j = sc.cast(x[2], dtype) * 1024 + sc.cast(x[3], dtype)
    values = (i + j, i - j, i * j, -i)
    # Two exactly representable halves check every bit through the current float64 entry signature.
    halves = [sc.cast(part, "float64") for value in values for part in (value // base, value % base)]
    return sc.stack([*halves, sc.cast(i + 1 > i, "float64")]).with_lowering(variant)

  proc = main_proc(lower_function(arithmetic.instantiate()))
  assert bool(proc.attrs.get("scalarized")) == (variant == "scalar")
  with np.errstate(over="ignore"):
    values = (a + b, a - b, a * b, -a)
    expected = np.array([*(part for value in values for part in (value // base, value % base)), (a + 1 > a)], dtype=np.float64)

  _assert_compiled_result(tmp_path, render_c_module(arithmetic), inputs, expected, command, sanitize)
  np.testing.assert_array_equal(arithmetic(inputs), expected)


@pytest.mark.parametrize("dtype", ["int64", "int32"])
def test_constant_integer_data_keeps_wrapping_in_scheduled_assignments(tmp_path: Path, dtype: str) -> None:
  values = np.array([np.iinfo(dtype).max - 1023, 0], dtype=dtype)

  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name=f"constant_data_{dtype}")
  def arithmetic(x: sc.Expr) -> sc.Expr:
    value = sc.print("{}", sc.const(values)) + 1024
    return sc.cast(value * value, "float64").block()

  with np.errstate(over="ignore"):
    value = values + 1024
    expected = (value * value).astype(np.float64)
  inputs = np.zeros(1)
  _assert_compiled_result(tmp_path, render_c_module(arithmetic), inputs, expected, named_compiler("gcc"), True)
  np.testing.assert_array_equal(arithmetic(inputs), expected)


def _assert_compiled_result(
  tmp_path: Path, module: CModule, inputs: np.ndarray, expected: np.ndarray, command: tuple[str, ...], sanitize: bool
) -> None:
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  size = expected.size
  source.write_text(
    module.source
    + f"""
#include <stdio.h>
int main(void) {{
  double x[] = {{{", ".join(repr(v) for v in inputs.reshape(-1).tolist())}}};
  double out[{size}], w[{max(1, module.workspace_size)}];
  const double *arg[] = {{x}};
  double *res[] = {{out}};
  int status = {module.fun.name}(arg, res, NULL, w, 0);
  if (status) return status;
  for (int k = 0; k < {size}; ++k) printf("%.17g\\n", out[k]);
  return 0;
}}
"""
  )
  executable = tmp_path / "arithmetic"
  flags = ["-fsanitize=signed-integer-overflow", "-fno-sanitize-recover=signed-integer-overflow"] if sanitize else []
  built = subprocess.run(
    [*command, "-std=c99", "-O2", "-DSCALY_PRINTF(...)=", *flags, str(source), "-lm", "-o", str(executable)], capture_output=True, text=True
  )
  assert built.returncode == 0, built.stderr
  run = subprocess.run([str(executable)], capture_output=True, text=True)
  assert run.returncode == 0, run.stderr
  assert run.stderr == ""
  np.testing.assert_array_equal(np.fromstring(run.stdout, sep="\n").reshape(expected.shape), expected)
