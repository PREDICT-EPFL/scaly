"""Signed arithmetic wraps at each dtype's limits in both lowerings, including sanitized C."""

from pathlib import Path
import subprocess
from typing import Literal

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen import CModule, render_c_module
from scaly.codegen.c import _includes, _render_raw_callee
from scaly.ir import program as p
from scaly.ir.types import CanonicalPattern
from scaly.passes.arith import constant
from scaly.passes.lowering import ctx, lower_function, main_proc
from tests.compilers import named_compiler


@pytest.mark.parametrize("dialect", ["gnu", "c"])
def test_hoisted_sparse_indices_keep_float64_c_spelling(dialect) -> None:
  n = 100
  q = sparse.diags_array([-np.ones(n - 1), 4 * np.ones(n), -np.ones(n - 1)], offsets=[-1, 0, 1], format="csc")
  coo = q.tocoo()
  pattern = CanonicalPattern(q.shape, tuple(map(int, coo.row)), tuple(map(int, coo.col)))

  @sc.function(sc.arg("q", sc.TensorType(q.shape, pattern=pattern), layout="csc"), sc.arg("x", n), name="banded")
  def loss(q, x):
    return x @ (q @ x)

  source = render_c_module(sc.sparse_hessian(loss, wrt="x"), dialect=dialect).source
  assert "uint64_t" not in source


@pytest.mark.parametrize("dtype", ["int64", "int32"])
@pytest.mark.parametrize("variant", ["block", "scalar"])
def test_signed_arithmetic_matches_numpy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, variant: Literal["block", "scalar"]) -> None:
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

  _assert_compiled_result(tmp_path, render_c_module(arithmetic), inputs, expected)
  np.testing.assert_array_equal(arithmetic(inputs), expected)


def test_constant_integer_data_keeps_wrapping_in_scheduled_assignments(tmp_path: Path) -> None:
  values = np.array([np.iinfo(np.int64).max - 1023, 0], dtype=np.int64)

  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="constant_data_int64")
  def arithmetic(x: sc.Expr) -> sc.Expr:
    value = sc.print("{}", sc.const(values)) + 1024
    return sc.cast(value * value, "float64").block()

  with np.errstate(over="ignore"):
    value = values + 1024
    expected = (value * value).astype(np.float64)
  inputs = np.zeros(1)
  _assert_compiled_result(tmp_path, render_c_module(arithmetic), inputs, expected)
  np.testing.assert_array_equal(arithmetic(inputs), expected)


@pytest.mark.parametrize("variant", ["block", "scalar"])
@pytest.mark.parametrize("axis", [0, 1])
def test_integer_axis_sums_wrap_in_strided_and_contiguous_schedules(tmp_path, variant, axis) -> None:
  info = np.iinfo(np.int64)
  values = np.tile(np.array([info.max - 1023, 1024, info.min, -1, 0], dtype=np.int64), (4, 1))
  base = 1 << (info.bits // 2)

  @sc.function(sc.arg("x", values.shape), outputs=sc.arg("y"), name=f"axis_sum_wrap_int64_{variant}_{axis}")
  def arithmetic(x):
    total = sc.cast(x, "int64").sum(axis)
    return sc.stack([sc.cast(total // base, "float64"), sc.cast(total % base, "float64")]).with_lowering(variant)

  with np.errstate(over="ignore"):
    reference = np.sum(values, axis=axis, dtype=np.int64)
  expected = np.array([reference // base, reference % base], dtype=np.float64)
  inputs = values.astype(np.float64)
  _assert_compiled_result(tmp_path, render_c_module(arithmetic), inputs, expected)
  np.testing.assert_array_equal(arithmetic(inputs), expected)


def test_nested_block_keeps_integer_wrapping(tmp_path) -> None:
  info = np.iinfo(np.int64)
  x, y = (p.buffer(name, sc.dtypes.int64, (1,)) for name in ("x", "y"))
  at = lambda buffer: p.view(buffer, [p.const_int(0)])
  body = [
    p.assign("seed", p.load(at(x)), declare=True),
    p.block(
      p.assign("incremented", p.add(p.var("seed", sc.dtypes.int64), constant(1, sc.dtypes.int64)), declare=True),
      p.store(at(y), p.var("incremented", sc.dtypes.int64)),
    ),
  ]
  proc = p.proc("block_wrap", [x, y], body)
  source = "\n".join([*_includes(dialect="c"), *_render_raw_callee(proc, dialect="c")])
  source += f"""
#include <stdio.h>
int main(void) {{
  int64_t x = {info.max}, y = 0;
  block_wrap_raw(&x, &y, NULL);
  printf("%lld\\n", (long long)y);
  return 0;
}}
"""
  path, executable = tmp_path / "block.c", tmp_path / "block"
  path.write_text(source)
  built = subprocess.run(
    [*named_compiler("gcc"), "-O2", "-fsanitize=signed-integer-overflow", "-fsanitize-undefined-trap-on-error", str(path), "-o", str(executable)],
    capture_output=True,
    text=True,
  )
  assert built.returncode == 0, built.stderr
  run = subprocess.run([str(executable)], capture_output=True, text=True)
  assert run.returncode == 0, run.stderr
  assert run.stderr == ""
  assert int(run.stdout) == info.min


def _assert_compiled_result(tmp_path: Path, module: CModule, inputs: np.ndarray, expected: np.ndarray) -> None:
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
  flags = ["-fsanitize=signed-integer-overflow", "-fsanitize-undefined-trap-on-error"]
  built = subprocess.run(
    [*named_compiler("gcc"), "-std=c99", "-O2", "-DSCALY_PRINTF(...)=", *flags, str(source), "-lm", "-o", str(executable)],
    capture_output=True,
    text=True,
  )
  assert built.returncode == 0, built.stderr
  run = subprocess.run([str(executable)], capture_output=True, text=True)
  assert run.returncode == 0, run.stderr
  assert run.stderr == ""
  np.testing.assert_array_equal(np.fromstring(run.stdout, sep="\n").reshape(expected.shape), expected)
