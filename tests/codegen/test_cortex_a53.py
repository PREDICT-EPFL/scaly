"""Generated C built for Cortex-A53, run under qemu-aarch64 and compared with NumPy."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from examples.embedded_oracle import oracle
from scaly.codegen import write_module
from scaly.ir.types import CanonicalPattern
from tests.codegen.test_embedded import INCLUDE, oracle_reference, undefined_symbols

PATTERN = CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (1, 3, 0, 1, 2))
MATRIX = sc.TensorType((3, 4), pattern=PATTERN)
EM_AARCH64 = 183


@sc.function(
  sc.arg("a", MATRIX, layout="csc"), sc.arg("x", 4), outputs=sc.group(sc.arg("y"), sc.arg("b", MATRIX, layout="csr")), name="sparse_kernel"
)
def sparse_kernel(a: sc.Expr, x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return a @ x.cos(), a.sin() * 2.0


@dataclass(frozen=True)
class Case:
  """A generated function, its inputs and NumPy outputs, and the libm it may leave undefined."""

  function: Any
  ctype: str
  inputs: dict[str, tuple[str, np.ndarray]]
  outputs: dict[str, tuple[str, np.ndarray]]
  rtol: float
  atol: float
  libm: dict[str, str]
  include: Path | None = None


def _oracle_case() -> Case:
  z = np.linspace(-0.7, 0.8, 8, dtype=np.float32)
  values, jac = oracle_reference(z)
  return Case(
    oracle,
    "float",
    {"zs": ("dense", z)},
    {"residuals": ("dense", values), "spjac_residuals_zs": ("csc", jac)},
    rtol=2e-6,
    atol=1e-7,
    libm={"sinf": "float sinf(float);", "cosf": "float cosf(float);"},
    include=INCLUDE,
  )


def _sparse_case() -> Case:
  a = sparse.coo_array((np.array([0.3, -1.2, 0.7, 2.0, -0.4]), (PATTERN.rows, PATTERN.cols)), shape=PATTERN.shape).toarray()
  x = np.array([0.1, -0.5, 0.9, 1.3])
  return Case(
    sparse_kernel,
    "double",
    {"a": ("csc", a), "x": ("dense", x)},
    {"y": ("dense", a @ np.cos(x)), "b": ("csr", np.sin(a) * 2.0)},
    rtol=1e-13,
    atol=1e-15,
    libm={"sin": "double sin(double);", "cos": "double cos(double);"},
  )


CASES = {"embedded_oracle": _oracle_case, "sparse_kernel": _sparse_case}


def _require(*tools: str) -> None:
  missing = [tool for tool in tools if shutil.which(tool) is None]
  if sys.platform != "linux" or missing:
    pytest.skip(f"the Cortex-A53 check needs Linux and {', '.join(tools)}")


def _qemu() -> str:
  for name in ("qemu-aarch64", "qemu-aarch64-static"):
    if (found := shutil.which(name)) is not None:
      return found
  pytest.skip("the Cortex-A53 check needs qemu-aarch64")


def _c_array(values: np.ndarray) -> str:
  return "{" + ", ".join(f"{float(v)!r}" for v in values.reshape(-1)) + "}"


def _loop(prefix: str, name: str, format_: str, body: str) -> str:
  """Visit every stored entry of a sparse leaf as ``(i, j, k)`` through the header's published tables."""
  ptr, ind = ("csc_col_ptr", "csc_row_ind") if format_ == "csc" else ("csr_row_ptr", "csr_col_ind")
  outer, inner = ("j", "i") if format_ == "csc" else ("i", "j")
  count = f"{prefix}_{name}_NCOL" if format_ == "csc" else f"{prefix}_{name}_NROW"
  return (
    f"  for (int {outer} = 0; {outer} < {count}; ++{outer})\n"
    f"    for (int k = {prefix}_{name}_{ptr}[{outer}]; k < {prefix}_{name}_{ptr}[{outer} + 1]; ++k) {{\n"
    f"      int {inner} = {prefix}_{name}_{ind}[k];\n      {body}\n    }}\n"
  )


def _main(case: Case, prefix: str) -> str:
  """A driver that fills the inputs, calls the typed entry once and prints each output entry as ``name [i] j value``."""
  digits = 9 if case.ctype == "float" else 17
  lines = ["#include <stdio.h>", f'#include "{prefix}.h"', "int main(int argc, char** argv) {", "  (void)argv;"]
  for name, (format_, values) in case.inputs.items():
    if format_ == "dense":
      lines.append(f"  {prefix}_{name}_t {name} = {{{_c_array(values)}}};")
    else:
      rows, cols = values.shape
      lines.append(f"  static const {case.ctype} {name}_dense[{rows}][{cols}] = {{{', '.join(_c_array(row) for row in values)}}};")
      lines.append(f"  {prefix}_{name}_t {name};")
      lines.append(_loop(prefix, name, format_, f"{name}.data[k] = {name}_dense[i][j];"))
  lines += [f"  {prefix}_{name}_t {name} = {{{{0}}}};" for name in case.outputs]
  lines.append(f"  static {prefix}_workspace_t workspace;")
  arguments = ", ".join(f"&{name}" for name in (*case.inputs, *case.outputs))
  lines.append(f"  int status = {prefix}_call({arguments}, &workspace);\n  if (status) return status;")
  # Any command-line argument perturbs one generated value, to prove the comparison can fail.
  lines.append(f"  if (argc > 1) {next(iter(case.outputs))}.data[0] *= ({case.ctype})1.001;")
  for name, (format_, values) in case.outputs.items():
    if format_ == "dense":
      lines.append(f'  for (int k = 0; k < {values.size}; ++k) printf("{name} %d %.{digits}g\\n", k, (double){name}.data[k]);')
    else:
      lines.append(_loop(prefix, name, format_, f'printf("{name} %d %d %.{digits}g\\n", i, j, (double){name}.data[k]);'))
  lines.append("  return 0;\n}\n")
  return "\n".join(lines)


def _parse(stdout: str, case: Case) -> dict[str, np.ndarray]:
  results = {name: np.zeros_like(values, dtype=np.float64) for name, (_, values) in case.outputs.items()}
  seen = {name: 0 for name in case.outputs}
  for line in stdout.splitlines():
    name, *index, value = line.split()
    results[name][tuple(map(int, index))] = float(value)
    seen[name] += 1
  # Every reference entry is nonzero, so a sparse output prints one line per entry of the dense reference.
  assert seen == {name: values.size if format_ == "dense" else np.count_nonzero(values) for name, (format_, values) in case.outputs.items()}
  return results


def _check(results: dict[str, np.ndarray], case: Case) -> None:
  for name, (_, expected) in case.outputs.items():
    np.testing.assert_allclose(results[name], expected, rtol=case.rtol, atol=case.atol, err_msg=name)


def _write(case: Case, tmp_path: Path):
  return write_module(case.function, tmp_path, dialect="c", lanes=1, frame_budget=0)


@pytest.mark.parametrize("name", CASES)
def test_cortex_a53_executable_matches_numpy_under_qemu(tmp_path, name) -> None:
  _require("zig")
  qemu = _qemu()
  case = CASES[name]()
  module = _write(case, tmp_path)
  prefix = module.header_name.removesuffix(".h")
  assert f"const {case.ctype}** arg" in module.header
  main, exe = tmp_path / "main.c", tmp_path / "main"
  main.write_text(_main(case, prefix))
  command = ["zig", "cc", "-target", "aarch64-linux-musl", "-mcpu=cortex_a53", "-O2", "-std=c11", "-static"]
  subprocess.run([*command, str(main), str(tmp_path / module.source_name), "-lm", "-o", str(exe)], check=True, capture_output=True, text=True)
  assert int.from_bytes(exe.read_bytes()[18:20], "little") == EM_AARCH64
  run = lambda *args: subprocess.run([qemu, "-cpu", "cortex-a53", str(exe), *args], check=True, capture_output=True, text=True, timeout=60).stdout
  _check(_parse(run(), case), case)
  with pytest.raises(AssertionError, match="Not equal to tolerance"):
    _check(_parse(run("perturb"), case), case)


@pytest.mark.parametrize("name", CASES)
def test_cortex_a53_freestanding_object_has_only_libm_symbols(tmp_path, name) -> None:
  _require("zig", "nm")
  case = CASES[name]()
  module = _write(case, tmp_path)
  include = case.include
  if include is None:
    include = tmp_path / "libm"
    include.mkdir()
    (include / "math.h").write_text("#pragma once\n" + "\n".join(case.libm.values()) + "\n")
  obj = tmp_path / "kernel.o"
  command = ["zig", "cc", "-target", "aarch64-freestanding", "-mcpu=cortex_a53", "-ffreestanding", "-Os"]
  flags = ["-fno-unwind-tables", "-fno-asynchronous-unwind-tables", f"-I{include}", "-c"]
  subprocess.run([*command, *flags, str(tmp_path / module.source_name), "-o", str(obj)], check=True, capture_output=True, text=True)
  assert int.from_bytes(obj.read_bytes()[18:20], "little") == EM_AARCH64
  assert undefined_symbols(obj) == set(case.libm)
