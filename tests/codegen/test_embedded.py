"""Float32 oracle objects for Cortex-M, with only target libm left to link."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly import linalg
from examples.embedded_oracle import oracle
from scaly.codegen import write_module
from scaly.ir.program import ProgramOp, walk_program

INCLUDE = Path(__file__).resolve().parents[2] / "examples" / "embedded"
LIBM = {"sinf", "cosf"}


def undefined_symbols(obj: Path) -> set[str]:
  output = subprocess.run(["nm", "--undefined-only", "--format=posix", str(obj)], check=True, capture_output=True, text=True).stdout
  return {line.split()[0] for line in output.splitlines() if line.strip()}


def _check_symbols(obj: Path) -> None:
  unexpected = undefined_symbols(obj) - LIBM
  assert not unexpected, f"symbols outside target libm: {sorted(unexpected)}"


def _compile(source: Path, obj: Path, cpu: str) -> None:
  subprocess.run(
    [
      "zig",
      "cc",
      "-target",
      "thumb-freestanding-eabihf",
      f"-mcpu={cpu}+vfp4d16sp",
      "-ffreestanding",
      "-Os",
      "-fno-unwind-tables",
      "-fno-asynchronous-unwind-tables",
      "-DSCALY_PRINTF(...)=",
      f"-I{INCLUDE}",
      "-c",
      str(source),
      "-o",
      str(obj),
    ],
    check=True,
    capture_output=True,
    text=True,
  )


def oracle_reference(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The embedded oracle's residuals and dense Jacobian, computed with NumPy."""
  pairs = z.reshape(4, 2)
  expected = np.stack([np.sin(pairs[:, 0]) * pairs[:, 1], pairs[:, 0] + pairs[:, 1] ** 2], axis=1).ravel()
  dense = np.zeros((8, 8), dtype=z.dtype)
  for i, (a, b) in enumerate(pairs):
    dense[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[np.cos(a) * b, np.sin(a)], [1.0, 2.0 * b]]
  return expected, dense


def test_embedded_values_and_jacobian_match_numpy() -> None:
  z = np.linspace(-0.7, 0.8, 8, dtype=np.float32)
  values, jac = oracle(z)
  expected, dense = oracle_reference(z)
  pattern = oracle.sparsity(of="spjac_residuals_zs")
  assert pattern is not None
  assert values.dtype == jac.dtype == np.float32
  np.testing.assert_allclose(values, expected, rtol=2e-6, atol=1e-7)
  np.testing.assert_allclose(jac.toarray(), dense, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("cpu", ["cortex_m4", "cortex_m7"])
def test_cortex_m_oracle_has_only_libm_symbols(tmp_path, cpu) -> None:
  if sys.platform != "linux" or not all(shutil.which(tool) for tool in ("zig", "nm", "size", "readelf")):
    pytest.skip("the Cortex-M check needs Linux, Zig and GNU binutils")
  module = write_module(oracle, tmp_path, dialect="c", lanes=1, frame_budget=0)
  assert module.workspace_size > 0
  assert "void embedded_oracle(const embedded_oracle_zs_t* zs, embedded_oracle_residuals_t* residuals," in module.header
  assert "embedded_oracle_workspace_t* w);" in module.header
  assert module.frame_bytes == 0
  local = [
    n
    for n in walk_program(module.program)
    if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private" and "workspace_offset" not in n.attrs and "alias_of" not in n.attrs
  ]
  assert not local, "zero frame budget must spill this oracle's float32 arrays"
  obj = tmp_path / "oracle.o"
  _compile(tmp_path / module.source_name, obj, cpu)
  attributes = subprocess.run(["readelf", "-A", str(obj)], check=True, capture_output=True, text=True).stdout
  assert "Tag_CPU_arch: v7E-M" in attributes
  assert "Tag_ABI_VFP_args: VFP registers" in attributes
  assert "Tag_ABI_HardFP_use: SP only" in attributes
  _check_symbols(obj)
  assert undefined_symbols(obj) == LIBM

  @sc.function(sc.arg("x", sc.TensorType((), dtype=sc.dtypes.float32)), outputs=sc.arg("y"))
  def diagnostic(x: sc.Expr) -> sc.Expr:
    return sc.where(sc.print("positive={}", x > 0.0), x, -x)

  printed = write_module(diagnostic, tmp_path, dialect="c", lanes=1, frame_budget=0)
  assert "SCALY_PRINTF(" in printed.source
  print_obj = tmp_path / "diagnostic.o"
  _compile(tmp_path / printed.source_name, print_obj, cpu)
  _check_symbols(print_obj)
  sizes = subprocess.run(["size", str(obj)], check=True, capture_output=True, text=True).stdout
  text, data, bss = map(int, sizes.splitlines()[-1].split()[:3])
  inputs = sum(e.type.storage_size * e.type.dtype.itemsize for e in module.fun.inputs)
  outputs = sum(e.type.storage_size * e.type.dtype.itemsize for e in module.fun.outputs)
  report = (
    f"## {cpu}\n\n```text\n{sizes}\n"
    f"Flash: {text + data} bytes; static RAM: {data + bss} bytes\n"
    f"Caller workspace: {module.workspace_size} bytes; inputs: {inputs} bytes; outputs: {outputs} bytes\n"
    f"Undefined symbols: {', '.join(sorted(undefined_symbols(obj)))}\n"
    "```\n\n"
    "Object text includes code and read-only data. Flash is text + data, static RAM is data + bss. "
    "Caller buffers are additional RAM. Target libm, startup code and application stack are excluded.\n"
  )
  (tmp_path / "report.txt").write_text(report)
  print(report)

  # A real extra object dependency proves that the same gate rejects non-libm symbols.
  bad_source, bad_obj = tmp_path / "needs_heap.c", tmp_path / "needs_heap.o"
  bad_source.write_text(module.source + "\nextern void* malloc(unsigned);\nvoid* needs_heap(unsigned n) { return malloc(n); }\n")
  _compile(bad_source, bad_obj, cpu)
  with pytest.raises(AssertionError, match="symbols outside target libm.*malloc"):
    _check_symbols(bad_obj)


F32 = sc.dtypes.float32


@sc.function(sc.arg("K", sc.TensorType((4, 4), F32)), sc.arg("b", sc.TensorType((4,), F32)), name="embedded_solve")
def embedded_solve(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  return linalg.solve(K, b, assume="pos")


def test_float32_solve_matches_numpy() -> None:
  rng = np.random.default_rng(0)
  m = rng.normal(size=(4, 4))
  k, b = (m @ m.T + 4 * np.eye(4)).astype(np.float32), rng.normal(size=4).astype(np.float32)
  x, status = embedded_solve(k, b)
  assert x.dtype == np.float32 and status.dtype == np.int32 and status == 0
  np.testing.assert_allclose(x, np.linalg.solve(k.astype(np.float64), b), rtol=1e-5)


# An arrow matrix, its hub last, as a constraint couples the variables of a small QP: the sparse
# factorization's loops, the work column and the analysis' tables all reach the generated C.
_ARROW = sc.SparsityPattern.from_mask(np.tril(np.eye(5, dtype=bool) | (np.arange(5) == 4)[:, None])).canonicalize()[0]


def _embedded_sparse_solve(assume: str) -> sc.Function:
  @sc.function(sc.arg("K", sc.TensorType((5, 5), F32, pattern=_ARROW)), sc.arg("b", sc.TensorType((5,), F32)), name=f"embedded_sparse_solve_{assume}")
  def solve(K: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return linalg.solve(K, b, assume=assume)

  return solve


@pytest.mark.parametrize("solve", ["dense", "sparse pos", "sparse sym"])
@pytest.mark.parametrize("cpu", ["cortex_m4", "cortex_m7"])
def test_cortex_m_float32_solve_links_no_double_precision_helper(tmp_path, cpu, solve) -> None:
  if sys.platform != "linux" or not all(shutil.which(tool) for tool in ("zig", "nm")):
    pytest.skip("the Cortex-M check needs Linux, Zig and GNU binutils")
  function = embedded_solve if solve == "dense" else _embedded_sparse_solve(solve.split()[1])
  module = write_module(function, tmp_path, dialect="c", lanes=1, frame_budget=0)
  assert "double" not in module.source
  obj = tmp_path / "solve.o"
  _compile(tmp_path / module.source_name, obj, cpu)
  assert undefined_symbols(obj) <= {"sqrtf", "fabsf"}, f"symbols outside target libm: {sorted(undefined_symbols(obj))}"
