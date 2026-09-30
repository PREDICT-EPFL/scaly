"""``sc.Target``: the presets and their derived choices, the host's detection, the target in force,
and that a render, the JIT and the CLI all follow it."""

from __future__ import annotations

import dataclasses
import shutil
import threading

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import aot, render_c_module, render_c_source
from scaly.ir import target as target_module
from scaly.ir.target import PRESETS, Target, _from_facts, host_target
from scaly.passes.lowering import lower_function
from scaly.utils.env import CpuFacts, _sysfs_size

_HAVE_CC = shutil.which("cc") is not None


@pytest.fixture(autouse=True)
def _restore_default():
  yield
  sc.set_target(None)


def _matmul(name: str, m: int, k: int, n: int) -> sc.Function:
  a, b = sc.sym("a", (m, k)), sc.sym("b", (k, n))
  return sc.Function.from_exprs(name, [a, b], [a @ b], ["a", "b"], ["c"])


def test_the_m3_derives_the_row_blocks_measured_on_it() -> None:
  m3 = Target.preset("apple-m3")
  assert m3.vector_doubles == 2
  assert m3.row_blocks == (16, 8, 4)
  assert m3.row_blocked_max == 64


def test_the_row_blocks_scale_with_the_vector_width() -> None:
  assert Target.preset("generic").row_blocks == (8, 4, 2)
  assert Target.preset("x86-64-v3").row_blocks == (32, 16, 8)
  assert Target.preset("x86-64-v4").row_blocks == (64, 32, 16)
  assert Target.preset("x86-64-v4").row_blocked_max == 256


def test_portable_rounding_makes_the_reference_machines_choices() -> None:
  for preset in PRESETS.values():
    portable = dataclasses.replace(preset, rounding="portable")
    assert portable.choices is target_module.PORTABLE
    assert portable.row_blocks == (16, 8, 4) and portable.row_blocked_max == 64
  assert PRESETS["x86-64-v3"].choices is PRESETS["x86-64-v3"]


@pytest.mark.parametrize("m, k, n", [(None, 256, 6), (20, 12, 40), (3, 70, 130), (8, 9, 100)])
def test_portable_rounding_renders_the_same_c_for_every_target(m, k, n) -> None:
  a, b = sc.sym("a", k) if m is None else sc.sym("a", (m, k)), sc.sym("b", (k, n))
  fn = sc.Function.from_exprs(f"portable_{m}_{k}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["c"])
  sources = {render_c_source(fn, target=dataclasses.replace(preset, rounding="portable")) for preset in PRESETS.values()}
  assert sources == {render_c_source(fn, target="apple-m3")}


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_portable_rounding_computes_the_same_bits_for_every_target() -> None:
  """A 256-long vector times a 256 x 6 matrix: the M3 blocks four columns and streams two, a
  256-bit target streams all six, and clang fuses the multiply-adds of one and not the other."""
  x, b = sc.sym("x", 256), sc.sym("b", (256, 6))
  fn = sc.Function.from_exprs("portable_bits", [x, b], [(x @ b).block()], ["x", "b"], ["y"])
  rng = np.random.default_rng(3)
  xv, bv = rng.standard_normal(256), rng.standard_normal((256, 6))
  with sc.target("apple-m3"):
    reference = fn((xv, bv)).copy()
  for preset in PRESETS.values():
    with sc.target(dataclasses.replace(preset, rounding="portable")):
      np.testing.assert_array_equal(fn((xv, bv)), reference)


def test_every_preset_is_valid_and_named_by_its_key() -> None:
  assert set(Target.presets()) == set(PRESETS)
  for name, preset in PRESETS.items():
    assert preset.name == name
    assert Target.preset(name) is preset
  assert Target.preset("apple-m3", rounding="portable") == dataclasses.replace(PRESETS["apple-m3"], rounding="portable")


@pytest.mark.parametrize(
  "changes, error",
  [
    ({"vector_bytes": 24}, ValueError),
    ({"rounding": "fast"}, ValueError),
    ({"arch": "riscv"}, ValueError),
    ({"fma_units": 0}, ValueError),
    ({"l1d_bytes": True}, ValueError),
    ({"cflags": ["-mcpu=native"]}, TypeError),
  ],
)
def test_an_invalid_field_is_refused(changes, error) -> None:
  with pytest.raises(error):
    Target(**changes)


def test_an_unknown_preset_names_the_known_ones() -> None:
  with pytest.raises(ValueError, match="apple-m3"):
    Target.preset("pentium-4")
  with pytest.raises(TypeError):
    sc.set_target(3)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
  "facts, preset",
  [
    (CpuFacts("aarch64", brand="Apple M3 Max"), "apple-m3"),
    (CpuFacts("aarch64", brand="Apple M1"), "apple-m1"),
    (CpuFacts("aarch64", brand="Apple M9 Ultra"), "apple-m4"),
    (CpuFacts("aarch64", part="0xd03"), "cortex-a53"),
    (CpuFacts("aarch64", part="0xD4F"), "neoverse-v2"),
    (CpuFacts("aarch64", part="0xfff"), "cortex-a76"),
    (CpuFacts("x86_64", features=frozenset({"avx2", "fma", "avx512f"})), "x86-64-v4"),
    (CpuFacts("x86_64", features=frozenset({"avx2", "fma"})), "x86-64-v3"),
    (CpuFacts("x86_64", features=frozenset({"avx2"})), "x86-64"),
    (CpuFacts("riscv64"), "generic"),
  ],
)
def test_the_host_is_the_preset_that_matches_it(facts, preset) -> None:
  assert _from_facts(facts) == PRESETS[preset]


def test_the_host_takes_the_caches_its_system_reports() -> None:
  facts = CpuFacts("aarch64", brand="Apple M3 Pro", l1i=131072, l2=None, line=128)
  host = _from_facts(facts)
  assert host.name == "apple-m3" and host.l1i_bytes == 131072
  assert host.l1d_bytes == PRESETS["apple-m3"].l1d_bytes and host.l2_bytes == PRESETS["apple-m3"].l2_bytes


def test_this_machine_is_detected_as_its_own_architecture() -> None:
  import platform

  machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
  host = host_target()
  assert host.arch == (machine if machine in ("aarch64", "x86_64") else "generic")
  assert Target.host() is host and Target.preset("host") is host


@pytest.mark.parametrize("text, size", [("48K", 49152), ("1M", 1048576), ("64", 64), ("32k\n", 32768), ("n/a", None)])
def test_linux_cache_sizes_parse(text, size) -> None:
  assert _sysfs_size(text) == size


def test_the_target_in_force_is_scoped_and_restored() -> None:
  outside = sc.get_target()
  with sc.target("x86-64-v3") as inner:
    assert inner is PRESETS["x86-64-v3"] and sc.get_target() is inner
    with sc.target(Target.preset("generic")):
      assert sc.get_target().name == "generic"
    assert sc.get_target() is inner
  assert sc.get_target() is outside
  sc.set_target("cortex-a53")
  assert sc.get_target().name == "cortex-a53"
  sc.set_target(None)
  assert sc.get_target() is outside


def test_a_block_is_local_to_its_thread() -> None:
  sc.set_target("cortex-a53")
  seen: list[str] = []
  with sc.target("generic"):
    worker = threading.Thread(target=lambda: seen.append(sc.get_target().name))
    worker.start()
    worker.join()
  assert seen == ["cortex-a53"]  # another thread sees the process default, not this block


def test_the_environment_names_the_default(monkeypatch) -> None:
  monkeypatch.setenv("SCALY_TARGET", "cortex-a76")
  monkeypatch.setattr(target_module, "_default", None)
  assert sc.get_target() is PRESETS["cortex-a76"]
  monkeypatch.setenv("SCALY_TARGET", "host")
  monkeypatch.setattr(target_module, "_default", None)
  assert sc.get_target() is host_target()


def test_lowering_records_the_target_and_follows_it() -> None:
  fn = _matmul("target_mm", 20, 12, 40)
  for name in ("apple-m3", "generic", "x86-64-v3"):
    assert lower_function(fn, target=name).attrs["target"] is PRESETS[name]
  with sc.target("generic"):
    assert lower_function(fn).attrs["target"] is PRESETS["generic"]
  # 40 columns: 16 + 16 + 8 on the M3, 32 + 8 on AVX2, 8 x 5 scalar
  sources = {name: render_c_source(fn, target=name) for name in ("apple-m3", "generic", "x86-64-v3")}
  assert len(set(sources.values())) == 3
  # 100 columns: past the M3's 64, so streamed there, but within AVX2's 128, so blocked there
  wide = _matmul("target_mm_wide", 20, 12, 100)
  m3, v3 = (lower_function(wide, target=name) for name in ("apple-m3", "x86-64-v3"))
  assert _private_scalars(m3) == 0
  assert _private_scalars(v3) == 32  # three blocks of 32 columns sharing one set of sums; the last 4 stream


def _private_scalars(prog) -> int:
  """The rank-0 private buffers of a lowered product: one per column of a block kept in registers."""
  from scaly.ir.program import ProgramOp

  seen = {id(n): n for n in _walk(prog)}
  return sum(
    1 for n in seen.values() if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private" and tuple(n.attrs["shape"]) in ((), (1,))
  )


def _walk(node):
  stack = [node]
  while stack:
    n = stack.pop()
    yield n
    stack.extend(n.args)


def test_the_module_carries_its_target_and_the_flags_for_it() -> None:
  fn = _matmul("target_flags", 4, 4, 4)
  module = render_c_module(fn, target="cortex-a76")
  assert module.target is PRESETS["cortex-a76"]
  assert module.compile_flags == ("-O2", "-mcpu=cortex-a76", "-fno-math-errno")
  assert render_c_module(fn, target="generic").compile_flags == ("-O2", "-fno-math-errno")


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
@pytest.mark.parametrize("m, k, n", [(20, 12, 40), (3, 70, 130), (1, 9, 33)])
def test_every_target_computes_the_product(m, k, n) -> None:
  """Integer-valued inputs keep every partial sum exact, so every target's blocking gives NumPy's result."""
  fn = _matmul(f"target_product_{m}_{k}_{n}", m, k, n)
  rng = np.random.default_rng(0)
  a, b = rng.integers(-8, 9, (m, k)).astype(float), rng.integers(-8, 9, (k, n)).astype(float)
  for name in (*PRESETS, "host"):
    with sc.target(name):
      np.testing.assert_array_equal(fn((a, b)), a @ b)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_the_jit_renders_again_under_another_target() -> None:
  fn = _matmul("target_jit", 6, 6, 40)
  a, b = np.ones((6, 6)), np.ones((6, 40))
  with sc.target("apple-m3"):
    fn((a, b))
    m3_build = fn._compiled
  with sc.target("generic"):
    fn((a, b))
    assert fn._compiled_for is PRESETS["generic"] and fn._compiled is not m3_build
    generic_build = fn._compiled
    fn((a, b))
    assert fn._compiled is generic_build
  with sc.target(dataclasses.replace(PRESETS["generic"])):  # equal but not the same object
    fn((a, b))
    assert fn._compiled is generic_build
  fn((a, b))
  assert fn._compiled_for == sc.get_target()


def test_the_cli_takes_a_target(tmp_path, monkeypatch, capsys) -> None:
  (tmp_path / "scaly_target_cli.py").write_text(
    "import scaly as sc\n\n\ndef build():\n  a, b = sc.sym('a', (20, 12)), sc.sym('b', (12, 40))\n"
    "  return sc.Function.from_exprs('target_cli', [a, b], [a @ b], ['a', 'b'], ['y'])\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  aot.main(["scaly_target_cli:build", "-o", str(tmp_path / "m3"), "--target", "apple-m3"])
  assert "target: apple-m3 (compile with -O2 -mcpu=apple-m3 -fno-math-errno)" in capsys.readouterr().out
  aot.main(["scaly_target_cli:build", "-o", str(tmp_path / "v3"), "--target", "x86-64-v3"])
  assert (tmp_path / "m3" / "target_cli.c").read_text() != (tmp_path / "v3" / "target_cli.c").read_text()
  with pytest.raises(SystemExit):
    aot.main(["scaly_target_cli:build", "-o", str(tmp_path), "--target", "pentium-4"])
  assert "unknown target preset 'pentium-4'" in capsys.readouterr().err
