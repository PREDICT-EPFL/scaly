"""``sc.Target``: the presets and their derived choices, the host's detection, the target in force,
and that a render, the JIT and the CLI all follow it."""

from __future__ import annotations

import contextvars
import dataclasses
import shutil
import sys
import threading

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import aot, render_c_module, render_c_source, workspace_size
from scaly.ir import target as target_module
from scaly.ir.target import PORTABLE, PRESETS, Target, _from_facts, host_target
from scaly.passes.lowering import lower_function
from scaly.utils.env import CpuFacts, _darwin_facts, _linux_facts, _sysfs_size

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
    assert (portable.straight_line_ops, portable.body_bytes, portable.panel_bytes) == (4096, 98304, 65536)
  assert PRESETS["x86-64-v3"].choices is PRESETS["x86-64-v3"]
  assert PRESETS["x86-64-v3"].body_bytes == 16384


@pytest.mark.parametrize("m, k, n", [(None, 256, 6), (20, 12, 40), (3, 70, 130), (8, 9, 100), (20, 128, 40)])
def test_portable_rounding_renders_the_same_c_for_every_target(m, k, n) -> None:
  a, b = sc.sym("a", k) if m is None else sc.sym("a", (m, k)), sc.sym("b", (k, n))
  fn = sc.Function.from_exprs(f"portable_{m}_{k}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["c"])
  sources = {render_c_source(fn, target=dataclasses.replace(preset, rounding="portable")) for preset in PRESETS.values()}
  assert sources == {render_c_source(fn, target="apple-m3")}


def test_portable_rounding_makes_every_choice_the_reference_machines() -> None:
  """Every choice a target makes, the straight-line factorizations and the small products among
  them, renders the reference machine's C under portable rounding."""
  from scaly.linalg import cholesky, solve_triangular

  a, b, x = sc.sym("a", (20, 20)), sc.sym("b", 20), sc.sym("x", (8, 8))
  fn = sc.Function.from_exprs("portable_choices", [a, b, x], [solve_triangular(cholesky(a), b), x @ x], ["a", "b", "x"], ["y", "z"])
  sources = {render_c_source(fn, target=dataclasses.replace(preset, rounding="portable")) for preset in PRESETS.values()}
  assert sources == {render_c_source(fn, target="apple-m3")}
  assert render_c_source(fn, target="generic") != render_c_source(fn, target="apple-m3")


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
    ({"vector_bytes": 16.0}, ValueError),
    ({"rounding": "fast"}, ValueError),
    ({"l1i_bytes": 0}, ValueError),
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
    (CpuFacts("aarch64", "darwin", brand="Apple M3 Max"), "apple-m3"),
    (CpuFacts("aarch64", "darwin", brand="Apple M1"), "apple-m1"),
    (CpuFacts("aarch64", "darwin", brand="Apple M9 Ultra"), "apple-m4"),
    (CpuFacts("aarch64", "darwin"), "apple-m4"),  # sysctl did not answer: still an Apple core
    (CpuFacts("aarch64", "linux", implementer="0x61", part="0x033"), "apple-m2"),
    (CpuFacts("aarch64", "linux", implementer="0x61", part="0x999"), "apple-m4"),
    (CpuFacts("aarch64", "linux", part="0xd03"), "cortex-a53"),
    (CpuFacts("aarch64", "linux", part="0xD4F"), "neoverse-v2"),
    (CpuFacts("aarch64", "linux", part="0xd0c"), "armv8-a"),  # a Neoverse N1: no preset, no -mcpu guess
    (CpuFacts("x86_64", features=frozenset({"avx2", "fma", "avx512f", "avx512bw", "avx512dq", "avx512vl"})), "x86-64-v4"),
    (CpuFacts("x86_64", features=frozenset({"avx2", "fma", "avx512f"})), "x86-64-v3"),  # AVX-512F alone is not level 4
    (CpuFacts("x86_64", features=frozenset({"avx2", "fma"})), "x86-64-v3"),
    (CpuFacts("x86_64", features=frozenset({"avx2"})), "x86-64"),
    (CpuFacts("riscv64"), "generic"),
  ],
)
def test_the_host_is_the_preset_that_matches_it(facts, preset) -> None:
  assert _from_facts(facts) is PRESETS[preset]


def test_the_host_takes_the_caches_its_system_reports() -> None:
  host = _from_facts(CpuFacts("aarch64", "darwin", brand="Apple M3 Pro", l1i=131072))
  assert host.name == "apple-m3" and host.l1i_bytes == 131072 and host.l1d_bytes == PRESETS["apple-m3"].l1d_bytes
  assert _from_facts(CpuFacts("aarch64", "darwin", brand="Apple M3 Max", l1i=196608, l1d=131072)) is PRESETS["apple-m3"]


_SYSCTL = """machdep.cpu.brand_string: Apple M3 Max
hw.perflevel0.l1icachesize: 196608
hw.perflevel0.l1dcachesize: 131072
hw.l1icachesize: 131072
hw.l1dcachesize: 65536
hw.optional.avx2_0: 0
"""


def test_sysctl_output_parses() -> None:
  facts = _darwin_facts("aarch64", _SYSCTL)
  assert facts == CpuFacts("aarch64", "darwin", brand="Apple M3 Max", l1i=196608, l1d=131072)  # the performance cores' caches
  assert _darwin_facts("x86_64", "hw.l1dcachesize: 32768\nhw.optional.avx2_0: 1\nhw.optional.fma: 1\n").features == {"avx2", "fma"}
  assert _darwin_facts("aarch64", "").l1i is None  # a system that answers nothing


def test_linux_proc_and_sys_parse(tmp_path) -> None:
  (tmp_path / "proc").mkdir()
  (tmp_path / "proc/cpuinfo").write_text(
    "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd0b\nFeatures\t: fp asimd\n\nprocessor\t: 1\nCPU part\t: 0xd05\n"
  )
  caches = tmp_path / "sys/devices/system/cpu/cpu0/cache"
  for index, (level, kind, size) in enumerate([("1", "Data", "64K"), ("1", "Instruction", "64K"), ("2", "Unified", "512K")]):
    (caches / f"index{index}").mkdir(parents=True)
    for name, value in (("level", level), ("type", kind), ("size", size)):
      (caches / f"index{index}" / name).write_text(value + "\n")
  facts = _linux_facts("aarch64", tmp_path)
  assert (facts.implementer, facts.part, facts.l1i, facts.l1d) == ("0x41", "0xd0b", 65536, 65536)  # the first core's part
  assert _from_facts(facts) is PRESETS["cortex-a76"]
  (tmp_path / "proc/cpuinfo").write_text("model name\t: AMD EPYC\nflags\t\t: sse2 avx2 fma avx512f avx512bw avx512dq avx512vl\n")
  assert _from_facts(_linux_facts("x86_64", tmp_path)).name == "x86-64-v4"
  assert _linux_facts("x86_64", tmp_path / "missing") == CpuFacts("x86_64", "linux")


def test_this_machine_is_detected() -> None:
  host = host_target()
  assert Target.preset("host") is host
  if sys.platform == "darwin" and host.vector_bytes == 16:
    assert host.name.startswith("apple-")
    assert host.row_blocks == PORTABLE.row_blocks  # the reference machine's choices on Apple silicon


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


def test_a_block_is_local_to_its_context() -> None:
  sc.set_target("cortex-a53")
  seen: list[str] = []
  with sc.target("generic"):
    # A thread started in its own context (not inheriting this one, as free-threaded builds do)
    # sees the process default, not this block.
    worker = threading.Thread(target=lambda: contextvars.Context().run(lambda: seen.append(sc.get_target().name)))
    worker.start()
    worker.join()
  assert seen == ["cortex-a53"]


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
    assert lower_function(fn, target=name).attrs["tuned_for"] is PRESETS[name]
  with sc.target("generic"):
    assert lower_function(fn).attrs["tuned_for"] is PRESETS["generic"]
  # 40 columns: 16 + 16 + 8 on the M3, 32 + 8 on AVX2, 8 x 5 scalar
  sources = {name: render_c_source(fn, target=name) for name in ("apple-m3", "generic", "x86-64-v3")}
  assert len(set(sources.values())) == 3
  # A vector times 100 columns: past the M3's 64, so streamed there, but within AVX2's 128, so blocked there
  x, b = sc.sym("x", 12), sc.sym("b", (12, 100))
  wide = sc.Function.from_exprs("target_vm_wide", [x, b], [(x @ b).block()], ["x", "b"], ["y"])
  m3, v3 = (lower_function(wide, target=name) for name in ("apple-m3", "x86-64-v3"))
  assert _running_sums(m3) == 0
  assert _running_sums(v3) == 32  # three blocks of 32 columns sharing one set of sums; the last 4 stream


def _running_sums(prog) -> int:
  """The running sums of a lowered product's blocks kept in registers, one per column: private
  buffers of one vector's lanes (a scalar each on a target of one lane)."""
  from scaly.ir.program import ProgramOp

  seen = {id(n): n for n in _walk(prog)}
  buffers = [n for n in seen.values() if n.op == ProgramOp.BUFFER and n.attrs.get("address_space") == "private"]
  return sum(int(np.prod(n.attrs["shape"])) for n in buffers if int(np.prod(n.attrs["shape"])) <= 8)


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
@pytest.mark.parametrize("m, k, n", [(20, 12, 40), (3, 70, 130), (1, 9, 100)])
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
  fn = _matmul("target_jit", 20, 12, 40)
  a, b = np.ones((20, 12)), np.ones((12, 40))
  with sc.target("apple-m3"):
    fn((a, b))
    m3_build = fn._compiled
  with sc.target("generic"):
    fn((a, b))
    generic_build = fn._compiled
    assert generic_build.target is PRESETS["generic"] and generic_build.cache_key != m3_build.cache_key  # another C, another library
    fn((a, b))
    assert fn._compiled is generic_build
  with sc.target(dataclasses.replace(PRESETS["generic"])):  # equal but not the same object
    fn((a, b))
    assert fn._compiled is generic_build
  fn((a, b))
  assert fn._compiled.target == sc.get_target()


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_a_handle_is_built_for_the_target_it_is_given() -> None:
  from scaly.codegen.jit import get_compiled

  fn = _matmul("target_handle", 20, 12, 40).concrete
  with sc.target("generic"):
    generic_key = get_compiled(fn).cache_key
  handle = get_compiled(fn, PRESETS["generic"])  # outside the block: its own target, not the one in force
  assert handle.target is PRESETS["generic"] and handle.cache_key == generic_key


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_recompile_drops_the_library_of_the_target_it_was_built_for() -> None:
  fn = _matmul("target_recompile", 20, 12, 40)
  a, b = np.ones((20, 12)), np.ones((12, 40))
  with sc.target("generic"):
    fn((a, b))
    library = fn._compiled.lib_path
  assert library.exists()
  fn.recompile()  # outside the block: it still drops generic's library, the one it had built
  assert not library.exists() and fn._compiled is None


def test_the_workspace_follows_the_target() -> None:
  fn = _matmul("target_workspace", 20, 12, 40)
  for name in ("apple-m3", "generic"):
    assert workspace_size(fn, target=name) == render_c_module(fn, target=name).workspace_size


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
  # A module may set the process default when imported; without --target the CLI renders for it.
  (tmp_path / "scaly_target_cli_set.py").write_text("import scaly as sc\nfrom scaly_target_cli import build\n\nsc.set_target('generic')\n")
  aot.main(["scaly_target_cli_set:build", "-o", str(tmp_path / "set")])
  assert "target: generic (compile with -O2 -fno-math-errno)" in capsys.readouterr().out


def test_the_cli_builds_the_graph_under_its_target(tmp_path, monkeypatch) -> None:
  """AD reads the target while it builds a derivative (``ad.forward._seed_groups``), so the module's
  import and its factory both run under ``--target``, and the target in force is restored after."""
  (tmp_path / "scaly_target_cli_build.py").write_text(
    "import scaly as sc\n\nat_import = sc.get_target().name\n\n\ndef build():\n  global at_build\n  at_build = sc.get_target().name\n"
    "  x = sc.sym('x', 4)\n  return sc.Function.from_exprs('target_cli_build', [x], [x * x], ['x'], ['y'])\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  before = sc.get_target()
  aot.main(["scaly_target_cli_build:build", "-o", str(tmp_path / "out"), "--target", "x86-64-v3"])
  module = sys.modules["scaly_target_cli_build"]
  assert (module.at_import, module.at_build) == ("x86-64-v3", "x86-64-v3")
  assert sc.get_target() is before
