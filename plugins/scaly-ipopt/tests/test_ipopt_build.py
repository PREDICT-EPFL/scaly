"""The wheel vendors the complete Windows runtime dependency chain."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def build_hook():
  spec = importlib.util.spec_from_file_location("ipopt_build", Path(__file__).parents[1] / "hatch_build.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_windows_runtime_bundles_transitive_dependencies(build_hook, tmp_path, monkeypatch):
  lib = tmp_path / "wheel"
  tools = tmp_path / "compiler"
  system = tmp_path / "Windows" / "System32"
  for directory in (lib, tools, system):
    directory.mkdir(parents=True)
  (lib / "ipopt.dll").write_bytes(b"solver")
  (system / "KERNEL32.dll").write_bytes(b"system")
  (system / "libgfortran-5.dll").write_bytes(b"unrelated installation")
  for name in ("libgfortran-5.dll", "libgcc_s_seh-1.dll"):
    (tools / name).write_bytes(name.encode())
  dependencies = {
    "ipopt.dll": ("KERNEL32.dll", "libgfortran-5.dll"),
    "libgfortran-5.dll": ("libgcc_s_seh-1.dll",),
    "libgcc_s_seh-1.dll": ("libgfortran-5.dll",),
  }
  monkeypatch.setenv("SystemRoot", str(system.parent))
  monkeypatch.setattr(build_hook, "_windows_deps", lambda path: dependencies[path.name])
  hook = SimpleNamespace(app=SimpleNamespace(display_info=lambda message: None))
  build_hook._bundle_windows_runtime(hook, lib, "ipopt.dll", tools)
  assert sorted(p.name for p in lib.iterdir()) == ["ipopt.dll", "libgcc_s_seh-1.dll", "libgfortran-5.dll"]
  for name in dependencies.keys() - {"ipopt.dll"}:
    assert (lib / name).read_bytes() == (tools / name).read_bytes()
  (tools / "libgcc_s_seh-1.dll").unlink()
  (lib / "libgcc_s_seh-1.dll").unlink()
  with pytest.raises(RuntimeError, match="libgcc_s_seh-1.dll"):
    build_hook._bundle_windows_runtime(hook, lib, "ipopt.dll", tools)


@pytest.mark.parametrize("toolchain", ["standalone", "wrong-version", "relative"])
@pytest.mark.parametrize("mode", ["auto", "required"])
def test_windows_unsuitable_toolchain_skips_only_editable_auto(build_hook, tmp_path, monkeypatch, toolchain, mode):
  compiler = tmp_path / "standalone" / "bin" / "gfortran.exe"
  compiler.parent.mkdir(parents=True)
  compiler.touch()
  messages = []
  hook = SimpleNamespace(
    root=str(Path(__file__).parents[1]),
    target_name="wheel",
    app=SimpleNamespace(display_info=messages.append),
  )
  monkeypatch.setattr(build_hook, "os", SimpleNamespace(**(vars(os) | {"name": "nt", "environ": dict(os.environ, SCALY_BUILD_SOLVERS=mode)})))
  monkeypatch.setattr(build_hook, "_find_fortran_compiler", lambda required=True: str(compiler))
  monkeypatch.setattr(build_hook, "_tool_identity", lambda cmd: cmd)
  monkeypatch.setattr(build_hook.shutil, "which", lambda cmd: str(compiler))
  monkeypatch.setattr(build_hook, "_wheel_tag", lambda: "py3-none-win_amd64")
  monkeypatch.setattr(build_hook, "_cache_root", lambda: tmp_path / "cache")
  monkeypatch.setattr(importlib.metadata, "version", lambda name: "test")

  @contextmanager
  def unlocked(*args, **kwargs):
    yield True

  monkeypatch.setattr(build_hook, "_locked", unlocked)
  if toolchain == "relative":
    monkeypatch.chdir(compiler.parent)
    monkeypatch.setattr(build_hook, "_find_fortran_compiler", lambda required=True: compiler.name)
  if toolchain == "wrong-version":
    bash = compiler.parents[2] / "usr" / "bin" / "bash.exe"
    bash.parent.mkdir(parents=True)
    bash.touch()
    monkeypatch.setattr(build_hook.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="mingw-w64-ucrt-x86_64-gcc 15.0.0-1\n"))
  if mode == "required":
    with pytest.raises(RuntimeError, match="missing native toolchain"):
      build_hook.BuildHook.initialize(hook, "editable", {})
  else:
    build_hook.BuildHook.initialize(hook, "editable", {})
    assert any("Skipping IPOPT build for editable install" in message for message in messages)
  assert not any((tmp_path / "cache").glob("build-*"))
