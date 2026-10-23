"""The wheel vendors the complete Windows runtime dependency chain."""

from __future__ import annotations

import importlib.util
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
