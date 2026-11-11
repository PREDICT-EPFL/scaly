"""The compilers tests use to build generated C by hand: the JIT's own, so every platform runs them."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scaly.codegen.toolchain import find_c_compiler


def c_compiler() -> tuple[str, ...]:
  """The JIT's C compiler command, skipping the test when there is none."""
  compiler = find_c_compiler()
  if compiler is None:
    pytest.skip("no C compiler")
  return compiler.command


def check_c_syntax(path: Path) -> subprocess.CompletedProcess[str]:
  """Run a C99 syntax check with the selected compiler."""
  command = c_compiler()
  args, source = [str(path)], None
  if Path(command[0]).stem == "zig":
    # zig cc expects an object even with -fsyntax-only: ziglang/zig#17361.
    source = subprocess.run([*command, "-std=c99", "-E", str(path)], check=True, capture_output=True, text=True).stdout
    target = subprocess.run([*command, "-dumpmachine"], check=True, capture_output=True, text=True).stdout.strip()
    command = (command[0], "clang", "-target", target)
    args = ["-x", "cpp-output", "-"]
  return subprocess.run([*command, "-std=c99", "-fsyntax-only", *args], input=source, capture_output=True, text=True)


def cxx_compiler() -> tuple[str, ...]:
  """``zig c++`` beside the JIT's zig, otherwise ``CXX`` or ``c++`` from ``PATH``."""
  cc = c_compiler()
  if Path(cc[0]).stem == "zig":
    return (cc[0], "c++")
  found = shutil.which(os.environ.get("CXX") or "c++")
  if found is None:
    pytest.skip("no C++ compiler")
  return (found,)


def named_compiler(name: str) -> tuple[str, ...]:
  """GCC or Clang from PATH, or packaged Zig; standalone Windows Clang requires the unsupported MSVC toolchain."""
  if sys.platform == "win32" and name == "clang":
    pytest.skip("Windows Clang requires the unsupported MSVC toolchain; these codegen cases also run with packaged Zig")
  if name == "zig":
    compiler = find_c_compiler()
    if compiler is None or compiler.source != "zig-bin":
      pytest.skip("the toolchain extra is not installed")
    return compiler.command
  found = shutil.which(name)
  if found is None:
    pytest.skip(f"{name} unavailable")
  return (found,)
