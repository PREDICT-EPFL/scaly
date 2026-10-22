"""Check the Windows wheel's compiler target independently of the build host's CPU."""

from pathlib import Path
import runpy
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Zig wheel target")
@pytest.mark.parametrize("language", ["C", "CXX"])
def test_windows_wheel_keeps_haswell_target_on_avx512_host(language):
  hook = runpy.run_path(str(Path(__file__).parents[1] / "hatch_build.py"))
  default = hook["_DEFAULT_CC"] if language == "C" else hook["_DEFAULT_CXX"][0]
  command = hook["_compiler_command"](default)
  command[2:2] = ["-mcpu=skylake_avx512"]
  result = subprocess.run([*command, "-dM", "-E", "-x", "c" if language == "C" else "c++", "-"], input="", text=True, capture_output=True, check=True)
  macros = {line.split()[1] for line in result.stdout.splitlines() if line.startswith("#define ")}
  assert {"__AVX2__", "__FMA__"} <= macros
  assert not any(name.startswith("__AVX512") or name == "__AVXVNNI__" for name in macros)
