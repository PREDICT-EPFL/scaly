# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "zig-bin==0.17.*"]
# ///
"""Compile a DLL shaped like Scaly's generated code with the zig-bin wheel, load it through ctypes, and time it.

Run with `uv run internal/notes/windows_native_2026_10_22/probe.py` on native Windows x86-64. The script
needs nothing but the wheel: the compiler runs with a PATH of only the Windows system directories.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import numpy as np

# Scaly's entry signature, GNU vector types passed by value to inline helpers, scalar libm calls in
# the lane loops, and a separate exported symbol, as in a generated module.
SOURCE = r"""
#include <math.h>
typedef double vec4 __attribute__((vector_size(32)));
typedef vec4 vec4_mem __attribute__((aligned(8), may_alias));

static inline __attribute__((always_inline)) vec4 vec4_sin(vec4 x) {
  vec4 r;
  for (int i = 0; i < 4; ++i) r[i] = sin(x[i]);
  return r;
}

static inline __attribute__((always_inline)) vec4 vec4_tanh(vec4 x) {
  vec4 r;
  for (int i = 0; i < 4; ++i) r[i] = tanh(x[i]);
  return r;
}

int probe(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw; (void)w; (void)mem;
  if (!arg || !res) return 1;
  const double* x = arg[0];
  double* y = res[0];
  long long i = 0;
  for (; i + 4 <= 1003; i += 4) {
    vec4 v = *(const vec4_mem*)&x[i];
    *(vec4_mem*)&y[i] = vec4_sin(v) * exp(x[0]) + vec4_tanh(v * v) + pow(1.0 + x[1] * x[1], 1.5);
  }
  for (; i < 1003; ++i) y[i] = sin(x[i]) * exp(x[0]) + tanh(x[i] * x[i]) + pow(1.0 + x[1] * x[1], 1.5);
  res[1][0] = 0.0;
  for (i = 0; i < 1003; ++i) res[1][0] += y[i];
  return 0;
}

int probe_lanes(void) {
#if defined(__AVX512F__)
  return 8;
#elif defined(__AVX__)
  return 4;
#else
  return 2;
#endif
}
"""

FLAGS = ["-O2", "-march=native", "-fno-math-errno", "-fPIC", "-shared"]


def imports(path: Path) -> list[str]:
  """The DLL names in a PE file's import directory."""
  data = path.read_bytes()
  pe = struct.unpack_from("<I", data, 0x3C)[0]
  sections_count, optional_size = struct.unpack_from("<H", data, pe + 6)[0], struct.unpack_from("<H", data, pe + 20)[0]
  optional = pe + 24
  directories = optional + (112 if struct.unpack_from("<H", data, optional)[0] == 0x20B else 96)
  sections = [struct.unpack_from("<IIII", data, optional + optional_size + 40 * i + 8) for i in range(sections_count)]

  def offset(rva: int) -> int:
    return next(rva - va + raw for size, va, raw_size, raw in sections if va <= rva < va + max(size, raw_size))

  names, entry = [], offset(struct.unpack_from("<I", data, directories + 8)[0])
  while name := struct.unpack_from("<I", data, entry + 12)[0]:
    start = offset(name)
    names.append(data[start : data.index(b"\0", start)].decode())
    entry += 20
  return names


def main() -> None:
  assert sys.platform == "win32", "this probe checks native Windows"
  zig = shutil.which("zig", path=sysconfig.get_path("scripts"))
  assert zig is not None, "zig-bin did not install zig into this environment"
  root = Path(tempfile.mkdtemp(prefix="scaly zig probe "))
  system = os.environ["SystemRoot"]
  base_env = {"PATH": f"{system}\\System32;{system}", "SystemRoot": system, "TEMP": str(root), "TMP": str(root)}
  (root / "probe.c").write_text(SOURCE)
  (root / "probe.def").write_text("EXPORTS\nprobe\nprobe_lanes\n")

  def compile_once(cache: Path, name: str) -> float:
    env = {**base_env, "ZIG_GLOBAL_CACHE_DIR": str(cache / "global"), "ZIG_LOCAL_CACHE_DIR": str(cache / "local")}
    command = [zig, "cc", *FLAGS, (root / "probe.c").as_posix(), (root / "probe.def").as_posix(), "-lm", "-o", (root / name).as_posix()]
    start = time.perf_counter()
    subprocess.run(command, env=env, check=True, capture_output=True)
    return time.perf_counter() - start

  cold, warm = [], []
  for run in range(5):
    cache = root / f"cache {run}"
    cold.append(compile_once(cache, f"cold{run}.dll"))
    warm.append(compile_once(cache, f"warm{run}.dll"))

  library = root / "warm0.dll"
  lib = ctypes.CDLL(str(library))
  entry = lib.probe
  pointer = ctypes.POINTER(ctypes.c_double)
  entry.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_int), pointer, ctypes.c_int]
  entry.restype = ctypes.c_int
  x = np.linspace(-3.0, 4.0, 1003)
  y, total = np.empty(1003), np.empty(1)
  args = (pointer * 1)(x.ctypes.data_as(pointer))
  results = (pointer * 2)(y.ctypes.data_as(pointer), total.ctypes.data_as(pointer))
  status = entry(args, results, None, None, 0)
  reference = np.sin(x) * np.exp(x[0]) + np.tanh(x * x) + (1.0 + x[1] * x[1]) ** 1.5
  version = subprocess.run([zig, "version"], capture_output=True, text=True, check=True).stdout.strip()
  clang = subprocess.run([zig, "cc", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
  target = subprocess.run([zig, "cc", "-dumpmachine"], capture_output=True, text=True, check=True).stdout.strip()
  report = {
    "windows": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "zig": version,
    "clang": clang,
    "target": target,
    "flags": FLAGS,
    "status": status,
    "lanes": lib.probe_lanes(),
    "max_abs_error": float(np.max(np.abs(y - reference))),
    "sum_error": float(abs(total[0] - reference.sum())),
    "imports": imports(library),
    "cold_s": [round(t, 3) for t in cold],
    "warm_s": [round(t, 3) for t in warm],
    "cold_median_s": round(statistics.median(cold), 3),
    "warm_median_s": round(statistics.median(warm), 3),
  }
  print(json.dumps(report, indent=2))
  assert status == 0 and report["max_abs_error"] < 1e-12 and report["sum_error"] < 1e-9


if __name__ == "__main__":
  main()
