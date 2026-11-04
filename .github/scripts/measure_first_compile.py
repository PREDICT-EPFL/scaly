"""Temporary measurement for #252: what zig's first compile costs on a fresh cache. Removed before review."""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from scaly.codegen.toolchain import find_c_compiler

WIN = sys.platform == "win32"
EXT = ".dll" if WIN else ".so"
NATIVE = "-march=native"

OPS = {
  "sin": "sin(v)",
  "cos": "cos(v)",
  "exp": "exp(-v)",
  "log1p": "log1p(v * v)",
  "pow": "pow(v, 1.3)",
  "tanh": "tanh(v)",
  "atan2": "atan2(v, 2.0)",
  "fma": "fma(v, v, 1.0)",
  "sqrt": "sqrt(v)",
}
KERNEL = "#include <math.h>\n#ifdef _WIN32\n#define EXPORT __declspec(dllexport)\n#else\n#define EXPORT\n#endif\n" + "".join(
  f"EXPORT void k_{name}(const double* x, double* y, int n) {{ for (int i = 0; i < n; ++i) {{ double v = x[i]; y[i] = {expr}; }} }}\n"
  for name, expr in OPS.items()
)

PROGRAM = r"""
import time, numpy as np, scaly as sc
t = time.perf_counter()
f = sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name=NAME)(lambda x: (x.sin() * x.exp()).sum())
f(np.ones(3))
print(f"{time.perf_counter() - t:.2f}")
"""

CPP = '#include <string>\n#include <iostream>\nint main() { std::cout << std::string("warm").size() << "\\n"; return 0; }\n'


def cpu_name() -> str:
  if WIN:
    import winreg

    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
  for line in Path("/proc/cpuinfo").read_text().splitlines():
    if line.startswith("model name"):
      return line.split(":", 1)[1].strip()
  return platform.processor()


def timed(command: list[str], cwd: Path, env: dict[str, str]) -> float:
  start = time.perf_counter()
  subprocess.run(command, cwd=cwd, env=env, check=True, capture_output=True, text=True)
  return time.perf_counter() - start


def scaly_call(name: str, env: dict[str, str], cwd: Path) -> tuple[float, float]:
  start = time.perf_counter()
  out = subprocess.run([sys.executable, "-c", PROGRAM.replace("NAME", repr(name))], cwd=cwd, env=env, check=True, capture_output=True, text=True)
  wall = time.perf_counter() - start
  return wall, float(out.stdout.split()[-1])


def kernel_speed(lib: ctypes.CDLL, name: str) -> float:
  fn = getattr(lib, f"k_{name}")
  fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
  x = np.linspace(0.1, 4.0, 1 << 14)
  y = np.empty_like(x)
  best = float("inf")
  for _ in range(40):
    start = time.perf_counter()
    fn(x.ctypes.data, y.ctypes.data, x.size)
    best = min(best, time.perf_counter() - start)
  return best * 1e9 / x.size


def main() -> None:
  compiler = find_c_compiler()
  assert compiler is not None and compiler.source in ("zig-bin", "ziglang"), compiler
  zig = compiler.command[0]
  version = subprocess.run([zig, "version"], capture_output=True, text=True, check=True).stdout.strip()
  builtin = subprocess.run([zig, "build-exe", "--show-builtin"], capture_output=True, text=True, check=True).stdout
  model = next(line.split(".")[-1].rstrip(",") for line in builtin.splitlines() if ".model = " in line)
  print(f"platform={platform.system()} {platform.machine()} cpu={cpu_name()!r} zig_model={model} cores={os.cpu_count()} zig={version}")

  with tempfile.TemporaryDirectory(prefix="first-compile-", ignore_cleanup_errors=True) as root:
    root = Path(root)
    work = root / "work"
    work.mkdir()
    (work / "k.c").write_text(KERNEL)
    (work / "warm.cpp").write_text(CPP)

    def env_for(cache: str) -> dict[str, str]:
      env = dict(os.environ)
      env |= {"ZIG_GLOBAL_CACHE_DIR": str(root / cache), "ZIG_LOCAL_CACHE_DIR": str(root / cache), "SCALY_CACHE_DIR": str(root / cache / "jit")}
      return env

    rows: list[tuple[str, float]] = []
    env = env_for("zig-a")
    wall, inside = scaly_call("first", env, work)
    rows.append(("scaly first call, cold zig cache (process wall)", wall))
    rows.append(("scaly first call, cold zig cache (inside process)", inside))
    wall, inside = scaly_call("second", env, work)
    rows.append(("scaly first call of a new function, warm zig cache (process wall)", wall))
    rows.append(("scaly first call of a new function, warm zig cache (inside process)", inside))
    wall, inside = scaly_call("second", env, work)
    rows.append(("scaly call, JIT cache hit (process wall)", wall))
    cpp = [zig, "c++", "warm.cpp", "-o", "warm.exe" if WIN else "warm"]
    rows.append(("zig c++ hello, cold libc++ (C runtime warm)", timed(cpp, work, env)))
    rows.append(("zig c++ hello, warm", timed(cpp, work, env)))

    one_step = [zig, "cc", "-O2", NATIVE, "-fno-math-errno", "-fPIC", "-shared", "k.c", "-lm"]
    env = env_for("zig-b")
    rows.append(("zig cc -march=native -shared, cold", timed([*one_step, "-o", f"one{EXT}"], work, env)))
    rows.append(("zig cc -march=native -shared, warm", timed([*one_step, "-o", f"one2{EXT}"], work, env)))

    # The kernel is compiled for this CPU, the runtime linked for the generic baseline.
    env = env_for("zig-c")
    compile_native = [zig, "cc", "-O2", NATIVE, "-fno-math-errno", "-fPIC", "-c", "k.c", "-o", "k.o"]
    link_generic = [zig, "cc", "-mcpu=baseline", "-shared", "-lm"]
    rows.append(("split: compile -c -march=native, cold", timed(compile_native, work, env)))
    rows.append(("split: link -mcpu=baseline, cold", timed([*link_generic, "k.o", "-o", f"split{EXT}"], work, env)))
    rows.append(("split: compile -c -march=native, warm", timed([*compile_native[:-1], "k2.o"], work, env)))
    rows.append(("split: link -mcpu=baseline, warm", timed([*link_generic, "k2.o", "-o", f"split2{EXT}"], work, env)))
    # Another CPU model's kernel reuses the generic runtime; a one-step build for it builds a new one.
    other = "-march=skylake" if model != "skylake" else "-march=znver3"
    if platform.machine().lower() in ("amd64", "x86_64"):
      rows.append(
        (
          f"split: compile {other} + link baseline, warm runtime",
          timed([zig, "cc", "-O2", other, "-fPIC", "-c", "k.c", "-o", "k3.o"], work, env)
          + timed([*link_generic, "k3.o", "-o", f"split3{EXT}"], work, env),
        )
      )
      rows.append(
        (f"one step {other} -shared, same cache", timed([zig, "cc", "-O2", other, "-fPIC", "-shared", "k.c", "-lm", "-o", f"other{EXT}"], work, env))
      )

    width = max(len(label) for label, _ in rows)
    for label, seconds in rows:
      print(f"{label:<{width}}  {seconds:7.2f} s")
    for path in (work / "zig-a", root / "zig-a"):
      if path.exists():
        print(f"zig cache size after the scaly calls: {sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) / 1e6:.1f} MB")
    # Interleaved and repeated, so frequency ramps and noise land on both builds alike.
    libs = ctypes.CDLL(str(work / f"one{EXT}")), ctypes.CDLL(str(work / f"split{EXT}"))
    for name in OPS:
      one = split = float("inf")
      for _ in range(7):
        one, split = min(one, kernel_speed(libs[0], name)), min(split, kernel_speed(libs[1], name))
      print(f"kernel {name:<6} one step native {one:6.2f} ns, native object + generic runtime {split:6.2f} ns ({100 * (split / one - 1):+.1f}%)")
    if objdump := shutil.which("objdump"):
      for lib in ("one", "split"):
        imports = subprocess.run([objdump, "-p", str(work / f"{lib}{EXT}")], capture_output=True, text=True).stdout
        found = sorted(op for op in OPS if re.search(rf"\b{op}\b", imports))
        print(f"{lib}: math imported from shared libraries: {found}")


if __name__ == "__main__":
  main()
