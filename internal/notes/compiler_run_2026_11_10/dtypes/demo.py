"""Section 4 demo: a float32 Function with predicates, select, cast and integer division, its typed entry
points, a Cortex-M build, and where select masks and casts stop.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/dtypes/demo.py``. The Cortex-M part needs
``zig``, ``nm`` and ``size`` and is skipped without them.
"""

import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import render_c_module, write_module

f32 = sc.TensorType((6,), dtype=sc.dtypes.float32)
i32 = sc.TensorType((6,), dtype=sc.dtypes.int32)

print("== 1. a float32 Function with where, cast, a comparison, // and % ==")


@sc.function(sc.arg("x", f32), sc.arg("k", i32), outputs=sc.group(sc.arg("y"), sc.arg("q"), sc.arg("r")))
def mixed(x, k):
  bucket = k // 3
  phase = k % 3
  y = sc.where((x > 0.5) & (phase < 2), x.sin() * sc.cast(bucket, "float32"), -x)
  return y, bucket, phase


x = np.linspace(-1.0, 2.0, 6, dtype=np.float32)
k = np.array([-7, -1, 0, 4, 5, 11], dtype=np.int32)
y, q, r = mixed(x, k)
ref = np.where((x > 0.5) & (k % 3 < 2), np.sin(x) * (k // 3).astype(np.float32), -x)
print(f"y {y.dtype} max |scaly - NumPy float32| {np.abs(y - ref).max():.1e}; k // 3 = {q} ({q.dtype}), k % 3 = {r} ({r.dtype})")
print(f"NumPy:                                               k // 3 = {k // 3}, k % 3 = {k % 3}")


def entry(fun, name):
  return re.search(rf"int {name}\((.*?)\);", render_c_module(fun).header).group(1)


@sc.function(sc.arg("x", f32), outputs=sc.arg("y"))
def clip32(x):
  return sc.where(x > 0.5, x.sin(), -x)


@sc.function(sc.arg("x", 6), outputs=sc.arg("y"))
def clip64(x):
  return sc.where(x > 0.5, x.sin(), -x)


print("float64 leaves:", entry(clip64, "clip64"))
print("float32 leaves:", entry(clip32, "clip32"))
print("mixed leaves:  ", entry(mixed, "mixed"))
body = render_c_module(clip32).body
print(
  "float32 C calls",
  sorted(set(re.findall(r"\b(sinf?|cosf?)\(", body))),
  "and writes literals like",
  sorted(set(re.findall(r"\d+\.\d+f\b", body)))[:2],
)


@sc.function(sc.arg("k", sc.TensorType((), dtype=sc.dtypes.int32)), outputs=sc.arg("y"))
def wraps(k):
  return k * k + 2147483647


with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  print(f"int32 overflow wraps as NumPy: scaly {wraps(np.int32(70000))}, NumPy {np.int32(70000) * np.int32(70000) + np.int32(2147483647)}")

print("\n== 2. the same float32 oracle, freestanding for a Cortex-M4F ==")
if all(shutil.which(tool) for tool in ("zig", "nm", "size")):
  oracle = clip32.factory("oracle", ["x"], ["y", sc.factory.SpJac("y", "x")])
  with tempfile.TemporaryDirectory() as tmp:
    module = write_module(oracle, Path(tmp), dialect="c", lanes=1, frame_budget=0)
    include = Path(__file__).resolve().parents[4] / "examples" / "embedded"
    obj = Path(tmp, "oracle.o")
    cmd = ["zig", "cc", "-target", "thumb-freestanding-eabihf", "-mcpu=cortex_m4+vfp4d16sp", "-ffreestanding", "-Os"]
    cmd += [
      "-fno-unwind-tables",
      "-fno-asynchronous-unwind-tables",
      "-DSCALY_PRINTF(...)=",
      f"-I{include}",
      "-c",
      str(Path(tmp, module.source_name)),
      "-o",
      str(obj),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    text, data, bss = map(int, subprocess.run(["size", str(obj)], check=True, capture_output=True, text=True).stdout.splitlines()[-1].split()[:3])
    undefined = subprocess.run(["nm", "--undefined-only", "--format=posix", str(obj)], check=True, capture_output=True, text=True).stdout
    print(" ".join(cmd[:8]), "...")
    print(f"flash {text + data} bytes, static RAM {data + bss} bytes, caller workspace {4 * module.workspace_size} bytes")
    print("undefined symbols:", ", ".join(sorted(line.split()[0] for line in undefined.splitlines() if line.strip())))
else:
  print("skipped: zig, nm or size is missing")

print("\n== 3. a select's condition travels with the derivative ==")


@sc.function(sc.arg("u", 2), outputs=sc.arg("y", 2))
def clip(u):
  return sc.where(u > 0.0, u, 0.0)


@sc.function(sc.arg("u", 2), outputs=sc.arg("y", 2))
def root(u):
  return u.sqrt()


def modes(build):
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("fwd"), sc.arg("rev")))
  def both(x):
    y = build(x).sum()
    return sc.jvp(y, x, sc.const(np.array([1.0, 0.0]))), sc.gradient(y, x)

  fwd, rev = both(np.array([-1.0, 4.0]))
  return f"forward d/dx0 {fwd}, reverse {rev}"


print("sqrt(where(x > 0, x, 0)) at x = [-1, 4]")
print("  inline:                     ", modes(lambda x: sc.where(x > 0.0, x, 0.0).sqrt()))
print("  select in a callee:         ", modes(lambda x: clip(x).sqrt()))
print("  sqrt in a callee:           ", modes(lambda x: root(sc.where(x > 0.0, x, 0.0))))
print("  both mapped over the entries:", modes(lambda x: sc.vmap(root, 1)(sc.vmap(clip, 1)(x.reshape((1, 2)))).vec()))
print("  through a dense product:    ", modes(lambda x: (sc.where(x > 0.0, x, 0.0) @ sc.const(np.eye(2))).sqrt()))

print("\n== 4. a structural zero with no select: #435's example ==")
C = np.array([[0.0, 1.0], [0.0, 1.0]])


def products(constant):
  @sc.function(sc.arg("x", 2), outputs=sc.group(sc.arg("value"), sc.arg("t"), sc.arg("g")))
  def f(x):
    y = (x @ constant).sqrt().sum()
    return y, sc.jvp(y, x, sc.const(np.ones(2))), sc.gradient(y, x)

  value, t, g = f(np.array([1.0, 4.0]))
  return f"value {value:.6f}, forward {t:.6f}, reverse {g}"


print("dense constant:    ", products(sc.const(C)))
print("sc.sparsify of it: ", products(sc.sparsify(sc.const(C))))
print(f"expected:           value {np.sqrt(5):.6f}, forward {1 / np.sqrt(5):.6f}, reverse {np.full(2, 0.5 / np.sqrt(5))}")

print("\n== 5. out-of-range float-to-integer casts (#365) and an integer mean (#430) ==")


@sc.function(sc.arg("x", 4), outputs=sc.arg("y", sc.TensorType((4,), dtype=sc.dtypes.int32)))
def to_int32(x):
  return sc.cast(x, "int32")


values = np.array([3e9, np.inf, np.nan, -7.9])
print(f"cast({values}, int32) = {to_int32(values)}")
try:
  sc.cast(sc.sym("m", (2, 3)), "int64").mean(axis=1)
except TypeError as e:
  print("mean of an int64 tensor:", type(e).__name__, e)
print("NumPy:", np.arange(6).reshape(2, 3).mean(axis=1).dtype)


@sc.function(sc.arg("x", (2, 3)), outputs=sc.arg("i", sc.TensorType((2,), dtype=sc.dtypes.int64)))
def row_argmax(x):
  return sc.argmax(x, axis=1)


print(f"an int64 result leaves a Function: argmax rows = {row_argmax(np.array([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]]))}")
