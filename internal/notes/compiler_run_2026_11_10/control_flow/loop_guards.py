"""Which pass guards of #29 real LOOP programs reach: remove each guard in a scratch copy of the source and count wrong results.

Every program is built with ``scan`` or ``while_loop``, rendered under lanes 1, 2 and auto with and without reciprocal
hoisting, compiled, and compared with a NumPy loop. A program of five steps must also keep its loop. Run from a checkout with
``uv run internal/notes/compiler_run_2026_11_10/control_flow/loop_guards.py``; ``--one`` runs the programs once against the
source on ``PYTHONPATH``.
"""

import ctypes
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

# Each guard as (file under src/scaly/passes/program, the guarded text, the text without the guard).
GUARDS = {
  "scalarize.work": ("scalarize.py", 'if rng.attrs.get("carried") or any(', "if any("),
  "unroll_unit_loops exit loops": ("unroll_unit_loops.py", 'if stmt.op != ProgramOp.FOR or stmt.attrs.get("exit_var"):', "if stmt.op != ProgramOp.FOR:"),
  "widen_ranges carried range": ("widen_ranges.py", 'if rng.attrs.get("carried") or count is None', "if count is None"),
  "widen_ranges range holding one": ("widen_ranges.py", '        or (n.op == ProgramOp.RANGE and n.attrs.get("carried"))\n', ""),
  "fuse_ranges": ("fuse_ranges.py", 'if rng.attrs.get("carried") or trip_count(rng) is None', "if trip_count(rng) is None"),
  "hoist_invariant loop attributes": ("hoist_invariant.py", "_call(hoisted, [*c.args[:n_in], *bufs, *c.args[n_in:]])), stmt.attrs,", '_call(hoisted, [*c.args[:n_in], *bufs, *c.args[n_in:]])), {"body_len": 1},'),
  "hoist_invariant carried work": ("hoist_invariant.py", "i not in opaque_calls | carried_work", "i not in opaque_calls"),
  "inline_producer": ("_common.py", '  if rng.attrs.get("carried"):\n    return None\n  if len(body) != 1', "  if len(body) != 1"),
  "fold_arith index splitting": ("fold_arith.py", '            and not rng.attrs.get("carried")\n', ""),
  "combine_scatter_sums": ("combine_scatter_sums.py", '    if scatter.args[0].attrs.get("carried"):\n      continue\n', ""),
  "pack_masks": ("pack_masks.py", '  if rng.attrs.get("carried"):\n    return None\n  start, stop, step', "  start, stop, step"),
}


def programs():
  import numpy as np

  import scaly as sc

  i64, flag = sc.TensorType((), sc.dtypes.int64), sc.TensorType((), sc.dtypes.bool_)

  @sc.function(sc.arg("z", 4), sc.arg("u", ()), outputs=sc.group(sc.arg("zn", 4), sc.arg("cost", ())))
  def rk4(z, u):
    def f(s):
      return sc.stack([s[2], s[3], u - 0.1 * s[2], -9.81 * s[1].sin() - u * s[1].cos()])

    k1 = f(z)
    k2 = f(z + 0.05 * k1)
    k3 = f(z + 0.05 * k2)
    k4 = f(z + 0.1 * k3)
    return z + (0.1 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), sc.sumsqr(z) + 0.01 * u * u

  def rk4_np(z, us):
    def f(s, u):
      return np.array([s[2], s[3], u - 0.1 * s[2], -9.81 * np.sin(s[1]) - u * np.cos(s[1])])

    costs = []
    for u in us:
      costs.append(z @ z + 0.01 * u * u)
      k1 = f(z, u)
      k2 = f(z + 0.05 * k1, u)
      k3 = f(z + 0.05 * k2, u)
      z = z + (0.1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + f(z + 0.1 * k3, u))
    return [z, np.array(costs)]

  @sc.function(sc.arg("c", 3), sc.arg("w", 3), sc.arg("x", ()), outputs=sc.group(sc.arg("cn", 3), sc.arg("y", ())))
  def lin(c, w, x):
    return c / 2.0 + w * w * x, c[0] * x

  def lin_np(c, w, xs, steps):
    ys = np.zeros(len(xs))
    for k in steps:
      ys[k] = c[0] * xs[k]
      c = c / 2.0 + w * w * xs[k]
    return [c, ys]

  @sc.function(sc.arg("c", 3), sc.arg("k", i64), sc.arg("w", 3), outputs=sc.arg("cn", 3))
  def counted(c, k, w):
    return c * 0.5 + w * k.cast(sc.dtypes.float64)

  @sc.function(sc.arg("c", 3), sc.arg("k", i64), sc.arg("w", 3), outputs=sc.arg("go", flag))
  def small(c, k, w):
    return sc.sumsqr(c) < 40.0

  def while_np(c, w, n):
    k = 0
    while k < n and c @ c < 40.0:
      c, k = c * 0.5 + w * k, k + 1
    return [c, np.array(float(k))]

  @sc.function(sc.arg("x", ()), sc.arg("w", 3), outputs=sc.arg("y", ()))
  def holds_scan(x, w):
    (c, (ys,)) = sc.scan(lin, sc.stack([x, x, x]), (sc.broadcast(w), sc.const(np.linspace(0.3, 1.1, 4))), length=4)
    return c.sum() + ys.sum()

  def rollout(n, hinted):
    @sc.function(sc.arg("z0", 4), sc.arg("us", n), name=f"rollout_{n}_{hinted.__name__}")
    def f(z0, us):
      zn, (costs,) = sc.scan(rk4, z0, (us,), length=n)
      return hinted(zn), costs

    return f

  def scanned(n, hinted, reverse, trip):
    @sc.function(sc.arg("c0", 3), sc.arg("w", 3), sc.arg("xs", n), name=f"lin_{n}_{hinted.__name__}_{reverse:d}_{trip}")
    def f(c0, w, xs):
      c, (ys,) = sc.scan(lin, c0, (sc.broadcast(w), xs), length=n, reverse=reverse, trip=trip)
      return hinted(c), ys

    return f

  def solved(n, hinted):
    @sc.function(sc.arg("c0", 3), sc.arg("w", 3), name=f"while_{n}_{hinted.__name__}")
    def f(c0, w):
      c, count = sc.while_loop(small, counted, c0, max_iter=n, params=(w,), index=True)
      return hinted(c), count.cast(sc.dtypes.float64)

    return f

  def auto(e):
    return e

  def scalar(e):
    return e.with_lowering("scalar")

  out = []
  for n, hinted in itertools.product((0, 1, 5), (auto, scalar)):
    us = np.sin(np.arange(n) * 0.7)
    z0 = np.array([0.1, 0.4, -0.2, 0.05])
    out.append((rollout(n, hinted), (z0, us), rk4_np(z0, us)))
    xs, w, c0 = np.linspace(0.3, 1.7, max(n, 1))[:n], np.array([0.7, -1.3, 0.2]), np.array([1.0, -2.0, 0.5])
    for reverse, trip in ((False, None), (True, None), (False, 3), (True, 3)):
      bound = n if trip is None else min(trip, n)
      out.append((scanned(n, hinted, reverse, trip), (c0, w, xs), lin_np(c0, w, xs, range(bound)[::-1] if reverse else range(bound))))
    out.append((solved(n, hinted), (c0, w), while_np(c0, w, n)))

  @sc.function(sc.arg("x", ()), sc.arg("w", 3), outputs=sc.group(sc.arg("y", ()), sc.arg("count", i64)))
  def holds_invariant_while(x, w):
    c, count = sc.while_loop(small, counted, w, max_iter=6, params=(w,), index=True)
    return x + c.sum(), count

  @sc.function(sc.arg("xs", 5), sc.arg("w", 3), name="map_of_invariant_while")
  def map_of_invariant_while(xs, w):
    y, count = sc.vmap(holds_invariant_while, 5)(xs, sc.broadcast(w))
    return y, count.cast(sc.dtypes.float64)

  c, count = while_np(w := np.array([0.7, -1.3, 0.2]), w, 6)
  out.append((map_of_invariant_while, (np.linspace(-1, 1, 5), w), [np.linspace(-1, 1, 5) + c.sum(), np.full(5, count)]))

  @sc.function(sc.arg("xs", 5), sc.arg("w", 3), name="map_of_scan")
  def map_of_scan(xs, w):
    return sc.vmap(holds_scan, 5)(xs, sc.broadcast(w))

  w = np.array([0.7, -1.3, 0.2])
  ref = []
  for x in np.linspace(-1, 1, 5):
    c, ys = lin_np(np.full(3, x), w, np.linspace(0.3, 1.1, 4), range(4))
    ref.append(c.sum() + ys.sum())
  out.append((map_of_scan, (np.linspace(-1, 1, 5), w), [np.array(ref)]))
  return out


def run_once() -> None:
  import numpy as np

  from scaly.codegen import render_c_source
  from scaly.codegen.aot import workspace_size
  from scaly.function.model import as_concrete

  bad = total = 0
  with tempfile.TemporaryDirectory() as tmp:
    for (fun, args, expected), lanes, recip in itertools.product(programs(), (1, 2, "auto"), (False, True)):
      total += 1
      tag = f"{fun.name} lanes={lanes} reciprocal={recip}"
      try:
        src = render_c_source(fun, lanes=lanes, reciprocal=recip)
        size = workspace_size(fun, lanes=lanes, reciprocal=recip)
      except Exception as e:  # noqa: BLE001
        bad += 1
        print("PASS FAILURE", tag, type(e).__name__, str(e)[:160])
        continue
      c, so = Path(tmp, f"m{total}.c"), Path(tmp, f"m{total}.so")
      c.write_text(src)
      built = subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-fno-math-errno", str(c), "-lm", "-o", str(so)], capture_output=True, text=True)
      if built.returncode:
        bad += 1
        print("COMPILE FAILURE", tag, built.stderr[:300])
        continue
      entry = getattr(ctypes.CDLL(str(so)), as_concrete(fun).name)
      ins = [np.ascontiguousarray(a, dtype=np.float64).reshape(-1) for a in args]
      outs = [np.full(max(np.size(e), 1), -7.0) for e in expected]
      dp = ctypes.POINTER(ctypes.c_double)
      arg = (dp * len(ins))(*(a.ctypes.data_as(dp) for a in ins))
      res = (dp * len(outs))(*(o.ctypes.data_as(dp) for o in outs))
      work = np.zeros(max(size, 1))
      entry(arg, res, None, work.ctypes.data_as(dp), 0)
      if ("_5_" in fun.name or fun.name.startswith("map")) and not re.search(r"for \((long long )?i_loop\w* = 0", src):
        bad += 1
        print("NO LOOP LEFT", tag)
        continue
      if not all(np.allclose(o[: np.size(e)], np.reshape(e, -1), rtol=1e-12, atol=1e-12) for o, e in zip(outs, expected, strict=True)):
        bad += 1
        print("MISMATCH", tag)
  print(f"{total} configurations, {bad} bad")


def main() -> None:
  if "--one" in sys.argv:
    run_once()
    return
  for name, (file, guarded, removed) in {"none": (None, None, None), **GUARDS}.items():
    with tempfile.TemporaryDirectory() as tmp:
      shutil.copytree(ROOT / "src", Path(tmp, "src"))
      if file is not None:
        path = Path(tmp, "src", "scaly", "passes", "program", file)
        text = path.read_text()
        assert text.count(guarded) == 1, name
        path.write_text(text.replace(guarded, removed))
      env = {**os.environ, "PYTHONPATH": str(Path(tmp, "src")), "SCALY_CACHE_DIR": str(Path(tmp, "cache"))}
      result = subprocess.run([sys.executable, __file__, "--one"], capture_output=True, text=True, env=env)
      lines = (result.stdout + result.stderr).strip().splitlines()
      print(f"{name}: {lines[-1] if lines else 'no output'}")
      for line in lines[:-1][:3]:
        print("   ", line[:200])


if __name__ == "__main__":
  main()
