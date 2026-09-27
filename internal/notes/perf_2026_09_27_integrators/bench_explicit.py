"""I1: the library's explicit RK4 against the hand-written scaly RK4 and CasADi, timed in C.

    uv run internal/notes/perf_2026_09_27_integrators/bench_explicit.py [--rounds 7] [--budget 0.05]

The model is `examples/nmpc_cartpole.py`'s cart-pole with its interval (two RK4 substeps of
0.025 s). Four pieces per variant: the step `(x, u) -> xnext`, its Jacobians in `x` and `u`, the
horizon's shooting defects over N = 40 stages as one function of `w = [xs; us]`, and their sparse
Jacobian in `w` (what IPOPT's constraint Jacobian oracle evaluates). Variants:

- `hand`: the example's plain-Python RK4 composed over `Expr` (the model inlined at every stage);
- `lib`: `sc.integrators.rk4(cartpole, dt=0.05, steps=2)`, the model a Function called per stage;
- `casadi_sx` / `casadi_mx`: the same step in CasADi 3.8, SX expanded and MX with a serial map,
  code-generated and compiled with the JIT's flags. The faster one is the CasADi row
  (`docs/results/fairness.md`: the fastest completed encoding, named).

Each cell is compiled to a shared library and timed by `time_entry.c` (the universal entry in a loop,
no Python). Rounds interleave the variants; a cell's time is its
fastest call over all rounds. Values are checked against the `hand` variant before timing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build" / "explicit"
DRIVER_SRC = HERE / "time_entry.c"
NX, NU, N, DT = 4, 1, 40, 0.05
M_CART, M_POLE, LENGTH, G = 1.0, 0.3, 0.5, 9.81


def dyn(x, u, sin, cos, stack):
  theta, pd, td = x[1], x[2], x[3]
  s, c = sin(theta), cos(theta)
  den = M_CART + M_POLE * s * s
  pdd = (u[0] + M_POLE * LENGTH * td * td * s - M_POLE * G * s * c) / den
  tdd = (G * s * (M_CART + M_POLE) - c * (u[0] + M_POLE * LENGTH * td * td * s)) / (LENGTH * den)
  return stack([pd, td, pdd, tdd])


def rk4(f, x, u, h):
  k1 = f(x, u)
  k2 = f(x + 0.5 * h * k1, u)
  k3 = f(x + 0.5 * h * k2, u)
  k4 = f(x + h * k3, u)
  return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def scaly_pieces(variant: str) -> dict:
  import scaly as sc
  from scaly import integrators as si

  sdyn = lambda x, u: dyn(x, u, lambda v: v.sin(), lambda v: v.cos(), sc.stack)  # noqa: E731
  if variant == "hand":

    @sc.function(NX, NU, output="xnext", name="hand_step")
    def step(x, u):
      return rk4(sdyn, rk4(sdyn, x, u, DT / 2), u, DT / 2)

  else:

    @sc.function(NX, NU, output="xdot")
    def cartpole(x, u):
      return sdyn(x, u)

    step = si.rk4(cartpole, dt=DT, steps=2, name="lib_step")

  @sc.function(NX, NU, NX, output="eq", name=f"{variant}_defect")
  def defect(x, u, xnext):
    return step(x, u) - xnext

  @sc.function(sc.L("w", (N + 1) * NX + N * NU), output="eq", name=f"{variant}_horizon")
  def horizon(w):
    xs, us = w[: (N + 1) * NX], w[(N + 1) * NX :]
    return sc.vmap(defect, N, [(xs, 0, NX), (us, 0, NU), (xs, NX, NX)])

  @sc.function(NX, NU, output=sc.G("jx", "ju"), name=f"{variant}_step_jac")
  def step_jac(x, u):
    y = step(x, u)
    return sc.jacobian(y, x), sc.jacobian(y, u)

  return {"step": step, "step_jac": step_jac, "horizon": horizon, "horizon_jac": sc.sparse_jacobian(horizon, name=f"{variant}_horizon_jac")}


def casadi_pieces(encoding: str) -> dict:
  import casadi as ca

  x, u, xn = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("xn", NX)
  f = ca.Function("f", [x, u], [dyn(x, u, ca.sin, ca.cos, lambda xs: ca.vertcat(*xs))])
  step = ca.Function(f"{encoding}_step", [x, u], [rk4(f, rk4(f, x, u, DT / 2), u, DT / 2)], ["x", "u"], ["xnext"])
  step_jac = ca.Function(f"{encoding}_step_jac", [x, u], [ca.jacobian(step(x, u), x), ca.jacobian(step(x, u), u)])
  defect = ca.Function("defect", [x, u, xn], [step(x, u) - xn])
  sym = ca.SX if encoding == "casadi_sx" else ca.MX
  w = sym.sym("w", (N + 1) * NX + N * NU)
  xs, us = ca.reshape(w[: (N + 1) * NX], NX, N + 1), ca.reshape(w[(N + 1) * NX :], NU, N)
  eq = ca.vec(defect.map(N, "serial")(xs[:, :N], us, xs[:, 1:]))
  horizon = ca.Function(f"{encoding}_horizon", [w], [eq])
  horizon_jac = ca.Function(f"{encoding}_horizon_jac", [w], [ca.jacobian(eq, w)])
  if encoding == "casadi_sx":
    step_jac = step_jac.expand() if hasattr(step_jac, "expand") else step_jac
  return {"step": step, "step_jac": step_jac, "horizon": horizon, "horizon_jac": horizon_jac}


def inputs(piece: str) -> list[np.ndarray]:
  rng = np.random.default_rng(3)
  if piece.startswith("step"):
    return [rng.normal(size=NX), rng.normal(size=NU)]
  return [rng.normal(size=(N + 1) * NX + N * NU)]


def blob(flat: list[np.ndarray], out_sizes: list[int], workspace: int) -> bytes:
  head = np.array([len(flat), len(out_sizes), workspace, *(a.size for a in flat), *out_sizes], dtype="<i8").tobytes()
  return head + b"".join(np.ascontiguousarray(a, dtype="<f8").tobytes() for a in flat)


def build_scaly(variant: str, piece: str, fn, out: Path) -> dict:
  from scaly.codegen import render_c_module
  from scaly.codegen.abi import c_ident
  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler

  t0 = time.perf_counter()
  module = render_c_module(fn)
  generate = time.perf_counter() - t0
  out.mkdir(parents=True, exist_ok=True)
  (out / "f.c").write_text(module.body)
  t0 = time.perf_counter()
  subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(out / "f.c"), "-lm", "-o", str(out / "lib.so")], check=True)
  compile_s = time.perf_counter() - t0
  args = inputs(piece)
  values = fn(*args)
  values = values if isinstance(values, tuple) else (values,)
  (out / "inputs.bin").write_bytes(blob(args, [np.size(v) for v in values], int(module.workspace_size)))
  meta = {"symbol": c_ident(fn.name), "generate_s": generate, "compile_s": compile_s, "c_lines": sum(1 for line in module.body.splitlines() if line.strip())}
  np.save(out / "expected.npy", np.concatenate([np.ravel(v) for v in values]))
  return meta


def build_casadi(encoding: str, piece: str, fn, out: Path) -> dict:
  import casadi as ca

  from scaly.codegen.jit import compile_flags
  from scaly.codegen.toolchain import find_c_compiler

  out.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  gen = ca.CodeGenerator("f.c", {"casadi_int": "long long", "casadi_real": "double"})
  gen.add(fn)
  gen.generate(str(out) + "/")
  generate = time.perf_counter() - t0
  t0 = time.perf_counter()
  subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(out / "f.c"), "-lm", "-o", str(out / "lib.so")], check=True)
  compile_s = time.perf_counter() - t0
  args = inputs(piece)
  values = [np.asarray(ca.DM(v).nonzeros() if isinstance(v, ca.DM) and v.nnz() < v.numel() else np.ravel(np.array(v), order="F")) for v in fn(*args)] if fn.n_out() > 1 else [np.ravel(np.array(fn(*args)), order="F")]
  # CasADi uses the argument and result arrays past n_in and n_out as pointer scratch for nested
  # calls: empty dummy slots up to sz_arg and sz_res give it room.
  sizes = [fn.nnz_out(i) for i in range(fn.n_out())] + [0] * (fn.sz_res() - fn.n_out())
  padded = [*args, *(np.zeros(0) for _ in range(fn.sz_arg() - fn.n_in()))]
  assert fn.sz_iw() <= 1 << 16, fn.sz_iw()
  (out / "inputs.bin").write_bytes(blob(padded, sizes, fn.sz_w()))
  return {"symbol": fn.name(), "generate_s": generate, "compile_s": compile_s, "c_lines": sum(1 for line in (out / "f.c").read_text().splitlines() if line.strip())}


def driver() -> Path:
  exe = BUILD / "time_entry"
  if not exe.exists() or exe.stat().st_mtime < DRIVER_SRC.stat().st_mtime:
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cc", "-O2", "-o", str(exe), str(DRIVER_SRC)], check=True)
  return exe


def time_cell(cell: Path, meta: dict, reps: int, inner: int) -> tuple[float, np.ndarray]:
  cmd = [str(driver()), str(cell / "lib.so"), meta["symbol"], str(cell / "inputs.bin"), str(reps), str(cell / "outputs.bin"), str(inner)]
  out = subprocess.run(cmd, capture_output=True, text=True, check=True)
  return float(out.stdout.split()[0]), np.fromfile(cell / "outputs.bin", dtype="<f8")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=7)
  parser.add_argument("--reps", type=int, default=400, help="samples per cell and round")
  args = parser.parse_args()
  variants = ["hand", "lib", "casadi_sx", "casadi_mx"]
  metas: dict[tuple[str, str], dict] = {}
  for variant in variants:
    pieces = scaly_pieces(variant) if variant in ("hand", "lib") else casadi_pieces(variant)
    for piece, fn in pieces.items():
      cell = BUILD / variant / piece
      build = build_scaly if variant in ("hand", "lib") else build_casadi
      metas[variant, piece] = build(variant, piece, fn, cell)
  best: dict[tuple[str, str], float] = {}
  for _ in range(args.rounds):
    for piece in ("step", "step_jac", "horizon", "horizon_jac"):
      for variant in variants:
        cell = BUILD / variant / piece
        inner = 200 if piece.startswith("step") else 10
        t, values = time_cell(cell, metas[variant, piece], args.reps, inner)
        best[variant, piece] = min(best.get((variant, piece), np.inf), t)
        if variant == "lib":
          expected = np.load(BUILD / "hand" / piece / "expected.npy")
          assert np.allclose(np.sort(values), np.sort(expected), rtol=1e-12, atol=1e-12), (piece, "lib disagrees with hand")
  rows = []
  print("| piece | hand ns | lib ns | lib/hand | CasADi (encoding) ns | lib/CasADi | C lines hand / lib / CasADi | generate s hand / lib |")
  print("| --- | --- | --- | --- | --- | --- | --- | --- |")
  for piece in ("step", "step_jac", "horizon", "horizon_jac"):
    enc = min(("casadi_sx", "casadi_mx"), key=lambda e: best[e, piece])
    h, lib, ca_t = best["hand", piece], best["lib", piece], best[enc, piece]
    lines = f"{metas['hand', piece]['c_lines']} / {metas['lib', piece]['c_lines']} / {metas[enc, piece]['c_lines']}"
    gen = f"{metas['hand', piece]['generate_s']:.2f} / {metas['lib', piece]['generate_s']:.2f}"
    print(f"| {piece} | {h:.1f} | {lib:.1f} | {lib / h:.3f} | {ca_t:.1f} ({enc}) | {lib / ca_t:.2f} | {lines} | {gen} |")
    rows.append({"piece": piece, **{f"{v}_ns": best[v, piece] for v in variants}, **{f"{v}_meta": metas[v, piece] for v in variants}})
  (BUILD / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
