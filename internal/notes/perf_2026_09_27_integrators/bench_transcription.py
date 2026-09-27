"""I5: a transcribed horizon's residuals and their sparse Jacobian, per transcription, timed in C.

    uv run internal/notes/perf_2026_09_27_integrators/bench_transcription.py [--rounds 5]

The cart-pole of `bench_explicit.py` over N = 40 intervals of 0.05 s, as a nonlinear program sees
it: all variables in one vector `w = [xs; us; zs]`, the residuals of every interval (one `vmap` of
the transcription's interval Function) and their sparse Jacobian in `w`. Transcriptions: multiple
shooting with RK4 (two substeps), Radau and Gauss collocation of degree 3, and Radau pseudospectral
segments of 4 nodes. CasADi 3.8 SX (a mapped interval, expanded) is the baseline for Radau
collocation, the formulation of its `direct_collocation` example. Timed by `time_entry.c`, rounds
interleaved, fastest sample kept; values are checked against a NumPy-free second evaluation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build" / "transcription"
NX, NU, N, DT = 4, 1, 40, 0.05


def scaly_cells() -> dict:
  import scaly as sc
  from scaly import integrators as si
  from bench_explicit import dyn

  @sc.function(NX, NU, output="xdot")
  def cartpole(x, u):
    return dyn(x, u, lambda v: v.sin(), lambda v: v.cos(), sc.stack)

  transcriptions = {
    "shooting_rk4x2": si.MultipleShooting(si.rk4, steps=2),
    "collocation_radau3": si.Collocation(3, "radau"),
    "collocation_legendre3": si.Collocation(3, "legendre"),
    "pseudospectral4": si.Pseudospectral(4),
  }
  cells = {}
  for label, transcription in transcriptions.items():
    interval = transcription.interval(cartpole, dt=DT, name=f"{label}_interval")
    k = interval.n_internal
    size = (N + 1) * NX + N * NU + N * k

    def residuals_of(interval=interval, k=k, size=size, label=label):
      @sc.function(size, output="r", name=f"{label}_residuals")
      def residuals(w):
        xs, us, zs = w[: (N + 1) * NX], w[(N + 1) * NX : (N + 1) * NX + N * NU], w[(N + 1) * NX + N * NU :]
        specs = [(xs, 0, NX), (us, 0, NU), *([(zs, 0, k)] if k else []), (xs, NX, NX)]
        return sc.vmap(interval.fn, N, specs)

      return residuals

    residuals = residuals_of()
    cells[label] = {"residuals": residuals, "jacobian": sc.sparse_jacobian(residuals, name=f"{label}_jac"), "size": size, "k": k}
  return cells


def casadi_cells() -> dict:
  import casadi as ca

  from scaly.integrators.polynomial import differentiation_matrix, radau_nodes
  from bench_explicit import dyn

  tau = np.concatenate([[0.0], radau_nodes(3)])
  d = differentiation_matrix(tau)[1:]
  x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
  z, xn = ca.SX.sym("z", 2 * NX), ca.SX.sym("xn", NX)
  f = ca.Function("f", [x, u], [dyn(x, u, ca.sin, ca.cos, lambda xs: ca.vertcat(*xs))])
  states = [x, z[:NX], z[NX:], xn]
  res = [sum(float(d[i, j]) * states[j] for j in range(4)) - DT * f(states[i + 1], u) for i in range(3)]
  interval = ca.Function("interval", [x, u, z, xn], [ca.vertcat(*res)])
  size = (N + 1) * NX + N * NU + N * 2 * NX
  w = ca.SX.sym("w", size)
  xs = ca.reshape(w[: (N + 1) * NX], NX, N + 1)
  us = ca.reshape(w[(N + 1) * NX : (N + 1) * NX + N * NU], NU, N)
  zs = ca.reshape(w[(N + 1) * NX + N * NU :], 2 * NX, N)
  r = ca.vec(interval.map(N, "serial")(xs[:, :N], us, zs, xs[:, 1:]))
  return {"casadi_sx_radau3": {"residuals": ca.Function("casadi_sx_radau3_residuals", [w], [r]), "jacobian": ca.Function("casadi_sx_radau3_jac", [w], [ca.jacobian(r, w)]), "size": size}}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--rounds", type=int, default=5)
  args = parser.parse_args()
  sys.path.insert(0, str(HERE))
  import bench_explicit as be

  scaly, casadi = scaly_cells(), casadi_cells()
  metas, best, rows = {}, {}, {}
  for label, cell in {**scaly, **casadi}.items():
    w = np.random.default_rng(0).normal(size=cell["size"]) * 0.2
    for piece in ("residuals", "jacobian"):
      fn, out = cell[piece], BUILD / label / piece
      out.mkdir(parents=True, exist_ok=True)
      t0 = time.perf_counter()
      if label.startswith("casadi"):
        import casadi as ca

        from scaly.codegen.jit import compile_flags
        from scaly.codegen.toolchain import find_c_compiler

        gen = ca.CodeGenerator("f.c", {"casadi_int": "long long", "casadi_real": "double"})
        gen.add(fn)
        gen.generate(str(out) + "/")
        subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(out / "f.c"), "-lm", "-o", str(out / "lib.so")], check=True)
        meta = {"symbol": fn.name(), "c_lines": sum(1 for line in (out / "f.c").read_text().splitlines() if line.strip())}
        build_s = time.perf_counter() - t0
        values = np.asarray(fn(w).nonzeros() if piece == "jacobian" else fn(w)).ravel()
        sizes = [fn.nnz_out(0)] + [0] * (fn.sz_res() - fn.n_out())
        (out / "inputs.bin").write_bytes(be.blob([w, *(np.zeros(0) for _ in range(fn.sz_arg() - 1))], sizes, fn.sz_w()))
      else:
        from scaly.codegen import render_c_module
        from scaly.codegen.abi import c_ident
        from scaly.codegen.jit import compile_flags
        from scaly.codegen.toolchain import find_c_compiler

        module = render_c_module(fn)
        (out / "f.c").write_text(module.body)
        subprocess.run([find_c_compiler().cc, *compile_flags(), "-fPIC", "-shared", str(out / "f.c"), "-lm", "-o", str(out / "lib.so")], check=True)
        build_s = time.perf_counter() - t0
        values = np.ravel(fn(w))  # through the JIT: its first dlopen of a new library costs macOS a scan of about a second
        (out / "inputs.bin").write_bytes(be.blob([w], [values.size], int(module.workspace_size)))
        meta = {"symbol": c_ident(fn.name), "c_lines": sum(1 for line in module.body.splitlines() if line.strip())}
      meta["build_s"], meta["values"] = build_s, values
      metas[label, piece] = meta
    rows[label] = {"size": cell["size"], "residuals": metas[label, "residuals"]["values"].size, "nnz": metas[label, "jacobian"]["values"].size}
  for _ in range(args.rounds):
    for label in rows:
      for piece in ("residuals", "jacobian"):
        t, values = be.time_cell(BUILD / label / piece, metas[label, piece], 200, 10)
        assert np.allclose(values[: metas[label, piece]["values"].size], metas[label, piece]["values"], rtol=1e-12, atol=1e-12), (label, piece)
        best[label, piece] = min(best.get((label, piece), np.inf), t)
  print("| transcription | variables | residuals | Jacobian nonzeros | residuals ns | Jacobian ns | C lines r / J | render and compile s r / J |")
  print("| --- | --- | --- | --- | --- | --- | --- | --- |")
  for label, row in rows.items():
    lines = f"{metas[label, 'residuals']['c_lines']} / {metas[label, 'jacobian']['c_lines']}"
    build = f"{metas[label, 'residuals']['build_s']:.2f} / {metas[label, 'jacobian']['build_s']:.2f}"
    print(f"| {label} | {row['size']} | {row['residuals']} | {row['nnz']} | {best[label, 'residuals']:.0f} | {best[label, 'jacobian']:.0f} | {lines} | {build} |")


if __name__ == "__main__":
  sys.path.insert(0, str(ROOT))
  main()
