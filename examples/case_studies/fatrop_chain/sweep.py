"""Chain-size sweep: IPOPT's Lagrangian Hessian and constraint Jacobian, Scaly against three CasADi encodings.

    uv run --with rockit-meco==0.6.7 examples/case_studies/fatrop_chain/sweep.py --dim 3 --masses 3 4 5 6 8 10 12 15 --out results/sweep_3d.json

Every cell runs in a fresh process with an empty JIT cache. For each chain size (`no_masses`, the
paper's instance is 6) and backend it builds the two kernels IPOPT calls, generates C, compiles it
through `baseline/cc_timed.sh` (wall time and peak memory of the compiler, 180 s budget as in
docs/results/fairness.md), times each kernel from C with `time_kernel.c`, and checks the values against
CasADi's SX evaluation at the same point. A backend that fails at one size is skipped at larger ones.

Backends:
  scaly         `sc.opt.solver(problem, "ipopt")`'s descriptor kernels (the IPOPT drop-in's oracles)
  scaly_stage   the Fatrop drop-in's oracles: dense stage Hessians and Jacobians, `vmap` over stages
  casadi_sx     rockit's NLP expanded to SX (`expand=True`), what the paper compiled
  casadi_mx     rockit's NLP as MX, one call node per stage (`expand=False`)
  casadi_map    the same NLP written with the stage map `F.map(N)` over an SX stage function
CasADi oracles are `transform`ed (CSE) and generated with `casadi_int = int`, like benchmarks/harness/sweep.py.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _common import wait_for_quiet  # noqa: E402

BUDGET = 180.0
NATIVE = "-mcpu=native" if platform.machine().lower() in {"arm64", "aarch64"} else "-march=native"
CFLAGS = ["-O2", NATIVE, "-fno-math-errno"]
BACKENDS = ("scaly", "scaly_stage", "casadi_sx", "casadi_mx", "casadi_map")


def compile_timed(sources: list[Path], lib: Path, extra: list[str] | None = None) -> dict:
  log = lib.with_suffix(".cclog")
  env = {**os.environ, "CC_TIMED_LOG": str(log)}
  cmd = [str(HERE / "baseline" / "cc_timed.sh"), *CFLAGS, "-shared", "-fPIC", "-o", str(lib), *map(str, sources), *(extra or [])]
  t0 = time.perf_counter()
  proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
  try:
    _, stderr = proc.communicate(timeout=BUDGET)
  except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGKILL)  # the wrapper, time(1) and the compiler under it
    proc.communicate()
    return {"ok": False, "reason": f"compile exceeded {BUDGET:.0f} s", "compile_wall": BUDGET}
  wall = time.perf_counter() - t0
  rss = max((int(line.split()[2]) for line in log.read_text().splitlines()), default=0) if log.exists() else 0
  if proc.returncode:
    return {"ok": False, "reason": stderr[-400:], "compile_wall": wall, "compile_peak_rss_bytes": rss}
  return {"ok": True, "compile_wall": wall, "compile_peak_rss_bytes": rss}


def text_bytes(lib: Path) -> int:
  """Bytes of machine code: the `__text` section on macOS, `.text` elsewhere."""
  out = subprocess.run(["size", "-m", str(lib)] if sys.platform == "darwin" else ["size", "-A", str(lib)], capture_output=True, text=True).stdout
  for line in out.splitlines():
    parts = line.replace(":", " ").split()
    if "__text" in parts or ".text" in parts:
      return int(next(p for p in parts[parts.index("__text" if "__text" in parts else ".text") + 1 :] if p.isdigit()))
  return 0


def time_call(
  lib: Path, symbol: str, inputs: list[np.ndarray], outputs: list[int], sizes: tuple[int, int, int, int], repeats: int, work: Path
) -> tuple[float, float, list]:
  harness = work / "time_kernel"
  if not harness.exists():
    subprocess.run(["cc", "-O2", "-o", str(harness), str(HERE / "time_kernel.c")], check=True)
  blob = np.array([len(inputs), len(outputs), *sizes, *(x.size for x in inputs), *outputs], dtype=np.int64).tobytes()
  blob += b"".join(np.ascontiguousarray(x, dtype=np.float64).tobytes() for x in inputs)
  (work / "in.bin").write_bytes(blob)
  out = subprocess.run(
    [str(harness), str(lib), symbol, str(work / "in.bin"), str(repeats), str(work / "out.bin")], capture_output=True, text=True, check=True
  )
  best, median = (float(v) * 1e-9 for v in out.stdout.split())
  raw = np.frombuffer((work / "out.bin").read_bytes(), dtype=np.float64)
  values, at = [], 0
  for n in outputs:
    if n >= 0:
      values.append(raw[at : at + n].copy())
      at += n
    else:
      values.append(None)
  return best, median, values


def point(dim: int, masses: int, horizon: int, seed: int = 0):
  """A point in rockit's layout: the initial guess plus noise, mixed-sign multipliers, and rockit's row order."""
  sys.path.insert(0, str(HERE))
  from scaly_impl import initial_guess, initial_state, sizes

  nx, nu = sizes(dim, masses)
  rng = np.random.default_rng(seed)
  z = initial_guess(dim, masses, horizon) + 0.05 * rng.standard_normal(horizon * (nx + nu) + nx)
  ng = horizon * nx + nx + horizon * nu
  lam_rockit = rng.uniform(-1.0, 1.0, ng)
  dyn = lambda k: list(range(k * nx, (k + 1) * nx))  # noqa: E731
  x0r = list(range(horizon * nx, horizon * nx + nx))
  ur = lambda k: list(range(horizon * nx + nx + k * nu, horizon * nx + nx + (k + 1) * nu))  # noqa: E731
  scaly_of_rockit = np.array(dyn(0) + x0r + ur(0) + sum((dyn(k) + ur(k) for k in range(1, horizon)), []))  # rockit row i is Scaly row perm[i]
  lam_scaly = np.empty(ng)
  lam_scaly[scaly_of_rockit] = lam_rockit
  return z, initial_state(dim, masses), lam_rockit, lam_scaly, scaly_of_rockit


def dense(values: np.ndarray, rows, cols, shape, symmetric: bool) -> np.ndarray:
  out = np.zeros(shape)
  out[np.asarray(rows), np.asarray(cols)] = values
  if symmetric:
    out = out + out.T - np.diag(np.diag(out))
  return out


def reference(dim: int, masses: int, horizon: int, z, lam_rockit):
  """CasADi's SX evaluation in Python of rockit's NLP: dense Hessian and Jacobian, rockit's row order."""
  import casadi as ca

  sys.path.insert(0, str(HERE / "baseline"))
  from run_casadi import build_nlp

  nlp, _, lbg, _ = build_nlp(dim, masses, horizon)
  x, f, g = nlp["x"], nlp["f"], nlp["g"]
  lam = ca.MX.sym("lam", g.shape[0])
  fn = ca.Function("ref", [x, lam], [ca.hessian(f + ca.dot(lam, g), x)[0], ca.jacobian(g, x)]).expand()
  h, j = fn(z, lam_rockit)
  return np.array(h), np.array(j)


def worker(backend: str, dim: int, masses: int, horizon: int, repeats: int, work: Path) -> dict:
  sys.path.insert(0, str(HERE))
  z, x0, lam_rockit, lam_scaly, perm = point(dim, masses, horizon)
  record: dict = {"backend": backend, "dim": dim, "masses": masses, "horizon": horizon}
  t0 = time.perf_counter()
  kernels = []  # (kind, symbol, inputs, outputs, (sz_w, sz_iw, sz_arg, sz_res), decode)
  if backend in ("scaly", "scaly_stage"):
    import scaly as sc
    from scaly.codegen import write_module
    from scaly.opt.external.graph import solver_descriptor
    from scaly_impl import build, sizes

    nx, nu = sizes(dim, masses)
    sources = []
    if backend == "scaly":
      b = build(dim, masses, horizon)
      desc = solver_descriptor(sc.opt.solver(b["problem"], "ipopt", name=f"sweep_{dim}d_M{masses}"))
      for kind, fn, sp in (("hess", desc.hess, desc.hess_sparsity), ("jac", desc.jac, desc.jac_sparsity)):
        assert isinstance(fn, sc.Function) and sp is not None
        module = write_module(fn, work)
        sources.append(work / module.source_name)
        args = [z, x0, np.array([1.0]), lam_scaly] if kind == "hess" else [z, x0]
        n = h_n = int(sp.nnz)

        def decode(v, sp=sp, kind=kind):
          m = dense(v[0], sp.rows, sp.cols, sp.shape, kind == "hess")
          return m if kind == "hess" else m[perm]  # Scaly's rows to rockit's order

        kernels.append((kind, fn.name, args, [h_n if kind == "hess" else n], (module.workspace_size, 0, 0, 0), decode))
    else:
      from fatrop_dropin import oracles, to_fatrop

      o = oracles(dim, masses, horizon)
      w = to_fatrop(z, nx, nu, horizon)
      lam_dyn = -lam_rockit[np.concatenate([np.arange(k * nx, (k + 1) * nx) + (0 if k == 0 else nx + k * nu) for k in range(horizon)])]
      for kind in ("hess", "jac"):
        fn = o[kind]
        module = write_module(fn, work)
        sources.append(work / module.source_name)
        args = [w, lam_dyn, np.array([1.0])] if kind == "hess" else [w]
        kernels.append(
          (
            kind,
            fn.name,
            args,
            [int(fn.outputs[0].size)],
            (module.workspace_size, 0, 0, 0),
            stage_hess_decoder(nx, nu, horizon) if kind == "hess" else None,
          )
        )
    record["t_build"] = time.perf_counter() - t0
  else:
    import casadi as ca

    sys.path.insert(0, str(HERE / "baseline"))
    from run_casadi import build_nlp

    if backend == "casadi_map":
      nlp, lbg = mapped_nlp(dim, masses, horizon)
      lam = lam_scaly
    else:
      nlp, _, lbg, _ = build_nlp(dim, masses, horizon)
      lam = lam_rockit
    fn_nlp = ca.nlpsol("s", "ipopt", nlp, {"expand": backend == "casadi_sx", "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False})
    sources = []
    for kind, name, out_index in (("hess", "nlp_hess_l", 0), ("jac", "nlp_jac_g", 1)):
      fn = fn_nlp.get_function(name).transform({})
      cwd = Path.cwd()
      os.chdir(work)
      try:
        gen = ca.CodeGenerator(f"casadi_{kind}.c", {"with_header": True, "casadi_int": "int"})
        gen.add(fn)
        gen.generate()
      finally:
        os.chdir(cwd)
      sources.append(work / f"casadi_{kind}.c")
      p = np.zeros(fn.numel_in(1))
      args = [z, p, np.array([1.0]), lam] if kind == "hess" else [z, p]
      outs = [(int(fn.nnz_out(i)) if i == out_index else -1) for i in range(fn.n_out())]
      sp = fn.sparsity_out(out_index)
      rows, cols = sp.get_triplet()

      def decode(v, rows=rows, cols=cols, shape=sp.shape, kind=kind, out_index=out_index):
        m = dense(v[out_index], rows, cols, shape, kind == "hess")
        if kind == "jac" and backend == "casadi_map":
          m = m[perm]
        return m

      kernels.append((kind, fn.name(), args, outs, (int(fn.sz_w()), int(fn.sz_iw()), int(fn.sz_arg()), int(fn.sz_res())), decode))
    record["t_build"] = time.perf_counter() - t0
  record["source_bytes"] = sum(s.stat().st_size for s in sources)
  lib = work / "kernels.so"
  record |= compile_timed(sources, lib)
  if not record["ok"]:
    return record
  record["text_bytes"] = text_bytes(lib)
  h_ref, j_ref = reference(dim, masses, horizon, z, lam_rockit)
  for kind, symbol, args, outs, sizes, decode in kernels:
    best, median, values = time_call(lib, symbol, args, outs, sizes, repeats, work)
    record[f"{kind}_best"], record[f"{kind}_median"] = best, median
    if decode is not None:
      got, ref = decode(values), (h_ref if kind == "hess" else j_ref)
      record[f"{kind}_err"] = float(np.max(np.abs(got - ref)) / max(1.0, np.max(np.abs(ref))))
  return record


def stage_hess_decoder(nx: int, nu: int, horizon: int):
  """The Fatrop oracle's stage blocks ([H_k (u, x order), grad_k] per stage) as the full Hessian in rockit's order."""
  nz = nx + nu
  ux_to_xu = np.r_[nu:nz, 0:nu]

  def decode(values):
    blocks = values[0].reshape(horizon, nz * nz + nz)[:, : nz * nz].reshape(horizon, nz, nz)
    out = np.zeros((horizon * nz + nx, horizon * nz + nx))
    for k in range(horizon):
      out[k * nz : (k + 1) * nz, k * nz : (k + 1) * nz] = blocks[k][np.ix_(ux_to_xu, ux_to_xu)]
    return out

  return decode


def mapped_nlp(dim: int, masses: int, horizon: int):
  """The chain NLP in Scaly's row order, the stages as `F.map(N)` of SX stage functions."""
  import casadi as ca

  from scaly_impl import ALPHA, BETA, D, GAMMA, GRAVITY, L_REST, MASS, PAPER, sizes

  nx, nu = sizes(dim, masses)
  nz, T = nx + nu, PAPER[dim]["T"]
  h = T / horizon
  x, u = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
  pos = ca.horzcat(ca.DM.zeros(dim), ca.reshape(x[: dim * (masses + 1)], dim, masses + 1))
  grav = ca.DM.zeros(dim)
  grav[1] = -GRAVITY
  forces = []
  for i in range(masses + 1):
    d = pos[:, i + 1] - pos[:, i]
    ss = ca.sumsqr(d)
    forces.append(D * (1 - L_REST / ca.if_else(ss > 0, ca.sqrt(ss), 0)) * d)
  acc = [1.0 / MASS * (forces[i + 1] - forces[i] + MASS * grav) for i in range(masses)]
  ode = ca.Function("ode", [x, u], [ca.vertcat(x[dim * (masses + 1) :], u, *acc)])
  k1 = ode(x, u)
  k2 = ode(x + h / 2 * k1, u)
  k3 = ode(x + h / 2 * k2, u)
  k4 = ode(x + h * k3, u)
  xn = ca.SX.sym("xn", nx)
  gap = ca.Function("gap", [x, u, xn], [xn - (x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))])
  x_end = ca.DM.zeros(dim)
  x_end[0] = 1.0
  end = dim * masses
  cost = ca.Function(
    "cost", [x, u], [ALPHA * ca.sumsqr(x[end : end + dim] - x_end) + BETA * ca.sumsqr(x[dim * (masses + 1) :]) + GAMMA * ca.sumsqr(u)]
  )
  zs = ca.MX.sym("z", horizon * nz + nx)
  x0 = ca.MX.sym("x0", nx)
  stages = ca.reshape(zs[: horizon * nz], nz, horizon)
  xs, us = stages[:nx, :], stages[nx:, :]
  xnext = ca.horzcat(xs[:, 1:], zs[horizon * nz :])
  gaps = gap.map(horizon, "serial")(xs, us, xnext)
  f = ca.sum2(cost.map(horizon, "serial")(xs, us))
  g = ca.vertcat(ca.reshape(gaps, -1, 1), zs[:nx] - x0, ca.reshape(us, -1, 1))
  lbg = np.concatenate([np.zeros(horizon * nx + nx), -np.ones(horizon * nu)])
  return {"x": zs, "p": x0, "f": f, "g": g}, lbg


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--dim", type=int, default=3)
  ap.add_argument("--masses", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12, 15])
  ap.add_argument("--horizon", type=int, default=25)
  ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
  ap.add_argument("--repeats", type=int, default=200)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--report", action="store_true", help="print the markdown table of --out and exit")
  ap.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
  args = ap.parse_args()
  if args.report:
    print(report(json.loads(args.out.read_text())))
    return
  if args.worker:
    backend, masses, work = args.worker[0], int(args.worker[1]), Path(args.worker[2])
    print(json.dumps(worker(backend, args.dim, masses, args.horizon, args.repeats, work)))
    return
  results = json.loads(args.out.read_text()) if args.out.exists() else []
  done = {(r["backend"], r["masses"]) for r in results}
  failed = {r["backend"] for r in results if not r.get("ok", False)}
  for masses in args.masses:
    for backend in args.backends:
      if (backend, masses) in done or backend in failed:
        continue
      with tempfile.TemporaryDirectory(prefix="fatrop-sweep-") as work:
        env = {**os.environ, "SCALY_CACHE_DIR": str(Path(work) / "cache"), "OMP_NUM_THREADS": "1"}
        cmd = [
          sys.executable,
          __file__,
          "--dim",
          str(args.dim),
          "--horizon",
          str(args.horizon),
          "--repeats",
          str(args.repeats),
          "--out",
          str(args.out),
        ]
        load = wait_for_quiet()
        t0 = time.perf_counter()
        proc = subprocess.run([*cmd, "--worker", backend, str(masses), work], env=env, capture_output=True, text=True)
        if proc.returncode:
          record = {"backend": backend, "masses": masses, "dim": args.dim, "ok": False, "reason": proc.stderr[-600:]}
        else:
          record = json.loads(proc.stdout.strip().splitlines()[-1])
        record["t_cell"], record["load"] = time.perf_counter() - t0, load
      results.append(record)
      if not record.get("ok"):
        failed.add(backend)
      args.out.parent.mkdir(parents=True, exist_ok=True)
      args.out.write_text(json.dumps(results, indent=1))
      keys = ("t_build", "compile_wall", "compile_peak_rss_bytes", "hess_best", "jac_best", "hess_err", "jac_err", "reason")
      print(backend, masses, {k: record[k] for k in keys if k in record}, flush=True)


def report(results: list[dict]) -> str:
  """One row per chain size and backend: kernel times, build, compile, memory, source and machine code."""
  names = {
    "scaly": "Scaly (IPOPT oracles)",
    "scaly_stage": "Scaly (stage oracles)",
    "casadi_sx": "CasADi SX",
    "casadi_mx": "CasADi MX",
    "casadi_map": "CasADi map(SX)",
  }
  rows = [
    "| Masses | States | Backend | Hessian, µs | Jacobian, µs | Build, s | Compile, s | Peak compiler memory, MB | Source, kB | Machine code, kB | Max rel. error |",
    "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for r in sorted(results, key=lambda r: (r["masses"], list(names).index(r["backend"]))):
    nx = r["dim"] * (2 * r["masses"] + 1)
    head = f"| {r['masses']} | {nx} | {names[r['backend']]} |"
    if not r.get("ok"):
      reason = r.get("reason", "").strip().splitlines()[-1][:60] if r.get("reason") else "failed"
      rows.append(f"{head} {reason} | | | | | | | | |")
      continue
    err = max(r.get("hess_err", 0.0), r.get("jac_err", 0.0))
    rows.append(
      f"{head} {1e6 * r['hess_best']:.1f} | {1e6 * r['jac_best']:.1f} | {r['t_build']:.1f} | {r['compile_wall']:.1f} | "
      f"{r['compile_peak_rss_bytes'] / 1e6:.0f} | {r['source_bytes'] / 1e3:.0f} | {r.get('text_bytes', 0) / 1e3:.0f} | {err:.0e} |"
    )
  return "\n".join(rows)


if __name__ == "__main__":
  main()
