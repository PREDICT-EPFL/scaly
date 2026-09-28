"""Fatrop with Scaly's oracles: the drop-in half of the Fatrop comparison.

CasADi's `fatrop` plugin drives libfatrop through its C interface (`fatrop/ocp/OCPCInterface.h`),
evaluating CasADi's whole-NLP oracles and scattering them into Fatrop's per-stage BLASFEO blocks.
`fatrop_scaly.c` fills the same interface from five Scaly Functions over the whole horizon, each one
`vmap` over the stages, and links the same `libfatrop` from the CasADi wheel. The solver, its options
and its linear algebra are identical; only the oracles change.

Fatrop's primal layout is `w = [u_0, x_0, u_1, x_1, ..., u_{N-1}, x_{N-1}, x_N]` and its Lagrangian
`s f + lam_dyn' (F(u_k, x_k) - x_{k+1}) + lam_eq' g + lam_ineq' g_ineq`, so the stage Hessian block is
that of `s l_k + lam_dyn_k' F_k` in `(u_k, x_k)`; the constraints other than the dynamics are linear.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaly_impl import ALPHA, BETA, GAMMA, PAPER, chain_model, initial_guess, initial_state, sizes  # noqa: E402

from scaly import integrators as si  # noqa: E402

HERE = Path(__file__).resolve().parent


def oracles(dim: int, no_masses: int = 6, horizon: int | None = None, T: float | None = None) -> dict[str, sc.Function]:
  horizon = PAPER[dim]["N"] if horizon is None else horizon
  T = PAPER[dim]["T"] if T is None else T
  nx, nu = sizes(dim, no_masses)
  nz = nx + nu
  nw = horizon * nz + nx
  step = si.rk4(chain_model(dim, no_masses), dt=T / horizon)
  x_end = np.zeros(dim)
  x_end[0] = 1.0
  end = dim * no_masses
  tag = f"{dim}d_M{no_masses}_N{horizon}"

  def cost(ux: sc.Expr) -> sc.Expr:
    u, x = ux[:nu], ux[nu:]
    return ALPHA * sc.sumsqr(x[end : end + dim] - sc.const(x_end)) + BETA * sc.sumsqr(x[dim * (no_masses + 1) :]) + GAMMA * sc.sumsqr(u)

  def dyn(ux: sc.Expr) -> sc.Expr:
    return step(ux[nu:], ux[:nu])

  @sc.function(sc.L("ux", nz), name=f"fs_cost_{tag}")
  def stage_cost(ux):
    return cost(ux).reshape((1,))

  @sc.function(sc.L("ux", nz), name=f"fs_cost_grad_{tag}")
  def stage_cost_grad(ux):
    return sc.gradient(cost(ux), ux)

  @sc.function(sc.L("ux", nz), sc.L("xnext", nx), name=f"fs_gap_{tag}")
  def stage_gap(ux, xnext):
    return dyn(ux) - xnext

  # One output per stage, so the vmap computes each stage once: [jac (nx, nz) row-major, b], where the
  # row-major Jacobian is BAbt's first nz rows in column-major order, and [hess (nz, nz), grad].
  @sc.function(sc.L("ux", nz), sc.L("xnext", nx), name=f"fs_jac_{tag}")
  def stage_jac(ux, xnext):
    f = dyn(ux)
    return sc.concat([sc.jacobian(f, ux).reshape((nx * nz,)), f - xnext])

  @sc.function(sc.L("ux", nz), sc.L("lam", nx), sc.L("s", 1), name=f"fs_hess_{tag}")
  def stage_hess(ux, lam, s):
    lag = s[0] * cost(ux) + (lam * dyn(ux)).sum()
    return sc.concat([sc.hessian(lag, ux).reshape((nz * nz,)), sc.gradient(lag, ux)])

  def xnext_of(w: sc.Expr) -> sc.Expr:
    xs = w[: horizon * nz].reshape((horizon, nz))[:, nu:]
    return sc.concat([xs[1:].reshape(((horizon - 1) * nx,)), w[horizon * nz :]])

  W, S, LAM = sc.L("w", nw), sc.L("s", 1), sc.L("lam_dyn", horizon * nx)

  @sc.function(W, S, name=f"fs_obj_{tag}")
  def obj(w, s):
    return (s[0] * sc.vmap(stage_cost, horizon, [(w, 0, nz)]).sum()).reshape((1,))

  @sc.function(W, S, name=f"fs_grad_{tag}")
  def grad(w, s):
    return sc.concat([s[0] * sc.vmap(stage_cost_grad, horizon, [(w, 0, nz)]), sc.const(np.zeros(nx))])

  @sc.function(W, name=f"fs_cv_{tag}")
  def cv(w):
    return sc.vmap(stage_gap, horizon, [(w, 0, nz), (xnext_of(w), 0, nx)])

  @sc.function(W, name=f"fs_jacs_{tag}")
  def jac(w):
    return sc.vmap(stage_jac, horizon, [(w, 0, nz), (xnext_of(w), 0, nx)])

  @sc.function(W, LAM, S, name=f"fs_hesss_{tag}")
  def hess(w, lam, s):
    return sc.vmap(stage_hess, horizon, [(w, 0, nz), (lam, 0, nx), (s, 0, 0)])

  return {"obj": obj, "grad": grad, "cv": cv, "jac": jac, "hess": hess, "nx": nx, "nu": nu, "horizon": horizon, "nw": nw, "tag": tag}


def casadi_dir() -> Path:
  import casadi

  return Path(casadi.__file__).resolve().parent


def build(dim: int, out: Path, no_masses: int = 6, horizon: int | None = None, cflags: tuple[str, ...] = ()) -> tuple[Path, dict]:
  """Generate the oracles, compile them with the adapter into one shared library. Returns the library and timings."""
  t0 = time.perf_counter()
  o = oracles(dim, no_masses, horizon)
  out.mkdir(parents=True, exist_ok=True)
  sizes_w, sources = {}, []
  for key in ("obj", "grad", "cv", "jac", "hess"):
    module = write_module(o[key], out)
    sizes_w[key] = module.workspace_size
    sources.append(out / module.source_name)
  t_gen = time.perf_counter() - t0
  config = [
    f"#define NX {o['nx']}",
    f"#define NU {o['nu']}",
    f"#define HORIZON {o['horizon']}",
    *(f"#define SCALY_{k.upper()} {o[k].name}" for k in ("obj", "grad", "cv", "jac", "hess")),
    *(f"#define SZ_W_{k.upper()} {max(v, 1)}" for k, v in sizes_w.items()),
  ]
  (out / "fatrop_scaly_config.h").write_text("\n".join(config) + "\n")
  lib = out / f"libfatrop_scaly_{o['tag']}.dylib"
  cdir = casadi_dir()
  flags = cflags or ("-O2", "-mcpu=native", "-fno-math-errno")
  cmd = [
    os.environ.get("SCALY_CC", "cc"),
    *flags,
    "-shared",
    "-fPIC",
    "-o",
    str(lib),
    str(HERE / "fatrop_scaly.c"),
    *map(str, sources),
    f"-I{out}",
    f"-I{cdir / 'include'}",
    f"-L{cdir}",
    "-lfatrop",
    "-lblasfeo",
    f"-Wl,-rpath,{cdir}",
  ]
  t0 = time.perf_counter()
  subprocess.run(cmd, check=True)
  t_cc = time.perf_counter() - t0
  return lib, {"t_generate": t_gen, "t_compile": t_cc, "source_bytes": sum(p.stat().st_size for p in sources)}


class Stats(ctypes.Structure):
  _fields_ = [
    (name, ctypes.c_double)
    for name in (
      "compute_sd_time",
      "duinf_time",
      "eval_hess_time",
      "eval_jac_time",
      "eval_cv_time",
      "eval_grad_time",
      "eval_obj_time",
      "initialization_time",
      "time_total",
    )
  ] + [
    (name, ctypes.c_int)
    for name in ("eval_hess_count", "eval_jac_count", "eval_cv_count", "eval_grad_count", "eval_obj_count", "iterations_count", "return_flag")
  ]


def solver(lib: Path, dim: int, no_masses: int = 6, horizon: int | None = None, options: dict[str, float | int] | None = None):
  """A Python callable `solve(x0, w_init) -> (w, info)` over the compiled adapter. `info` holds Fatrop's
  iteration count and return flag, the wall time of `fatrop_ocp_c_solve`, the time spent in Scaly's
  oracles (`t_oracle`) and in the callbacks around them, scatter included (`t_callback`), all from C."""
  horizon = PAPER[dim]["N"] if horizon is None else horizon
  nx, nu = sizes(dim, no_masses)
  nw = horizon * (nx + nu) + nx
  so = ctypes.CDLL(str(lib))
  vp, dp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)
  so.fs_create.restype = vp
  so.fs_solve.argtypes = [vp, dp, dp, dp]
  so.fs_set_option_double.argtypes = [vp, ctypes.c_char_p, ctypes.c_double]
  so.fs_set_option_int.argtypes = [vp, ctypes.c_char_p, ctypes.c_int]
  so.fs_stats.argtypes = [vp, ctypes.POINTER(Stats)]
  for name in ("fs_wall", "fs_t_oracle", "fs_t_callback"):
    getattr(so, name).argtypes, getattr(so, name).restype = [vp], ctypes.c_double
  so.fs_n_oracle.argtypes = [vp]
  handle = vp(so.fs_create())
  for key, value in {"print_level": 0, **(options or {})}.items():
    if isinstance(value, int):
      so.fs_set_option_int(handle, key.encode(), value)
    else:
      so.fs_set_option_double(handle, key.encode(), float(value))

  def solve(x0: np.ndarray, w_init: np.ndarray) -> tuple[np.ndarray, dict]:
    x0 = np.ascontiguousarray(x0, dtype=np.float64)
    w_init = np.ascontiguousarray(w_init, dtype=np.float64)
    w = np.zeros(nw)
    so.fs_solve(handle, x0.ctypes.data_as(dp), w_init.ctypes.data_as(dp), w.ctypes.data_as(dp))
    stats = Stats()
    so.fs_stats(handle, ctypes.byref(stats))
    info = {"iterations": stats.iterations_count, "return_flag": stats.return_flag, "wall": so.fs_wall(handle)}
    info |= {"t_oracle": so.fs_t_oracle(handle), "t_callback": so.fs_t_callback(handle), "n_oracle": so.fs_n_oracle(handle)}
    return w, info

  return solve


def to_fatrop(z: np.ndarray, nx: int, nu: int, horizon: int) -> np.ndarray:
  """`[x_0, u_0, x_1, ...]` to `[u_0, x_0, u_1, ...]`."""
  nz = nx + nu
  stages = z[: horizon * nz].reshape(horizon, nz)
  return np.concatenate([np.concatenate([stages[:, nx:], stages[:, :nx]], axis=1).reshape(-1), z[horizon * nz :]])


def from_fatrop(w: np.ndarray, nx: int, nu: int, horizon: int) -> np.ndarray:
  nz = nx + nu
  stages = w[: horizon * nz].reshape(horizon, nz)
  return np.concatenate([np.concatenate([stages[:, nu:], stages[:, :nu]], axis=1).reshape(-1), w[horizon * nz :]])


if __name__ == "__main__":
  build_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "build"
  for dim in (2, 3):
    lib, info = build(dim, build_dir / f"{dim}d")
    horizon = PAPER[dim]["N"]
    nx, nu = sizes(dim, 6)
    solve = solver(lib, dim, options={"tol": 1e-8})
    w0 = to_fatrop(initial_guess(dim, 6, horizon), nx, nu, horizon)
    runs = [solve(initial_state(dim), w0)[1] for _ in range(20)]
    w, last = solve(initial_state(dim), w0)
    best = min(runs, key=lambda r: r["wall"])
    print(
      f"{dim}D: {last['iterations']} iterations, flag {last['return_flag']}, solve {1e3 * best['wall']:.3f} ms, "
      f"oracles {1e3 * best['t_oracle']:.3f} ms ({best['n_oracle']} calls), callbacks {1e3 * best['t_callback']:.3f} ms, u_0 {w[:nu]}; "
      f"generate {info['t_generate']:.2f} s, compile {info['t_compile']:.2f} s"
    )
