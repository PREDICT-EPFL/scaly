"""FastSQP adapter - FastBench's own C++ real-time SQP.

On ``build`` the adapter:
  1. generates C code (CasADi) for the discrete dynamics + Jacobians and the
     cost gradients/Hessians of the given problem,
  2. compiles it together with the C++ SQP core (``cpp/fastsqp.cpp``) and a QP
     backend into a shared library,
  3. loads the library via ctypes.

This naturally yields code-generation and compilation metrics (generated
source size/LOC, generation time, compile time, compiled-artifact size) — the
embedded-codegen story, with no acados dependency.

QP backends are chosen at compile time; ``proxqp`` (header-only, bundled with
the CasADi wheel) is the default and the one verified here. Build with
``HAVE_PIQP``/``HAVE_OSQP``/``HAVE_QPOASES`` to enable the others.
"""
from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
import time

import numpy as np

from fastbench.core.registry import register_solver
from fastbench.solvers.base import BuildInfo, SolveStats, SolverAdapter

_BACKEND_ID = {"proxqp": 0, "piqp": 1, "osqp": 2, "qpoases": 3}
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CPP_DIR = os.path.join(_HERE, "cpp")
# Tag build artifacts by OS+arch so objects compiled on one platform are never
# reused on another (e.g. a Linux .o must not be linked on macOS).
_PLAT = f"{sys.platform}-{platform.machine()}"


def _casadi_include():
    import casadi
    return os.path.join(os.path.dirname(casadi.__file__), "include")


def _dir_file_stats(path):
    nbytes = os.path.getsize(path)
    with open(path, errors="ignore") as fh:
        loc = sum(1 for _ in fh)
    return nbytes, loc


class FastSqpAdapter(SolverAdapter):
    name = "fastsqp"

    def __init__(self, backend: str = "proxqp", sqp_iters: int = 1):
        self.backend = backend
        self.sqp_iters = sqp_iters
        self._lib = None
        self._h = None

    # ----------------------------------------------------------- availability
    def available(self) -> bool:
        try:
            import casadi  # noqa: F401
        except Exception:
            return False
        if subprocess.run(["which", "g++"], capture_output=True).returncode != 0:
            return False
        # proxqp headers ship with the casadi wheel
        return os.path.isdir(os.path.join(_casadi_include(), "proxsuite"))

    # ------------------------------------------------------------ code-gen
    def _codegen(self, problem, outdir) -> float:
        import casadi as ca
        nx, nu = problem.meta.nx, problem.meta.nu
        x = ca.SX.sym("x", nx); u = ca.SX.sym("u", nu)
        xr = ca.SX.sym("xr", nx); ur = ca.SX.sym("ur", nu)

        t0 = time.perf_counter()
        xnext = problem.discrete_dynamics(x, u)
        A = ca.densify(ca.jacobian(xnext, x))
        B = ca.densify(ca.jacobian(xnext, u))
        dyn = ca.Function("dyn", [x, u], [ca.densify(xnext), A, B])

        l = problem.stage_cost(x, u, xr, ur)
        gx = ca.densify(ca.gradient(l, x))
        gu = ca.densify(ca.gradient(l, u))
        Hxx = ca.densify(ca.hessian(l, x)[0])
        Huu = ca.densify(ca.hessian(l, u)[0])
        Hxu = ca.densify(ca.jacobian(ca.gradient(l, x), u))
        costs = ca.Function("costs", [x, u, xr, ur], [gx, gu, Hxx, Huu, Hxu])

        m = problem.terminal_cost(x, xr)
        gxN = ca.densify(ca.gradient(m, x))
        HxxN = ca.densify(ca.hessian(m, x)[0])
        costN = ca.Function("costN", [x, xr], [gxN, HxxN])

        cg = ca.CodeGenerator("probgen.c", {"with_header": False})
        cg.add(dyn); cg.add(costs); cg.add(costN)
        cg.generate(outdir + os.sep)
        return time.perf_counter() - t0

    # ----------------------------------------------------------------- build
    def build(self, problem) -> BuildInfo:
        outdir = os.path.abspath(os.path.join("fastsqp_build", _PLAT, problem.meta.name))
        os.makedirs(outdir, exist_ok=True)
        gen_time = self._codegen(problem, outdir)
        gen_c = os.path.join(outdir, "probgen.c")
        gen_bytes, gen_loc = _dir_file_stats(gen_c)

        casinc = _casadi_include()
        macros = [f"-DHAVE_{b.upper()}" for b in [self.backend]]
        so = os.path.join(outdir, f"libfastsqp_{problem.meta.name}.so")
        obj_c = os.path.join(outdir, "probgen.o")
        obj_cpp = self._core_object(macros, casinc)   # compiled once, cached

        t0 = time.perf_counter()
        # 1) compile generated C as C (unmangled symbols)
        _run(["gcc", "-O2", "-fPIC", "-c", gen_c, "-o", obj_c])
        # 2) link generated problem code with the cached C++ SQP core
        _run(["g++", "-shared", obj_cpp, obj_c, "-o", so])
        compile_time = time.perf_counter() - t0
        comp_bytes = os.path.getsize(so)

        self._load(so, problem)
        return BuildInfo(
            build_time_s=gen_time + compile_time, generation_time_s=gen_time,
            compile_time_s=compile_time, generated_code_bytes=gen_bytes,
            generated_code_loc=gen_loc, compiled_bytes=comp_bytes,
            notes=f"CasADi codegen + C++ sparse SQP ({self.backend}), "
                  f"sqp_iters={self.sqp_iters}")

    def _core_object(self, macros, casinc):
        """Compile the (problem-independent) C++ SQP core once and cache it."""
        core_dir = os.path.abspath(os.path.join("fastsqp_build", _PLAT, "_core"))
        os.makedirs(core_dir, exist_ok=True)
        obj = os.path.join(core_dir, f"fastsqp_{self.backend}.o")
        src = os.path.join(_CPP_DIR, "fastsqp.cpp")
        if (not os.path.exists(obj)
                or os.path.getmtime(obj) < os.path.getmtime(src)):
            _run(["g++", "-O2", "-std=c++17", "-fPIC", *macros,
                  f"-I{_CPP_DIR}", f"-I{casinc}", f"-I{casinc}/eigen3", "-c",
                  src, "-o", obj])
        return obj

    def _load(self, so, problem):
        lib = ctypes.CDLL(so)
        d = ctypes.POINTER(ctypes.c_double)
        lib.fastsqp_create.restype = ctypes.c_void_p
        lib.fastsqp_create.argtypes = [ctypes.c_int] * 3 + [d] * 4 + [ctypes.c_int]
        lib.fastsqp_reset.argtypes = [ctypes.c_void_p, d]
        lib.fastsqp_solve.restype = ctypes.c_int
        lib.fastsqp_solve.argtypes = [ctypes.c_void_p, d, d, d, ctypes.c_int, d, d]
        lib.fastsqp_backend_name.restype = ctypes.c_char_p
        lib.fastsqp_backend_name.argtypes = [ctypes.c_void_p]
        lib.fastsqp_destroy.argtypes = [ctypes.c_void_p]
        self._lib = lib
        self.problem = problem
        lbx, ubx, lbu, ubu = problem.bounds()
        self._bounds = [np.ascontiguousarray(v, float) for v in (lbx, ubx, lbu, ubu)]
        self._h = lib.fastsqp_create(
            problem.meta.nx, problem.meta.nu, problem.meta.N,
            *[v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for v in self._bounds],
            _BACKEND_ID.get(self.backend, 0))
        self._nx, self._nu, self._N = problem.meta.nx, problem.meta.nu, problem.meta.N
        self._reset_done = False

    # ----------------------------------------------------------------- solve
    def _p(self, a):
        a = np.ascontiguousarray(a, float)
        return a, a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def solve(self, x, k) -> SolveStats:
        nx, nu, N = self._nx, self._nu, self._N
        xa, xp = self._p(x)
        if not self._reset_done:
            self._lib.fastsqp_reset(self._h, xp)
            self._reset_done = True
        xref = np.zeros((N + 1) * nx); uref = np.zeros(N * nu)
        for j in range(N + 1):
            xref[j * nx:(j + 1) * nx] = self.problem.reference_traj(k + j)[0]
        for j in range(N):
            uref[j * nu:(j + 1) * nu] = self.problem.reference_traj(k + j)[1]
        _, xrp = self._p(xref); _, urp = self._p(uref)
        u0 = np.zeros(nu); stats = np.zeros(5)
        _, u0p = self._p(u0); _, stp = self._p(stats)
        status = self._lib.fastsqp_solve(self._h, xp, xrp, urp,
                                         self.sqp_iters, u0p, stp)
        return SolveStats(
            u0=u0.copy(), success=(status == 0),
            status=f"fastsqp_status={status}", solve_time_s=float(stats[3]),
            iterations=int(stats[0]), cost=float(stats[4]),
            kkt_residual=float(stats[2]), constraint_violation=float("nan"))

    def reset(self):
        self._reset_done = False


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")


@register_solver("fastsqp")
def _factory():
    return FastSqpAdapter(backend="proxqp", sqp_iters=1)


@register_solver("fastsqp_full")
def _factory_full():
    return FastSqpAdapter(backend="proxqp", sqp_iters=5)
