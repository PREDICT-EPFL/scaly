"""PR 6 (T2-6): generated dense kernels against LAPACK/BLAS (SciPy's OpenBLAS, called through ctypes).

Per operation and order ``n``: Scaly runs a ``vmap`` of ``K`` independent instances in one call, so
the ~3 µs call overhead is spread over ``K``; LAPACK is called ``K`` times from Python on
preallocated copies, and the cost of an empty call (``n = 0``) is subtracted. Both report
microseconds per instance.

``potrf``  Cholesky (``cholesky``); ``ldl`` the no-pivot ``L D L^T`` (compared with ``potrf``,
the same flop count: LAPACK has no unpivoted LDL^T); ``trsv`` one lower triangular solve;
``trsm`` a lower solve with ``n`` right-hand sides; ``gemm`` ``A @ B``.

Usage: ``SCALY_CC_OPT=-O3 python pr6_dense.py`` (``-O2`` is the JIT default).
"""

from __future__ import annotations

import ctypes
import os
import time

import numpy as np
from scipy.linalg import cython_blas, cython_lapack

import scaly as sc
from scaly.linalg import cholesky, ldl, solve_triangular

K = 64
INT = ctypes.POINTER(ctypes.c_int)
DBL = ctypes.POINTER(ctypes.c_double)
CHR = ctypes.c_char_p


def _function(module, name: str, *argtypes):
  cap = module.__pyx_capi__[name]
  ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
  ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
  ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
  ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
  return ctypes.CFUNCTYPE(None, *argtypes)(ctypes.pythonapi.PyCapsule_GetPointer(cap, ctypes.pythonapi.PyCapsule_GetName(cap)))


POTRF = _function(cython_lapack, "dpotrf", CHR, INT, DBL, INT, INT)
TRSV = _function(cython_blas, "dtrsv", CHR, CHR, CHR, INT, DBL, INT, DBL, INT)
TRSM = _function(cython_blas, "dtrsm", CHR, CHR, CHR, CHR, INT, INT, DBL, DBL, INT, DBL, INT)
GEMM = _function(cython_blas, "dgemm", CHR, CHR, INT, INT, INT, DBL, DBL, INT, DBL, INT, DBL, DBL, INT)


def ptr(a: np.ndarray):
  return a.ctypes.data_as(DBL)


def ref(x: int):
  return ctypes.byref(ctypes.c_int(x))


def lapack_us(op: str, n: int, mats: list[np.ndarray]) -> float:
  info, one, zero = ctypes.c_int(0), ctypes.c_double(1.0), ctypes.c_double(0.0)
  rhs = [np.ones((n, n)) for _ in mats]
  out = [np.zeros((n, n)) for _ in mats]
  nn, ld, inc = ctypes.c_int(n), ctypes.c_int(max(n, 1)), ctypes.c_int(1)
  calls = {
    "potrf": lambda a, b, c: POTRF(b"L", ctypes.byref(nn), ptr(a), ctypes.byref(ld), ctypes.byref(info)),
    "ldl": lambda a, b, c: POTRF(b"L", ctypes.byref(nn), ptr(a), ctypes.byref(ld), ctypes.byref(info)),
    "trsv": lambda a, b, c: TRSV(b"L", b"N", b"N", ctypes.byref(nn), ptr(a), ctypes.byref(ld), ptr(b), ctypes.byref(inc)),
    "trsm": lambda a, b, c: TRSM(b"L", b"L", b"N", b"N", ctypes.byref(nn), ctypes.byref(nn), ctypes.byref(one), ptr(a), ctypes.byref(ld), ptr(b), ctypes.byref(ld)),
    "gemm": lambda a, b, c: GEMM(b"N", b"N", ctypes.byref(nn), ctypes.byref(nn), ctypes.byref(nn), ctypes.byref(one), ptr(a), ctypes.byref(ld), ptr(b), ctypes.byref(ld), ctypes.byref(zero), ptr(c), ctypes.byref(ld)),
  }
  call = calls[op]
  best = np.inf
  for _ in range(15):
    copies = [m.copy(order="F") for m in mats]
    t0 = time.perf_counter()
    for a, b, c in zip(copies, rhs, out, strict=True):
      call(a, b, c)
    best = min(best, time.perf_counter() - t0)
  empty = np.inf
  n0 = ctypes.c_int(0)
  for _ in range(15):
    t0 = time.perf_counter()
    for _ in mats:
      POTRF(b"L", ctypes.byref(n0), ptr(mats[0]), ctypes.byref(ld), ctypes.byref(info))
    empty = min(empty, time.perf_counter() - t0)
  return (best - empty) / len(mats) * 1e6


def scaly_us(op: str, n: int, mats: list[np.ndarray]) -> float:
  a, b = sc.sym("a", (n, n)), sc.sym("b", (n, n))
  body = {
    "potrf": lambda: cholesky(a),
    "ldl": lambda: ldl(a),
    "trsv": lambda: solve_triangular(a, b[:, 0], lower=True),
    "trsm": lambda: solve_triangular(a, b, lower=True),
    "gemm": lambda: a @ b,
  }[op]()
  callee = sc.Function._from_exprs(f"p6_{op}{n}", [a, b], [body.reshape((body.size,))], ["a", "b"], ["o"])
  stack_a, stack_b = sc.sym("sa", K * n * n), sc.sym("sb", K * n * n)
  fn = sc.Function._from_exprs(f"p6v_{op}{n}", [stack_a, stack_b], [sc.vmap(callee, K, [(stack_a, 0, n * n), (stack_b, 0, n * n)])], ["sa", "sb"], ["o"])
  av = np.concatenate([m.reshape(-1) for m in mats])
  bv = np.ones(K * n * n)
  fn._flat_numerical_call(av, bv)
  x = sc.sym("x", ())
  trivial = sc.Function._from_exprs("p6_trivial", [x], [x * 2.0], ["x"], ["y"])
  best = min(_time(lambda: fn._flat_numerical_call(av, bv)) for _ in range(15))
  empty = min(_time(lambda: trivial._flat_numerical_call(np.array(1.0))) for _ in range(15))
  return (best - empty) / K * 1e6


def _time(f) -> float:
  t0 = time.perf_counter()
  for _ in range(5):
    f()
  return (time.perf_counter() - t0) / 5


def main() -> None:
  rng = np.random.default_rng(0)
  print(f"opt={os.environ.get('SCALY_CC_OPT', '-O2')}", flush=True)
  for n in (4, 8, 16, 32, 64):
    mats = []
    for _ in range(K):
      m = rng.standard_normal((n, n))
      mats.append(m @ m.T + n * np.eye(n))
    for op in ("potrf", "ldl", "trsv", "trsm", "gemm"):
      s, lp = scaly_us(op, n, mats), lapack_us(op, n, mats)
      print(f"{op:<6} n={n:>3} scaly={s:>8.3f} us  lapack={lp:>8.3f} us  ratio={s / max(lp, 1e-3):>5.2f}", flush=True)


if __name__ == "__main__":
  main()
