"""Sparse-constant products in ordinary code: each returns (inputs, outputs, input arrays)."""
import numpy as np
import scaly as sc

RNG = np.random.default_rng(5)
CASES = {}


def _mask(n, one_in, ones=True, rng=None):
  rng = rng or np.random.default_rng(n * 131 + one_in)
  m = np.zeros(n)
  at = rng.permutation(n)[: n // one_in]
  m[at] = 1.0 if ones else rng.uniform(0.5, 2.0, at.size)
  return m


def _add(name):
  def deco(f):
    CASES[name] = f
    return f
  return deco


for n in (64, 1024, 65536):
  for d in (8, 9, 16, 64):
    def plain(n=n, d=d):
      x = sc.sym("x", n)
      return [x], [(x * sc.const(_mask(n, d))).block()], [RNG.standard_normal(n)]
    CASES[f"plain_n{n}_d{d}"] = plain

    def scaled(n=n, d=d):
      x = sc.sym("x", n)
      return [x], [(x * sc.const(_mask(n, d, ones=False))).block()], [RNG.standard_normal(n)]
    CASES[f"scaled_n{n}_d{d}"] = scaled

    def addy(n=n, d=d):  # the masked product feeds an elementwise expression
      x, y = sc.sym("x", n), sc.sym("y", n)
      return [x, y], [(x * sc.const(_mask(n, d)) + y).block()], [RNG.standard_normal(n), RNG.standard_normal(n)]
    CASES[f"addy_n{n}_d{d}"] = addy

    def chain(n=n, d=d):  # (x*mask + y) * z : fused elementwise at base
      x, y, z = sc.sym("x", n), sc.sym("y", n), sc.sym("z", n)
      return [x, y, z], [((x * sc.const(_mask(n, d, ones=False)) + y) * z).block()], [RNG.standard_normal(n) for _ in range(3)]
    CASES[f"chain_n{n}_d{d}"] = chain

    def summed(n=n, d=d):
      x = sc.sym("x", n)
      return [x], [(x * sc.const(_mask(n, d, ones=False))).sum()], [RNG.standard_normal(n)]
    CASES[f"summed_n{n}_d{d}"] = summed

    def sinx(n=n, d=d):  # computed operand
      x = sc.sym("x", n)
      return [x], [(x.sin() * sc.const(_mask(n, d))).block()], [RNG.standard_normal(n)]
    CASES[f"sinx_n{n}_d{d}"] = sinx

    def twomask(n=n, d=d):  # two masked terms summed: x*m1 + y*m2
      x, y = sc.sym("x", n), sc.sym("y", n)
      m1, m2 = _mask(n, d), _mask(n, d, rng=np.random.default_rng(n + d + 7))
      return [x, y], [(x * sc.const(m1) + y * sc.const(m2)).block()], [RNG.standard_normal(n), RNG.standard_normal(n)]
    CASES[f"twomask_n{n}_d{d}"] = twomask

for n in (32, 256):
  for d in (8, 16):
    def matvec(n=n, d=d):  # A @ (x * mask)
      a, x = sc.sym("a", (n, n)), sc.sym("x", n)
      return [a, x], [(a @ (x * sc.const(_mask(n, d)))).block()], [RNG.standard_normal((n, n)), RNG.standard_normal(n)]
    CASES[f"matvec_n{n}_d{d}"] = matvec

    def bcast(n=n, d=d):  # column broadcast against a 2-D mask
      x = sc.sym("x", (n, 1))
      return [x], [(x * sc.const(_mask(n * n, d).reshape(n, n))).block()], [RNG.standard_normal((n, 1))]
    CASES[f"bcast_n{n}_d{d}"] = bcast

    def matmask(n=n, d=d):  # (A * mask) @ x : a masked matrix times a vector
      a, x = sc.sym("a", (n, n)), sc.sym("x", n)
      return [a, x], [((a * sc.const(_mask(n * n, d).reshape(n, n))) @ x).block()], [RNG.standard_normal((n, n)), RNG.standard_normal(n)]
    CASES[f"matmask_n{n}_d{d}"] = matmask
