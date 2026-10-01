"""Shapes for the C-217 A/B: each returns (inputs, outputs, input arrays)."""
import numpy as np
import scaly as sc

N = 1024
RNG = np.random.default_rng(11)


def _vec(names, n=N):
  return [sc.sym(k, n) for k in names]


def div_chain_rare():  # untaken branch expensive (three divisions), mask 2% true
  x, a, b, c, d = _vec("xabcd")
  data = [np.where(RNG.random(N) < 0.02, 1.0, -1.0), *(RNG.uniform(1.0, 2.0, N) for _ in range(4))]
  return [x, a, b, c, d], [sc.where(x > 0.0, a / b / c / d, 0.0).block()], data


def div_chain_half():  # same, mask random 50%
  x, a, b, c, d = _vec("xabcd")
  data = [RNG.standard_normal(N), *(RNG.uniform(1.0, 2.0, N) for _ in range(4))]
  return [x, a, b, c, d], [sc.where(x > 0.0, a / b / c / d, 0.0).block()], data


def div_chain_always():
  x, a, b, c, d = _vec("xabcd")
  data = [np.ones(N), *(RNG.uniform(1.0, 2.0, N) for _ in range(4))]
  return [x, a, b, c, d], [sc.where(x > 0.0, a / b / c / d, 0.0).block()], data


def relu():
  (x,) = _vec("x")
  return [x], [sc.where(x > 0.0, x, 0.0).block()], [RNG.standard_normal(N)]


def piecewise():  # three pieces, each with loads
  x, a, b, c = _vec("xabc")
  y = sc.where(x < 0.0, a * x, sc.where(x < 1.0, b * x * x + a, c * x + b / (1.0 + x * x)))
  return [x, a, b, c], [y.block()], [RNG.uniform(-1.0, 2.0, N), *(RNG.standard_normal(N) for _ in range(3))]


def piecewise_sorted():  # the same with x sorted: branches predict perfectly
  ins, outs, data = piecewise()
  data[0] = np.sort(data[0])
  return ins, outs, data


def masked_sum():  # a reduction over a select
  x, a, b = _vec("xab")
  return [x, a, b], [sc.where(x > 0.0, a / b, 0.0).sum()], [RNG.standard_normal(N), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


def masked_sum_rare():
  x, a, b = _vec("xab")
  return [x, a, b], [sc.where(x > 0.0, a / b, 0.0).sum()], [np.where(RNG.random(N) < 0.02, 1.0, -1.0), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


def short_circuit():
  x, a, b = _vec("xab")
  keep = sc.logical_and(x > 0.0, a / b > 0.5)
  return [x, a, b], [sc.where(keep, a, b).block()], [np.where(RNG.random(N) < 0.02, 1.0, -1.0), RNG.uniform(0.0, 2.0, N), RNG.uniform(1.0, 2.0, N)]


def safe_div():  # the guarded-division idiom
  x, a = _vec("xa")
  return [x, a], [sc.where(x.abs() > 1e-9, a / x, 0.0).block()], [np.where(RNG.random(N) < 0.5, 0.0, RNG.standard_normal(N)), RNG.standard_normal(N)]


def clip_lookup():  # a table lookup through a clamped index, under a select
  x, a = _vec("xa")
  table = np.linspace(0.0, 1.0, 257) ** 2
  idx = sc.cast(sc.minimum(sc.maximum((x * 256.0).floor(), 0.0), 255.0), "int64")
  lo, hi = sc.take(table, idx, in_range=True), sc.take(table, idx + 1, in_range=True)
  t = x * 256.0 - (x * 256.0).floor()
  y = sc.where(x < 0.0, 0.0, sc.where(x > 1.0, 1.0, lo + t * (hi - lo)))
  return [x, a], [(y * a).block()], [RNG.uniform(-0.2, 1.2, N), RNG.standard_normal(N)]


def scan_piecewise():  # a scalar-expanded body in a scan: states advance through a piecewise map
  n = 6
  c, u = sc.sym("c", n), sc.sym("u", n)
  nxt = sc.where(c > 0.5, c / (1.0 + u * u) / (2.0 + c * c), sc.where(c < -0.5, c * u + 0.1, 0.9 * c + u / (3.0 + u * u)))
  body = sc.Function.from_exprs("scan_body", [c, u], [nxt], ["c", "u"], ["n"])
  init, us = sc.sym("init", n), sc.sym("us", 200 * n)
  out = sc.scan(body, init, [(us, 0, n)], length=200)
  return [init, us], [out[0]], [RNG.uniform(-1.0, 1.0, n), RNG.uniform(-1.0, 1.0, 200 * n)]


def vmap_piecewise():  # a mapped scalar body: one element per call
  x, a, b = sc.sym("x", ()), sc.sym("a", ()), sc.sym("b", ())
  y = sc.where(x > 0.0, a / (1.0 + x) / (2.0 + b * b), sc.where(x < -1.0, a * b, a + b / (1.5 + x * x)))
  body = sc.Function.from_exprs("vmap_body", [x, a, b], [y], ["x", "a", "b"], ["y"])
  xs, as_, bs = _vec(["xs", "as", "bs"])
  out = sc.vmap(body, N, [(xs, 0, 1), (as_, 0, 1), (bs, 0, 1)])
  return [xs, as_, bs], [out], [RNG.uniform(-2.0, 1.0, N), RNG.standard_normal(N), RNG.standard_normal(N)]


def maxmin_reduce():
  (x,) = _vec("x")
  return [x], [sc.stack([x.max(), x.min()])], [RNG.standard_normal(N)]


CASES = {f.__name__: f for f in (div_chain_rare, div_chain_half, div_chain_always, relu, piecewise, piecewise_sorted, masked_sum, masked_sum_rare, short_circuit, safe_div, clip_lookup, scan_piecewise, vmap_piecewise, maxmin_reduce)}


def _rare(p=0.02):
  return np.where(RNG.random(N) < p, 1.0, -1.0)


def piecewise3_div():
  x, a, b, c = _vec("xabc")
  y = sc.where(x < 0.0, a / (1.0 + x * x), sc.where(x < 1.0, b / (2.0 + x), c / (3.0 + x)))
  return [x, a, b, c], [y.block()], [RNG.uniform(-1.0, 2.0, N), *(RNG.standard_normal(N) for _ in range(3))]


def piecewise3_div_sorted():
  ins, outs, data = piecewise3_div()
  data[0] = np.sort(data[0])
  return ins, outs, data


def masked_dot_rare():
  x, a, b, c = _vec("xabc")
  return [x, a, b, c], [sc.dot(sc.where(x > 0.0, a / b, 0.0), c)], [_rare(), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N), RNG.standard_normal(N)]


def masked_norm_inf_rare():
  x, a, b = _vec("xab")
  return [x, a, b], [sc.norm_inf(sc.where(x > 0.0, a / b, 0.0))], [_rare(), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


def masked_sum_mul_rare():
  x, a, b = _vec("xab")
  return [x, a, b], [sc.where(x > 0.0, a * b, 0.0).sum()], [_rare(), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


def masked_sum_div3_rare():
  x, a, b, c, d = _vec("xabcd")
  return [x, a, b, c, d], [sc.where(x > 0.0, a / b / c / d, 0.0).sum()], [_rare(), *(RNG.uniform(1.0, 2.0, N) for _ in range(4))]


def masked_sum_div3_half():
  x, a, b, c, d = _vec("xabcd")
  return [x, a, b, c, d], [sc.where(x > 0.0, a / b / c / d, 0.0).sum()], [RNG.standard_normal(N), *(RNG.uniform(1.0, 2.0, N) for _ in range(4))]


def masked_sum_never():  # the mask never true: the branch never taken
  x, a, b = _vec("xab")
  return [x, a, b], [sc.where(x > 0.0, a / b, 0.0).sum()], [-np.ones(N), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


def masked_matvec_rare():  # rows of a masked quotient times a vector
  m_, n_ = 32, 32
  x, a, b, v = sc.sym("x", (m_, n_)), sc.sym("a", (m_, n_)), sc.sym("b", (m_, n_)), sc.sym("v", n_)
  return [x, a, b, v], [sc.where(x > 0.0, a / b, 0.0) @ v], [np.where(RNG.random((m_, n_)) < 0.02, 1.0, -1.0), RNG.standard_normal((m_, n_)), RNG.uniform(1.0, 2.0, (m_, n_)), RNG.standard_normal(n_)]


CASES.update({f.__name__: f for f in (piecewise3_div, piecewise3_div_sorted, masked_dot_rare, masked_norm_inf_rare, masked_sum_mul_rare, masked_sum_div3_rare, masked_sum_div3_half, masked_sum_never, masked_matvec_rare)})


def _inv(n, flag):
  f, a, b = sc.sym("f", ()), sc.sym("a", n), sc.sym("b", n)
  return [f, a, b], [sc.where(f > 0.5, a, b).block()], [np.array([flag]), RNG.standard_normal(n), RNG.standard_normal(n)]


def inv_small():
  return _inv(1024, 1.0)


def inv_big():  # 16 MiB per array: past the caches
  return _inv(1 << 21, 1.0)


def inv_div_big():
  n = 1 << 21
  f, a, b, c = sc.sym("f", ()), sc.sym("a", n), sc.sym("b", n), sc.sym("c", n)
  return [f, a, b, c], [sc.where(f > 0.5, a, b / c).block()], [np.array([1.0]), RNG.standard_normal(n), RNG.standard_normal(n), RNG.uniform(1.0, 2.0, n)]


def masked_sum_always():
  x, a, b = _vec("xab")
  return [x, a, b], [sc.where(x > 0.0, a / b, 0.0).sum()], [np.ones(N), RNG.standard_normal(N), RNG.uniform(1.0, 2.0, N)]


CASES.update({f.__name__: f for f in (inv_small, inv_big, inv_div_big, masked_sum_always)})
