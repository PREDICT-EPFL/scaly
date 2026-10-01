"""Gathers of concatenations in ordinary code: each returns (inputs, outputs, input arrays)."""
import numpy as np
import scaly as sc

RNG = np.random.default_rng(9)
CASES = {}

for n in (16, 256, 4096):
  def perm3(n=n):  # a permutation of [x picked, constants, y picked]
    x, y = sc.sym("x", 2 * n), sc.sym("y", 2 * n)
    r = np.random.default_rng(n)
    joined = sc.concat([sc.gather(x, r.permutation(2 * n)[:n]), sc.const(r.standard_normal(n)), sc.gather(y, r.permutation(2 * n)[:n])])
    return [x, y], [sc.gather(joined, r.permutation(3 * n)).block()], [RNG.standard_normal(2 * n), RNG.standard_normal(2 * n)]
  CASES[f"perm3_n{n}"] = perm3

  def perm4(n=n):  # four picked parts, permuted
    xs = [sc.sym(f"x{i}", 2 * n) for i in range(4)]
    r = np.random.default_rng(n + 1)
    joined = sc.concat([sc.gather(x, r.permutation(2 * n)[:n]) for x in xs])
    return xs, [sc.gather(joined, r.permutation(4 * n)).block()], [RNG.standard_normal(2 * n) for _ in xs]
  CASES[f"perm4_n{n}"] = perm4

  def sub2(n=n):  # a sorted subset read from two picked parts
    x, y = sc.sym("x", 2 * n), sc.sym("y", 2 * n)
    r = np.random.default_rng(n + 2)
    joined = sc.concat([sc.gather(x, r.permutation(2 * n)[:n]), sc.gather(y, r.permutation(2 * n)[:n])])
    return [x, y], [sc.gather(joined, np.sort(r.permutation(2 * n)[:n])).block()], [RNG.standard_normal(2 * n), RNG.standard_normal(2 * n)]
  CASES[f"sub2_n{n}"] = sub2

  def interleave(n=n):  # x and a constant interleaved: [x0, c0, x1, c1, ...] from picked parts
    x = sc.sym("x", n)
    r = np.random.default_rng(n + 3)
    joined = sc.concat([sc.gather(x, np.arange(n)[::-1].copy()), sc.const(r.standard_normal(n))])
    picks = np.stack([np.arange(n), n + np.arange(n)], axis=1).reshape(-1)
    return [x], [sc.gather(joined, picks).block()], [RNG.standard_normal(n)]
  CASES[f"interleave_n{n}"] = interleave

  def perm3_then(n=n):  # the permuted vector feeds an elementwise expression
    x, y, z = sc.sym("x", 2 * n), sc.sym("y", 2 * n), sc.sym("z", 3 * n)
    r = np.random.default_rng(n)
    joined = sc.concat([sc.gather(x, r.permutation(2 * n)[:n]), sc.const(r.standard_normal(n)), sc.gather(y, r.permutation(2 * n)[:n])])
    return [x, y, z], [(sc.gather(joined, r.permutation(3 * n)) * z + z).block()], [RNG.standard_normal(2 * n), RNG.standard_normal(2 * n), RNG.standard_normal(3 * n)]
  CASES[f"perm3_then_n{n}"] = perm3_then

  def scat2(n=n):  # two scattered blocks joined, a subset read once
    x, y = sc.sym("x", n), sc.sym("y", n)
    r = np.random.default_rng(n + 5)
    joined = sc.concat([sc.scatter(x, r.permutation(2 * n)[:n], (2 * n,)), sc.scatter(y, r.permutation(2 * n)[:n], (2 * n,))])
    return [x, y], [sc.gather(joined, r.permutation(4 * n)[: 2 * n]).block()], [RNG.standard_normal(n), RNG.standard_normal(n)]
  CASES[f"scat2_n{n}"] = scat2
