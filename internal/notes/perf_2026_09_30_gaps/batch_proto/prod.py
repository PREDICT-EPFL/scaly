import sys, time
sys.path.insert(0, "internal/notes/perf_2026_09_30_gaps")
import numpy as np, scaly as sc
from e2_surrogate import best
rng = np.random.default_rng(0)
print(f"{'m x k x n':>16s} {'b':>9s} {'time':>10s} {'GF/s':>6s}")
for m, k, n in [(5, 512, 512), (10, 512, 512), (11, 512, 512), (20, 512, 512), (30, 512, 512), (32, 512, 512), (33, 512, 512), (64, 512, 512), (30, 128, 128), (30, 256, 256), (30, 500, 500), (30, 520, 520), (10, 500, 500)]:
  bv = rng.standard_normal((k, n))
  a, b = sc.sym("a", (m, k)), sc.sym("b", (k, n))
  for label, fn, args in (
    ("input", sc.Function.from_exprs(f"p_in_{m}_{k}_{n}", [a, b], [a @ b], ["a", "b"], ["c"]).concrete, lambda av: (av, bv)),
    ("constant", sc.Function.from_exprs(f"p_c_{m}_{k}_{n}", [a], [a @ sc.const(bv)], ["a"], ["c"]).concrete, lambda av: (av,)),
  ):
    av = rng.standard_normal((m, k))
    got = fn._flat_numerical_call(*args(av))[0]
    np.testing.assert_allclose(np.asarray(got).reshape(m, n), av @ bv, rtol=1e-9, atol=1e-9)
    t = best(lambda: fn._flat_numerical_call(*args(av)), floor=0.1)
    print(f"{m:4d}x{k:4d}x{n:4d} {label:>9s} {t*1e6:8.1f} us {m*k*n/t/1e9:6.1f}", flush=True)
