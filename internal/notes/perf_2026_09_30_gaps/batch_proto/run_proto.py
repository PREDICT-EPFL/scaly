import sys, time
sys.path.insert(0, "examples/case_studies/neural_mpc"); sys.path.insert(0, "internal/notes/perf_2026_09_30_gaps"); sys.path.insert(0, sys.argv[1])
import numpy as np, scaly as sc
import scaly_impl, e2_surrogate
from batch_proto import batched_map
rng = np.random.default_rng(0)
print(f"{'size':>8s} {'mapped':>11s} {'batched':>11s} {'numpy':>11s} {'batched/mapped':>14s} {'batched/numpy':>13s}")
for size in sys.argv[2].split(","):
  layers, width = (int(v) for v in size.split("x"))
  net = e2_surrogate.params(layers, width, rng)
  xs = rng.standard_normal(20)
  fn = scaly_impl.taylor_function(net).concrete
  (root,) = fn.outputs
  out = batched_map(root)
  assert out is not None
  fb = sc.Function.from_exprs(f"taylor_batched_{layers}_{width}", list(fn.inputs), [out], list(fn.input_names), ["y"]).concrete
  a, b = fn._flat_numerical_call(xs)[0], fb._flat_numerical_call(xs)[0]
  np.testing.assert_allclose(np.asarray(b).reshape(-1), np.asarray(a).reshape(-1), rtol=1e-9, atol=1e-10)
  tm, tb, tn = (e2_surrogate.best(c) for c in (lambda: fn._flat_numerical_call(xs), lambda: fb._flat_numerical_call(xs), lambda: e2_surrogate.batched(net, xs)))
  print(f"{size:>8s} {tm*1e6:9.1f} us {tb*1e6:9.1f} us {tn*1e6:9.1f} us {tb/tm:14.3f} {tb/tn:13.2f}", flush=True)
