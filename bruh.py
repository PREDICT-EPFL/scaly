import importlib.util
from pathlib import Path
import numpy as np
import alloy as al
from alloy.viz import clear_recordings, recording_path, visualize

spec = importlib.util.spec_from_file_location("tracking", Path("tests/alloy/test_tracking_workload.py"))
assert spec is not None and spec.loader is not None
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)

N = 5
clear_recordings(disk=True)

fn = tracking.tracking_eq_function_map(N)
spjf = al.spjacobian(fn, "z", "eq")

visualize(spjf, label=f"tracking map spjac N={N}")

rng = np.random.default_rng(0)
zv = rng.normal(size=tracking.NZ * (N + 1))
out = spjf(zv)
sparsity = spjf.output_sparsities[0]
assert isinstance(out, np.ndarray) and sparsity is not None

print("recording:", recording_path())
print("out shape:", out.shape)
print("nnz:", sparsity.nnz)
print("recording bytes:", recording_path().stat().st_size)
