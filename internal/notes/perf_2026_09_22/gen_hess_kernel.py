import sys, time
from pathlib import Path
sys.path.insert(0, ".")
from benchmarks.harness.sweep import _descriptor_kernel, _render_scaly
from benchmarks.problems.race_cars.closed_loop import EpisodeConfig, _race_car_nlp
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
t = time.perf_counter()
kernel, sparsity, cw = _descriptor_kernel(_race_car_nlp(EpisodeConfig(horizon=N)), "hess")
print("build", time.perf_counter() - t, "coloring width", cw, "nnz", sparsity.nnz)
module, ms = _render_scaly(kernel, kernel.name, Path(sys.argv[2] if len(sys.argv) > 2 else "."))
print("render ms", ms, "w", module.workspace_size, "lines", module.source.count("\n"))
print([ (i.name, i.shape) for i in kernel.inputs], [(o.name, o.shape) for o in kernel.outputs])
