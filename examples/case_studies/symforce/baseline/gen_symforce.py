"""Generate SymForce's robot 3D localization example for `n` poses, with SymForce's own code generator.

    <python with symforce> gen_symforce.py <symforce checkout> <n> <out dir> [--no-fixed]

Runs the example's `generate()` (symforce/examples/robot_3d_localization/robot_3d_localization.py)
with its module constant `NUM_POSES` set to `n`: the per-factor functions of the dynamic variant, the
whole-problem linearization of the fixed variant, the keys, and the measurements from the example's own
seeded generator, into `<out>/gen`. `--no-fixed` generates everything but the whole-problem
linearization. Prints the generation time as JSON.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

checkout, n, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
spec = importlib.util.spec_from_file_location(
  "robot_3d_localization", checkout / "symforce" / "examples" / "robot_3d_localization" / "robot_3d_localization.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.NUM_POSES = n
gen = out / "gen"
gen.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
if "--no-fixed" in sys.argv:
  mod.build_codegen_object = lambda num_poses, config=None: type("Skip", (), {"generate_function": lambda self, *a, **k: None})()
mod.generate(gen)
lin = gen / "linearization.h"
print(json.dumps({"n": n, "t_generate": time.perf_counter() - t0, "linearization_bytes": lin.stat().st_size if lin.exists() else 0}))
