"""Section 6 demo: the C does not depend on the hash seed, and a second process skips lowering.

Run from a checkout: `uv run internal/notes/compiler_run_2026_11_10/jit_cache/demo.py`.
Every child process gets its own `PYTHONHASHSEED`; all of them share one fresh cache.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WARM = ROOT / "tests/codegen/jit_warm_start.py"
RENDER = """
import hashlib, scaly as sc
from scaly.codegen.aot import render_c_module
from scaly.function.model import as_concrete
@sc.function(sc.group(sc.arg("x"), sc.arg("p", ())), outputs=sc.group(sc.arg("y"), sc.group(sc.arg("cost", ()))))
def stage(inputs):
  x, p = inputs
  return x * p, ((x * x).sum(),)
f = as_concrete(sc.vmap(stage, 3).instantiate(((3, 2), (3,))))
print(hashlib.sha256(render_c_module(f).body.encode()).hexdigest()[:16])
"""


def run(args, **env):
  out = subprocess.run(args, cwd=ROOT, env={**os.environ, **env}, capture_output=True, text=True, check=True)
  return out.stdout.strip().splitlines()[-1]


print("== #163: the mapped stage of the issue, rendered under five hash seeds ==")
digests = {seed: run([sys.executable, "-c", RENDER], PYTHONHASHSEED=seed) for seed in ("0", "1", "4", "5", "7")}
for seed, digest in digests.items():
  print(f"PYTHONHASHSEED={seed}: sha256 of the C {digest}")
print("one C for every seed:", len(set(digests.values())) == 1, "(on main before #317, seeds 4 and 5 gave another)")

print("\n== #64: the cross-process cache test of Milestone 1 ==")
with tempfile.TemporaryDirectory() as cache:
  runs = [
    ("first process, PYTHONHASHSEED=1", {"PYTHONHASHSEED": "1"}),
    ("second process, PYTHONHASHSEED=2", {"PYTHONHASHSEED": "2"}),
    ("third process, PYTHONHASHSEED=3, SCALY_JIT_KEY=verify", {"PYTHONHASHSEED": "3", "SCALY_JIT_KEY": "verify"}),
  ]
  results = []
  for label, env in runs:
    record = json.loads(run([sys.executable, str(WARM)], SCALY_CACHE_DIR=cache, **env))
    results.append(record)
    library = Path(record["library"])
    print(
      f"{label}: lower_function calls {record['lowered']}, result {record['result']:.12g}, library <cache>/{library.parent.name[:12]}…/{library.name}"
    )
  print("index entries:", len(list(Path(cache, "index").glob("*.json"))))
  same = len({(r["result"], r["library"]) for r in results}) == 1
  print("same result from the same library in all three:", same)
  print("second process lowered nothing:", results[1]["lowered"] == 0)
