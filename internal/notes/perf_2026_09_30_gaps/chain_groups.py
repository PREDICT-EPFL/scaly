"""The chain's stage Hessian with its seeds in groups that fit the instruction cache (C-211, Tier 3).

    uv run internal/notes/perf_2026_09_30_gaps/chain_groups.py [M ...]

For each number of masses M (default 5 7 9) and each body budget (the host's ``body_bytes``, one
body, and the budgets swept in the plan): the groups ``ad.forward._seed_groups`` makes, the work
the mapped bodies generate (``_body_ops`` times trips, summed), and the time from C of
``chain_cliff.py``'s Hessian, fastest of 9 samples after a 200 ms warm-up. Only the grouping varies:
the budget is patched into ``Target.body_bytes``, since changing ``l1i_bytes`` would also move the
straight-line budget. Writes ``results/chain_groups.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain_cliff import BUILD, hessian  # noqa: E402

BUDGETS = {"host": None, "one": 1 << 30, "full_l1i": 196608, "quarter_l1i": 49152, "l1i_32k": 16384}


def main() -> None:
  from corpus import blob
  from scaly.ad import forward
  from scaly.codegen import render_c_module
  from scaly.codegen.jit import compile_flags
  from scaly.ir import target as target_module
  from scaly.ir.expr import ExprOp, topo

  ms = [int(v) for v in sys.argv[1:]] or [5, 7, 9]
  BUILD.mkdir(parents=True, exist_ok=True)
  driver = HERE / "build" / "time_entry"
  if not driver.exists():
    subprocess.run(["cc", "-O2", str(HERE.parent / "perf_2026_09_30_codegen" / "time_entry.c"), "-o", str(driver)], check=True)
  host_bytes = target_module.Target.body_bytes
  groups: list[int] = []
  seed_groups = forward._seed_groups

  def spy(callee, fn, nseed):
    made = seed_groups(callee, fn, nseed)
    groups.append(len(made))
    return made

  forward._seed_groups = spy
  rows = []
  for m in ms:
    for tag, budget in BUDGETS.items():
      target_module.Target.body_bytes = host_bytes if budget is None else property(lambda self, b=budget: b)
      groups.clear()
      fn, args, _ = hessian(m)
      maps = [e for e in topo(list(fn.outputs)) if e.op == ExprOp.VMAP]
      work = sum(forward._body_ops(e.attrs["callee"]) * e.attrs["length"] for e in maps)
      module = render_c_module(fn)
      c, lib, data = BUILD / f"groups_{m}_{tag}.c", BUILD / f"groups_{m}_{tag}.so", BUILD / f"groups_{m}_{tag}.bin"
      c.write_text(module.body)
      subprocess.run(["cc", *compile_flags(), "-fPIC", "-shared", str(c), "-lm", "-o", str(lib)], check=True)
      data.write_bytes(blob([np.ravel(a) for a in args], [int(fn.outputs[0].size)], int(module.workspace_size)))
      t = time.perf_counter()
      while time.perf_counter() - t < 0.2:
        subprocess.run([str(driver), str(lib), fn.name, str(data), "3", "/dev/null"], capture_output=True, check=True)
      best = min(
        float(
          subprocess.run([str(driver), str(lib), fn.name, str(data), "5", "/dev/null"], capture_output=True, text=True, check=True).stdout.split()[0]
        )
        for _ in range(9)
      )
      row = {"M": m, "budget": tag, "body_bytes": budget, "groups": max(groups, default=1), "work": work, "time_us": best / 1e3}
      rows.append(row)
      print(json.dumps(row), flush=True)
  target_module.Target.body_bytes = host_bytes
  (HERE / "results").mkdir(exist_ok=True)
  (HERE / "results" / "chain_groups.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
