"""Save the arguments of 30 mid-episode SQP solver calls, for the C replay harness of this note."""

import os, json, numpy as np
from pathlib import Path
from scaly.codegen import jit
OUT = Path(os.environ["ZIGPROF_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
orig = jit.CompiledFunction.run
count = {"n": 0}
def run(self, args):
  name = self._fun.name
  if "sqp_nlp" in name:
    count["n"] += 1
    if 10 <= count["n"] < 40:
      d = OUT / f"call{count['n']:03d}"; d.mkdir(exist_ok=True)
      for i, a in enumerate(args):
        np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tofile(d / f"arg{i}.bin")
      opts = []
      if self._options:
        p = self._options[0]; k = 0
        while p[k].name:
          o = p[k]; opts.append([o.name.decode(), o.kind, o.integer, o.number, o.text.decode() if o.text else None]); k += 1
      meta = {"name": name, "lib": str(self.lib_path), "n_args": self._n_args, "arg_sizes": [int(np.asarray(a).size) for a in args],
              "res_sizes": list(self._output_sizes), "sz_w": self._sz_w, "options": opts}
      (d / "meta.json").write_text(json.dumps(meta))
  return orig(self, args)
jit.CompiledFunction.run = run
