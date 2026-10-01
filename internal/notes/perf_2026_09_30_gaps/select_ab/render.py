import json, sys
import numpy as np
import scaly as sc
from scaly.codegen import render_c_module
sys.path.insert(0, sys.argv[4])
mod = __import__(sys.argv[5] if len(sys.argv) > 5 else "cases")
name, out, data_out = sys.argv[1], sys.argv[2], sys.argv[3]
ins, res, data = mod.CASES[name]()
fn = sc.Function.from_exprs("ab_" + name, ins, res, [f"i{j}" for j in range(len(ins))], [f"o{j}" for j in range(len(res))])
module = render_c_module(fn)
open(out, "w").write(module.body)
sizes = [int(np.prod(o.shape)) if o.shape else 1 for o in fn.outputs]
head = np.array([len(data), len(sizes), int(module.workspace_size), *(np.asarray(a).size for a in data), *sizes], dtype="<i8").tobytes()
open(data_out, "wb").write(head + b"".join(np.ascontiguousarray(np.ravel(a), dtype="<f8").tobytes() for a in data))
print(json.dumps({"symbol": "ab_" + name, "scaly": sc.__file__}))
