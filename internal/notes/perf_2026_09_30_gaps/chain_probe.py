import sys, re, time
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import scaly as sc
from bench.problems import chain as ch
from scaly.opt.nlp import nlp_oracles
from scaly.function import ConcreteFunction
from scaly.codegen import render_c_module
M = int(sys.argv[1]); N = ch.HORIZON
nx, nz = ch.n_state(M), ch.n_state(M) + ch.NU
nv = ch.n_dec(M, N)
@sc.opt.problem(vars=sc.L("z", nv), params=sc.L("p", ch.n_param(M)), name=f"chain_M{M}")
def problem(z, p):
    return sc.opt.ProblemSpec(minimize=ch._objective(z, M, N), eq=(ch._chain_eq_expr(z, p, M, N),))
t=time.time()
o = nlp_oracles(problem)
h = o.hess.triangle("lower")
print("colouring width", h.coloring_width, "nnz", h.sparsity.nnz, "build s", round(time.time()-t,1))
fn = ConcreteFunction.from_exprs(f"chain_hess_M{M}", o.hess_inputs, (h.values,), tuple(f"in{i}" for i in range(len(o.hess_inputs))), ("h",))
t=time.time(); mod = render_c_module(fn); print("render s", round(time.time()-t,1))
src = mod.body
Path(f"{sys.argv[2]}/chain_{M}.c").write_text(src)
heads = [(m.start(), m.group(1)) for m in re.finditer(r"^(?:static )?(?:inline )?(?:__attribute__\(\([^)]*\)\) )*(?:void|int) (\w+)\([^;]*\{\s*$", src, re.M)]
heads.append((len(src), None))
for (a, name), (b, _) in zip(heads, heads[1:]):
    f = src[a:b]; lines = f.count("\n")
    if lines > 20:
        print(f"{name[:80]:80s} lines={lines:6d} for={len(re.findall(r'for ?\(', f)):4d} div={f.count('/'):5d} mul={f.count('*'):6d} sqrt={f.count('sqrt('):4d}")
print("workspace", mod.workspace_size)
