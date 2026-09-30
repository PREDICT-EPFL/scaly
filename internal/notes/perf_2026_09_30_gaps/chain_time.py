import sys, time
import numpy as np
sys.path.insert(0, ".")
import scaly as sc
from bench.problems import chain as ch
from scaly.opt.nlp import nlp_oracles
from scaly.function import ConcreteFunction
M = int(sys.argv[1]); N = ch.HORIZON
nv = ch.n_dec(M, N)
@sc.opt.problem(vars=sc.L("z", nv), params=sc.L("p", ch.n_param(M)), name=f"chain_M{M}")
def problem(z, p):
    return sc.opt.ProblemSpec(minimize=ch._objective(z, M, N), eq=(ch._chain_eq_expr(z, p, M, N),))
o = nlp_oracles(problem)
h = o.hess.triangle("lower")
fn = ConcreteFunction.from_exprs(f"chain_hess_M{M}", o.hess_inputs, (h.values,), tuple(f"in{i}" for i in range(len(o.hess_inputs))), ("h",))
rng = np.random.default_rng(0)
args = []
for e in o.hess_inputs:
    a = rng.standard_normal(e.shape) * 0.1
    args.append(a)
# physical params: mass, spring, rest, g, dt
p = args[1]; p[-5:] = [0.033, 1.0, 0.033, -9.81, 0.2]; args[0] = args[0] + 0.1
fn(tuple(args))
best = 1e9
for r in range(7):
    t = time.perf_counter(); K = 50
    for _ in range(K): fn(tuple(args))
    best = min(best, (time.perf_counter() - t) / K)
print(f"M={M} best {best*1e6:.1f} us")
