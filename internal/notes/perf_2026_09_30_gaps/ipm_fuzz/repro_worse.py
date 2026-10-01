"""One fuzz instance (fuzz.py's BASE and k) on whichever scaly is first on PYTHONPATH, with its trace;
with --piqp, vendored PIQP's own run of the same problem too.

  PYTHONPATH=<tree>/src SCALY_CACHE_DIR=<cache> uv run --no-sync python repro_worse.py BASE K [--piqp]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import sparse
import fuzz
import scaly as sc
import scaly.opt.ipm.kkt as kkt_mod
from scaly.opt.ipm import INFO_FIELDS, TRACE_FIELDS, QPValues, Settings, Solver
from scaly.testing.qp import QP, kkt_residuals
from tests.opt.ipm.problems import ipm_inputs

base, k = sys.argv[1], int(sys.argv[2])
qp0 = fuzz.base_qp(base)
s, v0 = ipm_inputs(qp0)
v, kind = fuzz.instance(s, v0, base, k)
n = s.n
P = sparse.coo_array((v["P"], (s.P_rows, s.P_cols)), shape=(n, n)).toarray()
P = P + np.triu(P, 1).T
qp = QP(name=f"fz_{k}", P=sparse.csc_array(P), c=v["c"], r=0.0,
        A=sparse.csc_array(sparse.coo_array((v["A"], (s.A_rows, s.A_cols)), shape=(s.p, n))), b=v["b"],
        G=sparse.csc_array(sparse.coo_array((v["G"], (s.G_rows, s.G_cols)), shape=(s.m, n))), h_l=v["h_l"], h_u=v["h_u"], x_l=v["x_l"], x_u=v["x_u"])
ORDER = fuzz.ORDER
RESULT = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "status", "iter", "trace", "trace_rows", "info")
syms = {key: sc.sym(key, np.shape(v0[key])) for key in ORDER}
tag = "".join(ch if ch.isalnum() else "_" for ch in base)
out = Solver(s, "sparse", Settings(), name=f"rw_{tag}").solve(QPValues.preprocess(s, **syms), trace=True)
fn = sc.Function.from_exprs(f"rw_{tag}_sparse", [syms[key] for key in ORDER], [out[key] for key in RESULT], list(ORDER), list(RESULT))
got = dict(zip(RESULT, fn(tuple(v[key] for key in ORDER)), strict=True))
tr = got["trace"][: int(got["trace_rows"])]
col = lambda name: tr[:, TRACE_FIELDS.index(name)]
pr, du = kkt_residuals(qp, **{key: got[key] for key in ("x", "y", "z_l", "z_u", "z_bl", "z_bu")})
tree = Path(kkt_mod.__file__).parts[-6]
rho = col("rho")
ups = int(np.sum(rho[1:] > rho[:-1] * 1.5))
steps = np.minimum(col("primal_step"), col("dual_step"))[1:]
print(f"[{tree}] {base} k={k} ({kind}): status {int(got['status'])} iter {int(got['iter'])} ir {got['info'][INFO_FIELDS.index('ir')]:.0f}  "
      f"objective {qp.objective(got['x']):.10g}  primal res {pr:.1e} dual res {du:.1e}")
print(f"[{tree}]   iterations in which a factorization retry raised rho: {ups}; max rho {rho.max():.1e}, final rho {rho[-1]:.1e}; steps under 1e-3: {int(np.sum(steps < 1e-3))}, smallest {steps.min():.1e}")
if "--trace" in sys.argv:
  for row in tr:
    print(f"[{tree}]   it {int(row[0]):3d} primal_res {row[TRACE_FIELDS.index('primal_res')]:.2e} dual_res {row[TRACE_FIELDS.index('dual_res')]:.2e} rho {row[TRACE_FIELDS.index('rho')]:.1e} delta {row[TRACE_FIELDS.index('delta')]:.1e} mu {row[TRACE_FIELDS.index('mu')]:.1e} steps {row[TRACE_FIELDS.index('primal_step')]:.1e} {row[TRACE_FIELDS.index('dual_step')]:.1e}")
if "--piqp" in sys.argv:
  from tests.opt.ipm import piqp_trace
  for dense in (False, True):
    t = piqp_trace.run(qp, dense=dense)
    print(f"[PIQP {t.backend}] status {t.status} iter {int(t.info['iter'])} max rho {t.column('rho').max():.1e}")
