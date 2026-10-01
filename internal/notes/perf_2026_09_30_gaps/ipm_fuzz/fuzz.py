"""One structure, many value sets, on whichever scaly is first on PYTHONPATH (sparse backend unless told).
The solver is specialized to the structure only, so one compile serves every instance.

  PYTHONPATH=<tree>/src SCALY_CACHE_DIR=<cache> uv run --no-sync python fuzz.py BASE COUNT OUT [backend] [settings-json]

BASE: a problem name (Maros-Meszaros, mpc_*, ex_*), or rnd:n:m:p:seed[:lp].
Instance k of a base is the same in every tree: the generator is seeded by (base, k).
"""
import json, sys, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "opt" / "qp_solvers"))
sys.path.insert(0, str(ROOT / "internal/notes/perf_2026_09_27_ipm_speed"))
import numpy as np
import scaly as sc
import scaly.opt.ipm.kkt as kkt_mod
from scaly.opt.ipm import INFO_FIELDS, QPValues, Settings, Solver
from scaly.testing.qp import random_qp
from tests.opt.ipm.problems import ipm_inputs

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")
import os
TRACE = bool(os.environ.get("FUZZ_TRACE"))
RESULT = ("x", "y", "z_l", "z_u", "z_bl", "z_bu", "status", "iter", "info") + (("trace", "trace_rows") if TRACE else ())


def base_qp(base):
  if base.startswith("rnd:"):
    parts = base.split(":")
    n, m, p, seed = (int(v) for v in parts[1:5])
    return random_qp(n, m, p, seed=seed, lp=len(parts) > 5)
  import gen
  return gen.problem(base)


def instance(s, v0, base, k):
  """Value set k: k = 0 is the base problem; the others scale variables, rows and the cost by powers
  of ten, perturb entries, zero the quadratic term, shift right-hand sides or pinch boxes."""
  v = {key: np.array(val, dtype=float, copy=True) for key, val in v0.items()}
  if k == 0:
    return v, "base"
  rng = np.random.default_rng([zlib.crc32(base.encode()), k])
  kind = ("perturb", "colscale", "rowscale", "costscale", "allscale", "lp", "tinyP", "hugeP", "rhs", "pinch", "dupcost", "zero_c")[k % 12]
  span = float(rng.choice([1.0, 3.0, 6.0, 8.0]))
  n = s.n
  fin = lambda a: np.isfinite(a) & (np.abs(a) < 1e19)
  def colscale(d):
    v["P"] *= d[s.P_rows] * d[s.P_cols]
    v["A"] *= d[s.A_cols]
    v["G"] *= d[s.G_cols]
    v["c"] *= d
    v["x_l"] = np.where(fin(v["x_l"]), v["x_l"] / d, v["x_l"])
    v["x_u"] = np.where(fin(v["x_u"]), v["x_u"] / d, v["x_u"])
  def rowscale(ea, eg):
    v["A"] *= ea[s.A_rows]
    v["b"] *= ea
    v["G"] *= eg[s.G_rows]
    v["h_l"] = np.where(fin(v["h_l"]), v["h_l"] * eg, v["h_l"])
    v["h_u"] = np.where(fin(v["h_u"]), v["h_u"] * eg, v["h_u"])
  if kind == "perturb":
    for key in ("P", "A", "G", "c"):
      v[key] *= 1.0 + 1e-3 * rng.standard_normal(v[key].shape)
    # keep P symmetric: P is given by its (row, col) list, both triangles when the structure stores both
    sym = {}
    for i, (r, c) in enumerate(zip(s.P_rows, s.P_cols)):
      sym.setdefault((min(r, c), max(r, c)), v["P"][i])
      v["P"][i] = sym[(min(r, c), max(r, c))]
  elif kind == "colscale":
    colscale(10.0 ** rng.uniform(-span, span, n))
  elif kind == "rowscale":
    rowscale(10.0 ** rng.uniform(-span, span, s.p), 10.0 ** rng.uniform(-span, span, s.m))
  elif kind == "costscale":
    f = 10.0 ** rng.uniform(-span, span)
    v["P"] *= f
    v["c"] *= f
  elif kind == "allscale":
    colscale(10.0 ** rng.uniform(-span, span, n))
    rowscale(10.0 ** rng.uniform(-span, span, s.p), 10.0 ** rng.uniform(-span, span, s.m))
    f = 10.0 ** rng.uniform(-span, span)
    v["P"] *= f
    v["c"] *= f
  elif kind == "lp":
    v["P"] *= 0.0
  elif kind == "tinyP":
    v["P"] *= 10.0 ** -rng.uniform(6, 16)
  elif kind == "hugeP":
    v["P"] *= 10.0 ** rng.uniform(6, 12)
  elif kind == "rhs":
    v["b"] = v["b"] + 10.0 ** rng.uniform(-6, 0) * rng.standard_normal(v["b"].shape)
    v["c"] = v["c"] + rng.standard_normal(n) * np.abs(v["c"]).max(initial=1.0) * 0.1
  elif kind == "pinch":  # fixed and nearly fixed variables where both bounds exist
    both = fin(v["x_l"]) & fin(v["x_u"])
    pick = both & (rng.random(n) < 0.5)
    mid = np.where(pick, 0.5 * (v["x_l"] + v["x_u"]), 0.0)
    gap = np.where(rng.random(n) < 0.5, 0.0, 1e-9)
    v["x_l"] = np.where(pick, mid - gap, v["x_l"])
    v["x_u"] = np.where(pick, mid + gap, v["x_u"])
  elif kind == "dupcost":  # the cost in the row space of A and G: every multiplier a solution of a degenerate dual
    w = np.zeros(n)
    if s.p:
      np.add.at(w, s.A_cols, v["A"] * rng.standard_normal(s.p)[s.A_rows])
    if s.m:
      np.add.at(w, s.G_cols, v["G"] * rng.standard_normal(s.m)[s.G_rows])
    v["c"] = w
    v["P"] *= 0.0
  elif kind == "zero_c":
    v["c"] *= 0.0
    v["P"] *= 0.0
  return v, kind


def main():
  base, count, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
  backend = sys.argv[4] if len(sys.argv) > 4 else "sparse"
  settings = Settings(**json.loads(sys.argv[5])) if len(sys.argv) > 5 else Settings()
  qp = base_qp(base)
  s, v0 = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(v0[k])) for k in ORDER}
  tag = "".join(ch if ch.isalnum() else "_" for ch in base)
  res = Solver(s, backend, settings, name=f"fz_{tag}").solve(QPValues.preprocess(s, **syms), trace=TRACE)
  fn = sc.Function.from_exprs(f"fz_{tag}_{backend}" + ("_tr" if TRACE else ""), [syms[k] for k in ORDER], [res[k] for k in RESULT], list(ORDER), list(RESULT))
  ir_at = INFO_FIELDS.index("ir")
  rows = []
  for k in range(count):
    v, kind = instance(s, v0, base, k)
    got = dict(zip(RESULT, fn(tuple(v[key] for key in ORDER)), strict=True))
    x = np.asarray(got["x"], dtype=float)
    dense_P = np.zeros((s.n, s.n)); dense_P[s.P_rows, s.P_cols] = v["P"]
    if np.allclose(dense_P, np.triu(dense_P)):  # the structure stores one triangle
      dense_P = dense_P + np.triu(dense_P, 1).T
    obj = float(0.5 * x @ dense_P @ x + v["c"] @ x) if np.all(np.isfinite(x)) else float("nan")
    extra = {}
    if TRACE:
      from scaly.opt.ipm import TRACE_FIELDS
      tr = np.asarray(got["trace"])[: int(got["trace_rows"])]
      st = np.minimum(tr[1:, TRACE_FIELDS.index("primal_step")], tr[1:, TRACE_FIELDS.index("dual_step")])
      rho = tr[:, TRACE_FIELDS.index("rho")]
      extra = {"min_step": float(st.min()) if st.size else 1.0, "tiny_steps": int(np.sum(st < 1e-10)), "rho_raised": int(np.sum(rho[1:] > 1.5 * rho[:-1]))}
    rows.append({**extra, "k": k, "kind": kind, "status": int(got["status"]), "iter": int(got["iter"]), "ir": float(got["info"][ir_at]), "obj": obj,
                 "xnorm": float(np.abs(x).max(initial=0.0)), "xhash": zlib.crc32(x.tobytes())})
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({"base": base, "backend": backend, "kkt_file": kkt_mod.__file__, "n": s.n, "p": s.p, "m": s.m, "rows": rows}))
  ir = sum(r["ir"] > 0 for r in rows)
  print(f"{base} {backend}: {count} instances, ir on in {ir}, statuses {sorted(set(r['status'] for r in rows))}", flush=True)

if __name__ == "__main__":
  main()
