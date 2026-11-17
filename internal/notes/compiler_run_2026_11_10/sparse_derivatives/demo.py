"""Section 3 demo: call support, known zeros in mixed tangents, sparse derivative leaves, and the chain M=9 Hessian stage.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/sparse_derivatives/demo.py``. It takes operation
counts, source size and build times, and no run times.
"""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import scaly as sc
from scaly.ad import finite_difference
from scaly.codegen import render_c_module
from scaly.function.model import as_concrete
from scaly.ir.expr import OP_INFO, is_map
from scaly.ir.support import specialize_support
from scaly.passes.expr import simplify_cse_fixpoint

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

print("== 1. a call's support follows its arguments (#221) ==")


@sc.function(sc.arg("t", (4, 3)), sc.arg("x", 3), outputs=sc.arg("y", (4, 3)))
def scale(t, x):
  return t * x.sin()


callee = as_concrete(scale)
own = callee.results[0].support
print(f"callee alone: {'every entry' if own is None else len(own)} of {callee.results[0].size} entries may be nonzero")
t = sc.scatter(sc.const(np.zeros(12)), [0, 4, 5, 11], sc.sym("s", (4,))).reshape((4, 3))
call = scale(t, sc.sym("x", (3,)))
print(f"called with a tangent of 4 nonzero entries: support {sorted(call.support)}")
lookup = specialize_support(callee, (t.support, None), OP_INFO, is_map)
print(f"specialisations cached on the callee: {len(callee._memo.supports)}, the product inside it has {len(lookup(callee.results[0]))} entries")

print("\n== 2. a selected zero tangent stays zero (#167) ==")
x, p = sc.sym("x", ()), sc.sym("p", ())
seeds = sc.sym("s", (2,))
direct = sc.jvp_many(sc.stack([x, p])[1] * sc.stack([x, p])[0], x, seeds)


@sc.function(sc.arg("x", ()), sc.arg("p", ()), outputs=sc.arg("y", ()))
def product(x, p):
  z = sc.stack([x, p])
  return z[1] * z[0]


through_call = sc.jvp_many(product(x, p), x, seeds)
for label, tangent in (("top level", direct), ("through a callee", through_call)):
  before, after = sc.jacobian_sparsity(tangent, x).nnz, sc.jacobian_sparsity(simplify_cse_fixpoint(tangent), x).nnz
  print(f"{label}: d(tangent)/dx has {before} entries before simplification and {after} after")


@sc.function(sc.arg("x", ()), sc.arg("p", ()), sc.arg("s", 2), outputs=sc.arg("t", 2))
def tangent_values(x, p, s):
  return sc.jvp_many(product(x, p), x, s)


print(f"values with run-time seeds [2, -3] at p = 0.7: {tangent_values(1.3, 0.7, np.array([2.0, -3.0]))} (NumPy {0.7 * np.array([2.0, -3.0])})")

print("\n== 3. a sparse derivative is a sparse leaf in a declared layout (#48) ==")
N = 5


@sc.function(sc.arg("z", 2), sc.arg("znext", 2), outputs=sc.arg("d", 2))
def defect(z, znext):
  return znext - sc.stack([z[0] + 0.1 * z[1].sin(), z[1] * z[0]])


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("ds"))
def shooting(zs):
  return sc.vmap(defect, N - 1)(sc.window(zs, 0, 2), sc.window(zs, 2, 2)).vec()


@sc.function(sc.arg("zs", 2 * N), outputs=sc.arg("cost"))
def coupled(zs):
  return (zs[:-1] * zs[1:]).sin().sum() + (zs**3).sum()


point = np.linspace(-0.8, 0.9, 2 * N)
dense_jac = finite_difference(shooting, point)
dense_hess = finite_difference(sc.gradient(coupled), point)
for layout in ("csc", "csr"):
  jac = sc.sparse_jacobian(shooting, layout=layout)
  out = as_concrete(jac).outputs[0]
  pattern = jac.sparsity()
  ref = sparse.coo_array((dense_jac[pattern.rows, pattern.cols], (pattern.rows, pattern.cols)), shape=pattern.shape)
  ref = ref.tocsc().data if layout == "csc" else ref.tocsr().data
  values = jac(point)
  print(
    f"sparse_jacobian layout={layout}: output type {out.type.shape} nnz {out.type.pattern.nnz}, result {type(values).__name__}, max |values - SciPy {layout}.data| {np.abs(np.asarray(values.data) - ref).max():.1e}"
  )
hess = sc.sparse_hessian(coupled, triangle="lower", layout="csr")
pattern = hess.sparsity()
ref = sparse.coo_array((np.tril(dense_hess)[pattern.rows, pattern.cols], (pattern.rows, pattern.cols)), shape=pattern.shape).tocsr().data
print(
  f"sparse_hessian lower, csr: nnz {pattern.nnz} of {4 * N * N}, max |values - finite differences| {np.abs(np.asarray(hess(point).data) - ref).max():.1e}"
)
header = render_c_module(sc.sparse_jacobian(shooting, name="shooting_spjac")).header
print(
  "header tables:",
  ", ".join(sorted(set(re.findall(r"shooting_spjac_\w*?_(NNZ|NDIM|STORAGE_SIZE|shape|col_ptr|row_ind|row_ptr|col_ind|val_perm\w*)\b", header)))),
)
print("output_sparsities on the Function:", hasattr(as_concrete(shooting), "output_sparsities"))
expr = sc.sym("v", (3,))
leaf = sc.sparse_jacobian(sc.stack([expr[0] * expr[2], expr[1].sin()]), expr)
print(f"expression form: an Expr of shape {leaf.shape} with pattern nnz {leaf.type.pattern.nnz}")
try:
  big_pattern = sc.SparsityPattern((100_000, 100_000), (0, 5), (1, 7)).canonicalize()[0]
  big = sc.arg("A", sc.TensorType(big_pattern.shape, pattern=big_pattern), layout="csr")
  sc.sparse_jacobian(sc.function(big, name="doubled")(lambda A: A * 2.0))
except ValueError as e:
  print("a 10^5 x 10^5 leaf with 2 stored entries, differentiated by itself:", e)

print("\n== 4. the chain M=9 Hessian stage, 40 stages, default budgets (#225, #214, #427) ==")
from benchmarks.harness.sweep import _descriptor_kernel, _dispatch_metrics  # noqa: E402
from benchmarks.problems import chain  # noqa: E402

started = time.perf_counter()
kernel, sparsity, colors = _descriptor_kernel(chain.chain_nlp(9, 40), "hess")
module = render_c_module(kernel)
source = module.body
generation = time.perf_counter() - started
trips, workspace, operations = _dispatch_metrics(module.fun, module.program)
with tempfile.TemporaryDirectory() as tmp:
  path = Path(tmp, "chain.c")
  path.write_text(source)
  started = time.perf_counter()
  subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(path), "-o", str(Path(tmp, "chain.so")), "-lm"], check=True)
  compile_time = time.perf_counter() - started
print(f"Hessian {sparsity.shape}, {len(sparsity.rows)} stored entries, {colors} colors")
print(f"operations per stage {operations} over {trips} stages (253,786 before #225, target about 60,000)")
print(
  f"generated C {len(source.encode())} bytes, generation {generation:.1f} s, cc -O2 {compile_time:.1f} s (la015, shared: observations, not timings)"
)
