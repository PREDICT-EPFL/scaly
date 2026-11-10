"""f(x) = x'Qx for a banded sparse Q, its gradient and Hessian, through every compiler stage.

Run from a checkout of the stack: uv run python banded_quadratic.py
"""

import traceback

import numpy as np
from scipy import sparse

import scaly as sc
import tempfile
from pathlib import Path

from scaly.codegen.aot import render_c_source, write_module
from scaly.function.model import as_concrete
from scaly.ir.expr import topo
from scaly.ir.types import SparsityPattern
from scaly.passes.expr import simplify_cse_fixpoint
from scaly.passes.lowering import lower_function

N = 5
Q_NUM = sparse.diags([[-1.0] * (N - 1), [4.0] * N, [-1.0] * (N - 1)], [-1, 0, 1], format="csc")
Q_PATTERN = SparsityPattern(Q_NUM.shape, *map(tuple, Q_NUM.tocoo().coords)).canonicalize()[0]


def stage(title: str, run) -> None:
  print(f"\n===== {title} =====")
  try:
    run()
  except Exception:
    print("STOPPED HERE:")
    traceback.print_exc(limit=6)


def show_support(fun) -> None:
  cf = as_concrete(fun)
  for node in topo(cf.outputs):
    s = node.support
    support = "unknown (any entry)" if s is None else f"{len(s)} of {node.type.size}: {sorted(s)[:20]}"
    pattern = node.type.pattern
    stored = "" if pattern is None else f" pattern nnz={pattern.nnz}"
    print(f"{node.op.name:<10} shape={node.type.shape}{stored}  support={support}")


q_leaf = sc.arg("Q", sc.TensorType(Q_NUM.shape, pattern=Q_PATTERN), layout="csc")
x_leaf = sc.arg("x", N)


@sc.function(q_leaf, x_leaf, outputs=sc.arg("f"))
def f(Q, x):
  return x @ (Q @ x)


grad_f = sc.gradient(f, of="f", wrt="x", name="grad_f")
hess_f = sc.hessian(f, of="f", wrt="x", name="hess_f")
sparse_hess_f = sc.sparse_hessian(f, of="f", wrt="x", name="sparse_hess_f")
x0 = np.arange(1.0, N + 1)

stage("1. Q's type and stored pattern", lambda: print(q_leaf.types[0], "\nrows", Q_PATTERN.rows, "\ncols", Q_PATTERN.cols, "\nlayout", q_leaf.layouts[0]))
for fun in (f, grad_f, hess_f, sparse_hess_f):
  name = fun.name
  stage(f"2. {name}: each node's support", lambda: show_support(fun))
  stage(f"3. {name}: traced expression", lambda: print(sc.render_expr_assembly(fun)))
  stage(f"3b. {name}: simplified expression", lambda: print(sc.render_expr_assembly([simplify_cse_fixpoint(o) for o in as_concrete(fun).outputs])))
  stage(f"4. {name}: program", lambda: print(sc.render_program_assembly(lower_function(fun))))
  stage(f"5. {name}: generated C", lambda: print(render_c_source(fun)))
  stage(f"5b. {name}: generated C header", lambda: print(write_module(fun, Path(tempfile.mkdtemp())).header))
  stage(f"6. {name}: numerical call vs NumPy", lambda: print(fun(Q_NUM, x0)))

import re
import time


def largest_buffer(n: int) -> None:
  qn = sparse.diags([[-1.0] * (n - 1), [4.0] * n, [-1.0] * (n - 1)], [-1, 0, 1], format="csc")
  pat = SparsityPattern(qn.shape, *map(tuple, qn.tocoo().coords)).canonicalize()[0]
  ql = sc.arg("Q", sc.TensorType(qn.shape, pattern=pat), layout="csc")

  @sc.function(ql, sc.arg("x", n), outputs=sc.arg("f"), name=f"f{n}")
  def fn(Q, x):
    return x @ (Q @ x)

  for g in (fn, sc.gradient(fn, of="f", wrt="x"), sc.sparse_hessian(fn, of="f", wrt="x")):
    t = time.perf_counter()
    asm = sc.render_program_assembly(lower_function(g))
    sizes = [int(m) for m in re.findall(r"memref<(\d+)xfloat64, private>", asm)]
    print(f"n={n} nnz(Q)={qn.nnz} {g.name}: largest private float64 buffer = {max(sizes, default=0)} entries; lowering took {time.perf_counter() - t:.1f} s")


stage("7. Scaling probe: largest private buffer per function", lambda: [largest_buffer(n) for n in (100, 400)])

dense = Q_NUM.toarray()
print("\nNumPy reference: f =", x0 @ dense @ x0, " grad =", (dense + dense.T) @ x0, "\nhess =\n", dense + dense.T)
