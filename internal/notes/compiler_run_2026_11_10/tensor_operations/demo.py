"""Section 7 demo: axis reductions, extrema, batched products and einsum against NumPy and finite differences.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/tensor_operations/demo.py``.
"""

import shutil
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import render_c_api_header, render_c_source
from scaly.ir.program import walk_program
from scaly.passes.lowering import lower_function

SHAPES = {"a": (2, 1, 3, 4), "b": (5, 4, 3), "v": (5, 3), "s": (3, 3)}
INPUTS = [sc.arg(name, shape) for name, shape in SHAPES.items()]


def tensors(a, b, v, s, lib=sc):
  """The same ten results from scaly expressions (``lib=sc``) or NumPy arrays (``lib=np``)."""
  product = a @ b  # (2, 1, 3, 4) @ (5, 4, 3): the leading axes broadcast to (2, 5)
  return {
    "a.sum(axis=-1, keepdims=True)": a.sum(axis=-1, keepdims=True),
    "a - a.mean(axis=(0, 3), keepdims=True)": a - a.mean(axis=(0, 3), keepdims=True),
    "reduce_max(a, axis=-1)": sc.reduce_max(a, axis=-1) if lib is sc else a.max(axis=-1),
    "norm_inf(s)": sc.norm_inf(s) if lib is sc else np.abs(s).max(),
    "a @ b": product,
    "einsum bij,bjk->bik": lib.einsum("bij,bjk->bik", b, product[1]),
    "einsum ...ij,...j->...i": lib.einsum("...ij,...j->...i", product, v),
    "einsum ii->": lib.einsum("ii->", s),
    "diag(einsum ii->i)": lib.diag(lib.einsum("ii->i", s)),
    "einsum ij,jk,kl->il": lib.einsum("ij,jk,kl->il", s, lib.diag(v[0]), s),
  }


LABELS = list(tensors(*(np.zeros(shape) for shape in SHAPES.values()), lib=np))


# A group holds at most eight trees, so the ten results come back as two groups of five.
@sc.function(*INPUTS, outputs=sc.group(*(sc.group(*(sc.arg(f"out{i}") for i in range(start, start + 5))) for start in (0, 5))))
def tensor_demo(a: sc.Expr, b: sc.Expr, v: sc.Expr, s: sc.Expr) -> tuple[tuple[sc.Expr, ...], ...]:
  results = tuple(tensors(a, b, v, s).values())
  return results[:5], results[5:]


@sc.function(*INPUTS, outputs=sc.arg("everything"))
def flat(a: sc.Expr, b: sc.Expr, v: sc.Expr, s: sc.Expr) -> sc.Expr:
  """Every result of ``tensor_demo`` in one vector, so one Jacobian per input covers them all."""
  return sc.concat([value.vec() for value in tensors(a, b, v, s).values()])


def single(index: int) -> sc.Function:
  @sc.function(*INPUTS, outputs=sc.arg("out"), name=f"single{index}")
  def one(a: sc.Expr, b: sc.Expr, v: sc.Expr, s: sc.Expr) -> sc.Expr:
    return list(tensors(a, b, v, s).values())[index]

  return one


def generated_c(fun: sc.Function) -> tuple[int, int, int, int, int, int, str]:
  """Operation counts of the lowered program, bytes of C, and the median compile time of three."""
  nodes = list(walk_program(lower_function(fun)))
  operations = Counter(node.op.value for node in nodes)
  floating = Counter(node.op.value for node in nodes if node.dtype.is_floating)
  source = render_c_source(fun)
  compiler = shutil.which("cc")
  seconds = []
  with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, f"{fun.name}.h").write_text(render_c_api_header(fun))
    Path(tmp, f"{fun.name}.c").write_text(source)
    for _ in range(3 if compiler else 0):
      start = time.perf_counter()
      subprocess.run([compiler, "-O3", "-fno-math-errno", "-c", f"{fun.name}.c", "-o", f"{fun.name}.o"], cwd=tmp, check=True)
      seconds.append(time.perf_counter() - start)
  compile_ms = f"{1e3 * statistics.median(seconds):.0f}" if seconds else "no cc"
  return len(nodes), operations["for"], floating["mul"], floating["add"], operations["buffer"], len(source.encode()), compile_ms


rng = np.random.default_rng(7)
values = [rng.normal(size=shape) for shape in SHAPES.values()]

print("== values against NumPy (np.sum, np.mean, np.max, np.matmul, np.einsum, np.diag) ==")
print("inputs:", ", ".join(f"{name} {shape}" for name, shape in SHAPES.items()))
reference = tensors(*values, lib=np)
for label, result in zip(LABELS, (result for half in tensor_demo(*values) for result in half), strict=True):
  expected = reference[label]
  assert result.shape == expected.shape, label
  error = float(np.max(np.abs(result - expected)))
  assert error < 1e-12, label
  print(f"{label:42} shape {expected.shape!s:14} max |scaly - numpy| {error:.1e}")

print("\n== Jacobians of all ten results against central finite differences ==")
jacobians = {name: sc.jacobian(flat, "everything", name) for name in SHAPES}
for position, (name, jacobian) in enumerate(jacobians.items()):
  exact = jacobian(*values)
  exact = exact.reshape(exact.shape[0], -1)
  numeric = np.empty_like(exact)
  step = 1e-6
  for column in range(exact.shape[1]):
    shifted = [[value.copy() for value in values] for _ in range(2)]
    shifted[0][position].reshape(-1)[column] += step
    shifted[1][position].reshape(-1)[column] -= step
    numeric[:, column] = (flat(*shifted[0]) - flat(*shifted[1])) / (2 * step)
  error = float(np.max(np.abs(exact - numeric)))
  assert error < 1e-7, name
  print(f"d everything / d {name}: shape {exact.shape!s:10} nonzeros {np.count_nonzero(exact):5} max |scaly - finite difference| {error:.1e}")

print("\n== the generated C: operation counts, source size and compile time ==")
print(f"{'Function':42}{'program nodes':>14}{'loops':>7}{'multiplies':>11}{'additions':>10}{'buffers':>9}{'C bytes':>9}{'cc -O3 -c, ms':>15}")
rows = [(label, single(index)) for index, label in enumerate(LABELS)]
rows += [("all ten results", tensor_demo)]
rows += [(f"Jacobian of all ten by {name}", jacobian) for name, jacobian in jacobians.items()]
for label, fun in rows:
  nodes, loops, multiplies, additions, buffers, size, compile_ms = generated_c(fun)
  print(f"{label:42}{nodes:>14}{loops:>7}{multiplies:>11}{additions:>10}{buffers:>9}{size:>9}{compile_ms:>15}")
print("Counts are nodes of the lowered program, and multiplications and additions count floating-point ones only, not")
print("index arithmetic. A loop body is counted once whatever its trip count. Compile times are the median of three on")
print("a shared machine.")
print("No run times are taken here: the run measures those once, in its Phase 4.")

print("\n== einsum beside the same contraction written with @ ==")
PAIRS = [
  ("bij,bjk->bik", "x @ y", {"x": (5, 4, 3), "y": (5, 3, 3)}, lambda x, y: sc.einsum("bij,bjk->bik", x, y), lambda x, y: x @ y),
  (
    "...ij,...j->...i",
    "(p @ v[..., None])[..., 0]",
    {"p": (2, 5, 3, 3), "v": (5, 3)},
    lambda p, v: sc.einsum("...ij,...j->...i", p, v),
    lambda p, v: (p @ v.reshape((5, 3, 1))).reshape((2, 5, 3)),
  ),
  ("ij,kj->ik", "s @ t.T", {"s": (3, 4), "t": (5, 4)}, lambda s, t: sc.einsum("ij,kj->ik", s, t), lambda s, t: s @ t.T),
  (
    "ij,jk,kl->il",
    "(s @ d) @ t",
    {"s": (3, 3), "d": (3, 3), "t": (3, 3)},
    lambda s, d, t: sc.einsum("ij,jk,kl->il", s, d, t),
    lambda s, d, t: (s @ d) @ t,
  ),
]
print(f"{'spec':18}{'written with @':28}{'loops':>14}{'buffers':>16}{'C bytes':>16}")
for index, (spec, label, shapes, contraction, product) in enumerate(PAIRS):
  args = [sc.arg(name, shape) for name, shape in shapes.items()]
  contraction, product = (sc.function(*args, outputs=sc.arg("out"), name=f"pair{index}")(build) for build in (contraction, product))
  point = [rng.normal(size=shape) for shape in shapes.values()]
  assert np.allclose(contraction(*point), product(*point), atol=1e-12)
  (_, loops, _, _, buffers, size, _), (_, loops_at, _, _, buffers_at, size_at, _) = generated_c(contraction), generated_c(product)
  print(f"{spec:18}{label:28}{f'{loops} and {loops_at}':>14}{f'{buffers} and {buffers_at}':>16}{f'{size} and {size_at}':>16}")
print("Each pair is einsum first. A transpose that changes nothing is simplified away, so einsum costs the copies of")
print("the permutations it needs and nothing else. The ellipsis spec permutes its left operand and its result.")

print("\nMilestone 2 (0.3.0) checks: this script is 'einsum against NumPy'. The #69 workspace gates and")
print("#71's matrix-vector gate belong to section 8 and are not shown here.")
