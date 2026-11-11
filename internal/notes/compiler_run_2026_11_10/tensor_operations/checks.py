"""Section 7 checks behind "What the coordinator was least sure of" in the README beside this file.

Run from a checkout with ``uv run internal/notes/compiler_run_2026_11_10/tensor_operations/checks.py``.
"""

import numpy as np

import scaly as sc
from scaly.ad import forward
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, topo
from scaly.ir.types import CanonicalPattern
from scaly.passes.expr import simplify

rng = np.random.default_rng(68)


def compiled(build, shape, lowering):
  x = sc.sym("x", shape)
  return ConcreteFunction._from_exprs("check", [x], [build(x).with_lowering(lowering)], ["x"], ["y"])


print("== 1. signed zeros through sums, against NumPy ==")
zeros = {
  "sum(axis=()) of [-0, 0, 1]": (np.array([-0.0, 0.0, 1.0]), lambda x: x.sum(axis=())),
  "sum() of [-0, -0]": (np.array([-0.0, -0.0]), lambda x: x.sum()),
  "sum(axis=0) of a (2, 2) of -0": (np.full((2, 2), -0.0), lambda x: x.sum(axis=0)),
  "mean(axis=()) of [-0, 2]": (np.array([-0.0, 2.0]), lambda x: x.mean(axis=())),
}
for label, (value, build) in zeros.items():
  want = np.asarray(build(value))
  for lowering in ("auto", "block", "scalar"):
    got = np.asarray(compiled(build, value.shape, lowering)(value))
    same = np.array_equal(got, want) and np.array_equal(np.signbit(got), np.signbit(want))
    print(f"{label:30} {lowering:6} scaly {got.tolist()!s:14} numpy {want.tolist()!s:14} {'same bits' if same else 'the sign of zero differs'}")

print("\n== 2. which forward rule a seed batch takes through a product ==")
taken = []
for name in ("_jvp_many_matmul_left", "_jvp_many_matmul_right"):

  def logged(x, y, d, nseed, original=getattr(forward, name), stored=name.endswith("left")):
    taken.append("folded" if (y if stored else x).type.pattern is not None else "batched")
    return original(x, y, d, nseed)

  setattr(forward, name, logged)


def finite_difference(f, value, eps=1e-6):
  columns = []
  for i in range(value.size):
    step = np.zeros(value.size)
    step[i] = eps
    step = step.reshape(value.shape)
    columns.append((np.asarray(f(value + step)).reshape(-1) - np.asarray(f(value - step)).reshape(-1)) / (2 * eps))
  return np.stack(columns, axis=1)


def check_product(label, fun, names, values):
  for position, name in enumerate(names):
    taken.clear()
    jacobian = sc.jacobian(fun, wrt=name)
    got = np.asarray(jacobian(*values)).reshape(-1, values[position].size)
    want = finite_difference(lambda w: fun(*(w if i == position else v for i, v in enumerate(values))), values[position])
    assert np.allclose(got, want, atol=1e-6), (label, name)
    products = [n for n in topo(tuple(as_concrete(jacobian).results)) if n.op == ExprOp.MATMUL]
    stored = [n for n in products if any(arg.type.pattern is not None for arg in n.args)]
    assert all(len(arg.shape) <= 2 for n in stored for arg in n.args), (label, name)
    ranks = sorted({max(len(arg.shape) for arg in n.args) for n in products})
    print(f"{label:28} by {name}: rule {sorted(set(taken)) or ['one product per seed']}, MATMUL operand ranks {ranks}, finite differences agree")


dense = [((3,), (3,)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((4, 1, 2, 3), (5, 3, 2)), ((3,), (5, 3, 2))]
for index, (xs, ys) in enumerate(dense):

  @sc.function(sc.arg("x", xs), sc.arg("y", ys), name=f"dense{index}")
  def dense_product(x, y):
    product = x.sin() @ y.cos()
    return product.reshape((product.size,))

  check_product(f"dense {xs} @ dense {ys}", dense_product, ("x", "y"), [rng.normal(size=xs), rng.normal(size=ys)])
  assert set(taken) == {"batched"}

pattern = CanonicalPattern((3, 4), (0, 1, 2, 2), (3, 0, 1, 3))
for index, (label, shape, left) in enumerate(
  [("sparse (3, 4) @ dense (4,)", (4,), True), ("sparse (3, 4) @ dense (4, 2)", (4, 2), True), ("dense (2, 3) @ sparse (3, 4)", (2, 3), False)]
):

  @sc.function(sc.arg("v", 4), sc.arg("b", shape), name=f"mixed{index}")
  def mixed_product(v, b):
    a = sc.pack(v.sin(), pattern)
    result = a @ b.cos() if left else b.cos() @ a
    return sc.values(result) if result.type.pattern is not None else result.reshape((result.size,))

  check_product(label, mixed_product, ("v", "b"), [rng.normal(size=4), rng.normal(size=shape)])
  assert set(taken) == {"folded"}

other = CanonicalPattern((4, 3), (0, 1, 3, 3), (2, 0, 0, 1))


@sc.function(sc.arg("v", 4), sc.arg("u", 4), name="both_sparse")
def sparse_product(v, u):
  return sc.values(sc.pack(v.sin(), pattern) @ sc.pack(u.cos(), other))


check_product("sparse (3, 4) @ sparse (4, 3)", sparse_product, ("v", "u"), [rng.normal(size=4), rng.normal(size=4)])
assert not taken

print("\n== 3. random einsum specs against np.einsum ==")


def random_spec():
  sizes = {c: int(rng.integers(2, 4)) for c in "ijklm"}
  ellipsis = tuple(int(rng.integers(1, 4)) for _ in range(int(rng.integers(0, 3)))) if rng.random() < 0.4 else None
  terms, shapes = [], []
  for _ in range(int(rng.integers(1, 4))):
    labels = [str(rng.choice(list("ijklm"[: int(rng.integers(2, 6))]))) for _ in range(int(rng.integers(0, 5)))]
    shape = [1 if rng.random() < 0.15 and labels.count(c) == 1 else sizes[c] for c in labels]
    term = "".join(labels)
    if ellipsis is not None and rng.random() < 0.8:
      cut = int(rng.integers(0, len(labels) + 1)) if rng.random() < 0.3 else 0
      broadcast = [1 if rng.random() < 0.2 else n for n in ellipsis[int(rng.integers(0, len(ellipsis) + 1)) :]]
      term, shape = term[:cut] + "..." + term[cut:], shape[:cut] + broadcast + shape[cut:]
    terms.append(term)
    shapes.append(tuple(shape))
  spec = ",".join(terms)
  if rng.random() < 0.6:
    keep = [c for c in sorted(set(spec) - set(".,")) if rng.random() < 0.6]
    rng.shuffle(keep)
    spec += "->" + ("..." if "..." in spec and rng.random() < 0.9 else "") + "".join(keep)
  return spec, shapes


counts = {
  "agree": 0,
  "both reject": 0,
  "disagree": 0,
  "an operand of rank 3 or more": 0,
  "a diagonal": 0,
  "a size-1 label or axis": 0,
  "an ellipsis": 0,
  "compiled to C": 0,
}
for trial in range(3000):
  spec, shapes = random_spec()
  values = [rng.normal(size=shape) for shape in shapes]
  results = []
  for build in (lambda: np.einsum(spec, *values), lambda: np.asarray(simplify(sc.einsum(spec, *map(sc.const, values))).value)):
    try:
      results.append(build())
    except (ValueError, TypeError, NotImplementedError):
      results.append(None)
  want, got = results
  if want is None or got is None:
    counts["both reject" if want is None and got is None else "disagree"] += 1
    continue
  if got.size != want.size or not np.allclose(got.reshape(want.shape), want, atol=1e-10):
    counts["disagree"] += 1
    continue
  counts["agree"] += 1
  inputs = [term.replace(".", "") for term in spec.split("->")[0].split(",")]
  counts["an operand of rank 3 or more"] += any(len(shape) >= 3 for shape in shapes)
  counts["a diagonal"] += any(len(set(term)) < len(term) for term in inputs)
  counts["a size-1 label or axis"] += any(1 in shape for shape in shapes)
  counts["an ellipsis"] += "..." in spec
  if counts["compiled to C"] < 40 and trial % 7 == 0 and any(len(shape) >= 3 for shape in shapes):
    xs = [sc.sym(f"x{i}", shape) for i, shape in enumerate(shapes)]
    fun = ConcreteFunction._from_exprs("einsum_check", xs, [sc.einsum(spec, *xs)], [f"x{i}" for i in range(len(xs))], ["y"])
    np.testing.assert_allclose(fun(*values), want, atol=1e-10, err_msg=spec)
    counts["compiled to C"] += 1
print("3000 specs of one to three operands, folded as constants:", ", ".join(f"{label} {count}" for label, count in counts.items()))
assert counts["disagree"] == 0

print("\n== 4. extrema: NaN, ties and slopes ==")
data = np.array([[1.0, np.nan, 3.0], [2.0, 2.0, -1.0]])


@sc.function(sc.arg("x", (2, 3)), outputs=sc.group(sc.arg("largest"), sc.arg("smallest"), sc.arg("where"), sc.arg("pairwise"), sc.arg("norm")))
def extrema(x):
  return sc.reduce_max(x, axis=1), sc.reduce_min(x, axis=1), sc.cast(sc.argmax(x, axis=1), "float64"), sc.maximum(x[0], x[1]), sc.norm_inf(x)


expected = (data.max(axis=1), data.min(axis=1), data.argmax(axis=1), np.maximum(data[0], data[1]), np.abs(data).max())
for label, got, want in zip(
  ("reduce_max(axis=1)", "reduce_min(axis=1)", "argmax(axis=1)", "maximum(x[0], x[1])", "norm_inf"), extrema(data), expected, strict=True
):
  np.testing.assert_array_equal(np.asarray(got), np.asarray(want, dtype=float))
  print(f"{label:20} scaly {np.asarray(got).tolist()!s:18} numpy {np.asarray(want).tolist()}")


@sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("tie"), sc.arg("pair"), sc.arg("absolute"), sc.arg("floor")))
def slopes(x):
  pair = sc.maximum(x[0], x[1])
  return sc.gradient(sc.reduce_max(x), x), sc.gradient(pair, x), sc.gradient(x.abs().sum(), x), sc.gradient(x.floor().sum(), x)


tie, pair, absolute, floor = slopes(np.array([2.0, 2.0, 0.0, -1.5]))
print("gradient of reduce_max at [2, 2, 0, -1.5]:", tie.tolist(), "| of maximum(x[0], x[1]):", pair.tolist())
print("gradient of sum(abs):", absolute.tolist(), "| of sum(floor):", floor.tolist())
assert tie.tolist() == [0.5, 0.5, 0.0, 0.0] and pair.tolist() == [0.5, 0.5, 0.0, 0.0]
assert absolute.tolist() == [1.0, 1.0, 0.0, -1.0] and floor.tolist() == [0.0, 0.0, 0.0, 0.0]
