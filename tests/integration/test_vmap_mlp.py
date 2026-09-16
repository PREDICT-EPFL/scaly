"""A dense-matmul stage body used through VMAP over a horizon, differentiated to first and second order.

This is the shape the neural-process MPC benchmark leans on and no other test covered: a small
multilayer perceptron whose weights arrive broadcast from a non-differentiable parameter tail,
used through VMAP over the stages of a prediction horizon, with `spjac` and `sphess` taken through the VMAP.
It lives here rather than in that benchmark's own checks so retiring the benchmark cannot drop the
coverage, and it is self-contained: the references are NumPy and an unrolled twin of the same
formulation, so nothing here depends on a trained checkpoint or on a solver plugin.
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly.codegen import render_c_source

# Two positions, two velocities and one input, so the stage closes its position rows by trapezoidal
# integration exactly as a second-order mechanical system does.
NX, NU, NY = 4, 1, 2
HIDDEN = 4
# input scaler, then two weight matrices, then the output bias
SHAPES = ((HIDDEN, NX + NU), (NY, HIDDEN))
N_PW = (NX + NU) + int(np.prod(SHAPES[0])) + int(np.prod(SHAPES[1])) + NY
OFFSETS = np.cumsum([0, NX + NU, int(np.prod(SHAPES[0])), int(np.prod(SHAPES[1])), NY])
DT = 0.1


def _slices() -> list[slice]:
  return [slice(int(OFFSETS[i]), int(OFFSETS[i + 1])) for i in range(4)]


def step_np(pw: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
  """One stage of the learned model in NumPy: scaled features, a sigmoid layer, then an affine one."""
  scale, w0, w1, bias = _slices()
  h = np.concatenate([x, u]) * pw[scale]
  h = 1.0 / (1.0 + np.exp(-(pw[w0].reshape(SHAPES[0]) @ h)))
  y = pw[w1].reshape(SHAPES[1]) @ h + pw[bias]
  return x + np.concatenate([DT * (x[NY:] + y / 2.0), y])


def stage_function() -> sc.Function:
  scale, w0, w1, bias = _slices()

  @sc.function(sc.G(sc.L("x", NX), sc.L("xnext", NX), sc.L("u", NU), sc.L("pw", N_PW)), sc.L("eq", ...), name="vmap_mlp_stage")
  def stage(inputs):  # type: ignore[no-untyped-def]
    x, xnext, u, pw = inputs
    h = sc.concat([x, u]) * pw[scale]
    h = 1.0 / (1.0 + (-(pw[w0].reshape(SHAPES[0]) @ h)).exp())
    y = pw[w1].reshape(SHAPES[1]) @ h + pw[bias]
    return x + sc.concat([DT * (x[NY:] + y / 2.0), y]) - xnext

  return stage


def n_dec(stages: int) -> int:
  return NX * (stages + 1) + NU * stages


def _cost_stage() -> sc.Function:
  @sc.function(sc.G(sc.L("x", NX), sc.L("xnext", NX), sc.L("u", NU)), sc.L("cost", ...), name="vmap_mlp_stage_cost")
  def cost(inputs):  # type: ignore[no-untyped-def]
    x, xnext, u = inputs
    difference = xnext - x
    return sc.sumsqr(x) + 2.0 * u[0] * u[0] + 0.5 * sc.sumsqr(difference)

  return cost


def vmapped(stages: int) -> sc.Function:
  """Objective and equalities together, both using VMAP, which is what a Lagrangian Hessian needs."""
  z = sc.sym("z", n_dec(stages))
  p = sc.sym("p", N_PW, diff=False)
  eq = sc.vmap(
    stage_function(),
    length=stages,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, NX * (stages + 1), NU), "pw": (p, 0, 0)},
  )
  cost = sc.vmap(
    _cost_stage(),
    length=stages,
    inputs={"x": (z, 0, NX), "xnext": (z, NX, NX), "u": (z, NX * (stages + 1), NU)},
  )
  return sc.Function._from_exprs(f"vmap_mlp_N{stages}", [z, p], [cost.sum(), eq], ["z", "p"], ["cost", "eq"])


def unrolled(stages: int) -> sc.Function:
  """The same formulation stage by stage, so the VMAP version can be tested against it."""
  z = sc.sym("z", n_dec(stages))
  p = sc.sym("p", N_PW, diff=False)
  stage, cost_stage = stage_function(), _cost_stage()
  offset = NX * (stages + 1)
  rows, terms = [], []
  for i in range(stages):
    x, xnext = z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)]
    u = z[offset + NU * i : offset + NU * (i + 1)]
    rows.append(stage((x, xnext, u, p)))
    terms.append(cost_stage((x, xnext, u)))
  cost = terms[0]
  for term in terms[1:]:
    cost = cost + term
  return sc.Function._from_exprs(f"vmap_mlp_unrolled_N{stages}", [z, p], [cost, sc.concat(rows)], ["z", "p"], ["cost", "eq"])


def sample(stages: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  scale, w0, w1, bias = _slices()
  pw = np.empty(N_PW)
  pw[scale] = rng.uniform(0.5, 1.5, NX + NU)
  pw[w0] = rng.normal(scale=np.sqrt(2.0 / (NX + NU)), size=int(np.prod(SHAPES[0])))
  pw[w1] = rng.normal(scale=np.sqrt(2.0 / HIDDEN), size=int(np.prod(SHAPES[1])))
  pw[bias] = rng.normal(scale=0.1, size=NY)
  return rng.normal(scale=0.4, size=n_dec(stages)), pw


def dense_jac_reference(stages: int, z: np.ndarray, pw: np.ndarray) -> np.ndarray:
  """Block-banded reference: differentiate the stage once and scatter its blocks.

  Each stage's residual touches only its own state, its control and its successor state, so the
  reference costs one small dense Jacobian per stage rather than one over the whole horizon — and,
  being built from the stage function alone, it never touches the VMAP graph under test.
  """
  jac = stage_function().factory(
    "vmap_mlp_stage_jac", ["x", "xnext", "u", "pw"], [sc.factory.Jac("eq", "x"), sc.factory.Jac("eq", "u"), sc.factory.Jac("eq", "xnext")]
  )
  offset = NX * (stages + 1)
  dense = np.zeros((NX * stages, n_dec(stages)))
  for i in range(stages):
    blocks = jac((z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)], z[offset + NU * i : offset + NU * (i + 1)], pw))
    d_x, d_u, d_xnext = (np.asarray(block).reshape(NX, -1) for block in blocks)
    rows = slice(NX * i, NX * (i + 1))
    dense[rows, NX * i : NX * (i + 1)] = d_x
    dense[rows, NX * (i + 1) : NX * (i + 2)] = d_xnext
    dense[rows, offset + NU * i : offset + NU * (i + 1)] = d_u
  return dense


def _scatter(sparse: sc.Function, z: np.ndarray, pw: np.ndarray, *extra: np.ndarray) -> np.ndarray:
  sparsity = sparse.output_sparsities[0]
  assert sparsity is not None
  dense = np.zeros(sparsity.shape)
  dense[np.asarray(sparsity.rows), np.asarray(sparsity.cols)] = np.asarray(sparse((z, *extra, pw))).reshape(-1)
  return dense


def test_vmapped_stage_matches_the_numpy_model() -> None:
  """The stage body's forward value is the NumPy one, with the weights read out of the tail."""
  z, pw = sample(4)
  stage = stage_function()
  for i in range(4):
    x, xnext = z[NX * i : NX * (i + 1)], z[NX * (i + 1) : NX * (i + 2)]
    u = z[NX * 5 + NU * i : NX * 5 + NU * (i + 1)]
    residual = np.asarray(stage((x, xnext, u, pw))).reshape(-1)
    np.testing.assert_allclose(residual, step_np(pw, x, u) - xnext, rtol=0.0, atol=1e-13)


def test_vmapped_and_unrolled_agree_in_value_jacobian_and_hessian() -> None:
  """Using VMAP for the horizon changes how the graph is built, not what it means.

  The unrolled twin is the independent construction: same stage function, same cost, no VMAP node
  anywhere. Value, sparse Jacobian and sparse Lagrangian Hessian all have to land on it exactly,
  and the Jacobian additionally has to land on a NumPy-scattered dense reference.
  """
  for stages in (1, 2, 5):
    z, pw = sample(stages)
    lam = np.linspace(-0.7, 0.9, NX * stages)
    vmap_fn, flat_fn = vmapped(stages), unrolled(stages)

    np.testing.assert_allclose(np.asarray(vmap_fn((z, pw))[0]), np.asarray(flat_fn((z, pw))[0]), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(vmap_fn((z, pw))[1]), np.asarray(flat_fn((z, pw))[1]), rtol=0.0, atol=1e-13)

    jacobians = [fn.factory(f"{fn.name}_spjac", ["z", "p"], [sc.factory.SpJac("eq", "z")]) for fn in (vmap_fn, flat_fn)]
    dense = [_scatter(fn, z, pw) for fn in jacobians]
    np.testing.assert_allclose(dense[0], dense[1], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(dense[0], dense_jac_reference(stages, z, pw), rtol=0.0, atol=1e-13)
    pattern = jacobians[0].output_sparsities[0]
    assert pattern is not None and pattern.nnz < dense[0].size

    hessians = [
      fn.factory(f"{fn.name}_sphess", ["z", "lam:cost", "lam:eq", "p"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["cost", "eq"]})
      for fn in (vmap_fn, flat_fn)
    ]
    hess = [_scatter(fn, z, pw, np.array(1.0), lam) for fn in hessians]
    np.testing.assert_allclose(hess[0], hess[1], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(hess[0], hess[0].T, rtol=0.0, atol=1e-12)


def test_lagrangian_hessian_through_the_vmap_matches_finite_differences() -> None:
  """Second order through a VMAP matmul body, checked without a second symbolic path.

  Differencing the Lagrangian's gradient is independent of how the Hessian is assembled, so this is
  what catches an error the VMAP-versus-unrolled comparison would make on both sides at once.
  """
  stages = 4
  z, pw = sample(stages)
  lam = np.linspace(-0.7, 0.9, NX * stages)
  fn = vmapped(stages)
  gradient = fn.factory("vmap_mlp_lag_grad", ["z", "lam:cost", "lam:eq", "p"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["cost", "eq"]})
  hessian = fn.factory("vmap_mlp_lag_sphess", ["z", "lam:cost", "lam:eq", "p"], [sc.factory.SpHess("gamma", "z")], aux={"gamma": ["cost", "eq"]})
  exact = _scatter(hessian, z, pw, np.array(1.0), lam)

  step = 1e-6
  approx = np.empty_like(exact)
  for column in range(z.size):
    shift = np.zeros_like(z)
    shift[column] = step
    forward = np.asarray(gradient((z + shift, np.array(1.0), lam, pw))).reshape(-1)
    backward = np.asarray(gradient((z - shift, np.array(1.0), lam, pw))).reshape(-1)
    approx[:, column] = (forward - backward) / (2.0 * step)
  np.testing.assert_allclose(exact, approx, rtol=2e-5, atol=2e-6)


def test_vmapped_source_is_constant_in_the_horizon_where_the_unrolled_twin_grows() -> None:
  """Both kernels stay one loop nest however many stages there are, and the twin shows it matters.

  The Hessian is the one that pins the *cost* to its VMAP form: written as a Python reduction over
  per-stage slices it unrolls. The control pins one retained caller dispatch per stage instead of
  measuring those calls together with the fixed helper bodies.
  """
  from scaly.ir.program import ProgramOp
  from scaly.passes.lowering import lower_function, main_proc

  sizes = (2, 8, 32)
  for kernel, request in (("spjac", sc.factory.SpJac("eq", "z")), ("sphess", sc.factory.SpHess("gamma", "z"))):
    lines = []
    for stages in sizes:
      names = ["z", "p"] if kernel == "spjac" else ["z", "lam:cost", "lam:eq", "p"]
      aux = None if kernel == "spjac" else {"gamma": ["cost", "eq"]}
      built = vmapped(stages).factory(f"vmap_mlp_{kernel}_N{stages}", names, [request], aux=aux)
      lines.append(len(render_c_source(built).splitlines()))
    assert max(lines) < 1.2 * min(lines), f"{kernel} source grew with the horizon: {dict(zip(sizes, lines, strict=True))}"

  unrolled_calls = []
  for stages in sizes:
    built = unrolled(stages).factory(f"vmap_mlp_unrolled_spjac_N{stages}", ["z", "p"], [sc.factory.SpJac("eq", "z")])
    proc = main_proc(lower_function(built))
    body = proc.args[int(proc.attrs["param_count"]) :]
    unrolled_calls.append(sum(stmt.op == ProgramOp.CALL for stmt in body))
  assert unrolled_calls == list(sizes), f"the unrolled twin should retain one caller dispatch per stage: {unrolled_calls}"
