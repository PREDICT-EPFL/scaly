"""Stage detection from ordinary problem sparsity and explicit builder layouts."""

from dataclasses import replace

import numpy as np
import pytest

import scaly as sc
from scaly.function.model import as_concrete
from scaly.solvers.stages import StageLayout, detect_stage_layout


def _tracking(*, initial: bool = True, terminal_controls: int = 0, controls: int = 1) -> sc.Problem:
  width = 2 + controls

  @sc.problem(vars=sc.arg("z", 2 * width + 2 + terminal_controls), params=sc.arg("initial", 2))
  def tracking(z: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    eq = [z[:2] - p] if initial else []
    for k in range(2):
      offset = k * width
      control = z[offset + 2 : offset + width].sum() if controls else sc.const(0.0)
      eq.append(z[offset + width : offset + width + 2] - z[offset : offset + 2] ** 2 - control)
    return sc.ProblemSpec(
      minimize=sc.sumsqr(z),
      eq=tuple(eq),
      ineq=(sc.bounded(sc.stack([z[-2 - terminal_controls], z[0]]), hi=1.0),),
    )

  return tracking


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("terminal_controls", [0, 1])
@pytest.mark.parametrize("controls", [0, 1, 2])
def test_detects_initial_rows_dynamics_and_separate_inequalities(initial: bool, terminal_controls: int, controls: int) -> None:
  problem = _tracking(initial=initial, terminal_controls=terminal_controls, controls=controls)
  descriptor = as_concrete(sc.solver(problem, "ipopt").function).descriptor
  layout = descriptor.stage_layout
  first = 2 if initial else 0
  assert layout == StageLayout(
    nx=(2, 2, 2),
    nu=(controls, controls, terminal_controls),
    dynamics=((first, first + 1), (first + 2, first + 3)),
    eq=((0, 1) if initial else (), (), ()),
    ineq=((problem.n_eq + 1,), (), (problem.n_eq,)),
  )
  assert layout.N == 2
  assert descriptor.stage_layout is layout


@pytest.mark.parametrize("initial", ["reversed", "partial_first", "partial_last", "combined"])
def test_detects_reordered_and_partial_initial_conditions(initial: str) -> None:
  problem = _tracking()
  z, p = problem._var_symbols[0], problem._param_symbols[0]
  residual = z[:2] - p
  if initial == "reversed":
    residual = sc.stack([residual[1], residual[0]])
  elif initial == "partial_first":
    residual = residual[:1]
  elif initial == "partial_last":
    residual = residual[1:]
  else:
    residual = z[0] + z[1] - p[0]
  problem = replace(problem, spec=replace(problem.spec, eq=(residual, *problem.spec.eq[1:])))
  layout = as_concrete(sc.solver(problem, "ipopt").function).descriptor.stage_layout
  first = residual.size
  assert layout == StageLayout(
    nx=(2, 2, 2),
    nu=(1, 1, 0),
    dynamics=((first, first + 1), (first + 2, first + 3)),
    eq=(tuple(range(first)), (), ()),
    ineq=((problem.n_eq + 1,), (), (problem.n_eq,)),
  )


def test_detects_local_equalities_and_changing_stage_sizes() -> None:
  @sc.problem(vars=sc.arg("z", 9))
  def varying(z: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(
      minimize=sc.sumsqr(z),
      eq=(
        z[:2],
        z[3:4] - z[:1] ** 2 - z[2],
        z[1] + z[2],
        z[5:8] - z[3] ** 2 - z[4],
        z[6] + z[7],
      ),
      ineq=(sc.bounded(z[4], hi=1.0),),
    )

  layout = as_concrete(sc.solver(varying, "ipopt").function).descriptor.stage_layout
  assert layout == StageLayout(
    nx=(2, 1, 3),
    nu=(1, 1, 1),
    dynamics=((2,), (4, 5, 6)),
    eq=((0, 1, 3), (), (7,)),
    ineq=((), (8,), ()),
  )


def test_mapped_dynamics_match_an_unrolled_layout() -> None:
  @sc.function(sc.arg("z", 3), sc.arg("znext", 3), outputs=sc.arg("gap", 2))
  def gap(z: sc.Expr, znext: sc.Expr) -> sc.Expr:
    return znext[:2] - z[:2] ** 2 - z[2]

  @sc.problem(vars=sc.arg("z", 9))
  def mapped(z: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(z), eq=(z[:2], sc.vmap(gap, 2)(sc.window(z, 0, 3), sc.window(z, 3, 3)).vec()))

  descriptor = as_concrete(sc.solver(mapped, "ipopt").function).descriptor
  reference = as_concrete(sc.solver(_tracking(terminal_controls=1), "ipopt").function).descriptor.stage_layout
  assert descriptor.stage_layout == replace(reference, ineq=((), (), ()))
  z = np.linspace(0.1, 0.9, 9)
  assert descriptor.base is not None
  _, actual = descriptor.base(z, ())
  expected = np.concatenate([z[:2], *(z[k + 3 : k + 5] - z[k : k + 2] ** 2 - z[k + 2] for k in (0, 3))])
  np.testing.assert_allclose(actual, expected)


def test_builder_supplied_layout_skips_detection(monkeypatch: pytest.MonkeyPatch) -> None:
  layout = StageLayout(nx=(2, 2, 2), nu=(1, 1, 0), dynamics=((2, 3), (4, 5)), eq=((0, 1), (), ()), ineq=((7,), (), (6,)))

  def unexpected(*args: object) -> StageLayout:
    raise AssertionError("explicit layouts must skip detection")

  monkeypatch.setattr("scaly.solvers.model.detect_stage_layout", unexpected)
  problem = replace(_tracking(), stage_layout=layout)
  descriptor = as_concrete(sc.solver(problem, "ipopt").function).descriptor
  assert descriptor.stage_layout is layout


@pytest.mark.parametrize(
  ("rule", "row", "variable"),
  [("earlier_stage", 4, 0), ("next_state_block", 3, 1), ("local_equality", 6, 0), ("local_inequality", 6, 0)],
)
def test_rejects_nonlocal_jacobian_entries(rule: str, row: int, variable: int) -> None:
  @sc.problem(vars=sc.arg("z", 8))
  def invalid(z: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    first = sc.stack([z[3] - z[0] ** 2 - z[2], z[4] - z[1] ** 2 - z[2]])
    if rule == "next_state_block":
      first = first + sc.stack([sc.const(0.0), z[3]])
    second = z[6:8] - z[3:5] ** 2 - z[5]
    if rule == "earlier_stage":
      second = second + sc.stack([z[0], sc.const(0.0)])
    local = z[0] + z[4]
    return sc.ProblemSpec(
      minimize=sc.sumsqr(z),
      eq=(z[:2], first, second, *((local,) if rule == "local_equality" else ())),
      ineq=(sc.bounded(local, hi=0.0),) if rule == "local_inequality" else (),
    )

  descriptor = as_concrete(sc.solver(invalid, "ipopt").function).descriptor
  with pytest.raises(ValueError, match=rf"constraint row {row}.*variable {variable}"):
    _ = descriptor.stage_layout


@pytest.mark.parametrize("backend", ["ipopt", "sqp"])
@pytest.mark.parametrize("source", ["objective", "dynamics"])
def test_rejects_cross_stage_lagrangian_hessian(backend: str, source: str) -> None:
  problem = _tracking()
  z = problem._var_symbols[0]
  if source == "objective":
    spec = replace(problem.spec, minimize=problem.spec.minimize + z[0] * z[3])
  else:
    initial, first, second = problem.spec.eq
    spec = replace(problem.spec, eq=(initial, first + sc.stack([z[0] * z[3], sc.const(0.0)]), second))
  problem = replace(problem, spec=spec)
  descriptor = as_concrete(sc.solver(problem, backend).function).descriptor
  with pytest.raises(ValueError, match="Hessian.*variables.*0.*3"):
    _ = descriptor.stage_layout


@pytest.mark.parametrize("position", ["first", "last"])
def test_rejects_shared_final_time_with_builder_guidance(position: str) -> None:
  variables = sc.group(sc.arg("time", ()), sc.arg("z", 8)) if position == "first" else sc.group(sc.arg("z", 8), sc.arg("time", ()))

  @sc.problem(vars=variables)
  def free_time(variables: tuple[sc.Expr, sc.Expr]) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    time, z = variables if position == "first" else variables[::-1]
    return sc.ProblemSpec(
      minimize=sc.sumsqr(z) + time**2,
      eq=(z[:2], z[3:5] - time * z[:2] - z[2], z[6:8] - time * z[3:5] - z[5]),
    )

  descriptor = as_concrete(sc.solver(free_time, "ipopt").function).descriptor
  with pytest.raises(ValueError, match=r"constraint row \d+.*variable \d+.*stage form.*#88"):
    _ = descriptor.stage_layout


def test_rejects_missing_equalities_and_empty_first_row() -> None:
  @sc.problem(vars=sc.arg("z", 2))
  def unconstrained(z: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(z))

  descriptor = as_concrete(sc.solver(unconstrained, "ipopt").function).descriptor
  with pytest.raises(ValueError, match="at least one equality"):
    _ = descriptor.stage_layout

  @sc.problem(vars=sc.arg("z", 2))
  def constant_row(z: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(z), eq=(sc.const(0.0), z[1] - z[0]))

  descriptor = as_concrete(sc.solver(constant_row, "ipopt").function).descriptor
  with pytest.raises(ValueError, match="constraint row 0.*no decision variable"):
    _ = descriptor.stage_layout


def test_detector_accepts_coordinate_storage_order() -> None:
  descriptor = as_concrete(sc.solver(_tracking(), "ipopt").function).descriptor
  jac, hess = descriptor.jac_sparsity, descriptor.hess_sparsity
  assert jac is not None and hess is not None
  permuted = sc.SparsityPattern(jac.shape, jac.rows[::-1], jac.cols[::-1])
  assert detect_stage_layout(permuted, hess, descriptor.n_eq) == descriptor.stage_layout
