"""Stage layouts and their detection from nonlinear-program sparsity patterns."""

from bisect import bisect_right
from dataclasses import dataclass

from ..ir.types import SparsityPattern


@dataclass(frozen=True)
class StageLayout:
  """State/control sizes and original constraint rows for stages ``0..N``.

  Variables are ordered ``[x_0, u_0, ..., x_N, u_N]``. ``dynamics[k]`` contains
  the ``nx[k+1]`` equality rows into stage ``k+1``, in next-state order.
  ``eq[k]`` and ``ineq[k]`` contain the remaining constraints of stage ``k``.
  All row indices refer to the stacked ``[equalities; inequalities]`` vector.
  The number of local constraints is the length of the corresponding tuple.
  """

  nx: tuple[int, ...]
  nu: tuple[int, ...]
  dynamics: tuple[tuple[int, ...], ...]
  eq: tuple[tuple[int, ...], ...]
  ineq: tuple[tuple[int, ...], ...]

  @property
  def N(self) -> int:
    """The number of transitions, one less than the number of stages."""
    return len(self.nx) - 1


_SHARED_VARIABLE = "Shared variables such as free final time require the stage form of the OCP builder (#88)."


def _jac_error(row: int, col: int, reason: str) -> ValueError:
  return ValueError(f"stage detection: constraint row {row}, variable {col}: {reason}. {_SHARED_VARIABLE}")


def detect_stage_layout(jac: SparsityPattern, hess: SparsityPattern, n_eq: int) -> StageLayout:
  """Walk equality-row rightmost nonzeros, then verify stage-local Jacobian and Hessian blocks.

  Dynamics rows must appear in stage and next-state order. The walk follows
  CasADi's ``structure_detection=auto`` in ``interfaces/fatrop/fatrop_interface.cpp``.
  Initial-state rows are local equalities rather than a transition from an empty
  stage. Inequalities are assigned separately because Scaly stacks them after
  every equality. Empty local rows belong to stage zero. No variables are reordered.
  """
  m, n = jac.shape
  if not 0 < n_eq <= m:
    raise ValueError("stage detection requires at least one equality constraint")
  if hess.shape != (n, n):
    raise ValueError(f"stage detection: Hessian shape {hess.shape} does not match {n} variables")
  row_ptr, columns, _ = jac.to_csr()
  rows = tuple(columns[row_ptr[i] : row_ptr[i + 1]] for i in range(m))
  if not rows[0]:
    raise ValueError("stage detection: constraint row 0 depends on no decision variable")

  start = pivot = rows[0][-1]
  starts = [start]
  groups = [[0]]
  for row in range(1, n_eq):
    cols = rows[row]
    right = cols[-1] if cols else -1
    second = cols[-2] if len(cols) > 1 else -1
    if right == pivot + 1 and second < start:
      groups[-1].append(row)
      pivot = right
    elif right > pivot:
      starts.append(right)
      groups.append([row])
      start = pivot = right

  nx = [min(starts[0], len(groups[0])), *(len(group) for group in groups)]
  offsets = [0, *starts]
  if starts[0] == 0:
    nx.pop(0)
    offsets.pop(0)
    groups.pop(0)
  nu = [end - begin - size for begin, end, size in zip(offsets, (*offsets[1:], n), nx, strict=True)]

  dynamics = tuple(tuple(group) for group in groups)
  dynamic_rows = {row for group in dynamics for row in group}
  for stage, group in enumerate(dynamics):
    lower, next_state = offsets[stage], offsets[stage + 1]
    for i, row in enumerate(group):
      for col in rows[row]:
        if not (lower <= col < next_state or col == next_state + i):
          raise _jac_error(row, col, f"dynamics into stage {stage + 1} must use stage {stage} and its own next-state variable")

  eq: list[list[int]] = [[] for _ in nx]
  ineq: list[list[int]] = [[] for _ in nx]
  for row, cols in enumerate(rows):
    if row in dynamic_rows:
      continue
    stage = bisect_right(offsets, cols[-1]) - 1 if cols else 0
    for col in cols:
      if col < offsets[stage]:
        raise _jac_error(row, col, f"local constraint in stage {stage} depends on another stage")
    (eq if row < n_eq else ineq)[stage].append(row)

  for row, col in zip(hess.rows, hess.cols, strict=True):
    if bisect_right(offsets, row) != bisect_right(offsets, col):
      left, right = sorted((row, col))
      raise ValueError(f"stage detection: Lagrangian Hessian couples variables {left} and {right} in different stages. {_SHARED_VARIABLE}")

  return StageLayout(tuple(nx), tuple(nu), dynamics, tuple(tuple(group) for group in eq), tuple(tuple(group) for group in ineq))
