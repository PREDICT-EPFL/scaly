"""Planning for a robot on a slippery floor: value iteration for a stochastic shortest-path problem.

The robot moves on the grid below, one cell per step, trying to reach ``G``. Each move goes the
intended way with probability ``1 - 2 p`` and slips to either side with probability ``p``; a move
into a wall leaves it where it is. A step costs 1, or ``puddle`` on a ``~`` cell, and ``G`` is
absorbing at no cost. The expected cost-to-go ``V`` is the fixed point of the Bellman equation

    V(s) = min_a [ c(s) + sum_s' P(s' | s, a) V(s') ],

which value iteration reaches from ``V = 0``. With the transitions stored as a list of
``(state-action, next state, probability)`` edges, one sweep is a ``segment_sum`` over the edges (the
expectation) and a ``segment_min`` over each state's four actions (the choice). The sweep repeats
in a ``while_loop`` until the largest change, ``norm_inf``, is below a tolerance. The greedy policy
is read off with ``where`` and returned as ``int64`` action numbers.

Everything is one generated ``Function`` of the slip probability and the puddle cost, so a sweep
over either parameter reuses the compiled solver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parent / "generated" / "mdp_value_iteration"

MAZE = (
  "##############",
  "#S....#......#",
  "#.##..#.####.#",
  "#..#......#..#",
  "#..#.~~~..#.G#",
  "#.....~~.....#",
  "#.##.....##..#",
  "##############",
)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # north, east, south, west
ARROWS = "^>v<"
TOL = 1e-10
MAX_SWEEPS = 2000

CELLS = [(r, c) for r, row in enumerate(MAZE) for c, ch in enumerate(row) if ch != "#"]
INDEX = {cell: i for i, cell in enumerate(CELLS)}
S = len(CELLS)
GOAL = next(i for i, (r, c) in enumerate(CELLS) if MAZE[r][c] == "G")
START = next(i for i, (r, c) in enumerate(CELLS) if MAZE[r][c] == "S")
PUDDLE = np.array([MAZE[r][c] == "~" for r, c in CELLS])


def _neighbour(s: int, move: int) -> int:
  r, c = CELLS[s]
  dr, dc = MOVES[move]
  return INDEX.get((r + dr, c + dc), s)


def edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """``(state-action, next state, kind)`` per edge, where kind 0 is the intended move and 1 a slip."""
  sa, nxt, kind = [], [], []
  for s in range(S):
    for a in range(4):
      if s == GOAL:
        sa.append(4 * s + a), nxt.append(s), kind.append(2)  # absorbing
        continue
      for k, move in ((0, a), (1, (a + 1) % 4), (1, (a + 3) % 4)):
        sa.append(4 * s + a), nxt.append(_neighbour(s, move)), kind.append(k)
  return np.array(sa), np.array(nxt), np.array(kind)


EDGE_SA, EDGE_NEXT, EDGE_KIND = edges()


def bellman(v: sc.Expr, slip: sc.Expr, puddle: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
  """``(Q, V_new)``: the state-action costs of one sweep and the best of each state's four."""
  prob = sc.where(sc.const(EDGE_KIND == 0, dtype="bool"), 1.0 - 2.0 * slip, sc.where(sc.const(EDGE_KIND == 1, dtype="bool"), slip, 1.0))
  expected = sc.segment_sum(prob * sc.gather(v, EDGE_NEXT), EDGE_SA, 4 * S)
  step_cost = np.repeat(np.where(np.arange(S) == GOAL, 0.0, 1.0), 4)
  in_puddle = sc.const(np.repeat(PUDDLE, 4), dtype="bool")
  q = sc.where(in_puddle, puddle, 1.0) * step_cost + expected
  return q, sc.segment_min(q, np.repeat(np.arange(S), 4), S)


# A loop body reads only its carry, so the two parameters travel in it next to V and the last change:
# carry = [V (S) | change | slip | puddle].
_carry = sc.sym("carry", S + 3)
_v, _change, _slip, _puddle = _carry[:S], _carry[S], _carry[S + 1], _carry[S + 2]
_, _v_new = bellman(_v, _slip, _puddle)
sweep = sc.Function.from_exprs(
  "vi_sweep", [_carry], [sc.concat([_v_new, sc.norm_inf(_v_new - _v).reshape((1,)), _carry[S + 1 :]])], ["carry"], ["next"]
)
not_converged = sc.Function.from_exprs("vi_not_converged", [_carry], [sc.greater(_change, TOL)], ["carry"], ["go_on"])


@sc.function((), (), output=sc.G("value", "policy", "sweeps"))
def solve(slip: sc.Expr, puddle: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  init = sc.concat([sc.const(np.zeros(S)), sc.const(np.ones(1)), slip.reshape((1,)), puddle.reshape((1,))])
  carry, sweeps = sc.while_loop(not_converged, sweep, init, max_iter=MAX_SWEEPS)
  v = carry[:S]
  q, best = bellman(v, slip, puddle)
  # The lowest-numbered action whose cost is within rounding of the best one.
  ties = sc.less_equal(q, sc.gather(best, np.repeat(np.arange(S), 4)) + 1e-9)
  actions = sc.const(np.tile(np.arange(4.0), S))
  policy = sc.segment_min(sc.where(ties, actions, 4.0), np.repeat(np.arange(S), 4), S).cast("int64")
  return v, policy, sweeps


def reference(slip: float, puddle: float) -> tuple[np.ndarray, np.ndarray]:
  """The same value iteration in NumPy: ``(V, Q)`` with ``Q`` of shape ``(S, 4)``."""
  prob = np.where(EDGE_KIND == 0, 1.0 - 2.0 * slip, np.where(EDGE_KIND == 1, slip, 1.0))
  cost = np.where(np.arange(S) == GOAL, 0.0, np.where(PUDDLE, puddle, 1.0))
  v = np.zeros(S)
  for _ in range(MAX_SWEEPS):
    q = np.repeat(cost, 4) + np.bincount(EDGE_SA, prob * v[EDGE_NEXT], minlength=4 * S)
    v_new = q.reshape(S, 4).min(axis=1)
    done = np.abs(v_new - v).max() <= TOL
    v = v_new
    if done:
      break
  return v, q.reshape(S, 4)


def render(policy: np.ndarray) -> str:
  rows = [list(row) for row in MAZE]
  for s, (r, c) in enumerate(CELLS):
    if MAZE[r][c] not in "G#":
      rows[r][c] = ARROWS[int(policy[s])]
  return "\n".join("".join(row) for row in rows)


def main(slip: float = 0.1, puddle: float = 6.0) -> dict[str, np.ndarray]:
  v, policy, sweeps = solve(np.array(slip), np.array(puddle))
  return {"value": v, "policy": policy, "sweeps": sweeps}


if __name__ == "__main__":
  for slip, puddle in ((0.0, 6.0), (0.1, 6.0), (0.2, 2.0)):
    out = main(slip, puddle)
    print(f"slip {slip}, puddle cost {puddle}: {int(out['sweeps'])} sweeps, expected cost from S {out['value'][START]:.2f}")
    print(render(out["policy"]))
  write_module(solve, GENERATED)
  print(f"generated C in {GENERATED}")
