"""A day of heat-pump MPC for a house, at 15 minutes, against hourly forecasts (Scaly).

    minimize   sum_k price(t_k) P_k dt + 1e-3 sum_k P_k^2
    subject to room and mass temperatures (T_r, T_m) by an explicit Euler step of a two-node model,
               heat delivered Q_k = COP(T_out(t_k), T_s,k) P_k = k_e (T_s,k - T_r,k),
               20 <= T_r <= 23, 0 <= P <= 4 kW, 25 <= T_s <= 55 C

The controls are the electrical power ``P`` and the supply temperature ``T_s``: a hotter supply heats
the room faster at a lower COP. The COP is a bicubic table in the outdoor and supply temperatures,
evaluated at the 96 stages as one batch. The hourly price and outdoor-temperature forecasts are
parameters of the problem, so a new day needs no new code: zero-order-hold tables over the hours,
read at the stage times with ``at()``, a constant selection of the 24 values each.
"""

import numpy as np

import scaly as sc
from _common import scaly_ipopt_options, show
from _data import HEAT_PUMP, cop_table, day_ahead
from scaly import interp

HP = HEAT_PUMP
N = HP.horizon
TIMES = HP.dt * np.arange(N)
COP = interp.interpolant(*cop_table(), kind="cubic")


@sc.problem(
  vars=sc.G(sc.L("P", N), sc.L("T_s", N), sc.L("T", (N + 1, 2))),
  params=sc.G(sc.L("x0", 2), sc.L("price", 24), sc.L("t_out", 24)),
)
def heat_pump(v, p):
  P, T_s, T = v
  x0, price, t_out = p
  hours = np.arange(24.0)
  price_k = interp.interpolant(hours, price, kind="zoh").at(TIMES)
  t_out_k = interp.interpolant(hours, t_out, kind="zoh").at(TIMES)
  Q = COP(sc.stack([t_out_k, T_s], axis=1)) * P
  room, mass = T[:-1, 0], T[:-1, 1]
  room_next = room + HP.dt / HP.c_air * (Q - HP.h_air_mass * (room - mass) - HP.h_air_out * (room - t_out_k))
  mass_next = mass + HP.dt / HP.c_mass * (HP.h_air_mass * (room - mass) - HP.h_mass_out * (mass - t_out_k))
  eq = (T[0] - x0, T[1:, 0] - room_next, T[1:, 1] - mass_next, Q - HP.k_emitter * (T_s - room))
  lo, hi = HP.t_room
  room_bounds = np.column_stack([np.full(N + 1, lo), np.full(N + 1, -np.inf)]), np.column_stack([np.full(N + 1, hi), np.full(N + 1, np.inf)])
  return sc.ProblemSpec(
    minimize=HP.dt * (price_k * P).sum() + 1e-3 * (P**2).sum(),
    eq=eq,
    lb=(sc.const(0.0), sc.const(HP.t_supply[0]), sc.const(room_bounds[0])),
    ub=(sc.const(HP.p_max), sc.const(HP.t_supply[1]), sc.const(room_bounds[1])),
  )


def build(verbose: bool = False):
  solve = sc.solver(heat_pump, "ipopt", options=scaly_ipopt_options(verbose, tol=1e-10))
  _, price, t_out = day_ahead()
  x0 = np.array(HP.x0)
  guess = (np.full(N, 1.0), np.full(N, 40.0), np.tile(x0, (N + 1, 1)))
  zeros = (np.zeros(N), np.zeros(N), np.zeros((N + 1, 2)))

  def run():
    (P, T_s, T), *_ = solve(guess, zeros, np.zeros(2 + 3 * N), np.zeros(0), (x0, price, t_out))
    stats = sc.solver_stats(solve)
    assert stats.to_solver_status().ok
    return {"P": P, "T_s": T_s, "T": T, "cost": np.array([stats.obj]), "iter": np.array([stats.iter])}

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
