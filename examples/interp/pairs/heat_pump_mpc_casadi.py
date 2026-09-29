# /// script
# requires-python = ">=3.12"
# dependencies = ["casadi"]
#
# [tool.ty.environment]
# extra-paths = ["."]  # the modules beside this script, which it imports
# ///
"""A day of heat-pump MPC for a house, at 15 minutes, against hourly forecasts (CasADi).

    minimize   sum_k price(t_k) P_k dt + 1e-3 sum_k P_k^2
    subject to room and mass temperatures (T_r, T_m) by an explicit Euler step of a two-node model,
               heat delivered Q_k = COP(T_out(t_k), T_s,k) P_k = k_e (T_s,k - T_r,k),
               20 <= T_r <= 23, 0 <= P <= 4 kW, 25 <= T_s <= 55 C

The controls are the electrical power ``P`` and the supply temperature ``T_s``. The COP is
``interpolant("bspline")`` on its 9 x 9 table, mapped over the 96 stages. The hourly price and
outdoor-temperature forecasts are ``Opti`` parameters, read at the stage times by ``interp1d`` with
``mode="floor"``, a zero-order hold.
"""

import casadi as ca
import numpy as np

from _common import as_arrays, casadi_ipopt_options, casadi_jit, show
from _data import HEAT_PUMP, cop_table, day_ahead

HP = HEAT_PUMP
N = HP.horizon


def build(verbose: bool = False):
  (t_out_grid, t_s_grid), cop_values = cop_table()
  cop = ca.interpolant("cop", "bspline", [list(t_out_grid), list(t_s_grid)], np.ravel(cop_values, order="F"))
  hours, price_data, t_out_data = day_ahead()
  times = list(HP.dt * np.arange(N))

  opti = ca.Opti()
  P, T_s, T = opti.variable(N), opti.variable(N), opti.variable(2, N + 1)
  x0, price, t_out = opti.parameter(2), opti.parameter(24), opti.parameter(24)
  price_k = ca.interp1d(list(hours), price, times, "floor")
  t_out_k = ca.interp1d(list(hours), t_out, times, "floor")
  Q = cop.map(N)(ca.horzcat(t_out_k, T_s).T).T * P
  room, mass = T[0, :-1].T, T[1, :-1].T
  room_next = room + HP.dt / HP.c_air * (Q - HP.h_air_mass * (room - mass) - HP.h_air_out * (room - t_out_k))
  mass_next = mass + HP.dt / HP.c_mass * (HP.h_air_mass * (room - mass) - HP.h_mass_out * (mass - t_out_k))
  opti.subject_to(T[:, 0] == x0)
  opti.subject_to(T[0, 1:].T == room_next)
  opti.subject_to(T[1, 1:].T == mass_next)
  opti.subject_to(Q == HP.k_emitter * (T_s - room))
  opti.subject_to(opti.bounded(0.0, P, HP.p_max))
  opti.subject_to(opti.bounded(HP.t_supply[0], T_s, HP.t_supply[1]))
  opti.subject_to(opti.bounded(HP.t_room[0], T[0, :], HP.t_room[1]))
  cost = HP.dt * ca.dot(price_k, P) + 1e-3 * ca.sumsqr(P)
  opti.minimize(cost)
  opti.solver("ipopt", {**casadi_ipopt_options(verbose, tol=1e-10), **casadi_jit()})
  opti.set_value(x0, np.array(HP.x0))
  opti.set_value(price, price_data)
  opti.set_value(t_out, t_out_data)

  def run():
    opti.set_initial(P, 1.0)
    opti.set_initial(T_s, 40.0)
    opti.set_initial(T, np.tile(np.array(HP.x0)[:, None], (1, N + 1)))
    sol = opti.solve()
    return as_arrays({"P": sol.value(P), "T_s": sol.value(T_s), "T": np.asarray(sol.value(T)).T, "cost": sol.value(cost), "iter": sol.stats()["iter_count"]})

  return run


if __name__ == "__main__":
  show(build(verbose=True)())
