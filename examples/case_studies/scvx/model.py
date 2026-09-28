"""The 6-DoF powered-descent problem of OpenSCvx's `examples/rocket/6DoF_pdg.py`, in Scaly expressions.

OpenSCvx (adapted from SCvxGEN) augments the rocket's 14 states with the time `t` and a penalty state `y`
whose rate is the squared violation of every continuous-time constraint, and poses everything in
normalized time `tau in [0, 1]` with the time dilation `s` (a zero-order-hold control) multiplying every
rate. States `[m, r (3), v (3), q (4, scalar last), w (3), t, y]`, controls `[T (3, body frame,
first-order hold), s]`. Every constant is the example's.
"""

from __future__ import annotations

import numpy as np

import scaly as sc
from scaly.integrators import Tableau

N_NODES = 5
NX, NU = 16, 4
G_I, ISP, G0, BETA = 1.0, 30.0, 1.0, 0.01
J_DIAG = np.array([0.168 * 2e-2, 0.168, 0.168])
R_ARM, R_CP = np.array([-0.25, 0.0, 0.0]), np.array([0.05, 0.0, 0.0])
RHO, S_A = 1.0, 0.5
CA = np.array([0.5, 1.0, 1.0])
M_DRY, V_MAX, W_MAX = 1.0, 3.0, 0.3752
DEL_MAX, GAMMA, THETA_MAX = np.radians(20.0), np.radians(75.0), np.radians(75.0)
T_MIN, T_MAX = 1.5, 6.5
X_MAX = np.r_[2.0, [10.0] * 3, [V_MAX] * 3, [1.0] * 4, [W_MAX] * 3]
X_MIN = np.r_[1.0, [-10.0] * 3, [-V_MAX] * 3, [-1.0] * 4, [-W_MAX] * 3]
T_BOUNDS = (0.0, 10.0)

# diffrax's Tsit5 (Tsitouras 2011), its stage coefficients and fifth-order weights.
_A_LOWER = [
  [161 / 1000],
  [
    -0.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2,
    0.3354806554923569885444268742502307746751211773934303915373692342452941929761641411569,
  ],
  [
    2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289,
    -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366,
    4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077,
  ],
  [
    5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791,
    -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589,
    7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753,
    -0.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1,
  ],
  [
    5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748,
    -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452,
    8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986,
    -0.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1,
    -0.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1,
  ],
]
_B = [
  0.9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1,
  1 / 100,
  0.4798896504144995747752495322905965199130404621990332488332634944254542060153074523509,
  1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331,
  -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677,
  2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841,
]


def tsit5() -> Tableau:
  a = np.zeros((6, 6))
  for i, row in enumerate(_A_LOWER, start=1):
    a[i, : len(row)] = row
  return Tableau(a, np.array(_B), a.sum(axis=1), 5, "tsit5", None)


def cross(a, b):
  return sc.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def norm(v):
  return (v * v).sum().sqrt()


def rocket(x, T):
  """The 14 physical rates in real time, as the example writes them."""
  m, v, q, w = x[0], x[4:7], x[7:11], x[11:14]
  q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
  C = [
    [q4 * q4 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 - q4 * q3), 2 * (q4 * q2 + q1 * q3)],
    [2 * (q4 * q3 + q1 * q2), q4 * q4 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 - q4 * q1)],
    [2 * (q1 * q3 - q4 * q2), 2 * (q4 * q1 + q2 * q3), q4 * q4 - q1 * q1 - q2 * q2 + q3 * q3],
  ]  # the example's block; its CBI is the transpose, and v' uses CBI.T = this matrix
  v_body = sc.stack([C[0][i] * v[0] + C[1][i] * v[1] + C[2][i] * v[2] for i in range(3)])  # CBI @ v
  A = -0.5 * RHO * norm(v) * S_A * sc.const(CA) * v_body
  f_body = T + A
  v_dot = sc.stack([C[i][0] * f_body[0] + C[i][1] * f_body[1] + C[i][2] * f_body[2] for i in range(3)]) / m + sc.const(np.array([-G_I, 0.0, 0.0]))
  q_dot = sc.stack(
    [
      0.5 * (w[0] * q4 - w[1] * q3 + w[2] * q2),
      0.5 * (w[0] * q3 - w[2] * q1 + w[1] * q4),
      0.5 * (w[1] * q1 - w[0] * q2 + w[2] * q4),
      -0.5 * (w[0] * q1 + w[1] * q2 + w[2] * q3),
    ]
  )
  Jw = sc.const(J_DIAG) * w
  w_dot = (cross(sc.const(R_ARM), T) + cross(sc.const(R_CP), A) - cross(w, Jw)) / sc.const(J_DIAG)
  m_dot = -(1.0 / (ISP * G0)) * norm(T) - BETA
  return sc.concat([m_dot.reshape((1,)), v, v_dot, q_dot, w_dot])


def path_constraints(x, T):
  """Every continuous-time constraint `g <= 0` the example declares, in its order and with its scalings:
  the state boxes (upper, then lower, per state), then the six nonlinear ones, then the time bounds."""
  m, r, v, q, w, t = x[0:1], x[1:4], x[4:7], x[7:11], x[11:14], x[14:15]
  g = []
  for lo, hi, s in ((0, 1, m), (1, 4, r), (4, 7, v), (7, 11, q), (11, 14, w)):
    g += [s - sc.const(X_MAX[lo:hi]), sc.const(X_MIN[lo:hi]) - s]
  q2, q3 = q[1], q[2]
  g += [
    -(1.0 * (m - M_DRY)),
    (0.1 * norm(r[1:]) - np.tan(GAMMA) * r[0]).reshape((1,)),
    (0.1 * (v * v).sum() - V_MAX**2).reshape((1,)),
    (1.0 * np.cos(THETA_MAX) - 1.0 + 2.0 * (q2 * q2 + q3 * q3)).reshape((1,)),
    (1.0 * (w * w).sum() - W_MAX**2).reshape((1,)),
    (0.1 * norm(T) - T[0] / np.cos(DEL_MAX)).reshape((1,)),
    (0.1 * (T * T).sum() - T_MAX**2).reshape((1,)),
    (0.1 * T_MIN**2 - (T * T).sum()).reshape((1,)),
    t - T_BOUNDS[1],
    T_BOUNDS[0] - t,
  ]
  return sc.concat(g)


def augmented(x, u):
  """`dx/dtau` of the 16 augmented states: every rate times the dilation `s`; `t' = 1`, and `y'` the sum
  of the squared positive parts of the constraints."""
  T, s = u[:3], u[3]
  g = path_constraints(x, T)
  relu = sc.maximum(g, 0.0)
  return s * sc.concat([rocket(x, T), sc.const(np.ones(1)), (relu * relu).sum().reshape((1,))])
