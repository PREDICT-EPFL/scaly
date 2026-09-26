"""Legendre-Gauss-Lobatto nodes, weights and differentiation matrix, for ``pseudospectral_collocation``.

Plain NumPy, shared by the CasADi and the Scaly version; copied from
casadi/docs/examples/python/pseudospectral_collocation.py.
"""

import numpy as np


def _legendre_all(N: int, x: np.ndarray):
  """
  Computes legendre polynomial P_N(x) and its derivative P'_N(x), and P''_N(x) using numerically
  stable three-term recurrence relations.
  """
  P0, P1 = np.ones_like(x), x.copy()
  dP0, dP1 = np.zeros_like(x), np.ones_like(x)
  d2P0, d2P1 = np.zeros_like(x), np.zeros_like(x)

  if N == 0:
    return P0, dP0, d2P0
  if N == 1:
    return P1, dP1, d2P1

  for k in range(1, N):
    P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1)
    dP2 = ((2 * k + 1) * (P1 + x * dP1) - k * dP0) / (k + 1)
    d2P2 = ((2 * k + 1) * (2 * dP1 + x * d2P1) - k * d2P0) / (k + 1)

    P0, P1 = P1, P2
    dP0, dP1 = dP1, dP2
    d2P0, d2P1 = d2P1, d2P2

  return P1, dP1, d2P1


def lgl_nodes(N: int, tol: float = 1e-15, max_iter: int = 100) -> np.ndarray:
  """
  Given N intervals, computes N+1 Legendre-Gauss-Lobatto (LGL) points on [-1, 1]

  Roots of (1-x**2)P'_N(x) are LGL points

  Parameters:
      N (int): intervals/polynomial degree (returns N+1 grid points)
      tol (float): tolerance for newton ralphson method
      max_iter (int): maximum newton-raphson iterations.

  Returns:
      x (ndarray): a 1d array of N+1 LGL points from -1 to 1
  """
  if N < 1:
    raise ValueError("Degree N must be at least 1.")

  # inital guess
  k = np.arange(N + 1)
  x = -np.cos(np.pi * k / N)

  x[0] = -1.0
  x[-1] = 1.0

  # begin newton iterations
  if N > 1:
    x_int = x[1:-1].copy()
    for _ in range(max_iter):
      _, dP, d2P = _legendre_all(N, x_int)
      dx = dP / d2P
      x_int -= dx
      if np.max(np.abs(dx)) < tol:
        break
    x[1:-1] = x_int

  x[0] = -1.0
  x[-1] = 1.0

  # use symmetry to remove round-off errors
  x = 0.5 * (x - x[::-1])
  return x


def lgl_weights(N: int) -> np.ndarray:
  """
  Computes LGL quadrature weights for N+1 nodes on [-1, 1].

  wi=2/[(N)(N+1)(P_N(x_i))**2]

  Parameters:
      N (int): Polynomial degree (returns N+1 weights).

  Returns:
      w (ndarray): Array of shape (N+1,) containing integration weights.
  """
  if N < 1:
    raise ValueError("Degree N must be at least 1.")
  x = lgl_nodes(N)
  P_N, _, _ = _legendre_all(N, x)
  w = 2.0 / (N * (N + 1) * (P_N**2))
  # symmetry to avoid round-off errors
  w = 0.5 * (w + w[::-1])
  return w


def lgl_diff_matrix(N: int) -> np.ndarray:
  """
  Computes the (N+1)x(N+1) LGL differentiation matrix using
  exact polynomial ratios and the negative-sum trick.

  Derivative of the Lagrange interpolation polynomial passing through points
  (x1,y1),.....(x_(N+1),y_(N+1)), evaluated at x_1...x_(N+1) given by

  Yd=DY, where Y=[y1....y_(N+1)].T and Yd is the derivative of Y at LGL points.

  Parameters:
      N (int): Polynomial degree.

  Returns:
      D (ndarray): High-precision differentiation matrix of shape (N+1, N+1).
  """
  if N < 1:
    raise ValueError("Degree N must be at least 1.")

  x = lgl_nodes(N)
  P_N, _, _ = _legendre_all(N, x)

  P_ratio = np.outer(P_N, 1.0 / P_N)
  dX = x[:, None] - x[None, :]
  np.fill_diagonal(dX, 1.0)  # Avoid division by zero on diagonal
  D = P_ratio / dX
  np.fill_diagonal(D, 0.0)
  np.fill_diagonal(D, -np.sum(D, axis=1))
  D = 0.5 * (D - D[::-1, ::-1])
  return D


def lgl_setup(N: int):
  """
  Convenience wrapper returning LGL nodes (x), quadrature weights (w),
  and differentiation matrix (D).
  """
  x = lgl_nodes(N)
  w = lgl_weights(N)
  D = lgl_diff_matrix(N)
  return x, w, D
