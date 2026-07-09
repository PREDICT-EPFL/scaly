"""The FastBench problem interface.

Every benchmark problem subclasses :class:`Problem` and provides a *symbolic*
description of the prediction model, cost and constraints (in CasADi), plus a
*numeric* plant model for closed-loop simulation.  Because the model is
symbolic, the same problem definition can be handed to any solver adapter.

Conventions
-----------
* States ``x`` (nx,), inputs ``u`` (nu,).
* Continuous dynamics ``xdot = dynamics_ct(x, u)``; the default discretization
  is fixed-step RK4 (override ``discrete_dynamics`` for analytic maps).
* Costs are stage-additive: ``sum_k stage_cost(x_k,u_k,xref_k,uref_k)
  + terminal_cost(x_N, xref_N)``.
* Reference is supplied numerically per absolute time index via
  ``reference_traj(k)`` (constant for regulation, time-varying for tracking).
* The *plant* used for simulation (``plant_step``) may deliberately differ from
  the prediction model (parameter mismatch) and may inject process/measurement
  noise, so closed-loop numbers reflect robustness, not just nominal accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from fastbench.core.dynamics import rk4


@dataclass
class ProblemMeta:
    name: str
    nx: int
    nu: int
    dt: float                 # sampling time [s]
    N: int                    # prediction horizon length [steps]
    n_sim: int                # closed-loop simulation length [steps]
    is_lti: bool = False      # True -> linear dynamics + quadratic cost (QP-able)
    convex: bool = False      # OCP is convex
    nonlinear: bool = True
    problem_class: str = ""   # e.g. "swing-up", "regulation", "tracking"
    description: str = ""
    sources: List[str] = field(default_factory=list)


class Problem:
    """Abstract base class for a benchmark control problem."""

    meta: ProblemMeta

    # True if the problem defines nonlinear path (inequality) constraints that
    # go beyond simple state/input box bounds.  Used by adapters to decide
    # support (only backends that handle general inequalities apply).
    has_path_constraints: bool = False

    # --- bounds (numpy arrays of shape (nx,)/(nu,) or None) ------------------
    lbx: Optional[np.ndarray] = None
    ubx: Optional[np.ndarray] = None
    lbu: Optional[np.ndarray] = None
    ubu: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ model
    def dynamics_ct(self, x, u):
        """Continuous-time dynamics xdot = f(x, u) as a CasADi expression."""
        raise NotImplementedError

    def discrete_dynamics(self, x, u):
        """One-step discrete prediction model.  Default: RK4 of dynamics_ct."""
        return rk4(self.dynamics_ct, x, u, self.meta.dt)

    # ------------------------------------------------------------------- cost
    def stage_cost(self, x, u, xref, uref):
        """Scalar stage cost l(x,u; xref,uref) as a CasADi expression."""
        raise NotImplementedError

    def terminal_cost(self, x, xref):
        """Scalar terminal cost m(x; xref) as a CasADi expression."""
        raise NotImplementedError

    # ------------------------------------------------------------ constraints
    def path_constraints(self, x, u):
        """Optional nonlinear path constraints.

        Return ``(h, lh, uh)`` with ``h`` a CasADi expression and ``lh,uh``
        numpy bounds such that ``lh <= h(x,u) <= uh``; or ``None``.
        """
        return None

    # -------------------------------------------------------------- reference
    def reference_traj(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Numeric reference (xref, uref) for absolute time index k."""
        return np.zeros(self.meta.nx), np.zeros(self.meta.nu)

    # ------------------------------------------------------ initial conditions
    def x0_nominal(self) -> np.ndarray:
        raise NotImplementedError

    def x0_samples(self, n: int, rng: np.random.Generator) -> List[np.ndarray]:
        """Initial states for Monte-Carlo episodes (default: jitter nominal)."""
        x0 = self.x0_nominal()
        if n == 1:
            return [x0]
        spread = 0.05 * (np.abs(x0) + 1.0)
        return [x0 + rng.uniform(-1, 1, size=x0.shape) * spread for _ in range(n)]

    # -------------------------------------------------------------- the plant
    def plant_step(self, x: np.ndarray, u: np.ndarray,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """True-system update used for simulation.

        Default is the nominal discrete model with no mismatch/noise.  Override
        to introduce parameter mismatch, disturbances or measurement noise.
        """
        if not hasattr(self, "_fd_plant"):
            from fastbench.core.dynamics import discrete_function
            self._fd_plant = discrete_function(self)
        xn = np.array(self._fd_plant(x, u)).flatten()
        return xn

    # -------------------------------------------------------------- success
    def episode_success(self, X: np.ndarray, U: np.ndarray) -> bool:
        """Whether a closed-loop episode achieved its objective.

        Default: final state within 5% (scaled) of the final reference.
        """
        xref_final, _ = self.reference_traj(self.meta.n_sim)
        err = np.abs(X[-1] - xref_final)
        tol = 0.05 * (np.abs(xref_final) + 1.0)
        return bool(np.all(err <= tol))

    # ------------------------------------------------------ LTI/QP description
    def lti_matrices(self) -> Optional[dict]:
        """For ``is_lti`` problems return ``{A,B,Q,R,Qf}`` (and optional ``c``).

        Used by pure-QP adapters (OSQP, PIQP).  Nonlinear problems return None.
        """
        return None

    # ----------------------------------------------------------------- bounds
    def bounds(self):
        nx, nu = self.meta.nx, self.meta.nu
        big = 1e6
        lbx = self.lbx if self.lbx is not None else -big * np.ones(nx)
        ubx = self.ubx if self.ubx is not None else big * np.ones(nx)
        lbu = self.lbu if self.lbu is not None else -big * np.ones(nu)
        ubu = self.ubu if self.ubu is not None else big * np.ones(nu)
        return (np.asarray(lbx, float), np.asarray(ubx, float),
                np.asarray(lbu, float), np.asarray(ubu, float))
