"""Closed-loop (and open-loop) simulation drivers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from fastbench.solvers.base import SolveStats


@dataclass
class EpisodeLog:
    X: np.ndarray                      # (n_sim+1, nx) closed-loop states
    U: np.ndarray                      # (n_sim, nu) applied inputs
    Xref: np.ndarray                   # (n_sim+1, nx) reference states
    stats: List[SolveStats] = field(default_factory=list)
    x0: Optional[np.ndarray] = None
    seed: Optional[int] = None


def run_closed_loop(problem, adapter, x0: np.ndarray,
                    rng: Optional[np.random.Generator] = None) -> EpisodeLog:
    """Run one closed-loop episode of ``problem`` controlled by ``adapter``.

    At each step the adapter solves the OCP for the current state; the first
    input is applied to the *plant* (which may differ from the model and add
    noise).  If a solve fails, the previous input is held (graceful fallback).
    """
    adapter.reset()
    nx, nu, n_sim = problem.meta.nx, problem.meta.nu, problem.meta.n_sim
    lbu, ubu = problem.bounds()[2], problem.bounds()[3]

    x = np.asarray(x0, float).copy()
    X = [x.copy()]
    Xref = [problem.reference_traj(0)[0]]
    U: List[np.ndarray] = []
    stats: List[SolveStats] = []
    u_prev = np.zeros(nu)

    for k in range(n_sim):
        st = adapter.solve(x, k)
        stats.append(st)
        if st.success and st.u0 is not None and np.all(np.isfinite(st.u0)):
            u = np.clip(np.asarray(st.u0, float).flatten(), lbu, ubu)
        else:
            u = u_prev.copy()       # hold last input on failure
        u_prev = u
        x = np.asarray(problem.plant_step(x, u, rng), float).flatten()
        U.append(u)
        X.append(x.copy())
        Xref.append(problem.reference_traj(k + 1)[0])

    return EpisodeLog(np.array(X), np.array(U), np.array(Xref),
                      stats=stats, x0=np.asarray(x0, float),
                      seed=None if rng is None else int(rng.bit_generator.seed_seq.entropy or 0))


def run_open_loop(problem, adapter, x0: np.ndarray) -> EpisodeLog:
    """Single OCP solve from ``x0`` (the predicted trajectory is the result)."""
    adapter.reset()
    st = adapter.solve(np.asarray(x0, float), 0)
    if st.x_pred is not None:
        X = np.asarray(st.x_pred)
        U = np.asarray(st.u_pred) if st.u_pred is not None else np.zeros((problem.meta.N, problem.meta.nu))
    else:
        X = np.array([x0]); U = np.zeros((0, problem.meta.nu))
    Xref = np.array([problem.reference_traj(k)[0] for k in range(len(X))])
    return EpisodeLog(X, U, Xref, stats=[st], x0=np.asarray(x0, float))
