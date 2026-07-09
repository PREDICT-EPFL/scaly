"""Common solver-adapter interface and result containers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BuildInfo:
    """One-off build/setup metrics for a (problem, solver) pair."""
    build_time_s: float = 0.0          # wall time to construct the solver
    generation_time_s: float = 0.0     # C-code generation time (codegen solvers)
    compile_time_s: float = 0.0        # C-code compile time
    generated_code_bytes: int = 0      # size of generated sources
    generated_code_loc: int = 0        # generated lines of code
    compiled_bytes: int = 0            # size of compiled artifacts (.so/.o)
    notes: str = ""


@dataclass
class SolveStats:
    """Per-call solve metrics, logged at every closed-loop step."""
    u0: Optional[np.ndarray] = None
    success: bool = False
    status: str = ""
    solve_time_s: float = float("nan")
    iterations: int = -1
    cost: float = float("nan")
    kkt_residual: float = float("nan")        # stationarity / optimality residual
    constraint_violation: float = float("nan")
    x_pred: Optional[np.ndarray] = None
    u_pred: Optional[np.ndarray] = None


class SolverAdapter:
    """Base class.  Subclasses wrap a concrete solver behind a common API."""

    name: str = "base"
    requires_lti: bool = False        # True for pure QP solvers (OSQP/PIQP)
    supports_path_constraints: bool = False  # handles nonlinear inequalities

    # ---- capability checks --------------------------------------------------
    def available(self) -> bool:
        """Whether the backend is importable/usable on this machine."""
        raise NotImplementedError

    def supports(self, problem) -> bool:
        """Whether this adapter can solve the given problem."""
        if getattr(problem, "has_path_constraints", False) \
                and not self.supports_path_constraints:
            return False
        if self.requires_lti:
            return bool(problem.meta.is_lti and problem.lti_matrices() is not None)
        return True

    # ---- lifecycle ----------------------------------------------------------
    def build(self, problem) -> BuildInfo:
        """Construct the solver for ``problem`` and return build metrics."""
        raise NotImplementedError

    def solve(self, x: np.ndarray, k: int) -> SolveStats:
        """Solve the OCP at state ``x`` and absolute time index ``k``."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear warm-start memory between episodes."""
        pass
