"""FastBench - a reproducible benchmark for optimization-based control.

FastBench provides a standard interface for posing optimal-control / MPC
problems once and solving them with many backends (CasADi+IPOPT, acados,
OSQP, PIQP, do-mpc, GRAMPC), running closed-loop (or open-loop) simulations,
and producing comparable benchmark metrics.
"""

__version__ = "0.1.0"

from fastbench.core.problem import Problem, ProblemMeta  # noqa: E402
from fastbench.core.registry import (  # noqa: E402
    register_problem,
    register_solver,
    get_problem,
    get_solver,
    list_problems,
    list_solvers,
)

__all__ = [
    "Problem",
    "ProblemMeta",
    "register_problem",
    "register_solver",
    "get_problem",
    "get_solver",
    "list_problems",
    "list_solvers",
]
