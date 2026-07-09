"""Solver adapters.  Importing this package registers all adapters."""
from __future__ import annotations

# Each import registers a factory via @register_solver and is import-safe even
# if the underlying solver is not installed (availability is checked lazily).
from fastbench.solvers import casadi_ipopt  # noqa: F401
from fastbench.solvers import osqp_solver   # noqa: F401
from fastbench.solvers import piqp_solver   # noqa: F401
from fastbench.solvers import proxqp_solver  # noqa: F401
from fastbench.solvers import acados_solver  # noqa: F401
from fastbench.solvers import dompc_solver   # noqa: F401
from fastbench.solvers import grampc_solver  # noqa: F401
from fastbench.solvers import fastsqp_solver  # noqa: F401
