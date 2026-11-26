"""A sparse QP interior-point solver generated with Scaly that reproduces PIQP 0.7.0 with every constraint eliminated from its KKT system.

The modules follow PIQP's sources: ``settings``, ``results``, ``variables``, ``data``,
``preconditioner`` (Ruiz equilibration), ``kkt`` (``sparse::KKT`` in mode ``KKT_ALL_ELIMINATED``),
``kkt_system`` (``KKTSystem``) and ``solver`` (``SparseSolver``, ``solver.tpp``, and ``build_solver``,
which returns the generated Functions).
``reference`` runs the PIQP library the ``scaly-piqp`` plugin vendors, for comparison.
"""

from .results import Info, Result, Status
from .settings import Settings
from .solver import SparseSolver, build_solver, pack_raw

__all__ = ["Info", "Result", "Settings", "SparseSolver", "Status", "build_solver", "pack_raw"]
