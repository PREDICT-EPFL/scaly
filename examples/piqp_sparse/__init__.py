"""A sparse QP interior-point solver generated with Scaly that reproduces PIQP 0.7.0 with every constraint eliminated from its KKT system.

The modules follow PIQP's sources: ``settings``, ``results``, ``variables``, ``data``,
``preconditioner`` (Ruiz equilibration), ``kkt`` (``sparse::KKT`` in mode ``KKT_ALL_ELIMINATED``),
``kkt_system`` (``KKTSystem``) and ``solver`` (``SparseSolver`` and ``solver.tpp``).
``reference`` runs the PIQP library the ``scaly-piqp`` plugin vendors, for comparison.
"""

from .results import Info, Result, Status
from .settings import Settings
from .solver import SparseSolver

__all__ = ["Info", "Result", "Settings", "SparseSolver", "Status"]
