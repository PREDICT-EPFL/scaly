"""Linear algebra as library Functions: triangular solves, Cholesky and ``solve``."""

from .dense import cholesky, solve_triangular
from .entry import solve

__all__ = ["cholesky", "solve", "solve_triangular"]
