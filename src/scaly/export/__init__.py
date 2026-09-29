"""Code for other tools: the C++ header (the ``cpp`` output adapter), CasADi external functions (the ``casadi`` adapter) and the acados drop-in."""

from .acados import acados_functions, install_dropin
from .casadi import CASADI_QUERIES, casadi_scratch, casadi_sparsity, check_casadi_layout
from .cpp import render_cpp_header

__all__ = ["CASADI_QUERIES", "acados_functions", "casadi_scratch", "casadi_sparsity", "check_casadi_layout", "install_dropin", "render_cpp_header"]
