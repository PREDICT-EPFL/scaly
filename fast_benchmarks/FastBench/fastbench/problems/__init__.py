"""Benchmark problems.  Importing this package registers all problems."""
from __future__ import annotations

from fastbench.problems.pendulum_swingup import problem as _pendulum  # noqa: F401
from fastbench.problems.chain_mass import problem as _chain  # noqa: F401
from fastbench.problems.kinematic_vehicle import problem as _vehicle  # noqa: F401
# LTI / QP
from fastbench.problems.double_integrator import problem as _di  # noqa: F401
from fastbench.problems.dc_motor import problem as _dcm  # noqa: F401
from fastbench.problems.rocket_landing_1d import problem as _rl1  # noqa: F401
from fastbench.problems.oscillating_masses import problem as _osc  # noqa: F401
# Nonlinear
from fastbench.problems.van_der_pol import problem as _vdp  # noqa: F401
from fastbench.problems.unicycle import problem as _uni  # noqa: F401
from fastbench.problems.planar_quadrotor import problem as _pq  # noqa: F401
from fastbench.problems.cstr import problem as _cstr  # noqa: F401
from fastbench.problems.quadruple_tank import problem as _qt  # noqa: F401
from fastbench.problems.vehicle_obstacle import problem as _vo  # noqa: F401
from fastbench.problems.two_link_arm import problem as _arm  # noqa: F401
