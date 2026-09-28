"""The helpers of ``examples/casadi/_common.py``, for the pairs in this directory.

The files here import ``_common`` as the pairs in ``examples/casadi`` do, and this module loads
that one by its path, so a pair runs on its own from any directory as well as through
``compare.py --dir examples/interp/pairs``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_casadi_common", Path(__file__).resolve().parents[2] / "casadi" / "_common.py")
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

Run = _module.Run
as_arrays = _module.as_arrays
show = _module.show
casadi_jit = _module.casadi_jit
casadi_ipopt_options = _module.casadi_ipopt_options
scaly_ipopt_options = _module.scaly_ipopt_options
