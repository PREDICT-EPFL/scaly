"""Solve Maros–Mészáros problems with the PIQP example in ``examples/piqp_sparse`` and with PIQP, and compare them.

Two sets of problems:

- the 48 small ones in ``tests/integration/data``, with ``n + m <= 1000``, which the tests use;
- qpbenchmark's sparse subset, the 76 problems with more than 1000 variables or 1000 constraints
  (``maros_meszaros_sparse.py`` in https://github.com/qpsolvers/maros_meszaros_qpbenchmark). Their
  ``.mat`` files are downloaded on first use from the pinned commit ``QPBENCHMARK_COMMIT`` into
  ``SCALY_MAROS_MESZAROS_DIR``, by default ``~/.cache/scaly/maros_meszaros``.

Run ``uv run examples/case_studies/maros_meszaros/maros_meszaros.py`` for the 48 small problems,
``... maros_meszaros.py --sparse`` for the 76 sparse ones, or name some:
``... maros_meszaros.py HS21 QAFIRO``. Each new sparsity pattern generates and compiles a solver
the first time, which takes from seconds to hours for the largest sparse problems; later runs
read Scaly's cache. ``benchmark.py`` next to this file measures both solvers on either set.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

DATA = Path(__file__).resolve().parents[3] / "tests" / "integration" / "data" / "maros_meszaros"
# The collection writes an absent bound as +-1e20.
_ABSENT = 1e19


# qpbenchmark's sparse subset at this commit (2026-10-01): its rule, n > 1000 or more than 1000
# constraints counting equalities, each finite side of an inequality and n box rows, picks these 76.
QPBENCHMARK_COMMIT = "afa738d9a66aca7f5ae223034c72f2e9c6866aff"
SPARSE = (
  "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP", "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP", "BOYD1", "BOYD2",
  "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300", "CVXQP1_L", "CVXQP1_M", "CVXQP2_L", "CVXQP2_M",
  "CVXQP3_L", "CVXQP3_M", "DTOC3", "EXDATA", "GOULDQP2", "GOULDQP3", "HUES-MOD", "HUESTIS", "KSIP", "LASER",
  "LISWET1", "LISWET10", "LISWET11", "LISWET12", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6", "LISWET7",
  "LISWET8", "LISWET9", "MOSARQP1", "MOSARQP2", "POWELL20", "PRIMAL4", "Q25FV47", "QETAMACR", "QFFFFF80", "QGFRDXPN",
  "QGROW22", "QPILOTNO", "QSCFXM2", "QSCFXM3", "QSCRS8", "QSCSD6", "QSCSD8", "QSCTAP2", "QSCTAP3", "QSEBA",
  "QSHELL", "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S", "QSHIP12L", "QSHIP12S", "QSIERRA", "QSTANDAT", "STADAT1",
  "STADAT2", "STADAT3", "STCQP1", "STCQP2", "UBH1", "YAO",
)  # fmt: skip


def problems(sparse_subset: bool = False) -> list[str]:
  """The 48 small problems, or with ``sparse_subset`` qpbenchmark's 76 sparse ones."""
  if sparse_subset:
    return list(SPARSE)
  return sorted(p.stem for p in DATA.glob("*.npz") if not p.stem.endswith("_patterns"))


def sparse_dir() -> Path:
  return Path(os.environ.get("SCALY_MAROS_MESZAROS_DIR") or Path.home() / ".cache" / "scaly" / "maros_meszaros") / QPBENCHMARK_COMMIT[:12]


def fetch(name: str) -> Path:
  """The ``.mat`` file of a sparse problem, downloaded from qpbenchmark's pinned commit if missing."""
  path = sparse_dir() / f"{name}.mat"
  if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/{QPBENCHMARK_COMMIT}/data/{name}.mat"
    partial = path.with_suffix(f".{os.getpid()}.part")
    with urllib.request.urlopen(url) as response:
      partial.write_bytes(response.read())
    partial.replace(path)
  return path


def _raw(name: str) -> dict[str, Any]:
  """The fields of one problem: ``P`` (both triangles), ``A``, ``q``, ``l`` and ``u``."""
  if (DATA / f"{name}.npz").exists():
    with np.load(DATA / f"{name}.npz") as raw:

      def matrix(key: str) -> sparse.csc_array:
        return sparse.csc_array((raw[f"{key}_data"], raw[f"{key}_indices"], raw[f"{key}_indptr"]), shape=tuple(raw[f"{key}_shape"]))

      return {"P": matrix("P"), "A": matrix("A"), "q": np.array(raw["q"]), "l": np.array(raw["l"]), "u": np.array(raw["u"])}
  from scipy import io

  mat = io.loadmat(str(fetch(name)))
  vector = lambda key: np.asarray(mat[key], dtype=float).ravel()  # noqa: E731
  return {"P": sparse.csc_array(mat["P"], dtype=float), "A": sparse.csc_array(mat["A"], dtype=float), "q": vector("q"), "l": vector("l"), "u": vector("u")}


def load(name: str) -> dict[str, Any]:
  """One problem, ``min 1/2 x'Px + q'x s.t. l <= Ax <= u``, in PIQP's form.

  Rows with ``l = u`` become equalities, rows with a single unit entry become bounds on that
  variable, and the rest inequalities.
  """
  raw = _raw(name)
  P, constraints, q, l, u = raw["P"], raw["A"].tocsr(), raw["q"], raw["l"], raw["u"]
  l[l <= -_ABSENT], u[u >= _ABSENT] = -np.inf, np.inf
  n = P.shape[0]
  counts = np.diff(constraints.indptr)
  single = counts == 1
  single[single] &= constraints.data[constraints.indptr[:-1][single]] == 1.0
  equality = l == u
  boxes = single & ~equality
  x_l, x_u = np.full(n, -np.inf), np.full(n, np.inf)
  for row in np.flatnonzero(boxes):
    j = constraints.indices[constraints.indptr[row]]
    x_l[j], x_u[j] = max(x_l[j], l[row]), min(x_u[j], u[row])
  inequality = ~equality & ~boxes
  return {
    "P": P,
    "c": q,
    "A": constraints[equality].tocsc(),
    "b": l[equality],
    "G": constraints[inequality].tocsc(),
    "h_l": l[inequality],
    "h_u": u[inequality],
    "x_l": x_l,
    "x_u": x_u,
  }


def compare(name: str) -> dict[str, Any]:
  """Solve one problem both ways: the statuses, the iteration counts, and the objectives' relative difference."""
  from piqp_sparse import SparseSolver, reference

  problem = load(name)
  solver = SparseSolver()
  solver.setup(**problem)
  status = solver.solve()
  ref = reference.solve(**problem)
  objective, objective_ref = solver.result.info.primal_obj, ref.info["primal_obj"]
  return {
    "name": name,
    "status": status,
    "piqp_status": ref.status,
    "iter": solver.result.info.iter,
    "piqp_iter": int(ref.info["iter"]),
    "objective_error": abs(objective - objective_ref) / max(1.0, abs(objective_ref)),
  }


def main(args: list[str]) -> None:
  sparse_subset = "--sparse" in args
  names = [a for a in args if a != "--sparse"]
  print(f"{'problem':10s} {'status':18s} {'PIQP status':18s} {'iterations':>10s}  {'objective error':>15s}")
  for name in names or problems(sparse_subset):
    row = compare(name)
    iterations = f"{row['iter']}/{row['piqp_iter']}"
    print(f"{name:10s} {row['status'].name:18s} {row['piqp_status'].name:18s} {iterations:>10s}  {row['objective_error']:15.1e}", flush=True)


if __name__ == "__main__":
  # piqp_sparse is a package in examples/, two levels up.
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  main(sys.argv[1:])
