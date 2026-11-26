"""Compare the port's extract_arrow_structure with PIQP's on generated problems and PIQP's own multistage test problems.

Build the oracle with oracle.sh first, then

    uv run internal/notes/piqp_multistage_arrow_2026-11-26/compare.py [path/to/piqp/tests/data]

prints one line per problem and the blocks the oracle printed for the problems the tests pin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import io, sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "examples"))

from piqp_multistage import MultistageKKT  # noqa: E402
from piqp_multistage.problems import mpc_qp  # noqa: E402
from piqp_sparse.data import Structure  # noqa: E402


def _triplets(M: sparse.coo_array) -> str:
  lines = [f"{M.shape[0]} {M.shape[1]} {M.nnz}"]
  lines += [f"{i} {j} {float(v)!r}" for i, j, v in zip(M.row, M.col, M.data, strict=True)]
  return "\n".join(lines)


def oracle(P, A, G) -> list[tuple[int, int, int]]:
  """PIQP's block_info for the stored patterns of ``P``'s upper triangle, ``A`` and ``G``."""
  P = sparse.triu(sparse.coo_array(P)).tocoo()
  A, G = sparse.coo_array(A), sparse.coo_array(G)
  text = f"{P.shape[0]} {A.shape[0]} {G.shape[0]}\n{_triplets(P)}\n{_triplets(A.T.tocoo())}\n{_triplets(G.T.tocoo())}\n"
  out = subprocess.run([str(HERE / "block_info")], input=text, capture_output=True, text=True, check=True).stdout
  return [tuple(int(x) for x in line.split()) for line in out.splitlines()]


def port(P, A, G) -> list[tuple[int, int, int]]:
  structure, *_ = Structure.from_matrices(sparse.csc_array(P), sparse.csc_array(A), sparse.csc_array(G))
  return [tuple(b) for b in MultistageKKT(structure).block_info]


def problems(data_dir: Path | None):
  for horizon, nx, nu, n_global, seed in [(5, 3, 2, 0, 0), (8, 4, 2, 2, 1), (20, 6, 3, 3, 2), (10, 2, 1, 0, 3), (40, 5, 2, 1, 4)]:
    qp = mpc_qp(horizon, nx, nu, n_global, seed=seed)
    yield f"mpc_qp({horizon}, {nx}, {nu}, {n_global}, seed={seed})", qp.P, qp.A, qp.G
  if data_dir is not None:
    for path in sorted(data_dir.glob("*.mat")):
      mat = io.loadmat(path)
      yield path.stem, *(sparse.csc_array(mat[k]) if mat[k].size else sparse.csc_array(mat[k].shape) for k in ("P", "A", "G"))


def main() -> None:
  data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  failures = 0
  for name, P, A, G in problems(data_dir):
    if P.shape[0] > 2000:
      print(f"{name:40s} skipped, n = {P.shape[0]}")
      continue
    expected, got = oracle(P, A, G), port(P, A, G)
    same = expected == got
    failures += not same
    print(f"{name:40s} n = {P.shape[0]:4d}  blocks = {len(expected):3d}  arrow = {expected[-1][1]:3d}  {'same' if same else 'DIFFERENT'}")
    if not same:
      print("  PIQP:", expected)
      print("  port:", got)
    elif name.startswith("mpc_qp"):
      print("  ", expected)
  print(f"{failures} differ")
  sys.exit(1 if failures else 0)


if __name__ == "__main__":
  np.set_printoptions(linewidth=200)
  main()
