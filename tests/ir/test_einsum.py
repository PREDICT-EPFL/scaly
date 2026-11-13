import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction


@pytest.mark.parametrize(
  ("spec", "shapes"),
  [
    ("ij,jk->ik", [(2, 3), (3, 4)]),
    ("bij,bjk->bik", [(2, 3, 4), (2, 4, 2)]),
    ("i,j->ij", [(3,), (2,)]),
    ("i,i->", [(3,), (3,)]),
    ("ij->ji", [(2, 3)]),
    ("ii->i", [(3, 3)]),
    ("ii->", [(3, 3)]),
    ("ijik->ijk", [(2, 3, 2, 4)]),
    ("...ij,...j->...i", [(2, 1, 3, 4), (5, 4)]),
    ("...ij,j->...i", [(2, 3, 4), (4,)]),
    ("ji", [(2, 3)]),
    ("ij,jk", [(2, 3), (3, 4)]),
    ("ij,ij->ij", [(1, 3), (2, 1)]),
    ("ij,jk->ik", [(2, 0), (0, 3)]),
    ("ij,jk->ik", [(0, 2), (2, 3)]),
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)]),
    ("i,i,i->", [(3,), (1,), (3,)]),
    (",i->i", [(), (3,)]),
  ],
)
def test_einsum_numpy(spec, shapes):
  rng = np.random.default_rng(68)
  values = [rng.normal(size=s) for s in shapes]
  xs = [sc.sym(f"x{i}", s) for i, s in enumerate(shapes)]
  result = sc.einsum(spec, *xs)
  fn = ConcreteFunction._from_exprs("einsum_numpy", xs, [result], [f"x{i}" for i in range(len(xs))], ["y"])
  np.testing.assert_allclose(fn(*values), np.einsum(spec, *values), atol=1e-12)


@pytest.mark.parametrize(
  ("spec", "shapes"),
  [
    ("ij->ii", [(2, 3)]),
    ("ij->k", [(2, 3)]),
    ("i..->i", [(2,)]),
    ("i->i->i", [(2,)]),
    ("ii->i", [(2, 3)]),
    ("ij,jk->ik", [(2, 3), (4, 2)]),
    ("ij", [(2,)]),
    ("i,j", [(2,)]),
    ("...i->i", [(2, 3)]),
  ],
)
def test_einsum_invalid(spec, shapes):
  with pytest.raises(ValueError):
    sc.einsum(spec, *(sc.sym(f"x{i}", s) for i, s in enumerate(shapes)))


def test_einsum_public_export():
  from scaly.ir.einsum import einsum

  assert sc.einsum is einsum
  assert "einsum" in sc.__all__
