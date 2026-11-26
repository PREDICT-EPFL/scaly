# PIQP's multistage KKT system in Scaly

This example ports PIQP 0.7.0's `sparse_multistage` KKT solver, `sparse/multistage_kkt.hpp` and
`.tpp`, onto `examples/piqp_sparse`. PIQP shares its data, equilibration and iteration between
its KKT solvers, and so does this example: only the KKT system is new. So far it reaches the
factorization of the condensed matrix

```
Psi = P + diag(x_reg) + A^T A / delta + G^T diag(z_reg)^-1 G
```

in the given order of the variables.

## The arrow structure

For a problem written stage after stage, `Psi` is block tridiagonal, and the variables coupled to
every stage, such as parameters shared by all stages, form a dense border at the end:

```
[D_1                                  ]
[B_1 D_2                              ]
[    B_2 D_3                          ]
[            ...                      ]
[                      B_N-2 D_N-1    ]
[E_1 E_2 E_3 ... E_N-3 E_N-2 E_N-1 D_N]
```

`extract_arrow_structure` finds the blocks from the pattern of `Psi` alone, as PIQP's constructor
does, deciding for each entry whether to grow the current block or to widen the arrow by comparing
the flops of the two factorizations. A Cholesky factor of `Psi` in this order keeps the structure:
its fill stays inside the blocks.

```python
from piqp_multistage import MultistageKKT
from piqp_sparse.data import Structure

structure, *_ = Structure.from_matrices(P, A, G)
kkt = MultistageKKT(structure)
kkt.block_info      # PIQP's block_info: (start, diag_size, off_diag_size) per block, the arrow's corner last
kkt.blocks          # the D_i, B_i and E_i blocks of PIQP's BlockKKT, as row and column ranges
```

To draw the structure on `Psi` and on its factor for a random 12-stage MPC problem:

```bash
uv run examples/piqp_multistage/plot_arrow_structure.py arrow_structure.png
```

## Where PIQP's code went

| PIQP 0.7.0 | Here | What it holds |
|---|---|---|
| `sparse/blocksparse/block_info.hpp` | `multistage_kkt.py` | `BlockInfo` |
| `sparse/multistage_kkt.hpp`, `.tpp` | `multistage_kkt.py` | `MultistageKKT`: `extract_arrow_structure`, its `flops_*` counts, the blocks of `BlockKKT`, `update_scalings_and_factor` |
| everything else | `examples/piqp_sparse` | the data, the assembly of `Psi`, the equilibration and the iteration |

`tests/integration/test_piqp_multistage.py` checks `block_info` against PIQP's on generated
problems.

## Not yet

The factorization is the general sparse one, `scaly.linalg.cholesky` in natural order, not
PIQP's recursion over the blocks, and there is no solve, no generated solver and no comparison
with PIQP's iterations.
