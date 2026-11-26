"""PIQP 0.7.0's multistage KKT system generated with Scaly, so far up to the factorization of its condensed matrix.

``multistage_kkt`` follows PIQP's ``sparse/multistage_kkt.hpp`` and ``.tpp``: the arrow structure
of ``Psi`` and its factorization. Everything else, the data, the equilibration and the iteration,
is ``piqp_sparse``'s, as PIQP shares them between its KKT solvers. ``problems`` builds random
multistage QPs and ``plot_arrow_structure`` draws the structure on ``Psi`` and on its factor.
"""

from .multistage_kkt import Block, BlockInfo, Factor, MultistageKKT, extract_arrow_structure

__all__ = ["Block", "BlockInfo", "Factor", "MultistageKKT", "extract_arrow_structure"]
