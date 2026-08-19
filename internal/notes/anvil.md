# Relationship to anvil

Alloy spun out of [anvil](https://github.com/PREDICT-EPFL/anvil), a tinygrad-based code-generation
framework for optimal control, in May 2026. They share design lineage but no runtime code: alloy is
pure Python with NumPy as its only required dependency. The anvil repository remains a useful
reference for SQP solver architecture, the multistage formulation pattern, and tinygrad-style graph
rewriting.
