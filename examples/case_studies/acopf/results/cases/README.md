# Two PGLib-OPF cases, as PowerModels processes them

`case14_ieee.json` and `case118_ieee.json` are `pglib_opf_case14_ieee.m` and `pglib_opf_case118_ieee.m`
from PGLib-OPF v23.07 (IEEE PES Task Force on Benchmarks for Validation of Emerging Power System
Algorithms, <https://github.com/power-grid-lib/pglib-opf>), licensed under the Creative Commons
Attribution 4.0 International license (<http://creativecommons.org/licenses/by/4.0/>).

Changed: they are not the original files but the data `baseline/export_case.jl` writes after
PowerModels' processing (per-unit values, corrected angle limits, quadratic costs, thermal limits,
active components only), so that `acopf.ipynb` can build the models without Julia.
