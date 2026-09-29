# Export

Code for other tools. The output adapters register under their names, `write_module(fn, directory,
adapters=("cpp",))` or `("casadi",)`; the acados drop-in builds on the CasADi layer.

## The C++ header

::: scaly.export.cpp.render_cpp_header

## The CasADi layer

::: scaly.export.casadi.check_casadi_layout

::: scaly.export.casadi.casadi_sparsity

::: scaly.export.casadi.casadi_scratch

## The acados drop-in

::: scaly.export.acados.acados_functions

::: scaly.export.acados.install_dropin
