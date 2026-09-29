"""The acados drop-in: a continuous-time model's explicit-integrator functions under the names and argument lists acados' generated code declares, written as CasADi-layer C over acados' own sources."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..codegen.aot import write_module
from ..function.api import function, jacobian
from ..function.model import Function
from ..function.tree import G, L
from ..ir.expr import Expr, concat


def acados_functions(
  xdot: Callable[[Expr, Expr, Expr], Expr], nx: int, nu: int, *, name: str, n_p: int = 0
) -> dict[str, Function[Any, Any, Any, Any]]:
  """The three functions acados' explicit Runge-Kutta integrator calls for the model
  ``xdot(x, u, p)``, keyed and named as acados' generated model header declares them:

  - ``{name}_expl_ode_fun(x, u, p) -> f``;
  - ``{name}_expl_vde_forw(x, Sx, Sp, u, p) -> (f, Sx_dot, Sp_dot)``, the forward sensitivities, with
    ``Sx`` (``nx x nx``) and ``Sp`` (``nx x nu``) and their derivatives flat and column-major, as
    acados copies a dense argument (Scaly is row-major, CasADi column-major, and Scaly's CasADi layer
    takes no dense matrix);
  - ``{name}_expl_vde_adj(x, lam, u, p) -> [f_x' lam; f_u' lam]``.

  ``p`` is acados' parameter vector, ``n_p`` values. ``xdot`` builds the model's expression in each
  body rather than being called as a Function: the derivative of a call node recomputes the callee's
  forward pass beside the call's own value, which a network-sized model pays for twice."""

  @function(L("x", nx), L("u", nu), L("p", n_p), name=f"{name}_expl_ode_fun")
  def ode(x: Expr, u: Expr, p: Expr) -> Expr:
    return xdot(x, u, p)

  @function(L("x", nx), L("Sx", nx * nx), L("Sp", nx * nu), L("u", nu), L("p", n_p), output=G("f", "Sx_dot", "Sp_dot"), name=f"{name}_expl_vde_forw")
  def vde_forw(x: Expr, sx: Expr, sp: Expr, u: Expr, p: Expr) -> tuple[Expr, Expr, Expr]:
    f = xdot(x, u, p)
    jx, ju = jacobian(f, x), jacobian(f, u)
    sx_rows = sx.reshape((nx, nx)).T  # column-major in, so the row-major reshape is the transpose
    if nu == 1:  # one control: Sp is a column, and its derivative is one too
      sp_dot = jx @ sp + ju.reshape((nx,))
    else:
      sp_dot = (jx @ sp.reshape((nu, nx)).T + ju).T.reshape((nx * nu,))
    return f, (jx @ sx_rows).T.reshape((nx * nx,)), sp_dot

  @function(L("x", nx), L("lam", nx), L("u", nu), L("p", n_p), name=f"{name}_expl_vde_adj")
  def vde_adj(x: Expr, lam: Expr, u: Expr, p: Expr) -> Expr:
    f = xdot(x, u, p)
    return concat([jacobian(f, x).T @ lam, jacobian(f, u).T @ lam])

  return {f"{name}_expl_ode_fun": ode, f"{name}_expl_vde_forw": vde_forw, f"{name}_expl_vde_adj": vde_adj}


def install_dropin(functions: dict[str, Function[Any, Any, Any, Any]], model_dir: Path) -> list[Path]:
  """Replace acados' generated model sources in ``model_dir`` (``<code_export_directory>/<model>_model``)
  by ``functions``' C, one ``<key>.c`` each: the CasADi layer's source with its header inlined, prefixed
  by ``#define casadi_int int``, as acados builds its C with ``casadi_int`` an ``int`` where the layer
  defaults to CasADi's ``long long``. The generated modules are also written, as they are, to
  ``model_dir/scaly``. Run it after acados has generated its code and before it builds."""
  written = []
  for key, fn in functions.items():
    module = write_module(fn, model_dir / "scaly", adapters=("casadi",))
    header = (model_dir / "scaly" / module.header_name).read_text()
    source = (model_dir / "scaly" / module.source_name).read_text().replace(f'#include "{module.header_name}"', header)
    target = model_dir / f"{key}.c"
    target.write_text("#define casadi_int int\n" + source)
    written.append(target)
  return written


__all__ = ["acados_functions", "install_dropin"]
