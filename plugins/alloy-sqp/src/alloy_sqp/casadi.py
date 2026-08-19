from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from alloy.ir.types import SparsityType
from .external import external_nlp


def _adapter(fn, raw_symbol: str) -> str:
  inputs = [f"const double* in{i}" for i in range(fn.n_in())]
  outputs = [f"double* out{i}" for i in range(fn.n_out())]
  lines = [f"static void {raw_symbol}({', '.join([*inputs, *outputs, 'double* alloy_w'])}) {{", "  (void)alloy_w;"]
  lines.append(f"  const casadi_real* arg[{max(fn.sz_arg(), 1)}] = {{0}};")
  lines.append(f"  casadi_real* res[{max(fn.sz_res(), 1)}] = {{0}};")
  lines.append(f"  static casadi_int iw[{max(fn.sz_iw(), 1)}]; static casadi_real cw[{max(fn.sz_w(), 1)}];")
  lines += [f"  arg[{i}] = in{i};" for i in range(fn.n_in())]
  lines += [f"  res[{i}] = out{i};" for i in range(fn.n_out())]
  lines.append(f"  if ({fn.name()}(arg, res, iw, cw, 0) != 0) {{")
  for i in range(fn.n_out()):
    lines.append(f"    for (int k = 0; k < {fn.nnz_out(i)}; ++k) out{i}[k] = NAN;")
  lines += ["  }", "}"]
  return "\n".join(lines)


def build_casadi_external_sqp(
  *,
  name: str,
  base,
  grad,
  jac,
  hess,
  n_eq: int,
  n_ineq: int,
  x_lb: np.ndarray,
  x_ub: np.ndarray,
  l_ineq: np.ndarray,
  u_ineq: np.ndarray,
  options: dict[str, str | int | float] | None = None,
):
  """Embed CasADi-codegenerated oracles behind the same SQP descriptor as Alloy."""
  import casadi as ca

  n, np_ = int(base.size1_in(0)), int(base.size1_in(1))
  total_constraints = n_eq + n_ineq

  def shape(fn, *, inputs: tuple[tuple[int, int], ...], outputs: tuple[tuple[int, int], ...]) -> None:
    actual_inputs = tuple((int(fn.size1_in(i)), int(fn.size2_in(i))) for i in range(fn.n_in()))
    actual_outputs = tuple((int(fn.size1_out(i)), int(fn.size2_out(i))) for i in range(fn.n_out()))
    if actual_inputs != inputs or actual_outputs != outputs:
      raise ValueError(f"CasADi oracle {fn.name()!r} has {actual_inputs}->{actual_outputs}, expected {inputs}->{outputs}")

  base_outputs = ((1, 1), *(((total_constraints, 1),) if total_constraints else ()))
  shape(base, inputs=((n, 1), (np_, 1)), outputs=base_outputs)
  shape(grad, inputs=((n, 1), (np_, 1)), outputs=((n, 1),))
  if total_constraints:
    if jac is None:
      raise ValueError("a CasADi Jacobian oracle is required when constraints are present")
    shape(jac, inputs=((n, 1), (np_, 1)), outputs=((total_constraints, n),))
  elif jac is not None:
    shape(jac, inputs=((n, 1), (np_, 1)), outputs=((0, n),))
  hess_inputs = ((n, 1), (1, 1), *(((total_constraints, 1),) if total_constraints else ()), (np_, 1))
  shape(hess, inputs=hess_inputs, outputs=((n, n),))
  x_lb, x_ub = np.asarray(x_lb, dtype=np.float64), np.asarray(x_ub, dtype=np.float64)
  l_ineq, u_ineq = np.asarray(l_ineq, dtype=np.float64), np.asarray(u_ineq, dtype=np.float64)
  for label, value, expected in (
    ("x_lb", x_lb, (n,)),
    ("x_ub", x_ub, (n,)),
    ("l_ineq", l_ineq, (n_ineq,)),
    ("u_ineq", u_ineq, (n_ineq,)),
  ):
    if value.shape != expected:
      raise ValueError(f"{label} has shape {value.shape}, expected {expected}")
  p = ca.MX.sym(f"{name}_bounds_p", np_)
  bound_outputs = [ca.DM(x_lb), ca.DM(x_ub)]
  if n_ineq:
    bound_outputs += [ca.DM(l_ineq), ca.DM(u_ineq)]
  bounds = ca.Function(f"{name}_bounds", [p], bound_outputs)
  functions = {"base": base, "grad": grad, "hess": hess, "bounds": bounds}
  if jac is not None:
    functions["jac"] = jac
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    cwd = Path.cwd()
    os.chdir(root)
    try:
      generator = ca.CodeGenerator(f"{name}_oracles.c", {"with_header": False, "casadi_int": "int"})
      for fn in functions.values():
        generator.add(fn)
      generator.generate()
    finally:
      os.chdir(cwd)
    generated = (root / f"{name}_oracles.c").read_text()
  raw_symbols = {key: f"{name}_{key}_raw_external" for key in functions}
  source = generated + "\n\n" + "\n\n".join(_adapter(fn, raw_symbols[key]) for key, fn in functions.items())
  jac_rows, jac_cols = jac.sparsity_out(0).get_triplet() if jac is not None else ([], [])
  hess_rows, hess_cols = hess.sparsity_out(0).get_triplet()
  return external_nlp(
    name=name,
    n=n,
    n_eq=n_eq,
    n_ineq=n_ineq,
    params=(("p", (np_,)),),
    source=source,
    raw_symbols=raw_symbols,
    jac_sparsity=SparsityType((n_eq + n_ineq, n), tuple(int(v) for v in jac_rows), tuple(int(v) for v in jac_cols)),
    hess_sparsity=SparsityType((n, n), tuple(int(v) for v in hess_rows), tuple(int(v) for v in hess_cols)),
    options=options,
  )
