"""C codegen for ``Ops.SOLVER_CALL`` — emits a raw function that drives the
vendored PIQP / IPOPT C interfaces.

The :class:`SolverFunction` renderer bypasses the standard tape-based scalar
emitter: a solver Function's body is shape-specific enough that the cleanest
implementation is a small hand-written template per backend, parameterised by
the descriptor.

Outer functions that contain a solver as a callee still go through the
standard ``Ops.CALL`` machinery — the solver renders to a `static void
qp_xxx_raw(...)` body, and the caller's tape just emits a `qp_xxx_raw(...)`
invocation. The solver Function is included automatically via
``solver_callees(...)``, and the JIT detects the PIQP/IPOPT linkage need
through ``uses_piqp(...)`` / ``uses_ipopt(...)``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from alloy.function import Function
from alloy.ops import Ops

if TYPE_CHECKING:
  from alloy.solvers.solver_function import SolverDescriptor


def _c_ident(name: str) -> str:
  ident = re.sub(r"\W", "_", name)
  return f"_{ident}" if ident[:1].isdigit() else ident


def _raw_symbol(fun: Function) -> str:
  return f"{_c_ident(fun.name)}_raw"


def is_solver_function(fun: Function) -> bool:
  return hasattr(fun, "descriptor") and getattr(getattr(fun, "descriptor", None), "backend", None) in {"piqp", "ipopt"}


def _descriptor(fun: Function) -> SolverDescriptor:
  return fun.descriptor  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Predicates / dependencies the outer renderer consumes.
# ---------------------------------------------------------------------------


def solver_callees(fun: Function) -> list[Function]:
  """Return the inner Functions a solver Function depends on at codegen time."""
  if not is_solver_function(fun):
    return []
  desc = _descriptor(fun)
  out: list[Function] = []
  for cand in (desc.oracle, desc.base, desc.grad, desc.jac, desc.hess, desc.bounds):
    if cand is not None and cand not in out:
      out.append(cand)
  return out


def _walk_tape_callees(fun: Function, seen: set[int]) -> bool:
  """True iff ``fun`` (or any transitive callee) is or contains a SolverFunction."""
  if id(fun) in seen:
    return False
  seen.add(id(fun))
  if is_solver_function(fun):
    return True
  for inst in fun.tape():
    if inst.op in {Ops.CALL, Ops.MAP}:
      if _walk_tape_callees(inst.attrs["callee"], seen):
        return True
    if inst.op == Ops.SOLVER_CALL:
      # SolverFunction-internal SOLVER_CALL nodes — the SolverFunction itself
      # is handled at the is_solver_function check above; this branch is for
      # safety in case a SOLVER_CALL appears outside a SolverFunction.
      return True
  return False


def uses_any_solver(fun: Function) -> bool:
  return _walk_tape_callees(fun, set())


def uses_piqp(fun: Function) -> bool:
  return _uses_backend(fun, "piqp", set())


def uses_ipopt(fun: Function) -> bool:
  return _uses_backend(fun, "ipopt", set())


def _uses_backend(fun: Function, backend: str, seen: set[int]) -> bool:
  if id(fun) in seen:
    return False
  seen.add(id(fun))
  if is_solver_function(fun) and _descriptor(fun).backend == backend:
    return True
  for inst in fun.tape():
    if inst.op in {Ops.CALL, Ops.MAP}:
      if _uses_backend(inst.attrs["callee"], backend, seen):
        return True
    if inst.op == Ops.SOLVER_CALL:
      if inst.attrs["solver"].backend == backend:
        return True
  return False


def solver_includes(fun: Function) -> list[str]:
  out: list[str] = []
  if uses_piqp(fun):
    out.append('#include "piqp/piqp.h"')
  if uses_ipopt(fun):
    out.append('#include "coin-or/IpStdCInterface.h"')
  return out


def solver_workspace(fun: Function) -> int:
  """Doubles required in ``w[]`` to drive this solver Function.

  We stack-allocate the QP data inside the raw body, so ``w[]`` only carries
  whatever the oracle needs. Pre-existing helpers handle CALL/MAP workspace
  recursion; this is just the solver-specific add-on.
  """
  if not is_solver_function(fun):
    return 0
  desc = _descriptor(fun)
  # Caller is the outer Function: it has already accounted for the oracle's
  # own workspace via the regular CALL accounting. The solver wrapper itself
  # adds no extra workspace today (QP data on stack).
  _ = desc
  return 0


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_solver_raw(fun: Function) -> list[str]:
  desc = _descriptor(fun)
  if desc.backend == "piqp":
    return _render_piqp_raw(fun, desc)
  raise NotImplementedError(f"C codegen for solver backend {desc.backend!r} not yet implemented")


def _render_piqp_raw(fun: Function, desc: SolverDescriptor) -> list[str]:
  symbol = _c_ident(fun.name)
  raw = _raw_symbol(fun)
  n, p, m = desc.n, desc.n_eq, desc.n_ineq
  oracle = desc.oracle
  assert oracle is not None, "PIQP solver descriptor must carry an oracle Function"

  # Map solver input order to C parameter names: in0..in{n-1}.
  # Inputs: x0, lam_eq0, lam_ineq0, then params in order.
  param_count = len(desc.param_names)
  oracle_param_args = [f"in{3 + i}" for i in range(param_count)]

  # Oracle output ordering (set by qp.py): P, c, [A_eq, b_eq], [G_ineq, l_ineq, u_ineq], x_lb, x_ub.
  oracle_outs: list[tuple[str, int]] = []  # (buffer C name, size)
  oracle_outs.append(("P_buf", n * n))
  oracle_outs.append(("c_buf", n))
  if p:
    oracle_outs.append(("A_buf", p * n))
    oracle_outs.append(("b_buf", p))
  if m:
    oracle_outs.append(("G_buf", m * n))
    oracle_outs.append(("lineq_buf", m))
    oracle_outs.append(("uineq_buf", m))
  oracle_outs.append(("xlb_buf", n))
  oracle_outs.append(("xub_buf", n))

  # Parameters for the raw signature.
  c_inputs = [f"const double* in{i}" for i in range(len(desc.input_signature))]
  c_outputs = [f"double* out{i}" for i in range(len(desc.output_signature))]
  params = [*c_inputs, *c_outputs, "double* w"]

  lines: list[str] = []
  lines.append(f"// PIQP dense solver wrapper for {fun.name} (n={n}, p={p}, m={m}).")
  lines.append(f"static void {raw}({', '.join(params)}) {{")
  # Suppress unused warnings (warm-start inputs are not consumed yet).
  lines.append("  (void)w;")
  lines.append("  (void)in0;")  # x0
  lines.append("  (void)in1;")  # lam_eq0
  lines.append("  (void)in2;")  # lam_ineq0

  # 1. Local QP data buffers (stack-allocated; row-major from the oracle).
  for buf, size in oracle_outs:
    lines.append(f"  double {buf}[{size}];")
  # Column-major staging buffers used for actual PIQP calls (transpose).
  lines.append(f"  double Pcol[{n * n}];")
  if p:
    lines.append(f"  double Acol[{p * n}];")
  if m:
    lines.append(f"  double Gcol[{m * n}];")

  # 2. Call the oracle. The oracle's input order is the param list; its
  # output order matches oracle_outs above.
  oracle_raw = _raw_symbol(oracle)
  oracle_call = ", ".join([*oracle_param_args, *(name for name, _ in oracle_outs), "w"])
  lines.append(f"  {oracle_raw}({oracle_call});")

  # 3. Row-major -> column-major transpose for P, A, G.
  lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {n}; ++i) Pcol[i + j * {n}] = P_buf[i * {n} + j];")
  if p:
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {p}; ++i) Acol[i + j * {p}] = A_buf[i * {n} + j];")
  if m:
    lines.append(f"  for (int j = 0; j < {n}; ++j) for (int i = 0; i < {m}; ++i) Gcol[i + j * {m}] = G_buf[i * {n} + j];")

  # 4. Static PIQP workspace, lazily set up on first call.
  lines.append(f"  static piqp_workspace* {symbol}_ws = NULL;")
  lines.append(f"  static piqp_settings {symbol}_settings;")
  lines.append(f"  if ({symbol}_ws == NULL) {{")
  lines.append(f"    piqp_set_default_settings_dense(&{symbol}_settings);")
  for key, val in desc.options:
    if isinstance(val, bool):
      lines.append(f"    {symbol}_settings.{key} = {1 if val else 0};")
    elif isinstance(val, (int, float)):
      lines.append(f"    {symbol}_settings.{key} = {val};")
    else:
      # String options are not part of PIQP's settings struct.
      raise NotImplementedError(f"PIQP option {key}={val!r} cannot be lowered to C")
  # Build setup data struct.
  lines.append("    piqp_data_dense setup_data;")
  lines.append(f"    setup_data.n = {n};")
  lines.append(f"    setup_data.p = {p};")
  lines.append(f"    setup_data.m = {m};")
  lines.append("    setup_data.P = Pcol;")
  lines.append("    setup_data.c = c_buf;")
  lines.append(f"    setup_data.A = {'Acol' if p else 'NULL'};")
  lines.append(f"    setup_data.b = {'b_buf' if p else 'NULL'};")
  lines.append(f"    setup_data.G = {'Gcol' if m else 'NULL'};")
  lines.append(f"    setup_data.h_l = {'lineq_buf' if m else 'NULL'};")
  lines.append(f"    setup_data.h_u = {'uineq_buf' if m else 'NULL'};")
  lines.append("    setup_data.x_l = xlb_buf;")
  lines.append("    setup_data.x_u = xub_buf;")
  lines.append(f"    piqp_setup_dense(&{symbol}_ws, &setup_data, &{symbol}_settings);")
  lines.append("  } else {")
  lines.append(
    f"    piqp_update_dense({symbol}_ws, Pcol, c_buf, "
    f"{'Acol' if p else 'NULL'}, {'b_buf' if p else 'NULL'}, "
    f"{'Gcol' if m else 'NULL'}, {'lineq_buf' if m else 'NULL'}, {'uineq_buf' if m else 'NULL'}, "
    "xlb_buf, xub_buf);"
  )
  lines.append("  }")

  # 5. Solve and read results.
  lines.append(f"  piqp_solve({symbol}_ws);")
  lines.append(f"  piqp_result* res = {symbol}_ws->result;")
  # Outputs: x, cost, lam_eq, lam_ineq, lam_box.
  lines.append(f"  for (int i = 0; i < {n}; ++i) out0[i] = res->x[i];")
  lines.append("  out1[0] = res->info.primal_obj;")
  if p:
    lines.append(f"  for (int i = 0; i < {p}; ++i) out2[i] = res->y[i];")
  if m:
    lines.append(f"  for (int i = 0; i < {m}; ++i) out3[i] = res->z_u[i] - res->z_l[i];")
  lines.append(f"  for (int i = 0; i < {n}; ++i) out4[i] = res->z_bu[i] - res->z_bl[i];")

  lines.append("}")
  return lines
