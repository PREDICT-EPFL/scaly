"""Write mapped dense derivatives directly into Fatrop's stage blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from scaly.codegen.solver import SolverWrapperCtx
  from scaly.solvers.model import SolverDescriptor


def render_callbacks(desc: SolverDescriptor, ctx: SolverWrapperCtx, call: Callable) -> list[str]:
  """Render Jacobian and Hessian callbacks sharing the whole-problem wrapper's bookkeeping."""
  stage = desc.stage_oracles
  assert stage is not None
  layout = desc.stage_layout
  N, nx, nu = layout.N, layout.nx[0], layout.nu[0]
  nz, ni, np_ = nx + nu, len(layout.eq[0]), len(layout.ineq[0])
  m, s = desc.n_eq + desc.n_ineq, ctx.symbol
  params = [f"d->p{i}" for i in range(len(desc.param_names))]
  jac_call = call(stage.jac, ["d->x", *params], ["d->stage_dyn", "d->stage_initial", "d->stage_path", "d->stage_terminal"], "n_jac")
  hess_call = call(stage.hess, ["d->x", *params, "&scale", "d->lam"], ["d->stage_h", "d->terminal_h"], "n_hess")
  return f"""
  static int {s}_jacobian(const double* x, const double* p, const double* gp,
                         struct blasfeo_dmat* dyn, struct blasfeo_dmat* eq, struct blasfeo_dmat* ineq,
                         const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_base(d);
    {jac_call}
    for (int k = 0; k <= {N}; ++k) {{
      int start = {s}_start[k], width = {s}_nx[k]+{s}_nu[k];
      for (int j = 0; j < d->ng[k]; ++j) for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&eq[k], i, j) = 0.0;
      for (int j = 0; j < d->ngi[k]; ++j) for (int i = 0; i <= width; ++i) BLASFEO_DMATEL(&ineq[k], i, j) = 0.0;
      if (k < {N}) for (int j = 0; j < {nx}; ++j) {{
        for (int i = 0; i < width; ++i)
          BLASFEO_DMATEL(&dyn[k], {s}_vlocal[start+i], j) = d->stage_dyn[(k*{nx}+j)*{nz}+i];
        BLASFEO_DMATEL(&dyn[k], width, j) = d->g[{ni}+k*{nx}+j];
      }}
      for (int j = {s}_order_start[k]; j < {s}_order_start[k+1]; ++j) {{
        int row = {s}_order[j], kind = d->kind[row], column = d->column[row];
        if (kind < 0) continue;
        struct blasfeo_dmat* block = kind == 1 ? &eq[k] : &ineq[k];
        if (row >= {m}) BLASFEO_DMATEL(block, {s}_vlocal[row-{m}], column) = 1.0;
        else {{
          int cols = row < {ni} || k == {N} ? {nx} : {nz};
          const double* values = row < {ni} ? d->stage_initial+row*{nx} :
            k == {N} ? d->stage_terminal+(row-{desc.n_eq + N * np_})*{nx} : d->stage_path+(row-{desc.n_eq})*{nz};
          for (int i = 0; i < cols; ++i) BLASFEO_DMATEL(block, {s}_vlocal[start+i], column) = values[i];
        }}
        BLASFEO_DMATEL(block, width, column) = {s}_value(d, row)-(kind == 1 ? d->lower[row] : 0.0);
      }}
    }}
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
  static int {s}_hessian(double scale, const double* x, const double* lam, const double* p, const double* gp,
                        struct blasfeo_dmat* out, const struct FatropOcpCDims* dims, void* ud) {{
    {s}_data* d = ({s}_data*)ud; double begin = scaly_clock_s();
    {s}_read(d, x); {s}_multipliers(d, lam);
    {hess_call}
    for (int k = 0; k <= {N}; ++k) {{
      int start = {s}_start[k], width = {s}_nx[k]+{s}_nu[k];
      const double* block = k == {N} ? d->terminal_h : d->stage_h+k*({nz}+1)*{nz};
      for (int i = 0; i < width; ++i) {{
        for (int j = 0; j < width; ++j)
          BLASFEO_DMATEL(&out[k], {s}_vlocal[start+i], {s}_vlocal[start+j]) = block[i*width+j];
        BLASFEO_DMATEL(&out[k], width, {s}_vlocal[start+i]) = block[width*width+i]+d->lam[{m}+start+i];
      }}
    }}
    d->t_callback += scaly_clock_s()-begin; return 1;
  }}
""".strip().splitlines()
