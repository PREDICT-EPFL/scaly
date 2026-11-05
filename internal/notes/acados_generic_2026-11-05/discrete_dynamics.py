"""Run Scaly discrete dynamics through acados' native generic interface and compare with CasADi."""

import ctypes
import json
import os
from pathlib import Path
import tempfile

import casadi as ca
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

import scaly as sc
from scaly.codegen import render_c_module
from scaly.function.concrete import ConcreteFunction
from scaly.ir.types import SparsityPattern


def main():
  work = Path(tempfile.mkdtemp(prefix="scaly-acados-discrete-"))
  os.chdir(work)
  ux, pi, p = sc.sym("ux", 3), sc.sym("pi", 2), sc.sym("p", 1)
  u, x = ux[:1], ux[1:]
  phi = sc.stack([x[0] + 0.2 * x[1] + 0.05 * u[0] ** 2 + p[0], 0.9 * x[1] + 0.3 * u[0]])
  jac = sc.sparse_jacobian(phi, ux)
  jt_sp = SparsityPattern((3, 2), jac.sparsity.cols, jac.sparsity.rows)
  hess = sc.sparse_hessian((phi * pi).sum(), ux)
  fun = ConcreteFunction.build(
    "scaly_disc",
    sc.group(sc.arg("ux", 3), sc.arg("pi", 2), sc.arg("p", 1)),
    (ux, pi, p),
    sc.group(sc.arg("phi", phi.type), sc.arg("jt", jac.values.type), sc.arg("hess", hess.values.type)),
    (phi, jac.values, hess.values),
    output_sparsities=(None, jt_sp, hess.sparsity),
  )
  module = render_c_module(fun, casadi=True)
  assert jt_sp.nnz == 5 and hess.sparsity.nnz == 1
  source = (
    module.body
    + f"\n#define SCALY_WORKSPACE_SIZE {max(module.workspace_size, 1)}\n"
    + r"""
#undef NDEBUG
#include <assert.h>
#include "acados/utils/external_function_generic.h"
#include "blasfeo_d_aux.h"

static int calls[3];
int scaly_disc_calls(int order) { return calls[order]; }

static void pack_sparse(const double *v, const int *sp, struct blasfeo_dmat_args *dst)
{
    assert(sp[2] == 0);
    int nr = sp[0], nc = sp[1];
    const int *col = sp + 2, *row = sp + 3 + nc;
    blasfeo_dgese(nr, nc, 0.0, dst->A, dst->ai, dst->aj);
    for (int j = 0; j < nc; ++j)
        for (int k = col[j]; k < col[j+1]; ++k)
            blasfeo_dgein1(v[k], dst->A, dst->ai + row[k], dst->aj + j);
}

static int evaluate(void **in, void **out, void *params, int order)
{
    struct blasfeo_dvec_args *x = in[0], *u = in[1], *phi = out[0];
    double ux[3], pi[2] = {0.0, 0.0}, value[2], jt[5], hess[1];
    double w[SCALY_WORKSPACE_SIZE];
    blasfeo_unpack_dvec(1, u->x, u->xi, ux, 1);
    blasfeo_unpack_dvec(2, x->x, x->xi, ux + 1, 1);
    if (order == 2) {
        struct blasfeo_dvec_args *weight = in[2];
        blasfeo_unpack_dvec(2, weight->x, weight->xi, pi, 1);
    }
    const double *arg[] = {ux, pi, params};
    double *res[] = {value, jt, hess};
    int status = scaly_disc(arg, res, NULL, w, 0);
    assert(status == 0);
    blasfeo_pack_dvec(2, value, 1, phi->x, phi->xi);
    if (order >= 1) pack_sparse(jt, scaly_disc_sparsity_out(1), out[1]);
    if (order == 2) pack_sparse(hess, scaly_disc_sparsity_out(2), out[2]);
    ++calls[order];
    return status;
}

int generic_disc_fun(void **in, void **out, void *params) { return evaluate(in, out, params, 0); }
int generic_disc_fun_jac(void **in, void **out, void *params) { return evaluate(in, out, params, 1); }
int generic_disc_fun_jac_hess(void **in, void **out, void *params) { return evaluate(in, out, params, 2); }
"""
  )
  Path("generic_disc.c").write_text(source)

  def problem(name, generic):
    ocp = AcadosOcp()
    model = ocp.model
    model.name = name
    model.x, model.u, model.p = ca.SX.sym("x", 2), ca.SX.sym("u", 1), ca.SX.sym("p", 1)
    x, u, p = model.x, model.u, model.p
    model.disc_dyn_expr = ca.vertcat(x[0] + 0.2 * x[1] + 0.05 * u[0] ** 2 + p[0], 0.9 * x[1] + 0.3 * u[0])
    if generic:
      model.dyn_ext_fun_type = "generic"
      model.dyn_generic_source = "generic_disc.c"
      model.dyn_disc_fun = "generic_disc_fun"
      model.dyn_disc_fun_jac = "generic_disc_fun_jac"
      model.dyn_disc_fun_jac_hess = "generic_disc_fun_jac_hess"
    ocp.parameter_values = np.array([0.03])
    ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W, ocp.cost.W_e = np.diag([1.0, 2.0, 0.2]), np.diag([3.0, 4.0])
    ocp.cost.Vx = np.vstack([np.eye(2), np.zeros((1, 2))])
    ocp.cost.Vu = np.array([[0.0], [0.0], [1.0]])
    ocp.cost.Vx_e = np.eye(2)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(3), np.zeros(2)
    ocp.constraints.x0 = np.array([1.0, -0.2])
    opts = ocp.solver_options
    opts.N_horizon, opts.tf = 4, 1.0
    opts.integrator_type, opts.nlp_solver_type = "DISCRETE", "SQP"
    opts.hessian_approx, opts.globalization = "EXACT", "MERIT_BACKTRACKING"
    opts.nlp_solver_max_iter = 30
    opts.nlp_solver_tol_stat = opts.nlp_solver_tol_eq = opts.nlp_solver_tol_ineq = opts.nlp_solver_tol_comp = 1e-9
    opts.store_iterates = True
    ocp.code_gen_options.code_export_directory = str(work / name)
    ocp.code_gen_options.json_file = f"{name}.json"
    ocp.code_gen_options.ext_fun_compile_flags = "-O2 -Dcasadi_int=int"
    return ocp

  solvers = []
  for name, generic in (("scaly_generic", True), ("casadi_reference", False)):
    solver = AcadosOcpSolver(problem(name, generic))
    status = solver.solve()
    assert status == 0, (name, status)
    solvers.append(solver)

  actual, reference = solvers
  for field, stages in (("x", 5), ("u", 4), ("pi", 4)):
    for stage in range(stages):
      np.testing.assert_allclose(actual.get(stage, field), reference.get(stage, field), atol=1e-8, rtol=1e-8)
  np.testing.assert_allclose(actual.get_cost(), reference.get_cost(), atol=1e-9, rtol=1e-9)
  assert actual.get_stats("sqp_iter") == reference.get_stats("sqp_iter")
  for iteration in range(actual.get_stats("sqp_iter") + 1):
    np.testing.assert_allclose(actual.get_iterate(iteration).x, reference.get_iterate(iteration).x, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual.get_iterate(iteration).u, reference.get_iterate(iteration).u, atol=1e-8, rtol=1e-8)
  lib = actual.shared_lib
  sqp_iter = int(actual.get_stats("sqp_iter"))
  set_at_stage = lib.ocp_nlp_solver_opts_set_at_stage
  set_at_stage.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p]
  set_at_stage.restype = None
  no_hess = ctypes.c_int(0)
  for stage in range(4):
    set_at_stage(actual.nlp_config, actual.nlp_opts, stage, b"dynamics_compute_hess", ctypes.byref(no_hess))
  assert actual.solve() == 0
  assert np.max(actual.get_residuals()) < 1e-9
  lib.scaly_disc_calls.argtypes = [ctypes.c_int]
  counts = [lib.scaly_disc_calls(k) for k in range(3)]
  assert all(count > 0 for count in counts), counts
  print(
    json.dumps(
      {
        "status": 0,
        "sqp_iter": sqp_iter,
        "cost": actual.get_cost(),
        "calls": counts,
        "jac_shape": jt_sp.shape,
        "jac_nnz": jt_sp.nnz,
        "hess_nnz": hess.sparsity.nnz,
        "work": str(work),
      }
    )
  )


if __name__ == "__main__":
  main()
