"""Probe generic cost generation and run native Scaly costs through acados' custom-update hook."""

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


def main():
  work = Path(tempfile.mkdtemp(prefix="scaly-acados-cost-"))
  os.chdir(work)
  modules = []
  for stage, factor, nu in (("initial", 0.7, 1), ("path", 1.3, 1), ("terminal", 2.1, 0)):
    ux, p = sc.sym("ux", 2 + nu), sc.sym("p", 1)
    x = ux[nu:]
    value = 0.5 * factor * ((x[0] - p[0]) ** 2 + 2.0 * x[1] ** 2 + (ux[0] ** 2 if nu else 0.0))
    grad, hess = sc.gradient(value, ux), sc.sparse_hessian(value, ux)
    fun = ConcreteFunction.build(
      f"scaly_cost_{stage}",
      sc.group(sc.arg("ux", ux.type), sc.arg("p", p.type)),
      (ux, p),
      sc.group(sc.arg("value", value.type), sc.arg("grad", grad.type), sc.arg("hess", hess.values.type)),
      (value, grad, hess.values),
      output_sparsities=(None, None, hess.sparsity),
    )
    modules.append(render_c_module(fun, casadi=True))
  source = "\n".join(module.body for module in modules)
  source += f"\n#define SCALY_WORKSPACE_SIZE {max(1, *(module.workspace_size for module in modules))}\n"
  source += r"""
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>
#include "acados_solver_scaly_cost.h"
#include "acados_c/external_function_interface.h"
#include "blasfeo_d_aux.h"

static int calls[3][3];
int scaly_cost_calls(int stage, int order) { return calls[stage][order]; }

static int evaluate(void **in, void **out, void *params, int stage, int order,
                    int (*fun)(const double **, double **, int *, double *, int), const int *sp)
{
    int nu = stage == 2 ? 0 : 1, n = nu + 2;
    struct blasfeo_dvec_args *x = in[0], *u = in[1];
    double ux[3], value, grad[3], hess[3], w[SCALY_WORKSPACE_SIZE];
    if (nu) blasfeo_unpack_dvec(nu, u->x, u->xi, ux, 1);
    blasfeo_unpack_dvec(2, x->x, x->xi, ux + nu, 1);
    const double *arg[] = {ux, params};
    double *res[] = {&value, grad, hess};
    int status = fun(arg, res, NULL, w, 0);
    assert(status == 0);
    *(double *)out[0] = value;
    if (order >= 1) blasfeo_pack_dvec(n, grad, 1, out[1], 0);
    if (order == 2) {
        assert(sp[2] == 0);
        const int *col = sp + 2, *row = sp + 3 + n;
        struct blasfeo_dmat *dst = out[2];
        blasfeo_dgese(n, n, 0.0, dst, 0, 0);
        for (int j = 0; j < n; ++j)
            for (int k = col[j]; k < col[j+1]; ++k)
                blasfeo_dgein1(hess[k], dst, row[k], j);
    }
    ++calls[stage][order];
    return status;
}
"""
  for stage, name in enumerate(("initial", "path", "terminal")):
    for order, suffix in enumerate(("value", "jac", "hess")):
      source += f"""
static int {name}_{suffix}(void **in, void **out, void *p)
{{ return evaluate(in, out, p, {stage}, {order}, scaly_cost_{name}, scaly_cost_{name}_sparsity_out(2)); }}
"""
  source += r"""
typedef struct { external_function_external_param_generic functions[5][3]; } cost_memory;

int custom_update_init_function(scaly_cost_solver_capsule *capsule)
{
    cost_memory *mem = calloc(1, sizeof(*mem));
    assert(mem);
    capsule->custom_update_memory = mem;
    int (*functions[3][3])(void **, void **, void *) = {
        {initial_value, initial_jac, initial_hess},
        {path_value, path_jac, path_hess},
        {terminal_value, terminal_jac, terminal_hess}
    };
    const char *fields[] = {"ext_cost_fun", "ext_cost_fun_jac", "ext_cost_fun_jac_hess"};
    external_function_opts opts;
    external_function_opts_set_to_default(&opts);
    for (int i = 0; i <= 4; ++i) {
        int stage = i == 0 ? 0 : (i == 4 ? 2 : 1);
        for (int k = 0; k < 3; ++k) {
            external_function_external_param_generic *f = &mem->functions[i][k];
            f->fun = functions[stage][k];
            external_function_external_param_generic_create(f, &opts);
            int status = ocp_nlp_cost_model_set_external_param_fun(
                capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, i, fields[k], f);
            assert(status == 0);
        }
    }
    return 0;
}

int custom_update_function(scaly_cost_solver_capsule *capsule, double *data, int data_len)
{
    int numerical_hess = 1;
    for (int i = 0; i <= 4; ++i)
        ocp_nlp_solver_opts_set_at_stage(capsule->nlp_config, capsule->nlp_opts, i,
                                       "cost_numerical_hessian", &numerical_hess);
    return 0;
}

int custom_update_terminate_function(scaly_cost_solver_capsule *capsule)
{
    cost_memory *mem = capsule->custom_update_memory;
    for (int i = 0; i <= 4; ++i)
        for (int k = 0; k < 3; ++k)
            external_function_external_param_generic_free(&mem->functions[i][k]);
    free(mem);
    return 0;
}
"""
  Path("cost_update.c").write_text(source)
  Path("cost_update.h").write_text("""#include "acados_solver_scaly_cost.h"
int custom_update_init_function(scaly_cost_solver_capsule *capsule);
int custom_update_function(scaly_cost_solver_capsule *capsule, double *data, int data_len);
int custom_update_terminate_function(scaly_cost_solver_capsule *capsule);
""")

  def problem(name, native):
    ocp = AcadosOcp()
    ocp.name = name
    model = ocp.model
    model.name = name
    model.x, model.u, model.p = ca.SX.sym("x", 2), ca.SX.sym("u", 1), ca.SX.sym("p", 1)
    x, u, p = model.x, model.u, model.p
    model.disc_dyn_expr = ca.vertcat(x[0] + 0.2 * x[1] + 0.1 * u[0] + p[0], 0.9 * x[1] + 0.3 * u[0])
    ocp.cost.cost_type_0 = ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    base = 0.5 * ((x[0] - p[0]) ** 2 + 2.0 * x[1] ** 2)
    model.cost_expr_ext_cost_0 = 0 if native else 0.7 * (base + 0.5 * u[0] ** 2)
    model.cost_expr_ext_cost = 0 if native else 1.3 * (base + 0.5 * u[0] ** 2)
    model.cost_expr_ext_cost_e = 0 if native else 2.1 * base
    ocp.parameter_values = np.array([0.03])
    ocp.constraints.x0 = np.array([1.0, -0.2])
    opts = ocp.solver_options
    opts.N_horizon, opts.tf = 4, 1.0
    opts.integrator_type, opts.nlp_solver_type = "DISCRETE", "SQP"
    opts.hessian_approx, opts.globalization = "EXACT", "MERIT_BACKTRACKING"
    opts.store_iterates = True
    opts.nlp_solver_tol_stat = opts.nlp_solver_tol_eq = opts.nlp_solver_tol_ineq = opts.nlp_solver_tol_comp = 1e-9
    if native:
      opts.custom_update_filename = "cost_update.c"
      opts.custom_update_header_filename = "cost_update.h"
    ocp.code_gen_options.code_export_directory = str(work / name)
    ocp.code_gen_options.json_file = f"{name}.json"
    ocp.code_gen_options.ext_fun_compile_flags = "-O2 -Dcasadi_int=int"
    return ocp

  probe = problem("generic_cost_probe", False)
  for suffix in ("_0", "", "_e"):
    setattr(probe.cost, f"cost_ext_fun_type{suffix}", "generic")
    setattr(probe.model, f"cost_expr_ext_cost{suffix}", [])
  try:
    probe.make_consistent()
  except TypeError as exc:
    assert str(exc) == "cost_expr_ext_cost_0 should be casadi expression.", str(exc)
    print(f"Direct Python generic cost blocked: {exc}")
  else:
    raise AssertionError("The pinned generic cost expression check changed")

  solvers = []
  for name, native in (("scaly_cost", True), ("casadi_cost_reference", False)):
    solver = AcadosOcpSolver(problem(name, native))
    status = solver.solve()
    assert status == 0, (name, status)
    solvers.append(solver)
  actual, reference = solvers
  for field, stages in (("x", 5), ("u", 4), ("pi", 4)):
    for stage in range(stages):
      np.testing.assert_allclose(actual.get(stage, field), reference.get(stage, field), atol=1e-8, rtol=1e-8)
  np.testing.assert_allclose(actual.get_cost(), reference.get_cost(), atol=1e-9, rtol=1e-9)
  np.testing.assert_allclose(actual.get_residuals(recompute=True), reference.get_residuals(recompute=True), atol=1e-8)
  assert actual.get_stats("sqp_iter") == reference.get_stats("sqp_iter")
  for iteration in range(actual.get_stats("sqp_iter") + 1):
    np.testing.assert_allclose(actual.get_iterate(iteration).x, reference.get_iterate(iteration).x, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual.get_iterate(iteration).u, reference.get_iterate(iteration).u, atol=1e-8, rtol=1e-8)
  sqp_iter = int(actual.get_stats("sqp_iter"))
  for stage in range(5):
    factor = 0.7 if stage == 0 else (2.1 if stage == 4 else 1.3)
    actual.cost_set(stage, "ext_cost_num_hess", factor * np.diag([1.0, 2.0] if stage == 4 else [1.0, 1.0, 2.0]))
  actual.custom_update(np.empty(0))
  assert actual.solve() == 0
  assert np.max(actual.get_residuals()) < 1e-9
  lib = actual.shared_lib
  lib.scaly_cost_calls.argtypes = [ctypes.c_int, ctypes.c_int]
  counts = [[lib.scaly_cost_calls(stage, order) for order in range(3)] for stage in range(3)]
  assert all(count > 0 for stage in counts for count in stage), counts
  print(json.dumps({"status": 0, "sqp_iter": sqp_iter, "cost": actual.get_cost(), "calls": counts, "work": str(work)}))


if __name__ == "__main__":
  main()
