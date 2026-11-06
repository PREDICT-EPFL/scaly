#include <math.h>
#include <fatrop/ocp/OCPCInterface.h>

static int nx(int k, void *data) { return 1; }
static int nu(int k, void *data) { return k == 0; }
static int ng(int k, void *data) { return k == 0; }
static int no_ineq(int k, void *data) { return 0; }
static int horizon(void *data) { return 2; }

/* min u0^2 + (x1 - 3)^2, with x0 = 1 and x1 = x0 + u0. */
static int hess(double scale, const double *x, const double *lam, const double *p,
                const double *gp, struct blasfeo_dmat *out,
                const struct FatropOcpCDims *dims, void *data) {
  BLASFEO_DMATEL(&out[0], 0, 0) = 2 * scale;
  BLASFEO_DMATEL(&out[0], 0, 1) = 0;
  BLASFEO_DMATEL(&out[0], 1, 0) = 0;
  BLASFEO_DMATEL(&out[0], 1, 1) = 0;
  BLASFEO_DMATEL(&out[0], 2, 0) = 2 * scale * x[0];
  BLASFEO_DMATEL(&out[0], 2, 1) = 0;
  BLASFEO_DMATEL(&out[1], 0, 0) = 2 * scale;
  BLASFEO_DMATEL(&out[1], 1, 0) = 2 * scale * (x[2] - 3);
  return 1;
}

static int jac(const double *x, const double *p, const double *gp,
               struct blasfeo_dmat *dyn, struct blasfeo_dmat *eq,
               struct blasfeo_dmat *ineq, const struct FatropOcpCDims *dims, void *data) {
  BLASFEO_DMATEL(&dyn[0], 0, 0) = 1;
  BLASFEO_DMATEL(&dyn[0], 1, 0) = 1;
  BLASFEO_DMATEL(&dyn[0], 2, 0) = x[0] + x[1] - x[2];
  BLASFEO_DMATEL(&eq[0], 0, 0) = 0;
  BLASFEO_DMATEL(&eq[0], 1, 0) = 1;
  BLASFEO_DMATEL(&eq[0], 2, 0) = x[1] - 1;
  return 1;
}

static int violation(const double *x, const double *p, const double *gp, double *out,
                     const struct FatropOcpCDims *dims, void *data) {
  out[dims->dyn_eq_offs[0]] = x[0] + x[1] - x[2];
  out[dims->g_offs[0]] = x[1] - 1;
  return 1;
}

static int grad(double scale, const double *x, const double *p, const double *gp,
                double *out, const struct FatropOcpCDims *dims, void *data) {
  out[0] = 2 * scale * x[0];
  out[1] = 0;
  out[2] = 2 * scale * (x[2] - 3);
  return 1;
}

static int objective(double scale, const double *x, const double *p, const double *gp,
                     double *out, const struct FatropOcpCDims *dims, void *data) {
  *out = scale * (x[0] * x[0] + (x[2] - 3) * (x[2] - 3));
  return 1;
}

int run_solve(void) {
  struct FatropOcpCInterface problem = {0};
  problem.get_nx = nx;
  problem.get_nu = nu;
  problem.get_ng = ng;
  problem.get_ng_ineq = no_ineq;
  problem.get_horizon_length = horizon;
  problem.full_eval_lag_hess = hess;
  problem.full_eval_constr_jac = jac;
  problem.full_eval_contr_viol = violation;
  problem.full_eval_obj_grad = grad;
  problem.full_eval_obj = objective;
  struct FatropOcpCSolver *solver = fatrop_ocp_c_create(&problem, 0, 0);
  if (!solver) return 1;
  fatrop_ocp_c_set_option_int(solver, "print_level", 0);
  int status = fatrop_ocp_c_solve(solver);
  const struct blasfeo_dvec *solution = fatrop_ocp_c_get_primal(solver);
  const struct FatropOcpCStats *stats = fatrop_ocp_c_get_stats(solver);
  int failed = status != 0 || stats->return_flag != 0 || stats->iterations_count <= 0;
  const double expected[] = {1, 1, 2};
  for (int i = 0; i < 3; ++i)
    failed |= !isfinite(solution->pa[i]) || fabs(solution->pa[i] - expected[i]) > 1e-6;
  fatrop_ocp_c_destroy(solver);
  return failed;
}

#ifndef PROBE_LIBRARY
int main(void) { return run_solve(); }
#endif
