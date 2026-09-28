/* Fatrop's C interface filled from Scaly-generated oracles (see fatrop_dropin.py).
 *
 * The same callbacks CasADi's fatrop plugin installs (fatrop_runtime.hpp in CasADi 3.8): whole-horizon
 * evaluations that return 1 so Fatrop skips its per-stage calls. Scaly computes the objective, its
 * gradient, the dynamics residuals, the stage Jacobians [B A]' and the stage Lagrangian Hessians with
 * their gradient rows; this file only scatters them into Fatrop's BLASFEO blocks and fills the two
 * linear constraints (x_0 = x0 at stage 0, the box on u_k) exactly as CasADi's runtime does. */
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <blasfeo.h>

#include "fatrop/ocp/OCPCInterface.h"
#include "fatrop_scaly_config.h"

#define NZ (NX + NU)
#define KK (HORIZON + 1)
#define NW (HORIZON * NZ + NX)

typedef int scaly_fn(const double **arg, double **res, int *iw, double *w, int mem);
scaly_fn SCALY_OBJ, SCALY_GRAD, SCALY_CV, SCALY_JAC, SCALY_HESS;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* Time the oracle call itself (t_oracle) and the whole callback with its scatter (t_callback). */
#define TIMED_CALL(fn, ...) \
  do { \
    double t_ = now(); \
    fn(__VA_ARGS__); \
    d->t_oracle += now() - t_; \
    d->n_oracle++; \
  } while (0)

typedef struct {
  struct FatropOcpCInterface itf;
  struct FatropOcpCSolver *solver;
  double x0[NX], w_init[NW];
  double b[HORIZON * NX], jac[HORIZON * (NX * NZ + NX)], hess[HORIZON * (NZ * NZ + NZ)], lam_dyn[HORIZON * NX];
  double w_obj[SZ_W_OBJ], w_grad[SZ_W_GRAD], w_cv[SZ_W_CV], w_jac[SZ_W_JAC], w_hess[SZ_W_HESS];
  double wall, t_oracle, t_callback;
  int n_oracle;
} fs_data;

static fatrop_int get_nx(fatrop_int k, void *u) { return NX; }
static fatrop_int get_nu(fatrop_int k, void *u) { return k < HORIZON ? NU : 0; }
static fatrop_int get_ng(fatrop_int k, void *u) { return k == 0 ? NX : 0; }
static fatrop_int get_ng_ineq(fatrop_int k, void *u) { return k < HORIZON ? NU : 0; }
static fatrop_int get_n_stage_params(fatrop_int k, void *u) { return 0; }
static fatrop_int get_n_global_params(void *u) { return 0; }
static fatrop_int get_default_stage_params(double *p, fatrop_int k, void *u) { return 0; }
static fatrop_int get_default_global_params(double *p, void *u) { return 0; }
static fatrop_int get_horizon_length(void *u) { return KK; }

static fatrop_int get_bounds(double *lower, double *upper, fatrop_int k, void *u) {
  for (int i = 0; i < NU; ++i) lower[i] = -1.0, upper[i] = 1.0;
  return 0;
}

static fatrop_int get_initial_xk(double *xk, fatrop_int k, void *u) {
  fs_data *d = u;
  memcpy(xk, d->w_init + k * NZ + (k < HORIZON ? NU : 0), NX * sizeof(double));
  return 0;
}

static fatrop_int get_initial_uk(double *uk, fatrop_int k, void *u) {
  fs_data *d = u;
  if (k < HORIZON) memcpy(uk, d->w_init + k * NZ, NU * sizeof(double));
  return 0;
}

static fatrop_int full_eval_obj(double s, const double *primal, const double *sp, const double *gp, double *res, const struct FatropOcpCDims *dims, void *u) {
  fs_data *d = u;
  double t_cb = now();
  const double *arg[] = {primal, &s};
  double *out[] = {res};
  TIMED_CALL(SCALY_OBJ, arg, out, NULL, d->w_obj, 0);
  d->t_callback += now() - t_cb;
  return 1;
}

static fatrop_int full_eval_obj_grad(double s, const double *primal, const double *sp, const double *gp, double *res, const struct FatropOcpCDims *dims,
                                     void *u) {
  fs_data *d = u;
  double t_cb = now();
  const double *arg[] = {primal, &s};
  double *out[] = {res};
  TIMED_CALL(SCALY_GRAD, arg, out, NULL, d->w_grad, 0);
  d->t_callback += now() - t_cb;
  return 1;
}

static fatrop_int full_eval_contr_viol(const double *primal, const double *sp, const double *gp, double *res, const struct FatropOcpCDims *dims, void *u) {
  fs_data *d = u;
  double t_cb = now();
  const double *arg[] = {primal};
  double *out[] = {d->b};
  TIMED_CALL(SCALY_CV, arg, out, NULL, d->w_cv, 0);
  for (int k = 0; k < HORIZON; ++k) memcpy(res + dims->dyn_eq_offs[k], d->b + k * NX, NX * sizeof(double));
  for (int i = 0; i < NX; ++i) res[dims->g_offs[0] + i] = primal[NU + i] - d->x0[i];
  for (int k = 0; k < HORIZON; ++k)
    for (int i = 0; i < NU; ++i) res[dims->g_ineq_offs[k] + i] = primal[k * NZ + i];
  d->t_callback += now() - t_cb;
  return 1;
}

static fatrop_int full_eval_constr_jac(const double *primal, const double *sp, const double *gp, struct blasfeo_dmat *BAbt, struct blasfeo_dmat *Ggt,
                                       struct blasfeo_dmat *Ggt_ineq, const struct FatropOcpCDims *dims, void *u) {
  fs_data *d = u;
  double t_cb = now();
  const double *arg[] = {primal};
  double *out[] = {d->jac};
  TIMED_CALL(SCALY_JAC, arg, out, NULL, d->w_jac, 0);
  for (int k = 0; k < HORIZON; ++k) { /* per stage: the row-major (NX, NZ) Jacobian, which is column-major (NZ, NX), then b */
    const double *stage = d->jac + k * (NX * NZ + NX);
    blasfeo_pack_dmat(NZ, NX, (double *)stage, NZ, BAbt + k, 0, 0);
    blasfeo_pack_dmat(1, NX, (double *)stage + NX * NZ, 1, BAbt + k, NZ, 0);
  }
  blasfeo_dgese(NZ + 1, NX, 0.0, Ggt, 0, 0);
  for (int i = 0; i < NX; ++i) {
    blasfeo_dgein1(1.0, Ggt, NU + i, i);
    blasfeo_dgein1(primal[NU + i] - d->x0[i], Ggt, NZ, i);
  }
  for (int k = 0; k < HORIZON; ++k) {
    blasfeo_dgese(NZ + 1, NU, 0.0, Ggt_ineq + k, 0, 0);
    for (int i = 0; i < NU; ++i) {
      blasfeo_dgein1(1.0, Ggt_ineq + k, i, i);
      blasfeo_dgein1(primal[k * NZ + i], Ggt_ineq + k, NZ, i);
    }
  }
  d->t_callback += now() - t_cb;
  return 1;
}

static fatrop_int full_eval_lag_hess(double s, const double *primal, const double *lam, const double *sp, const double *gp, struct blasfeo_dmat *RSQrqt,
                                     const struct FatropOcpCDims *dims, void *u) {
  fs_data *d = u;
  double t_cb = now();
  for (int k = 0; k < HORIZON; ++k) memcpy(d->lam_dyn + k * NX, lam + dims->dyn_eq_offs[k], NX * sizeof(double));
  const double *arg[] = {primal, d->lam_dyn, &s};
  double *out[] = {d->hess};
  TIMED_CALL(SCALY_HESS, arg, out, NULL, d->w_hess, 0);
  for (int k = 0; k < HORIZON; ++k) { /* per stage: the Hessian, then the Lagrangian's gradient row */
    double *stage = d->hess + k * (NZ * NZ + NZ), *grad = stage + NZ * NZ;
    if (k == 0)
      for (int i = 0; i < NX; ++i) grad[NU + i] += lam[dims->g_offs[0] + i];
    for (int i = 0; i < NU; ++i) grad[i] += lam[dims->g_ineq_offs[k] + i];
    blasfeo_pack_dmat(NZ, NZ, stage, NZ, RSQrqt + k, 0, 0);
    blasfeo_pack_dmat(1, NZ, grad, 1, RSQrqt + k, NZ, 0);
  }
  blasfeo_dgese(NX + 1, NX, 0.0, RSQrqt + HORIZON, 0, 0);
  d->t_callback += now() - t_cb;
  return 1;
}

void *fs_create(void) {
  fs_data *d = calloc(1, sizeof(fs_data));
  struct FatropOcpCInterface *o = &d->itf;
  o->get_nx = get_nx, o->get_nu = get_nu, o->get_ng = get_ng, o->get_ng_ineq = get_ng_ineq;
  o->get_n_stage_params = get_n_stage_params, o->get_n_global_params = get_n_global_params;
  o->get_default_stage_params = get_default_stage_params, o->get_default_global_params = get_default_global_params;
  o->get_horizon_length = get_horizon_length, o->get_bounds = get_bounds;
  o->get_initial_xk = get_initial_xk, o->get_initial_uk = get_initial_uk;
  o->full_eval_obj = full_eval_obj, o->full_eval_obj_grad = full_eval_obj_grad, o->full_eval_contr_viol = full_eval_contr_viol;
  o->full_eval_constr_jac = full_eval_constr_jac, o->full_eval_lag_hess = full_eval_lag_hess;
  o->user_data = d;
  d->solver = fatrop_ocp_c_create(o, 0, 0);
  const struct FatropOcpCDims *dims = fatrop_ocp_c_get_dims(d->solver);
  for (int k = 0; k < KK; ++k)
    if (dims->ux_offs[k] != k * NZ) abort(); /* the Scaly oracles assume [u_0 x_0 u_1 x_1 ... x_N] */
  return d;
}

int fs_set_option_double(void *h, const char *name, double v) { return fatrop_ocp_c_set_option_double(((fs_data *)h)->solver, name, v); }
int fs_set_option_int(void *h, const char *name, int v) { return fatrop_ocp_c_set_option_int(((fs_data *)h)->solver, name, v); }

int fs_solve(void *h, const double *x0, const double *w_init, double *w_out) {
  fs_data *d = h;
  memcpy(d->x0, x0, sizeof d->x0);
  memcpy(d->w_init, w_init, sizeof d->w_init);
  d->t_oracle = d->t_callback = 0.0, d->n_oracle = 0;
  double t0 = now();
  int ret = fatrop_ocp_c_solve(d->solver);
  d->wall = now() - t0;
  blasfeo_unpack_dvec(NW, (struct blasfeo_dvec *)fatrop_ocp_c_get_primal(d->solver), 0, w_out, 1);
  return ret;
}

double fs_wall(void *h) { return ((fs_data *)h)->wall; }
double fs_t_oracle(void *h) { return ((fs_data *)h)->t_oracle; }
double fs_t_callback(void *h) { return ((fs_data *)h)->t_callback; }
int fs_n_oracle(void *h) { return ((fs_data *)h)->n_oracle; }

void fs_stats(void *h, struct FatropOcpCStats *out) { *out = *fatrop_ocp_c_get_stats(((fs_data *)h)->solver); }
