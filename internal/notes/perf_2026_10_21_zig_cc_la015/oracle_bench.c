
#include <stdio.h>
static double zp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
void zp_oracles(const double** a, int reps, const scaly_solver_option* const* o, double* w, double* out) {
  double x[76], f, g[60], grad[76], jac[236], hess[136], xlb[76], xub[76], gl[60], gu[60], one = 1.0, sink = 0;
  for (int i = 0; i < 16; ++i) x[i] = a[0][i];
  for (int i = 0; i < 60; ++i) x[16 + i] = a[1][i];
  double t[5] = {0};
  for (int r = 0; r < reps; ++r) {
    double t0 = zp_now(); ctdt_scaly_problem_N8_walls_base_raw(x, a[6], a[7], a[8], a[9], a[10], &f, g, w, o);
    double t1 = zp_now(); ctdt_scaly_problem_N8_walls_grad_raw(x, a[6], a[7], a[8], a[9], a[10], grad, w, o);
    double t2 = zp_now(); ctdt_scaly_problem_N8_walls_jac_raw(x, a[6], a[7], a[8], a[9], a[10], jac, w, o);
    double t3 = zp_now(); ctdt_scaly_problem_N8_walls_hess_upper_raw(x, a[6], a[7], a[8], a[9], a[10], &one, a[5], hess, w, o);
    double t4 = zp_now(); ctdt_scaly_problem_N8_walls_bounds_raw(a[6], a[7], a[8], a[9], a[10], xlb, xub, gl, gu, w, o);
    double t5 = zp_now();
    t[0] += t1 - t0; t[1] += t2 - t1; t[2] += t3 - t2; t[3] += t4 - t3; t[4] += t5 - t4;
    sink += f + grad[3] + jac[7] + hess[11];
  }
  for (int i = 0; i < 5; ++i) out[i] = 1e6 * t[i] / reps;
  if (sink == 12345.678) puts("");
}
