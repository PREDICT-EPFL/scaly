/* Reading an episode written by run_benchmark.py and writing the per-step results, shared by the
 * TinyMPC and the Scaly drivers so that both replay exactly the same sequence of solves.
 *
 * Episode file (little-endian): int32 header[12] = {magic, version, nx, nu, N, steps, max_iter,
 * en_state_bound, en_input_bound, n_state_cones, n_input_cones, fixed_bounds}, then float64: rho, abs_pri_tol,
 * abs_dua_tol, A (row-major), B (row-major), f, Q, R, (start, dim, mu) per state cone then per
 * input cone, x_min, x_max (N x nx, stage-major), u_min, u_max ((N-1) x nu), then per step x0,
 * xref (N x nx), uref ((N-1) x nu).
 *
 * fixed_bounds says the generated solver has the bounds built in (the library still reads them).
 *
 * Result file: int32 {steps, nu, reps}, then per step float64 {iterations, solved, best time in ns
 * over the repetitions, u0[nu]}, then per repetition the total solve time in ns.
 */
#ifndef TINYMPC_BENCH_EPISODE_H
#define TINYMPC_BENCH_EPISODE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPISODE_MAGIC 0x594e4954 /* "TINY" */

typedef struct {
  int nx, nu, N, steps, max_iter, en_state_bound, en_input_bound, n_state_cones, n_input_cones, fixed_bounds;
  double rho, abs_pri_tol, abs_dua_tol;
  double *A, *B, *f, *Q, *R, *state_cones, *input_cones;
  double *x_min, *x_max, *u_min, *u_max;
  double *x0, *xref, *uref; /* steps x nx, steps x N*nx, steps x (N-1)*nu */
} episode_t;

static double *episode_take(FILE *fp, size_t n) {
  double *p = (double *)malloc((n ? n : 1) * sizeof(double));
  if (!p || fread(p, sizeof(double), n, fp) != n) {
    fprintf(stderr, "episode: short read\n");
    exit(2);
  }
  return p;
}

static void episode_read(const char *path, episode_t *e) {
  FILE *fp = fopen(path, "rb");
  int32_t h[12];
  if (!fp || fread(h, sizeof(int32_t), 12, fp) != 12 || h[0] != EPISODE_MAGIC || h[1] != 1) {
    fprintf(stderr, "episode: cannot read %s\n", path);
    exit(2);
  }
  e->nx = h[2], e->nu = h[3], e->N = h[4], e->steps = h[5], e->max_iter = h[6];
  e->en_state_bound = h[7], e->en_input_bound = h[8], e->n_state_cones = h[9], e->n_input_cones = h[10], e->fixed_bounds = h[11];
  double *s = episode_take(fp, 3);
  e->rho = s[0], e->abs_pri_tol = s[1], e->abs_dua_tol = s[2];
  free(s);
  int nx = e->nx, nu = e->nu, N = e->N;
  e->A = episode_take(fp, (size_t)nx * nx);
  e->B = episode_take(fp, (size_t)nx * nu);
  e->f = episode_take(fp, nx);
  e->Q = episode_take(fp, nx);
  e->R = episode_take(fp, nu);
  e->state_cones = episode_take(fp, 3 * (size_t)e->n_state_cones);
  e->input_cones = episode_take(fp, 3 * (size_t)e->n_input_cones);
  e->x_min = episode_take(fp, (size_t)N * nx);
  e->x_max = episode_take(fp, (size_t)N * nx);
  e->u_min = episode_take(fp, (size_t)(N - 1) * nu);
  e->u_max = episode_take(fp, (size_t)(N - 1) * nu);
  size_t per = (size_t)nx + (size_t)N * nx + (size_t)(N - 1) * nu;
  double *all = episode_take(fp, per * e->steps);
  e->x0 = (double *)malloc(sizeof(double) * nx * e->steps);
  e->xref = (double *)malloc(sizeof(double) * N * nx * e->steps);
  e->uref = (double *)malloc(sizeof(double) * (N - 1) * nu * e->steps);
  for (int k = 0; k < e->steps; ++k) {
    const double *p = all + per * k;
    memcpy(e->x0 + (size_t)k * nx, p, sizeof(double) * nx);
    memcpy(e->xref + (size_t)k * N * nx, p + nx, sizeof(double) * N * nx);
    memcpy(e->uref + (size_t)k * (N - 1) * nu, p + nx + N * nx, sizeof(double) * (N - 1) * nu);
  }
  free(all);
  fclose(fp);
}

static inline double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

typedef struct {
  int steps, nu, reps;
  double *iterations, *solved, *best_ns, *u0, *total_ns;
} results_t;

static void results_init(results_t *r, int steps, int nu, int reps) {
  r->steps = steps, r->nu = nu, r->reps = reps;
  r->iterations = (double *)calloc(steps, sizeof(double));
  r->solved = (double *)calloc(steps, sizeof(double));
  r->best_ns = (double *)malloc(steps * sizeof(double));
  for (int k = 0; k < steps; ++k) r->best_ns[k] = 1e300;
  r->u0 = (double *)calloc((size_t)steps * nu, sizeof(double));
  r->total_ns = (double *)calloc(reps, sizeof(double));
}

static void results_write(const char *path, const results_t *r) {
  FILE *fp = fopen(path, "wb");
  int32_t h[3] = {r->steps, r->nu, r->reps};
  fwrite(h, sizeof(int32_t), 3, fp);
  for (int k = 0; k < r->steps; ++k) {
    double row[3] = {r->iterations[k], r->solved[k], r->best_ns[k]};
    fwrite(row, sizeof(double), 3, fp);
    fwrite(r->u0 + (size_t)k * r->nu, sizeof(double), r->nu, fp);
  }
  fwrite(r->total_ns, sizeof(double), r->reps, fp);
  fclose(fp);
}

#endif
