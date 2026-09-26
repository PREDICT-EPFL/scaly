/* Run the vendored PIQP on one problem with verbose output and print its result at full precision.
 *
 * Usage: piqp_trace <problem.bin>. The file, written by tests/ipm/piqp_trace.py, is little endian:
 * int64 header[10] = {n, p, m, nnz(P upper), nnz(A), nnz(G), dense, max_iter,
 * iterative_refinement_always_enabled, preconditioner_scale_cost} (a negative setting keeps PIQP's default),
 * then P, A and G each as CSC (int32 col_ptr[n + 1], int32 row_ind[nnz], double values[nnz]) with
 * the vectors after them: P, c[n], A, b[p], G, h_l[m], h_u[m], x_l[n], x_u[n]. An absent bound
 * is an IEEE infinity, which PIQP reads as absent (anything at or beyond PIQP_INF is).
 *
 * PIQP prints its per-iteration table itself. After it, this prints "SCALY_TRACE_RESULT", one
 * "info <name> <value>" line per field and one "vec <name> <values...>" line per result vector.
 *
 * With SCALY_TRACE_REPEAT=k in the environment, the solve then runs k more times in the same
 * process, quietly, and "info solve_time_min" reports the fastest of all: a warmed-up timing. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "piqp/piqp.h"

static void *read_exact(FILE *f, size_t count, size_t size) {
  void *buf = calloc(count ? count : 1, size);
  if (!buf || fread(buf, size, count, f) != count) {
    fprintf(stderr, "piqp_trace: truncated input\n");
    exit(2);
  }
  return buf;
}

static piqp_csc *read_csc(FILE *f, piqp_int rows, piqp_int cols, piqp_int nnz) {
  piqp_int *p = read_exact(f, (size_t)cols + 1, sizeof(piqp_int));
  piqp_int *i = read_exact(f, (size_t)nnz, sizeof(piqp_int));
  piqp_float *x = read_exact(f, (size_t)nnz, sizeof(piqp_float));
  return piqp_csc_matrix(rows, cols, nnz, p, i, x);
}

static piqp_float *read_vec(FILE *f, piqp_int n) { return read_exact(f, (size_t)n, sizeof(piqp_float)); }

/* Row-major dense copy of a CSC matrix, as PIQP's dense C interface reads it. */
static piqp_float *dense_row_major(const piqp_csc *a) {
  piqp_float *d = calloc((size_t)(a->m * a->n) + 1, sizeof(piqp_float));
  for (piqp_int j = 0; j < a->n; ++j) {
    for (piqp_int k = a->p[j]; k < a->p[j + 1]; ++k) d[(size_t)a->i[k] * a->n + j] = a->x[k];
  }
  return d;
}

static void print_vec(const char *name, const piqp_float *v, piqp_int n) {
  printf("vec %s", name);
  for (piqp_int k = 0; k < n; ++k) printf(" %.17g", (double)v[k]);
  printf("\n");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: piqp_trace <problem.bin>\n");
    return 2;
  }
  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    perror("piqp_trace");
    return 2;
  }
  int64_t header[10];
  if (fread(header, sizeof(int64_t), 10, f) != 10) {
    fprintf(stderr, "piqp_trace: truncated header\n");
    return 2;
  }
  piqp_int n = (piqp_int)header[0], p = (piqp_int)header[1], m = (piqp_int)header[2];
  int dense = (int)header[6];
  piqp_csc *P = read_csc(f, n, n, (piqp_int)header[3]);
  piqp_float *c = read_vec(f, n);
  piqp_csc *A = read_csc(f, p, n, (piqp_int)header[4]);
  piqp_float *b = read_vec(f, p);
  piqp_csc *G = read_csc(f, m, n, (piqp_int)header[5]);
  piqp_float *h_l = read_vec(f, m), *h_u = read_vec(f, m), *x_l = read_vec(f, n), *x_u = read_vec(f, n);
  fclose(f);

  piqp_settings settings;
  if (dense) {
    piqp_set_default_settings_dense(&settings);
  } else {
    piqp_set_default_settings_sparse(&settings);
  }
  settings.verbose = 1;
  settings.compute_timings = 1;
  if (header[7] >= 0) settings.max_iter = (piqp_int)header[7];
  if (header[8] >= 0) settings.iterative_refinement_always_enabled = (piqp_int)header[8];
  if (header[9] >= 0) settings.preconditioner_scale_cost = (piqp_int)header[9];

  piqp_workspace *work = NULL;
  if (dense) {
    piqp_data_dense data = {n, p, m, dense_row_major(P), c, dense_row_major(A), b, dense_row_major(G), h_l, h_u, x_l, x_u};
    piqp_setup_dense(&work, &data, &settings);
  } else {
    piqp_data_sparse data = {n, p, m, P, c, A, b, G, h_l, h_u, x_l, x_u};
    piqp_setup_sparse(&work, &data, &settings);
  }
  if (!work) {
    fprintf(stderr, "piqp_trace: setup failed\n");
    return 1;
  }
  piqp_status status = piqp_solve(work);
  fflush(stdout);
  double solve_time_min = (double)work->result->info.solve_time;
  const char *repeat = getenv("SCALY_TRACE_REPEAT");
  if (repeat && atoi(repeat) > 0) {
    settings.verbose = 0;
    piqp_update_settings(work, &settings);
    for (int k = 0; k < atoi(repeat); ++k) {
      piqp_solve(work);
      double t = (double)work->result->info.solve_time;
      if (t < solve_time_min) solve_time_min = t;
    }
  }
  const piqp_result *r = work->result;
  const piqp_info *info = &r->info;
  printf("SCALY_TRACE_RESULT\n");
  printf("info status %d\n", (int)status);
  printf("info iter %d\n", (int)info->iter);
#define INFO(field) printf("info " #field " %.17g\n", (double)info->field)
  INFO(rho);
  INFO(delta);
  INFO(mu);
  INFO(sigma);
  INFO(primal_step);
  INFO(dual_step);
  INFO(primal_res);
  INFO(primal_res_rel);
  INFO(dual_res);
  INFO(dual_res_rel);
  INFO(primal_res_reg);
  INFO(dual_res_reg);
  INFO(primal_prox_inf);
  INFO(dual_prox_inf);
  INFO(primal_obj);
  INFO(dual_obj);
  INFO(duality_gap);
  INFO(duality_gap_rel);
  INFO(reg_limit);
  INFO(setup_time);
  INFO(solve_time);
#undef INFO
  printf("info solve_time_min %.17g\n", solve_time_min);
  printf("info factor_retires %d\n", (int)info->factor_retires);
  printf("info no_primal_update %d\n", (int)info->no_primal_update);
  printf("info no_dual_update %d\n", (int)info->no_dual_update);
  print_vec("x", r->x, n);
  print_vec("y", r->y, p);
  print_vec("z_l", r->z_l, m);
  print_vec("z_u", r->z_u, m);
  print_vec("z_bl", r->z_bl, n);
  print_vec("z_bu", r->z_bu, n);
  print_vec("s_l", r->s_l, m);
  print_vec("s_u", r->s_u, m);
  print_vec("s_bl", r->s_bl, n);
  print_vec("s_bu", r->s_bu, n);
  piqp_cleanup(work);
  return 0;
}
