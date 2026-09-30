/* Scaly-generated dense kernels against BLASFEO (panel-major native API and its column-major
 * BLAS/LAPACK API), timed from C.
 *
 *   bench <op> <n> <scaly_jit.so> <scaly_o3.so> <symbol> <workspace> [sample_ms] [samples]
 *
 * op: gemm | potrf | trsm. Every variant is timed in batches of calls filling >= sample_ms (20)
 * per sample; samples are interleaved across variants round by round; the minimum over the
 * samples (9) is reported. Consecutive calls are independent (same inputs), so this is a
 * throughput time, identical for every variant. In-place column-major LAPACK/BLAS calls
 * (lapack_dpotrf, blas_dtrsm) restore their input with a memcpy each call; that memcpy is timed
 * alone and subtracted. The trsm variants reset BLASFEO's cached inverse diagonal (use_dA = 0) on
 * each call, so a fresh factor is assumed as for Scaly.
 * Prints one line per variant: name ns_per_call gflops max_abs_diff_vs_scaly_jit.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "blasfeo.h"

typedef int (*entry_t)(const double **, double **, int *, double *, int);

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

static int n;
static double *A_cm, *B_cm, *A_rm, *B_rm, *Bt_cm, *out_rm, *out_rm2, *w, *C_cm, *tmp_cm, *X_cm;
static entry_t f_jit, f_o3;
static struct blasfeo_dmat sA, sB, sBt, sD, sDt;
static const char *op;

static void scaly_jit(void) { const double *arg[2] = {A_rm, B_rm}; double *res[1] = {out_rm}; f_jit(arg, res, NULL, w, 0); }
static void scaly_o3(void) { const double *arg[2] = {A_rm, B_rm}; double *res[1] = {out_rm2}; f_o3(arg, res, NULL, w, 0); }
static void bf_gemm_nn(void) { blasfeo_dgemm_nn(n, n, n, 1.0, &sA, 0, 0, &sB, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0); }
static void bf_gemm_nt(void) { blasfeo_dgemm_nt(n, n, n, 1.0, &sA, 0, 0, &sBt, 0, 0, 0.0, &sD, 0, 0, &sD, 0, 0); }
static void bf_blas_gemm(void) { char t = 'N'; double one = 1.0, zero = 0.0; blasfeo_blas_dgemm(&t, &t, &n, &n, &n, &one, A_cm, &n, B_cm, &n, &zero, C_cm, &n); }
static void bf_potrf(void) { blasfeo_dpotrf_l(n, &sA, 0, 0, &sD, 0, 0); }
static void bf_lapack_potrf(void) { char u = 'L'; int info; memcpy(C_cm, A_cm, sizeof(double) * n * n); blasfeo_lapack_dpotrf(&u, &n, C_cm, &n, &info); }
static void bf_trsm_llnn(void) { sA.use_dA = 0; blasfeo_dtrsm_llnn(n, n, 1.0, &sA, 0, 0, &sB, 0, 0, &sD, 0, 0); }
/* X = L^{-1} B  <=>  X^T = B^T L^{-T}: rltn on B^T gives X^T */
static void bf_trsm_rltn(void) { sA.use_dA = 0; blasfeo_dtrsm_rltn(n, n, 1.0, &sA, 0, 0, &sBt, 0, 0, &sDt, 0, 0); }
static void bf_blas_trsm(void) { char s = 'L', u = 'L', t = 'N', d = 'N'; double one = 1.0; memcpy(C_cm, B_cm, sizeof(double) * n * n); blasfeo_blas_dtrsm(&s, &u, &t, &d, &n, &n, &one, A_cm, &n, C_cm, &n); }
static void copy_only(void) { memcpy(C_cm, (op[0] == 'p' ? A_cm : B_cm), sizeof(double) * n * n); }

typedef struct { const char *name; void (*fn)(void); long reps; double best; int subtract_copy; double diff; } variant;

static double time_batch(void (*fn)(void), long reps) {
  double t0 = now_ns();
  for (long r = 0; r < reps; ++r) fn();
  return now_ns() - t0;
}

static long calibrate(void (*fn)(void), double sample_ns) {
  for (int k = 0; k < 1000; ++k) fn();
  long reps = 16;
  for (;;) {
    double t = time_batch(fn, reps);
    if (t >= sample_ns) return reps;
    reps = (long)(reps * (t > 0 ? 1.2 * sample_ns / t : 2.0)) + 1;
  }
}

/* max |scaly_rm - X_cm^T| over the entries compared (lower triangle for potrf) */
static double diff_rm_cm(const double *rm, const double *cm, int lower) {
  double m = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < (lower ? i + 1 : n); ++j) m = fmax(m, fabs(rm[i * n + j] - cm[i + j * n]));
  return m;
}

int main(int argc, char **argv) {
  if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
  op = argv[1]; n = atoi(argv[2]);
  double sample_ns = 1e6 * (argc > 7 ? atof(argv[7]) : 20.0);
  int samples = argc > 8 ? atoi(argv[8]) : 9;
  void *l1 = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL), *l2 = dlopen(argv[4], RTLD_NOW | RTLD_LOCAL);
  if (!l1 || !l2) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  f_jit = (entry_t)dlsym(l1, argv[5]); f_o3 = (entry_t)dlsym(l2, argv[5]);
  if (!f_jit || !f_o3) { fprintf(stderr, "dlsym\n"); return 1; }
  long ws = atol(argv[6]);
  size_t nn = (size_t)n * n;
  A_cm = calloc(nn, 8); B_cm = calloc(nn, 8); Bt_cm = calloc(nn, 8); A_rm = calloc(nn, 8); B_rm = calloc(nn, 8);
  out_rm = calloc(nn, 8); out_rm2 = calloc(nn, 8); C_cm = calloc(nn, 8); tmp_cm = calloc(nn, 8); X_cm = calloc(nn, 8);
  w = calloc((size_t)ws + 1, 8);
  srand(12345);
  for (size_t k = 0; k < nn; ++k) { A_cm[k] = (double)rand() / RAND_MAX - 0.5; B_cm[k] = (double)rand() / RAND_MAX - 0.5; }
  if (strcmp(op, "potrf") == 0) { /* SPD: A = M M^T + n I */
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { double s = 0; for (int k = 0; k < n; ++k) s += A_cm[i + k * n] * A_cm[j + k * n]; tmp_cm[i + j * n] = s + (i == j ? n : 0); }
    memcpy(A_cm, tmp_cm, nn * 8);
  } else if (strcmp(op, "trsm") == 0) { /* well-conditioned lower triangle */
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A_cm[i + j * n] = i < j ? 0.0 : (i == j ? 1.0 + 0.5 * (double)rand() / RAND_MAX : A_cm[i + j * n] / n);
  }
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { A_rm[i * n + j] = A_cm[i + j * n]; B_rm[i * n + j] = B_cm[i + j * n]; Bt_cm[i + j * n] = B_cm[j + i * n]; }
  void *m1, *m2, *m3, *m4, *m5;
  blasfeo_malloc_align(&m1, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sA, m1);
  blasfeo_malloc_align(&m2, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sB, m2);
  blasfeo_malloc_align(&m3, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sBt, m3);
  blasfeo_malloc_align(&m4, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sD, m4);
  blasfeo_malloc_align(&m5, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sDt, m5);
  blasfeo_pack_dmat(n, n, A_cm, n, &sA, 0, 0); blasfeo_pack_dmat(n, n, B_cm, n, &sB, 0, 0); blasfeo_pack_dmat(n, n, Bt_cm, n, &sBt, 0, 0);
  memset(m4, 0, blasfeo_memsize_dmat(n, n)); blasfeo_create_dmat(n, n, &sD, m4);

  variant v[8]; int nv = 0;
  v[nv++] = (variant){"scaly_jit", scaly_jit, 0, 1e300, 0, 0};
  v[nv++] = (variant){"scaly_o3", scaly_o3, 0, 1e300, 0, 0};
  int lower = 0;
  if (strcmp(op, "gemm") == 0) {
    v[nv++] = (variant){"bf_dgemm_nn", bf_gemm_nn, 0, 1e300, 0, 0};
    v[nv++] = (variant){"bf_dgemm_nt", bf_gemm_nt, 0, 1e300, 0, 0};
    v[nv++] = (variant){"bf_blas_dgemm", bf_blas_gemm, 0, 1e300, 0, 0};
  } else if (strcmp(op, "potrf") == 0) {
    lower = 1;
    v[nv++] = (variant){"bf_dpotrf_l", bf_potrf, 0, 1e300, 0, 0};
    v[nv++] = (variant){"bf_lapack_dpotrf", bf_lapack_potrf, 0, 1e300, 1, 0};
  } else {
    v[nv++] = (variant){"bf_dtrsm_llnn", bf_trsm_llnn, 0, 1e300, 0, 0};
    v[nv++] = (variant){"bf_dtrsm_rltn", bf_trsm_rltn, 0, 1e300, 0, 0};
    v[nv++] = (variant){"bf_blas_dtrsm", bf_blas_trsm, 0, 1e300, 1, 0};
  }
  v[nv++] = (variant){"memcpy", copy_only, 0, 1e300, 0, 0};

  /* correctness: run each once, compare to scaly_jit */
  for (int k = 0; k < nv; ++k) {
    v[k].fn();
    const char *nm = v[k].name;
    if (!strcmp(nm, "scaly_jit") || !strcmp(nm, "memcpy")) continue;
    if (!strcmp(nm, "scaly_o3")) { double m = 0; for (size_t q = 0; q < nn; ++q) m = fmax(m, fabs(out_rm2[q] - out_rm[q])); v[k].diff = m; continue; }
    if (!strncmp(nm, "bf_blas", 7) || !strncmp(nm, "bf_lapack", 9)) { v[k].diff = diff_rm_cm(out_rm, C_cm, lower); continue; }
    if (!strcmp(nm, "bf_dtrsm_rltn")) { blasfeo_unpack_dmat(n, n, &sDt, 0, 0, tmp_cm, n); for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) X_cm[i + j * n] = tmp_cm[j + i * n]; }
    else blasfeo_unpack_dmat(n, n, &sD, 0, 0, X_cm, n);
    v[k].diff = diff_rm_cm(out_rm, X_cm, lower);
  }
  /* reference check of scaly_jit itself against a naive column-major computation */
  double ref = 0;
  if (!strcmp(op, "gemm")) {
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { double s = 0; for (int k = 0; k < n; ++k) s += A_cm[i + k * n] * B_cm[k + j * n]; ref = fmax(ref, fabs(s - out_rm[i * n + j])); }
  } else if (!strcmp(op, "potrf")) { /* || L L^T - A || on the lower triangle */
    for (int i = 0; i < n; ++i) for (int j = 0; j <= i; ++j) { double s = 0; for (int k = 0; k <= j; ++k) s += out_rm[i * n + k] * out_rm[j * n + k]; ref = fmax(ref, fabs(s - A_cm[i + j * n])); }
  } else { /* || L X - B || */
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) { double s = 0; for (int k = 0; k <= i; ++k) s += A_cm[i + k * n] * out_rm[k * n + j]; ref = fmax(ref, fabs(s - B_cm[i + j * n])); }
  }

  for (int k = 0; k < nv; ++k) v[k].reps = calibrate(v[k].fn, sample_ns);
  for (int s = 0; s < samples; ++s)
    for (int k = 0; k < nv; ++k) { double t = time_batch(v[k].fn, v[k].reps) / v[k].reps; if (t < v[k].best) v[k].best = t; }
  double copy_ns = v[nv - 1].best;
  double flops = !strcmp(op, "gemm") ? 2.0 * n * n * n : !strcmp(op, "potrf") ? n * (double)n * n / 3.0 : (double)n * n * n;
  printf("# op=%s n=%d scaly_residual=%.3g copy_ns=%.2f\n", op, n, ref, copy_ns);
  for (int k = 0; k < nv - 1; ++k) {
    double t = v[k].best - (v[k].subtract_copy ? copy_ns : 0.0);
    printf("%s %s %d %.2f %.3f %.3g\n", op, v[k].name, n, t, flops / t, v[k].diff);
  }
  return 0;
}
