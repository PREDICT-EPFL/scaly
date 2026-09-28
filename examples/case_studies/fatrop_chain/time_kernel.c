/* Times one generated kernel in C through the shared entry point, with no Python in the loop.
 *
 *   time_kernel <library> <symbol> <inputs.bin> <repeats> <outputs.bin>
 *
 * Scaly and CasADi (generated with casadi_int = int) both export
 *   int f(const double** arg, double** res, int* iw, double* w, int mem).
 * inputs.bin holds, as little-endian int64: the number of inputs, the number of outputs, the double
 * workspace size, the int workspace size, the lengths of the arg and res pointer arrays (CasADi's
 * generated MX code uses the slots past its inputs and outputs as scratch for nested calls, so they
 * must be at least sz_arg and sz_res long), each input's size, each output's size (-1 for an output the
 * caller does not request, passed as NULL, as IPOPT's callbacks do with CasADi's nlp_jac_g); then the
 * inputs as doubles. Prints the best and the median call in nanoseconds and writes the requested
 * outputs of the last call to outputs.bin. Same protocol as examples/qp_solvers/time_entry.c, except that
 * each sample times a batch of calls long enough (1 ms) for the clock, which on macOS ticks in µs. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int (*entry_t)(const double **, double **, int *, double *, int);

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s <library> <symbol> <inputs.bin> <repeats> <outputs.bin>\n", argv[0]);
    return 2;
  }
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) return fprintf(stderr, "dlopen: %s\n", dlerror()), 1;
  entry_t f = (entry_t)dlsym(lib, argv[2]);
  if (!f) return fprintf(stderr, "dlsym: %s\n", dlerror()), 1;
  FILE *fp = fopen(argv[3], "rb");
  if (!fp) return 1;
  int64_t head[6];
  if (fread(head, sizeof(int64_t), 6, fp) != 6) return 1;
  int64_t n_arg = head[0], n_res = head[1], sz_w = head[2], sz_iw = head[3], sz_arg = head[4], sz_res = head[5];
  if (sz_arg < n_arg) sz_arg = n_arg;
  if (sz_res < n_res) sz_res = n_res;
  int64_t *sizes = malloc(sizeof(int64_t) * (size_t)(n_arg + n_res));
  if (fread(sizes, sizeof(int64_t), (size_t)(n_arg + n_res), fp) != (size_t)(n_arg + n_res)) return 1;
  const double **arg = calloc((size_t)(sz_arg + 1), sizeof(double *));
  double **res = calloc((size_t)(sz_res + 1), sizeof(double *));
  for (int64_t i = 0; i < n_arg; ++i) {
    double *a = malloc(sizeof(double) * (size_t)(sizes[i] + 1));
    if (fread(a, sizeof(double), (size_t)sizes[i], fp) != (size_t)sizes[i]) return 1;
    arg[i] = a;
  }
  fclose(fp);
  for (int64_t i = 0; i < n_res; ++i) res[i] = sizes[n_arg + i] < 0 ? NULL : malloc(sizeof(double) * (size_t)(sizes[n_arg + i] + 1));
  double *w = malloc(sizeof(double) * (size_t)(sz_w + 1));
  int *iw = malloc(sizeof(int) * (size_t)(sz_iw + 1));

  /* CasADi's generated code can use res as scratch for nested calls and clobber the leading slots
   * (docs/results/fairness.md), so both pointer arrays are restored before every call. */
  const double **arg0 = malloc(sizeof(double *) * (size_t)(sz_arg + 1));
  double **res0 = malloc(sizeof(double *) * (size_t)(sz_res + 1));
  for (int64_t i = 0; i <= sz_arg; ++i) arg0[i] = arg[i];
  for (int64_t i = 0; i <= sz_res; ++i) res0[i] = res[i];
#define RESTORE() \
  for (int64_t i_ = 0; i_ < n_arg; ++i_) arg[i_] = arg0[i_]; \
  for (int64_t i_ = 0; i_ < n_res; ++i_) res[i_] = res0[i_]
  int repeats = atoi(argv[4]);
  double *t = malloc(sizeof(double) * (size_t)repeats);
  for (int k = 0; k < 3; ++k) { /* warm up caches and clocks */
    RESTORE();
    f(arg, res, iw, w, 0);
  }
  double t0 = now_ns();
  for (int k = 0; k < 64; ++k) {
    RESTORE();
    f(arg, res, iw, w, 0);
  }
  double per_call = (now_ns() - t0) / 64.0;
  long batch = per_call > 1e6 ? 1 : (long)(1e6 / (per_call > 1.0 ? per_call : 1.0)) + 1;
  for (int k = 0; k < repeats; ++k) {
    double t1 = now_ns();
    for (long b = 0; b < batch; ++b) {
      RESTORE();
      int status = f(arg, res, iw, w, 0);
      if (status != 0) return fprintf(stderr, "entry returned %d\n", status), 1;
    }
    t[k] = (now_ns() - t1) / (double)batch;
  }
  qsort(t, (size_t)repeats, sizeof(double), cmp);
  printf("%.1f %.1f\n", t[0], t[repeats / 2]);
  fp = fopen(argv[5], "wb");
  if (!fp) return 1;
  for (int64_t i = 0; i < n_res; ++i)
    if (res0[i]) fwrite(res0[i], sizeof(double), (size_t)sizes[n_arg + i], fp);
  fclose(fp);
  return 0;
}
