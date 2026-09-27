/* Times one generated Scaly entry point in C, with no Python in the loop (a copy of
 * examples/qp_solvers/time_entry.c with a nanosecond clock on macOS, where CLOCK_MONOTONIC ticks
 * in microseconds).
 *
 *   time_entry <library> <symbol> <inputs.bin> <repeats> <outputs.bin>
 *
 * Every Scaly function exports the same universal entry,
 *   int f(const double** arg, double** res, int* iw, double* w, int mem),
 * so one driver serves every solver. inputs.bin holds, as little-endian int64: the number of
 * inputs, the number of outputs, the workspace size, each input's size, each output's size; then
 * the inputs as doubles. The driver calls the entry `repeats` times on the same inputs (a cold
 * solve each time: PIQP 0.6.2 has no warm start, and neither the generated solver nor IPOPT is given
 * one), prints the best and the median wall time in nanoseconds, and writes the last call's outputs
 * to outputs.bin.
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 200809L
#endif
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int (*entry_t)(const double **, double **, int *, double *, int);

static double now_ns(void) {
#ifdef __APPLE__
  return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#endif
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
  if (!lib) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  entry_t f = (entry_t)dlsym(lib, argv[2]);
  if (!f) {
    fprintf(stderr, "dlsym: %s\n", dlerror());
    return 1;
  }
  FILE *fp = fopen(argv[3], "rb");
  if (!fp) return 1;
  int64_t head[3];
  if (fread(head, sizeof(int64_t), 3, fp) != 3) return 1;
  int64_t n_arg = head[0], n_res = head[1], sz_w = head[2];
  int64_t *sizes = malloc(sizeof(int64_t) * (size_t)(n_arg + n_res));
  if (fread(sizes, sizeof(int64_t), (size_t)(n_arg + n_res), fp) != (size_t)(n_arg + n_res)) return 1;
  const double **arg = malloc(sizeof(double *) * (size_t)(n_arg + 1));
  double **res = malloc(sizeof(double *) * (size_t)(n_res + 1));
  for (int64_t i = 0; i < n_arg; ++i) {
    double *a = malloc(sizeof(double) * (size_t)(sizes[i] + 1));
    if (fread(a, sizeof(double), (size_t)sizes[i], fp) != (size_t)sizes[i]) return 1;
    arg[i] = a;
  }
  fclose(fp);
  for (int64_t i = 0; i < n_res; ++i) res[i] = malloc(sizeof(double) * (size_t)(sizes[n_arg + i] + 1));
  double *w = malloc(sizeof(double) * (size_t)(sz_w + 1));

  int repeats = atoi(argv[4]);
  double *t = malloc(sizeof(double) * (size_t)repeats);
  for (int k = 0; k < 3; ++k) f(arg, res, NULL, w, 0); /* warm up caches and clocks */
  for (int k = 0; k < repeats; ++k) {
    double t0 = now_ns();
    int status = f(arg, res, NULL, w, 0);
    t[k] = now_ns() - t0;
    if (status != 0) {
      fprintf(stderr, "entry returned %d\n", status);
      return 1;
    }
  }
  qsort(t, (size_t)repeats, sizeof(double), cmp);
  printf("%.0f %.0f\n", t[0], t[repeats / 2]);

  FILE *out = fopen(argv[5], "wb");
  if (!out) return 1;
  for (int64_t i = 0; i < n_res; ++i) fwrite(res[i], sizeof(double), (size_t)sizes[n_arg + i], out);
  fclose(out);
  return 0;
}
