/* Times the generated PIQP example on one problem, with no Python in the process.
 *
 * benchmark.py exports the setup and solve Functions with scaly.codegen.write_module and builds
 * this file with them, defining SETUP and SOLVE as their pointer-array entries (<symbol>_abi).
 * The setup's outputs, the equilibrated data, its scaling and A^T A, are the solve's inputs.
 *
 * DIR/meta.txt holds the byte sizes of the setup's inputs, of its outputs (the solve's inputs)
 * and of the solve's outputs, then the two workspace sizes. DIR/inputs.bin holds the setup's
 * inputs. The program runs each function once to warm up and then until BUDGET seconds have
 * passed (at least 3 and at most 1000 calls), writes the solve's outputs to DIR/outputs.bin, and
 * prints the median and minimum time of each call in microseconds as one JSON line.
 *
 * Usage: ./run DIR [BUDGET]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int SETUP(const void* const* arg, void* const* res, void* w);
extern int SOLVE(const void* const* arg, void* const* res, void* w);

typedef int (*entry_t)(const void* const*, void* const*, void*);

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cmp(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return (x > y) - (x < y);
}

static void** buffers(FILE* meta, int* count) {
  if (fscanf(meta, "%d", count) != 1) exit(3);
  void** out = calloc(*count ? *count : 1, sizeof(void*));
  for (int i = 0; i < *count; ++i) {
    long bytes;
    if (fscanf(meta, "%ld", &bytes) != 1) exit(3);
    out[i] = aligned_alloc(64, (bytes + 63) / 64 * 64 + 64);
    ((long*)out[i])[0] = bytes; /* remembered until the buffer is filled */
  }
  return out;
}

static long* sizes_of(void** bufs, int count) {
  long* sizes = malloc(sizeof(long) * (count ? count : 1));
  for (int i = 0; i < count; ++i) sizes[i] = ((long*)bufs[i])[0];
  return sizes;
}

/* Calls f until budget seconds have passed, at least 3 and at most 1000 times, after one warm-up. */
static void timed(entry_t f, void** arg, void** res, void* w, double budget, double* median, double* minimum, int* reps) {
  double t0 = now();
  if (f((const void* const*)arg, res, w)) { fprintf(stderr, "nonzero status\n"); exit(4); }
  double first = now() - t0;
  int n = first > 0 ? (int)(budget / first) : 1000;
  n = n < 3 ? 3 : n > 1000 ? 1000 : n;
  double* t = malloc(sizeof(double) * n);
  for (int i = 0; i < n; ++i) {
    double a = now();
    f((const void* const*)arg, res, w);
    t[i] = now() - a;
  }
  qsort(t, n, sizeof(double), cmp);
  *median = t[n / 2] * 1e6, *minimum = t[0] * 1e6, *reps = n;
  free(t);
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s DIR [BUDGET]\n", argv[0]); return 2; }
  double budget = argc > 2 ? atof(argv[2]) : 2.0;
  char path[4096];
  snprintf(path, sizeof path, "%s/meta.txt", argv[1]);
  FILE* meta = fopen(path, "r");
  if (!meta) { perror(path); return 1; }
  int n_in, n_mid, n_out;
  long w_setup, w_solve;
  void** in = buffers(meta, &n_in);
  void** mid = buffers(meta, &n_mid);
  void** out = buffers(meta, &n_out);
  if (fscanf(meta, "%ld %ld", &w_setup, &w_solve) != 2) return 3;
  fclose(meta);
  long* in_bytes = sizes_of(in, n_in);
  long* out_bytes = sizes_of(out, n_out);

  snprintf(path, sizeof path, "%s/inputs.bin", argv[1]);
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); return 1; }
  for (int i = 0; i < n_in; ++i)
    if (fread(in[i], 1, in_bytes[i], f) != (size_t)in_bytes[i]) { fprintf(stderr, "short read\n"); return 1; }
  fclose(f);
  void* ws = aligned_alloc(64, (w_setup + 63) / 64 * 64 + 64);
  void* wv = aligned_alloc(64, (w_solve + 63) / 64 * 64 + 64);

  double setup_median, setup_min, solve_median, solve_min;
  int setup_reps, solve_reps;
  timed(SETUP, in, mid, ws, budget, &setup_median, &setup_min, &setup_reps);
  timed(SOLVE, mid, out, wv, budget, &solve_median, &solve_min, &solve_reps);

  snprintf(path, sizeof path, "%s/outputs.bin", argv[1]);
  f = fopen(path, "wb");
  for (int i = 0; i < n_out; ++i) fwrite(out[i], 1, out_bytes[i], f);
  fclose(f);
  printf("{\"setup_median_us\": %.3f, \"setup_min_us\": %.3f, \"setup_reps\": %d, \"solve_median_us\": %.3f, \"solve_min_us\": %.3f, \"solve_reps\": %d}\n",
         setup_median, setup_min, setup_reps, solve_median, solve_min, solve_reps);
  return 0;
}
