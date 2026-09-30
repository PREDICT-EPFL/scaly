/* Achievable double-precision FMA throughput of one core and the core clock, measured from C.
 *   gcc -O2 peak.c -o peak && ./peak
 * FMA: 24 independent float64x2 fmla chains (asm) (enough to cover 4 pipes x 4-cycle latency), best of 9.
 * Clock: a chain of dependent 1-cycle integer adds (inline asm), best of 9. */
#define _POSIX_C_SOURCE 200809L
#include <arm_neon.h>
#include <stdio.h>
#include <time.h>

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return 1e9 * ts.tv_sec + ts.tv_nsec; }

#define ACC 24
/* 24 independent fmla chains on fixed registers (inline asm: GCC keeps a C accumulator array on the
 * stack across iterations, which hides the FMA throughput). */
static double fma_loop(long iters) {
  double out;
  __asm__ volatile(
    "fmov d0, #1.0\n dup v0.2d, v0.d[0]\n dup v1.2d, v0.d[0]\n"
    "movi v2.2d, #0\n"
    "movi v3.2d, #0\n"
    "movi v4.2d, #0\n"
    "movi v5.2d, #0\n"
    "movi v6.2d, #0\n"
    "movi v7.2d, #0\n"
    "movi v8.2d, #0\n"
    "movi v9.2d, #0\n"
    "movi v10.2d, #0\n"
    "movi v11.2d, #0\n"
    "movi v12.2d, #0\n"
    "movi v13.2d, #0\n"
    "movi v14.2d, #0\n"
    "movi v15.2d, #0\n"
    "movi v16.2d, #0\n"
    "movi v17.2d, #0\n"
    "movi v18.2d, #0\n"
    "movi v19.2d, #0\n"
    "movi v20.2d, #0\n"
    "movi v21.2d, #0\n"
    "movi v22.2d, #0\n"
    "movi v23.2d, #0\n"
    "movi v24.2d, #0\n"
    "movi v25.2d, #0\n"
    "1:\n"
      "fmla v2.2d, v0.2d, v1.2d\n"
      "fmla v3.2d, v0.2d, v1.2d\n"
      "fmla v4.2d, v0.2d, v1.2d\n"
      "fmla v5.2d, v0.2d, v1.2d\n"
      "fmla v6.2d, v0.2d, v1.2d\n"
      "fmla v7.2d, v0.2d, v1.2d\n"
      "fmla v8.2d, v0.2d, v1.2d\n"
      "fmla v9.2d, v0.2d, v1.2d\n"
      "fmla v10.2d, v0.2d, v1.2d\n"
      "fmla v11.2d, v0.2d, v1.2d\n"
      "fmla v12.2d, v0.2d, v1.2d\n"
      "fmla v13.2d, v0.2d, v1.2d\n"
      "fmla v14.2d, v0.2d, v1.2d\n"
      "fmla v15.2d, v0.2d, v1.2d\n"
      "fmla v16.2d, v0.2d, v1.2d\n"
      "fmla v17.2d, v0.2d, v1.2d\n"
      "fmla v18.2d, v0.2d, v1.2d\n"
      "fmla v19.2d, v0.2d, v1.2d\n"
      "fmla v20.2d, v0.2d, v1.2d\n"
      "fmla v21.2d, v0.2d, v1.2d\n"
      "fmla v22.2d, v0.2d, v1.2d\n"
      "fmla v23.2d, v0.2d, v1.2d\n"
      "fmla v24.2d, v0.2d, v1.2d\n"
      "fmla v25.2d, v0.2d, v1.2d\n"
    "subs %1, %1, #1\n b.ne 1b\n"
    "fadd v2.2d, v2.2d, v25.2d\n fmov %0, d2\n"
    : "=r"(out), "+r"(iters)
    :
    : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "cc");
  return out;
}

static void add_chain(long iters) {
  long x = 0;
  for (long k = 0; k < iters; ++k) {
    __asm__ volatile(
      "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
      "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
      "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n"
      "add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n add %0, %0, #1\n" : "+r"(x));
  }
}

int main(void) {
  long it = 20000000;
  double best_f = 1e300, best_c = 1e300, sink = 0;
  for (int r = 0; r < 9; ++r) {
    double t0 = now_ns(); sink += fma_loop(it); double t = now_ns() - t0; if (t < best_f) best_f = t;
    t0 = now_ns(); add_chain(it); t = now_ns() - t0; if (t < best_c) best_c = t;
  }
  double ghz = 16.0 * it / best_c;
  double gflops = 4.0 * ACC * it / best_f;
  printf("clock_ghz %.3f\nfma_gflops %.2f\nflops_per_cycle %.2f\n(sink %g)\n", ghz, gflops, gflops / ghz, sink);
  return 0;
}
