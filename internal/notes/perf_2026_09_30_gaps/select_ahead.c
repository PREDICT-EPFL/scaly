#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 502
static const uint8_t k8[N] = {[0 ... 71] = 1, [72] = 0, [73 ... N-1] = 1};
static const double kd[N] = {[0 ... 71] = 1.0, [72] = 0.0, [73 ... N-1] = 1.0};
static const int64_t k64[N] = {[0 ... 71] = -1, [72] = 0, [73 ... N-1] = -1};
#define NOINL __attribute__((noinline))
// 1. constant mask select
NOINL void mask_u8(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) s[i] = (k8[i] ? (a[i] * (b[i] - (c[i] * a[i]))) : 0.0); }
NOINL void mask_dbl(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) s[i] = ((kd[i] != 0.0) ? (a[i] * (b[i] - (c[i] * a[i]))) : 0.0); }
NOINL void mask_i64(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) s[i] = (k64[i] ? (a[i] * (b[i] - (c[i] * a[i]))) : 0.0); }
NOINL void mask_u8_pragma(const double* a, const double* b, const double* c, double* s) {
#pragma clang loop vectorize(enable)
  for (long long i = 0; i < N; ++i) s[i] = (k8[i] ? (a[i] * (b[i] - (c[i] * a[i]))) : 0.0); }
NOINL void mask_div_u8(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) s[i] = (k8[i] ? (1.0 / ((a[i] * b[i]) + c[0])) : 0.0); }
NOINL void mask_div_dbl(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) s[i] = ((kd[i] != 0.0) ? (1.0 / ((a[i] * b[i]) + c[0])) : 0.0); }

// eager forms: every operand loaded, then the select
NOINL void mask_u8_eager(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double v = (a[i] * (b[i] - (c[i] * a[i]))); s[i] = (k8[i] ? v : 0.0); } }
NOINL void mask_dbl_eager(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double v = (a[i] * (b[i] - (c[i] * a[i]))); s[i] = ((kd[i] != 0.0) ? v : 0.0); } }
NOINL void mask_div_u8_eager(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double v = (1.0 / ((a[i] * b[i]) + c[0])); s[i] = (k8[i] ? v : 0.0); } }
NOINL void mask_div_dbl_eager(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double v = (1.0 / ((a[i] * b[i]) + c[0])); s[i] = ((kd[i] != 0.0) ? v : 0.0); } }
NOINL void inv_hoist_eager(const uint8_t* c0, const uint8_t* c1, const double* a, const double* b, const double* c, double* next) { const uint8_t h0 = c0[0], h1 = c1[0]; for (long long j = 0; j < N; ++j) { const double va = a[j], vb = b[j], vc = c[j]; next[9 + j] = (h0 ? va : (h1 ? vb : vc)); } }
NOINL void inv_load_eager(const uint8_t* c0, const uint8_t* c1, const double* a, const double* b, const double* c, double* next) { for (long long j = 0; j < N; ++j) { const double va = a[j], vb = b[j], vc = c[j]; next[9 + j] = (c0[0] ? va : (c1[0] ? vb : vc)); } }
NOINL void inv_local(const uint8_t* c0, const uint8_t* c1, const double* a, const double* b, const double* c, double* next) { double s[N]; const uint8_t h0 = c0[0], h1 = c1[0]; for (long long j = 0; j < N; ++j) { const double va = a[j], vb = b[j], vc = c[j]; s[j] = (h0 ? va : (h1 ? vb : vc)); } for (long long j = 0; j < N; ++j) next[9 + j] = s[j]; }

// loads only: the operands' loads named, the arithmetic left in the branch
NOINL void mask_u8_loads(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double va = a[i], vb = b[i], vc = c[i]; s[i] = (k8[i] ? (va * (vb - (vc * va))) : 0.0); } }
NOINL void mask_div_u8_loads(const double* a, const double* b, const double* c, double* s) { for (long long i = 0; i < N; ++i) { const double va = a[i], vb = b[i], vc = c[0]; s[i] = (k8[i] ? (1.0 / ((va * vb) + vc)) : 0.0); } }
NOINL void and_now(const double* a, uint8_t* s) { for (long long i = 0; i < N; ++i) s[i] = (k8[i] && (a[i] < 2.2204460492503131e-16)); }
NOINL void and_loads(const double* a, uint8_t* s) { for (long long i = 0; i < N; ++i) { const double va = a[i]; s[i] = (k8[i] && (va < 2.2204460492503131e-16)); } }
// 2. invariant select
NOINL void inv_load(const uint8_t* c0, const uint8_t* c1, const double* a, const double* b, const double* c, double* next) { for (long long j = 0; j < N; ++j) next[9 + j] = (c0[0] ? a[j] : (c1[0] ? b[j] : c[j])); }
NOINL void inv_hoist(const uint8_t* c0, const uint8_t* c1, const double* a, const double* b, const double* c, double* next) { const uint8_t h0 = c0[0], h1 = c1[0]; for (long long j = 0; j < N; ++j) next[9 + j] = (h0 ? a[j] : (h1 ? b[j] : c[j])); }
// 3. ratio min reduction, four chains (today) and vector lanes
NOINL double ratio_now(const double* s2, const double* t) {
  double m0, m1, m2, m3;
  { double v = s2[0]; m0 = (v < 0.0) ? (-(t[0] / v)) : 1e30; v = s2[1]; m1 = (v < 0.0) ? (-(t[1] / v)) : 1e30; v = s2[2]; m2 = (v < 0.0) ? (-(t[2] / v)) : 1e30; v = s2[3]; m3 = (v < 0.0) ? (-(t[3] / v)) : 1e30; }
  for (long long k = 4; k < 500; k += 4) {
    double v = s2[k], r = (v < 0.0) ? (-(t[k] / v)) : 1e30; m0 = (r < m0) ? r : m0;
    v = s2[k+1]; r = (v < 0.0) ? (-(t[k+1] / v)) : 1e30; m1 = (r < m1) ? r : m1;
    v = s2[k+2]; r = (v < 0.0) ? (-(t[k+2] / v)) : 1e30; m2 = (r < m2) ? r : m2;
    v = s2[k+3]; r = (v < 0.0) ? (-(t[k+3] / v)) : 1e30; m3 = (r < m3) ? r : m3;
  }
  for (long long k = 500; k < 502; ++k) { double v = s2[k], r = (v < 0.0) ? (-(t[k] / v)) : 1e30; m0 = (r < m0) ? r : m0; }
  m0 = m0 < m1 ? m0 : m1; m2 = m2 < m3 ? m2 : m3; return m0 < m2 ? m0 : m2;
}
NOINL double ratio_two(const double* s2, const double* t, double* tmp) {
  for (long long k = 0; k < N; ++k) { double v = s2[k]; tmp[k] = (v < 0.0) ? (-(t[k] / v)) : 1e30; }
  double m0 = tmp[0], m1 = tmp[1], m2 = tmp[2], m3 = tmp[3];
  for (long long k = 4; k < 500; k += 4) { m0 = tmp[k] < m0 ? tmp[k] : m0; m1 = tmp[k+1] < m1 ? tmp[k+1] : m1; m2 = tmp[k+2] < m2 ? tmp[k+2] : m2; m3 = tmp[k+3] < m3 ? tmp[k+3] : m3; }
  for (long long k = 500; k < 502; ++k) m0 = tmp[k] < m0 ? tmp[k] : m0;
  m0 = m0 < m1 ? m0 : m1; m2 = m2 < m3 ? m2 : m3; return m0 < m2 ? m0 : m2;
}

NOINL double ratio_eager(const double* s2, const double* t) {
  double m0, m1, m2, m3;
  { double v = s2[0]; m0 = (v < 0.0) ? (-(t[0] / v)) : 1e30; v = s2[1]; m1 = (v < 0.0) ? (-(t[1] / v)) : 1e30; v = s2[2]; m2 = (v < 0.0) ? (-(t[2] / v)) : 1e30; v = s2[3]; m3 = (v < 0.0) ? (-(t[3] / v)) : 1e30; }
  for (long long k = 4; k < 500; k += 4) {
    double v = s2[k], e = (-(t[k] / v)), r = (v < 0.0) ? e : 1e30; m0 = (r < m0) ? r : m0;
    v = s2[k+1]; e = (-(t[k+1] / v)); r = (v < 0.0) ? e : 1e30; m1 = (r < m1) ? r : m1;
    v = s2[k+2]; e = (-(t[k+2] / v)); r = (v < 0.0) ? e : 1e30; m2 = (r < m2) ? r : m2;
    v = s2[k+3]; e = (-(t[k+3] / v)); r = (v < 0.0) ? e : 1e30; m3 = (r < m3) ? r : m3;
  }
  for (long long k = 500; k < 502; ++k) { double v = s2[k], e = (-(t[k] / v)), r = (v < 0.0) ? e : 1e30; m0 = (r < m0) ? r : m0; }
  m0 = m0 < m1 ? m0 : m1; m2 = m2 < m3 ? m2 : m3; return m0 < m2 ? m0 : m2;
}
// 4. G^T z: as rendered (rows outer, 8 sums inner) and as 8 dot products with lanes
NOINL void gt_now(const double* G, const double* z, double* o) { for (int j = 0; j < 8; ++j) o[j] = 0.0; for (long long i0 = 0; i0 < N; ++i0) { for (long long i1 = 0; i1 < 8; ++i1) { o[i1] = (o[i1] + (G[(i0 + (N * i1))] * z[i0])); } } }
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));
NOINL void gt_dots(const double* G, const double* z, double* o) { for (long long j = 0; j < 8; ++j) { double2 a0 = {0, 0}, a1 = {0, 0}; const double* g = G + N * j; for (long long i = 0; i < 500; i += 4) { a0 += (*(double2*)(g + i)) * (*(double2*)(z + i)); a1 += (*(double2*)(g + i + 2)) * (*(double2*)(z + i + 2)); } a0 += a1; double s = a0[0] + a0[1]; for (long long i = 500; i < N; ++i) s += g[i] * z[i]; o[j] = s; } }
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e9 + ts.tv_nsec; }
#define TIME(name, call) { double best = 1e30; for (int r = 0; r < 30; ++r) { double t0 = now(); for (int q = 0; q < 20000; ++q) { call; } double t1 = (now() - t0) / 20000; if (t1 < best) best = t1; } printf("%-16s %7.1f ns  (%.3f ns/elem)\n", name, best, best / N); }
int main(void) {
  double* a = malloc(8 * 8 * N); double* b = a + N; double* c = b + N; double* s = c + N; double* G = malloc(8 * 8 * N); double* next = malloc(8 * 4 * N); double o[8]; uint8_t c0[1] = {0}, c1[1] = {0};
  for (int i = 0; i < N; ++i) { a[i] = 0.5 + (i % 7) * 0.25 - ((i % 3 == 0) ? 2.0 : 0.0); b[i] = 1.0 + (i % 5); c[i] = 0.1 * (i % 11) + 0.5; }
  for (int i = 0; i < 8 * N; ++i) G[i] = (i % 13) * 0.1 - 0.6;
  volatile double sink = 0; double t0 = now(); while (now() - t0 < 2e8) { mask_u8(a, b, c, s); sink += s[3]; }
  TIME("mask_u8", mask_u8(a, b, c, s)); TIME("mask_dbl", mask_dbl(a, b, c, s)); TIME("mask_i64", mask_i64(a, b, c, s)); TIME("mask_u8_pragma", mask_u8_pragma(a, b, c, s));
  TIME("mask_u8_eager", mask_u8_eager(a, b, c, s)); TIME("mask_dbl_eager", mask_dbl_eager(a, b, c, s)); TIME("mask_u8_loads", mask_u8_loads(a, b, c, s)); TIME("mask_div_u8_lds", mask_div_u8_loads(a, b, c, s)); TIME("and_now", and_now(a, (uint8_t*)s)); TIME("and_loads", and_loads(a, (uint8_t*)s)); TIME("mask_div_u8", mask_div_u8(a, b, c, s)); TIME("mask_div_u8_eag", mask_div_u8_eager(a, b, c, s)); TIME("mask_div_dbl_eag", mask_div_dbl_eager(a, b, c, s)); TIME("mask_div_dbl", mask_div_dbl(a, b, c, s));
  TIME("inv_load", inv_load(c0, c1, a, b, c, next)); TIME("inv_hoist", inv_hoist(c0, c1, a, b, c, next)); TIME("inv_load_eager", inv_load_eager(c0, c1, a, b, c, next)); TIME("inv_hoist_eager", inv_hoist_eager(c0, c1, a, b, c, next)); TIME("inv_local", inv_local(c0, c1, a, b, c, next));
  TIME("ratio_now", (a[5] += 1e-9, sink += ratio_now(a, b))); TIME("ratio_eager", (a[5] += 1e-9, sink += ratio_eager(a, b))); TIME("ratio_two", (a[5] += 1e-9, sink += ratio_two(a, b, s)));
  TIME("gt_now", gt_now(G, a, o)); double r0 = o[3]; TIME("gt_dots", gt_dots(G, a, o)); printf("gt diff %.2e  ratio %.6g %.6g\n", fabs(r0 - o[3]), ratio_now(a, b), ratio_two(a, b, s));
  return 0;
}
