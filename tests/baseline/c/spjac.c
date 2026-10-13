/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c shooting_spjac_eq_z.c
 * clang -O3 -fno-math-errno -c shooting_spjac_eq_z.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));
#ifndef SCALY_LANES
#if defined(__AVX512F__)
#define SCALY_LANES 8
#elif defined(__AVX__) || (defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 256)
#define SCALY_LANES 4
#elif defined(__SSE2__) || defined(__aarch64__)
#define SCALY_LANES 2
#else
#define SCALY_LANES 1
#endif
#endif
#if SCALY_LANES != 1 && SCALY_LANES != 2 && SCALY_LANES != 4 && SCALY_LANES != 8
#error "SCALY_LANES must be 1, 2, 4 or 8"
#endif
#if defined(__AVX512F__)
#define SCALY_REGISTER_SLOTS 256
#elif defined(__AVX__) || defined(__aarch64__)
#define SCALY_REGISTER_SLOTS 64
#elif defined(__SSE2__)
#define SCALY_REGISTER_SLOTS 32
#else
#define SCALY_REGISTER_SLOTS 16
#endif
#define SCALY_WIDTH_4_4_4_4 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
typedef double lanes_4_4_4_4_vec __attribute__((vector_size(8 * SCALY_WIDTH_4_4_4_4)));
typedef lanes_4_4_4_4_vec lanes_4_4_4_4_vec_mem __attribute__((aligned(8), may_alias));
static inline __attribute__((always_inline, unused)) lanes_4_4_4_4_vec lanes_4_4_4_4_vec_splat(double x) {
  lanes_4_4_4_4_vec v;
  for (int i = 0; i < SCALY_WIDTH_4_4_4_4; ++i) v[i] = x;
  return v;
}
static inline __attribute__((always_inline, unused)) void lanes_4_4_4_4_vec_st(double* p, lanes_4_4_4_4_vec v, long long n) {
  if (n == SCALY_WIDTH_4_4_4_4) *(lanes_4_4_4_4_vec_mem*)p = v;
  else for (long long i = 0; i < n; ++i) p[i] = v[i];
}

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void shooting_spjac_eq_z_lanes_1(long long v0_chunk, long long n, double* spjac_eq_z, const double* z) {
  int64_t v1[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) v1[v0_lane] = (4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2)));
  double ls[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) ls[v0_lane] = z[(v1[v0_lane] + 2)];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  lanes_4_4_4_4_vec v2 = l;
  double l_2s[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) l_2s[v0_lane] = z[(v1[v0_lane] + 3)];
  lanes_4_4_4_4_vec l_2 = *(const lanes_4_4_4_4_vec_mem*)l_2s;
  lanes_4_4_4_4_vec v4 = l_2;
  lanes_4_4_4_4_vec r = lanes_4_4_4_4_vec_splat(1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2)))) + 0] = r[v0_lane];
  lanes_4_4_4_4_vec r_2 = lanes_4_4_4_4_vec_splat(1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((3 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_2[v0_lane];
  lanes_4_4_4_4_vec r_3 = lanes_4_4_4_4_vec_splat(0.050000000000000003);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((1 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_3[v0_lane];
  lanes_4_4_4_4_vec r_4 = lanes_4_4_4_4_vec_splat(0.050000000000000003);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((4 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_4[v0_lane];
  lanes_4_4_4_4_vec r_5 = lanes_4_4_4_4_vec_splat(-1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((2 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_5[v0_lane];
  lanes_4_4_4_4_vec r_6 = lanes_4_4_4_4_vec_splat(-1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((5 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_6[v0_lane];
  lanes_4_4_4_4_vec r_7 = lanes_4_4_4_4_vec_splat(-1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((8 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_7[v0_lane];
  lanes_4_4_4_4_vec r_8 = lanes_4_4_4_4_vec_splat(-1.0);
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((11 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_8[v0_lane];
  lanes_4_4_4_4_vec v3 = (lanes_4_4_4_4_vec_splat(0.10000000000000001) * (lanes_4_4_4_4_vec_splat(2.0) * v2));
  lanes_4_4_4_4_vec v5 = (lanes_4_4_4_4_vec_splat(0.10000000000000001) * ((v2 * v2) + (v4 * v4)));
  lanes_4_4_4_4_vec r_9 = (lanes_4_4_4_4_vec_splat(1.0) - (lanes_4_4_4_4_vec_splat(0.050000000000000003) * ((v3 * v2) + v5)));
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((6 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_9[v0_lane];
  lanes_4_4_4_4_vec v6 = (lanes_4_4_4_4_vec_splat(0.10000000000000001) * (lanes_4_4_4_4_vec_splat(2.0) * v4));
  lanes_4_4_4_4_vec r_10 = (-(lanes_4_4_4_4_vec_splat(0.050000000000000003) * (v6 * v2)));
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((7 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_10[v0_lane];
  lanes_4_4_4_4_vec r_11 = (-(lanes_4_4_4_4_vec_splat(0.050000000000000003) * (v3 * v4)));
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((9 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_11[v0_lane];
  lanes_4_4_4_4_vec r_12 = (lanes_4_4_4_4_vec_splat(1.0) - (lanes_4_4_4_4_vec_splat(0.050000000000000003) * ((v6 * v4) + v5)));
  for (long long v0_lane = 0; v0_lane < n; ++v0_lane) spjac_eq_z[((10 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = r_12[v0_lane];
}
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  for (long long v0_chunk = 0; v0_chunk < 3 / SCALY_WIDTH_4_4_4_4; ++v0_chunk) {
    shooting_spjac_eq_z_lanes_1(v0_chunk, SCALY_WIDTH_4_4_4_4, res[0], arg[0]);
  }
#if (3 % SCALY_WIDTH_4_4_4_4) != 0
  shooting_spjac_eq_z_lanes_1(3 / SCALY_WIDTH_4_4_4_4, 3 % SCALY_WIDTH_4_4_4_4, res[0], arg[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
