/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c workspace.c
 * clang -O3 -fno-math-errno -c workspace.c
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
#define SCALY_WIDTH_8_8_8_8 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 8 : SCALY_REGISTER_SLOTS == 64 ? 8 : SCALY_REGISTER_SLOTS == 32 ? 8 : 8) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 8 : SCALY_REGISTER_SLOTS == 64 ? 8 : SCALY_REGISTER_SLOTS == 32 ? 8 : 8))
typedef double lanes_8_8_8_8_vec __attribute__((vector_size(8 * SCALY_WIDTH_8_8_8_8)));
typedef lanes_8_8_8_8_vec lanes_8_8_8_8_vec_mem __attribute__((aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) lanes_8_8_8_8_vec workspace_lanes_1_sin(lanes_8_8_8_8_vec x) {
  lanes_8_8_8_8_vec result;
  for (int i = 0; i < SCALY_WIDTH_8_8_8_8; ++i) result[i] = sin(x[i]);
  return result;
}
static inline __attribute__((always_inline)) void workspace_lanes_1(long long i_t0_chunk, long long workspace_lanes_1_valid, double* s0, const double* x) {
  lanes_8_8_8_8_vec workspace_lanes_1_load_1;
  if (workspace_lanes_1_valid == SCALY_WIDTH_8_8_8_8) workspace_lanes_1_load_1 = *(const lanes_8_8_8_8_vec_mem*)(&x[(0 + (((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + 0) < 2047 ? ((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + 0) : 2047))]);
  else for (long long i_t0_lane = 0; i_t0_lane < SCALY_WIDTH_8_8_8_8; ++i_t0_lane) workspace_lanes_1_load_1[i_t0_lane] = x[(0 + (((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + i_t0_lane) < 2047 ? ((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + i_t0_lane) : 2047))];
  lanes_8_8_8_8_vec workspace_lanes_1_store_2 = workspace_lanes_1_sin(workspace_lanes_1_load_1);
  if (workspace_lanes_1_valid == SCALY_WIDTH_8_8_8_8) *(lanes_8_8_8_8_vec_mem*)(&s0[((0 + (((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + 0) < 2047 ? ((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + 0) : 2047))) + 0]) = workspace_lanes_1_store_2;
  else for (long long i_t0_lane = 0; i_t0_lane < workspace_lanes_1_valid; ++i_t0_lane) s0[((0 + (((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + i_t0_lane) < 2047 ? ((i_t0_chunk * SCALY_WIDTH_8_8_8_8) + i_t0_lane) : 2047))) + 0] = workspace_lanes_1_store_2[i_t0_lane];
}
static inline __attribute__((always_inline)) void workspace_lanes_2(long long i_sum_chunk, long long workspace_lanes_2_valid, const double* s0, double* sum) {
  lanes_8_8_8_8_vec workspace_lanes_2_load_1;
  if (workspace_lanes_2_valid == SCALY_WIDTH_8_8_8_8) workspace_lanes_2_load_1 = *(const lanes_8_8_8_8_vec_mem*)(&s0[(0 + (((i_sum_chunk * SCALY_WIDTH_8_8_8_8) + 0) < 2047 ? ((i_sum_chunk * SCALY_WIDTH_8_8_8_8) + 0) : 2047))]);
  else for (long long i_sum_lane = 0; i_sum_lane < SCALY_WIDTH_8_8_8_8; ++i_sum_lane) workspace_lanes_2_load_1[i_sum_lane] = s0[(0 + (((i_sum_chunk * SCALY_WIDTH_8_8_8_8) + i_sum_lane) < 2047 ? ((i_sum_chunk * SCALY_WIDTH_8_8_8_8) + i_sum_lane) : 2047))];
  for (long long i_sum_lane = 0; i_sum_lane < workspace_lanes_2_valid; ++i_sum_lane) sum[0] = (sum[0] + workspace_lanes_2_load_1[i_sum_lane]);
}
int workspace(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  double* s0 = w + 0;
  for (long long i_t0_chunk = 0; i_t0_chunk < 2048 / SCALY_WIDTH_8_8_8_8; ++i_t0_chunk) {
    workspace_lanes_1(i_t0_chunk, SCALY_WIDTH_8_8_8_8, s0, arg[0]);
  }
#if (2048 % SCALY_WIDTH_8_8_8_8) != 0
  workspace_lanes_1(2048 / SCALY_WIDTH_8_8_8_8, 2048 % SCALY_WIDTH_8_8_8_8, s0, arg[0]);
#endif
  res[0][0] = 0.0;
  for (long long i_sum_chunk = 0; i_sum_chunk < 2048 / SCALY_WIDTH_8_8_8_8; ++i_sum_chunk) {
    workspace_lanes_2(i_sum_chunk, SCALY_WIDTH_8_8_8_8, s0, res[0]);
  }
#if (2048 % SCALY_WIDTH_8_8_8_8) != 0
  workspace_lanes_2(2048 / SCALY_WIDTH_8_8_8_8, 2048 % SCALY_WIDTH_8_8_8_8, s0, res[0]);
#endif
  res[1][0] = 0.0;
  for (long long i_sumsqr = 0; i_sumsqr < 2048; ++i_sumsqr) {
    double v0 = s0[i_sumsqr];
    res[1][0] = (res[1][0] + (v0 * v0));
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
