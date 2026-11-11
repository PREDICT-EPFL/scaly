/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics_float32.c
 * clang -O3 -fno-math-errno -c dynamics_float32.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

int dynamics_float32(const float** arg, float** res, int* iw, float* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  const float* t0 = arg[0];
  const float* t2 = arg[0] + 2;
  float s0[1];
  float s1[1];
  float acc_t7 = 0.0f;
  for (long long i_t7 = 0; i_t7 < 2; ++i_t7) {
    float v0 = t2[i_t7];
    acc_t7 = (acc_t7 + (v0 * v0));
  }
  s0[0] = acc_t7;
  s1[0] = (0.100000001f * s0[0]);
  for (long long j_znext_0 = 0; j_znext_0 < 2; ++j_znext_0) {
    res[0][j_znext_0] = (t0[j_znext_0] + (0.0500000007f * t2[j_znext_0]));
  }
  for (long long j_znext_1 = 0; j_znext_1 < 2; ++j_znext_1) {
    float v1 = t2[j_znext_1];
    res[0][(2 + j_znext_1)] = (v1 + (0.0500000007f * (arg[1][j_znext_1] - (s1[0] * v1))));
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
