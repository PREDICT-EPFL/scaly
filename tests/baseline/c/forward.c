/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics.c
 * clang -O3 -fno-math-errno -c dynamics.c
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

int dynamics(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  const double* t0 = arg[0];
  const double* t2 = arg[0] + 2;
  double s0[1];
  double s1[1];
  s0[0] = 0.0;
  for (long long i_t7 = 0; i_t7 < 2; ++i_t7) {
    double v0 = t2[i_t7];
    s0[0] = (s0[0] + (v0 * v0));
  }
  s1[0] = (0.10000000000000001 * s0[0]);
  for (long long j_znext_0 = 0; j_znext_0 < 2; ++j_znext_0) {
    res[0][j_znext_0] = (t0[j_znext_0] + (0.050000000000000003 * t2[j_znext_0]));
  }
  for (long long j_znext_1 = 0; j_znext_1 < 2; ++j_znext_1) {
    double v1 = t2[j_znext_1];
    res[0][(2 + j_znext_1)] = (v1 + (0.050000000000000003 * (arg[1][j_znext_1] - (s1[0] * v1))));
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
