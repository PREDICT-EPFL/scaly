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

int workspace(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  double* s0 = w + 0;
  double s1[1];
  double s2[1];
  double s3[1];
  double s4[1];
  for (long long i_t0 = 0; i_t0 < 2048; ++i_t0) {
    s0[i_t0] = sin(arg[0][i_t0]);
  }
  s1[0] = 0.0;
  s2[0] = 0.0;
  s3[0] = 0.0;
  s4[0] = 0.0;
  for (long long kb_sum = 0; kb_sum < 2048; kb_sum += 4) {
    s1[0] = (s1[0] + s0[kb_sum]);
    s2[0] = (s2[0] + s0[(kb_sum + 1)]);
    s3[0] = (s3[0] + s0[(kb_sum + 2)]);
    s4[0] = (s4[0] + s0[(kb_sum + 3)]);
  }
  res[0][0] = ((s1[0] + s2[0]) + (s3[0] + s4[0]));
  s1[0] = 0.0;
  s2[0] = 0.0;
  s3[0] = 0.0;
  s4[0] = 0.0;
  for (long long kb_sumsqr = 0; kb_sumsqr < 2048; kb_sumsqr += 4) {
    double v0 = s0[kb_sumsqr];
    s1[0] = (s1[0] + (v0 * v0));
    double v1 = s0[(kb_sumsqr + 1)];
    s2[0] = (s2[0] + (v1 * v1));
    double v2 = s0[(kb_sumsqr + 2)];
    s3[0] = (s3[0] + (v2 * v2));
    double v3 = s0[(kb_sumsqr + 3)];
    s4[0] = (s4[0] + (v3 * v3));
  }
  res[1][0] = ((s1[0] + s2[0]) + (s3[0] + s4[0]));
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
