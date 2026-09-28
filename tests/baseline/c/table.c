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

int table(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[8] = {0.5, 1, 0.75, -0.5, 0.125, -0.75, 0.875, 2.25};
  double s0[2];
  static const double k6[5] = {0, 1, 2, 3, 4};
  int64_t s1[1];
  int64_t s2[2];
  double s3[2];
  const double* t12 = s3;
  const double* t19 = s3 + 1;
  static const int64_t k29[2] = {0, 1};
  const double* t32 = s0 + 1;
  static const double k33[4] = {0.5, 1.5, 2.5, 3.5};
  const double* t35 = s3;
  const double* t38 = s0;
  s0[0] = fmin(fmax(floor(arg[0][0]), 0.0), 3.0);
  s1[0] = ((int64_t)s0[0]);
  s2[0] = s1[0];
  s2[1] = (s1[0] + 1);
  for (long long j_t11 = 0; j_t11 < 2; ++j_t11) {
    s3[j_t11] = k6[s2[j_t11]];
  }
  double v0 = s0[0];
  double v1 = arg[0][0];
  s1[0] = ((int64_t)((v0 - ((double)((!(t12[0] <= v1)) && (0.0 < v0)))) + ((double)((t19[0] <= v1) && (v0 < 3.0)))));
  s2[0] = (s1[0] * 2);
  for (long long j_t31 = 0; j_t31 < 2; ++j_t31) {
    s0[j_t31] = k0[(s2[0] + k29[j_t31])];
  }
  s3[0] = k33[s1[0]];
  res[0][0] = ((t32[0] * (arg[0][0] - t35[0])) + t38[0]);
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
