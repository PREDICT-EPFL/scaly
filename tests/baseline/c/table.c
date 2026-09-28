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
  static const double k1[4] = {((double)NAN), 1, 2, 3};
  int64_t s0[1];
  int64_t s1[1];
  int64_t s2[1];
  static const int64_t k17[2] = {0, 1};
  double s3[2];
  const double* t20 = s3 + 1;
  const double* t26 = s3;
  s0[0] = ((2.0 <= arg[0][0]) ? 2 : 0);
  s1[0] = (s0[0] + 1);
  int64_t v0 = s1[0];
  s2[0] = ((int64_t)((double)((k1[v0] <= arg[0][0]) ? v0 : s0[0])));
  s0[0] = s2[0];
  s1[0] = (s0[0] * 2);
  for (long long j_t19 = 0; j_t19 < 2; ++j_t19) {
    s3[j_t19] = k0[(s1[0] + k17[j_t19])];
  }
  res[0][0] = ((t20[0] * (arg[0][0] - (0.5 + ((double)s2[0])))) + t26[0]);
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
