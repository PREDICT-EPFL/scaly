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

static __attribute__((noinline)) void dynamics_fwd3c8e1b6ee2b3_znext_z_raw(const double* z, double* fwd_znext_z, double* w) {
  (void)w;
  double v0 = z[2];
  double v1 = (0.10000000000000001 * (2.0 * v0));
  double v2 = z[3];
  double v3 = (0.10000000000000001 * ((v0 * v0) + (v2 * v2)));
  double v4 = (0.10000000000000001 * (2.0 * v2));
  *(double2*)(fwd_znext_z) = (double2){1.0, 1.0};
  *(double2*)(fwd_znext_z + 2) = (double2){0.0, 0.0};
  *(double2*)(fwd_znext_z + 4) = (double2){0.050000000000000003, 0.0};
  *(double2*)(fwd_znext_z + 6) = (double2){(1.0 - (0.050000000000000003 * ((v1 * v0) + v3))), (-(0.050000000000000003 * (v1 * v2)))};
  *(double2*)(fwd_znext_z + 8) = (double2){0.0, 0.050000000000000003};
  *(double2*)(fwd_znext_z + 10) = (double2){(-(0.050000000000000003 * (v4 * v0))), (1.0 - (0.050000000000000003 * ((v4 * v2) + v3)))};
}

int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[48] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double s0[36];
  double s1[12];
  static const double k4[48] = {0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double s2[12];
  static const double k8[48] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0};
  double s3[12];
  static const double k12[48] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  static const int64_t k15[36] = {0, 1, 2, 4, 6, 5, 9, 10, 8, 13, 14, 15, 18, 16, 17, 21, 23, 20, 24, 27, 26, 28, 31, 29, 33, 34, 32, 36, 37, 38, 42, 41, 40, 46, 45, 44};
  for (long long it_t1 = 0; it_t1 < 3; ++it_t1) {
    dynamics_fwd3c8e1b6ee2b3_znext_z_raw((arg[0] + (4 * it_t1)), (s0 + (it_t1 * 12)), NULL);
  }
  for (long long i_t2_0 = 0; i_t2_0 < 3; ++i_t2_0) {
    for (long long i_t2_1 = 0; i_t2_1 < 4; ++i_t2_1) {
      s1[((4 * i_t2_0) + i_t2_1)] = s0[((12 * i_t2_0) + i_t2_1)];
    }
  }
  for (long long i_t5_0 = 0; i_t5_0 < 3; ++i_t5_0) {
    for (long long i_t5_1 = 0; i_t5_1 < 4; ++i_t5_1) {
      s2[((4 * i_t5_0) + i_t5_1)] = s0[((4 + (12 * i_t5_0)) + i_t5_1)];
    }
  }
  for (long long i_t9_0 = 0; i_t9_0 < 3; ++i_t9_0) {
    for (long long i_t9_1 = 0; i_t9_1 < 4; ++i_t9_1) {
      s3[((4 * i_t9_0) + i_t9_1)] = s0[((8 + (12 * i_t9_0)) + i_t9_1)];
    }
  }
  for (long long i_spjac_eq_z = 0; i_spjac_eq_z < 36; ++i_spjac_eq_z) {
    int64_t v0 = k15[i_spjac_eq_z];
    int64_t v1 = ((v0 / 4) + ((v0 % 4) * 12));
    int64_t v2 = ((((v1 / 4) % 3) * 4) + (v1 % 4));
    res[0][i_spjac_eq_z] = ((((k0[v1] * s1[v2]) + (k4[v1] * s2[v2])) + (k8[v1] * s3[v2])) - k12[v1]);
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
