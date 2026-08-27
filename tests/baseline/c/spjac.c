#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define ALLOY_SUCCESS 0
#define ALLOY_ERR_NULL_ABI 1
#define ALLOY_ERR_NULL_WORK 2
#define ALLOY_ERR_NULL_RESULT 3
#define ALLOY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static __attribute__((noinline)) void dynamics_fwd3c8e1b6ee2b3_znext_z_raw(const double* z, double* fwd_znext_z, double* w) {
  (void)w;
  static const double k0[4] = {1, 1, 0, 0};
  static const double k1[2] = {0.050000000000000003, 0};
  static const double k2[2] = {1, 0};
  static const double k3[1] = {0.050000000000000003};
  static const double k4[1] = {0.10000000000000001};
  static const double k5[1] = {2};
  const double* t6 = z + 2;
  double s0[4];
  double s1[4];
  double s2[1];
  static const double k21[2] = {0, 0.050000000000000003};
  static const double k22[2] = {0, 1};
  double s3[1];
  s0[0] = 0;
  for (long long i_t9 = 0; i_t9 < 2; ++i_t9) {
    s0[0] = (s0[0] + (k5[0] * (k2[i_t9] * t6[i_t9])));
  }
  s1[0] = (k4[0] * s0[0]);
  s0[0] = 0;
  for (long long i_t13 = 0; i_t13 < 2; ++i_t13) {
    s0[0] = (s0[0] + (t6[i_t13] * t6[i_t13]));
  }
  s2[0] = (k4[0] * s0[0]);
  for (long long j_t20_0 = 0; j_t20_0 < 2; ++j_t20_0) {
    s0[j_t20_0] = k1[j_t20_0];
  }
  for (long long j_t20_1 = 0; j_t20_1 < 2; ++j_t20_1) {
    s0[(2 + j_t20_1)] = (k2[j_t20_1] + (k3[0] * (-((s1[0] * t6[j_t20_1]) + (s2[0] * k2[j_t20_1])))));
  }
  s1[0] = 0;
  for (long long i_t25 = 0; i_t25 < 2; ++i_t25) {
    s1[0] = (s1[0] + (k5[0] * (k22[i_t25] * t6[i_t25])));
  }
  s3[0] = (k4[0] * s1[0]);
  for (long long j_t33_0 = 0; j_t33_0 < 2; ++j_t33_0) {
    s1[j_t33_0] = k21[j_t33_0];
  }
  for (long long j_t33_1 = 0; j_t33_1 < 2; ++j_t33_1) {
    s1[(2 + j_t33_1)] = (k22[j_t33_1] + (k3[0] * (-((s3[0] * t6[j_t33_1]) + (s2[0] * k22[j_t33_1])))));
  }
  for (long long j_fwd_znext_z_0 = 0; j_fwd_znext_z_0 < 4; ++j_fwd_znext_z_0) {
    fwd_znext_z[((0 * 4) + j_fwd_znext_z_0)] = k0[j_fwd_znext_z_0];
  }
  for (long long j_fwd_znext_z_1 = 0; j_fwd_znext_z_1 < 4; ++j_fwd_znext_z_1) {
    fwd_znext_z[((1 * 4) + j_fwd_znext_z_1)] = s0[j_fwd_znext_z_1];
  }
  for (long long j_fwd_znext_z_2 = 0; j_fwd_znext_z_2 < 4; ++j_fwd_znext_z_2) {
    fwd_znext_z[((2 * 4) + j_fwd_znext_z_2)] = s1[j_fwd_znext_z_2];
  }
}

int shooting_spjac_eq_z_sz_arg(void) { return 2; }
int shooting_spjac_eq_z_sz_res(void) { return 1; }
int shooting_spjac_eq_z_sz_iw(void) { return 0; }
int shooting_spjac_eq_z_sz_w(void) { return 0; }
void* shooting_spjac_eq_z_alloc_mem(void) { return NULL; }
int shooting_spjac_eq_z_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void shooting_spjac_eq_z_free_mem(void* mem) { (void)mem; }

int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!arg[1]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  static const double k0[48] = {1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double s0[48];
  double s1[12];
  static const double k4[48] = {0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double s2[12];
  static const double k8[48] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0};
  double s3[12];
  static const double k12[48] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  static const int64_t k15[36] = {0, 1, 2, 4, 6, 5, 9, 10, 8, 13, 14, 15, 18, 16, 17, 21, 23, 20, 24, 27, 26, 28, 31, 29, 33, 34, 32, 36, 37, 38, 42, 41, 40, 46, 45, 44};
  for (long long it_t1 = 0; it_t1 < 3; ++it_t1) {
    dynamics_fwd3c8e1b6ee2b3_znext_z_raw((arg[0] + (0 + (4 * it_t1))), (s0 + (it_t1 * 12)), NULL);
  }
  for (long long i_t2 = 0; i_t2 < 12; ++i_t2) {
    s1[i_t2] = s0[((((i_t2 / 4) * 12) + (0 * 4)) + (i_t2 % 4))];
  }
  for (long long i_t5 = 0; i_t5 < 12; ++i_t5) {
    s2[i_t5] = s0[((((i_t5 / 4) * 12) + (1 * 4)) + (i_t5 % 4))];
  }
  for (long long i_t9 = 0; i_t9 < 12; ++i_t9) {
    s3[i_t9] = s0[((((i_t9 / 4) * 12) + (2 * 4)) + (i_t9 % 4))];
  }
  for (long long d0_t14 = 0; d0_t14 < 12; ++d0_t14) {
    for (long long d1_t14 = 0; d1_t14 < 4; ++d1_t14) {
      s0[((d0_t14 * 4) + d1_t14)] = ((((k0[(d0_t14 + (d1_t14 * 12))] * s1[(((0 * 12) + ((((d0_t14 + (d1_t14 * 12)) / 4) % 3) * 4)) + ((d0_t14 + (d1_t14 * 12)) % 4))]) + (k4[(d0_t14 + (d1_t14 * 12))] * s2[(((0 * 12) + ((((d0_t14 + (d1_t14 * 12)) / 4) % 3) * 4)) + ((d0_t14 + (d1_t14 * 12)) % 4))])) + (k8[(d0_t14 + (d1_t14 * 12))] * s3[(((0 * 12) + ((((d0_t14 + (d1_t14 * 12)) / 4) % 3) * 4)) + ((d0_t14 + (d1_t14 * 12)) % 4))])) - k12[(d0_t14 + (d1_t14 * 12))]);
    }
  }
  for (long long i_spjac_eq_z = 0; i_spjac_eq_z < 36; ++i_spjac_eq_z) {
    res[0][i_spjac_eq_z] = s0[k15[i_spjac_eq_z]];
  }
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
