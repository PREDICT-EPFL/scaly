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

int dynamics_jac_znext_z_sz_arg(void) { return 1; }
int dynamics_jac_znext_z_sz_res(void) { return 1; }
int dynamics_jac_znext_z_sz_iw(void) { return 0; }
int dynamics_jac_znext_z_sz_w(void) { return 0; }
void* dynamics_jac_znext_z_alloc_mem(void) { return NULL; }
int dynamics_jac_znext_z_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void dynamics_jac_znext_z_free_mem(void* mem) { (void)mem; }

int dynamics_jac_znext_z(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  static const double k0[8] = {1, 0, 0, 1, 0.050000000000000003, 0, 0, 0.050000000000000003};
  static const double k1[8] = {0, 0, 0, 0, 1, 0, 0, 1};
  static const double k2[1] = {0.050000000000000003};
  static const double k3[1] = {0.10000000000000001};
  static const double k4[1] = {2};
  const double* t5 = arg[0] + 2;
  double s0[8];
  static const double k9[2] = {1, 1};
  double s1[16];
  double s2[4];
  double s3[1];
  for (long long j_t6_0 = 0; j_t6_0 < 2; ++j_t6_0) {
    s0[((0 * 2) + j_t6_0)] = t5[j_t6_0];
  }
  for (long long j_t6_1 = 0; j_t6_1 < 2; ++j_t6_1) {
    s0[((1 * 2) + j_t6_1)] = t5[j_t6_1];
  }
  for (long long j_t6_2 = 0; j_t6_2 < 2; ++j_t6_2) {
    s0[((2 * 2) + j_t6_2)] = t5[j_t6_2];
  }
  for (long long j_t6_3 = 0; j_t6_3 < 2; ++j_t6_3) {
    s0[((3 * 2) + j_t6_3)] = t5[j_t6_3];
  }
  for (long long i_t10 = 0; i_t10 < 4; ++i_t10) {
    s1[i_t10] = 0;
    for (long long k_t10 = 0; k_t10 < 2; ++k_t10) {
      s1[i_t10] = (s1[i_t10] + ((k4[0] * (k1[((i_t10 * 2) + k_t10)] * s0[((i_t10 * 2) + k_t10)])) * k9[k_t10]));
    }
  }
  for (long long i_t11 = 0; i_t11 < 4; ++i_t11) {
    s2[i_t11] = (k3[0] * s1[i_t11]);
  }
  s1[0] = 0;
  for (long long i_t14 = 0; i_t14 < 2; ++i_t14) {
    s1[0] = (s1[0] + (t5[i_t14] * t5[i_t14]));
  }
  s3[0] = (k3[0] * s1[0]);
  for (long long j_t21_0 = 0; j_t21_0 < 8; ++j_t21_0) {
    s1[(((j_t21_0 / 2) * 4) + (j_t21_0 % 2))] = k0[j_t21_0];
  }
  for (long long j_t21_1 = 0; j_t21_1 < 8; ++j_t21_1) {
    s1[(((j_t21_1 / 2) * 4) + (2 + (j_t21_1 % 2)))] = (k1[j_t21_1] + (k2[0] * (-((s2[((j_t21_1 / 2) + 0)] * s0[j_t21_1]) + (s3[0] * k1[j_t21_1])))));
  }
  for (long long d0_jac_znext_z = 0; d0_jac_znext_z < 4; ++d0_jac_znext_z) {
    for (long long d1_jac_znext_z = 0; d1_jac_znext_z < 4; ++d1_jac_znext_z) {
      res[0][((d0_jac_znext_z * 4) + d1_jac_znext_z)] = s1[(d0_jac_znext_z + (d1_jac_znext_z * 4))];
    }
  }
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
