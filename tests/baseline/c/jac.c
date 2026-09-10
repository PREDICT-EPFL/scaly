#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define ALLOY_SUCCESS 0
#define ALLOY_ERR_NULL_ABI 1
#define ALLOY_ERR_NULL_WORK 2
#define ALLOY_ERR_NULL_RESULT 3
#define ALLOY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

int dynamics_jac_znext_z_sz_arg(void) { return 2; }
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
  if (!arg[1]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  static const double k0[8] = {1, 0, 0, 1, 0.050000000000000003, 0, 0, 0.050000000000000003};
  static const double k1[8] = {0, 0, 0, 0, 1, 0, 0, 1};
  const double* t5 = arg[0] + 2;
  double s0[8];
  double s1[8];
  double s2[16];
  double s3[1];
  for (long long i_t6 = 0; i_t6 < 2; ++i_t6) {
    s0[i_t6] = (2.0 * t5[i_t6]);
  }
  for (long long j_t7_0 = 0; j_t7_0 < 2; ++j_t7_0) {
    s1[j_t7_0] = s0[j_t7_0];
  }
  for (long long j_t7_1 = 0; j_t7_1 < 2; ++j_t7_1) {
    s1[(2 + j_t7_1)] = s0[j_t7_1];
  }
  for (long long j_t7_2 = 0; j_t7_2 < 2; ++j_t7_2) {
    s1[(4 + j_t7_2)] = s0[j_t7_2];
  }
  for (long long j_t7_3 = 0; j_t7_3 < 2; ++j_t7_3) {
    s1[(6 + j_t7_3)] = s0[j_t7_3];
  }
  for (long long i_t10 = 0; i_t10 < 4; ++i_t10) {
    s0[i_t10] = 0.0;
  }
  for (long long k_t10 = 0; k_t10 < 2; ++k_t10) {
    s0[(0 * 4)] = (s0[(0 * 4)] + (s1[(((0 * 4) * 2) + k_t10)] * k1[(((0 * 4) * 2) + k_t10)]));
    s0[((0 * 4) + 1)] = (s0[((0 * 4) + 1)] + (s1[((((0 * 4) + 1) * 2) + k_t10)] * k1[((((0 * 4) + 1) * 2) + k_t10)]));
    s0[((0 * 4) + 2)] = (s0[((0 * 4) + 2)] + (s1[((((0 * 4) + 2) * 2) + k_t10)] * k1[((((0 * 4) + 2) * 2) + k_t10)]));
    s0[((0 * 4) + 3)] = (s0[((0 * 4) + 3)] + (s1[((((0 * 4) + 3) * 2) + k_t10)] * k1[((((0 * 4) + 3) * 2) + k_t10)]));
  }
  for (long long i_t11 = 0; i_t11 < 4; ++i_t11) {
    s1[i_t11] = (0.10000000000000001 * s0[i_t11]);
  }
  for (long long j_t12_0 = 0; j_t12_0 < 2; ++j_t12_0) {
    s0[j_t12_0] = t5[j_t12_0];
  }
  for (long long j_t12_1 = 0; j_t12_1 < 2; ++j_t12_1) {
    s0[(2 + j_t12_1)] = t5[j_t12_1];
  }
  for (long long j_t12_2 = 0; j_t12_2 < 2; ++j_t12_2) {
    s0[(4 + j_t12_2)] = t5[j_t12_2];
  }
  for (long long j_t12_3 = 0; j_t12_3 < 2; ++j_t12_3) {
    s0[(6 + j_t12_3)] = t5[j_t12_3];
  }
  s2[0] = 0.0;
  for (long long i_t15 = 0; i_t15 < 2; ++i_t15) {
    s2[0] = (s2[0] + (t5[i_t15] * t5[i_t15]));
  }
  s3[0] = (0.10000000000000001 * s2[0]);
  for (long long j_t21_0 = 0; j_t21_0 < 8; ++j_t21_0) {
    s2[(((j_t21_0 / 2) * 4) + (j_t21_0 % 2))] = k0[j_t21_0];
  }
  for (long long j_t21_1 = 0; j_t21_1 < 8; ++j_t21_1) {
    s2[(((j_t21_1 / 2) * 4) + (2 + (j_t21_1 % 2)))] = (k1[j_t21_1] - (0.050000000000000003 * ((s1[(j_t21_1 / 2)] * s0[j_t21_1]) + (s3[0] * k1[j_t21_1]))));
  }
  for (long long d0_jac_znext_z = 0; d0_jac_znext_z < 4; ++d0_jac_znext_z) {
    for (long long d1_jac_znext_z = 0; d1_jac_znext_z < 4; ++d1_jac_znext_z) {
      res[0][((d0_jac_znext_z * 4) + d1_jac_znext_z)] = s2[(d0_jac_znext_z + (d1_jac_znext_z * 4))];
    }
  }
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
