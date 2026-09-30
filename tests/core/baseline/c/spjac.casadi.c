#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifndef casadi_real
#define casadi_real double
#endif
#ifndef casadi_int
#define casadi_int long long int
#endif

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
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double* spjac_eq_z_native = w + 0;
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
    spjac_eq_z_native[i_spjac_eq_z] = ((((k0[v1] * s1[v2]) + (k4[v1] * s2[v2])) + (k8[v1] * s3[v2])) - k12[v1]);
  }
  static const int spjac_eq_z_csc_val_perm[36] = {0, 3, 1, 6, 9, 4, 7, 10, 2, 12, 5, 15, 8, 13, 18, 21, 11, 16, 19, 22, 14, 24, 17, 27, 20, 25, 30, 33, 23, 28, 31, 34, 26, 29, 32, 35};
  for (int k = 0; k < 36; ++k) res[0][k] = spjac_eq_z_native[spjac_eq_z_csc_val_perm[k]];
  return SCALY_SUCCESS;
}

static const casadi_int shooting_spjac_eq_z_sin0[3] = {16, 1, 1};
static const casadi_int shooting_spjac_eq_z_sin1[3] = {6, 1, 1};
static const casadi_int shooting_spjac_eq_z_sout0[55] = {12, 16, 0, 1, 2, 5, 8, 10, 12, 16, 20, 22, 24, 28, 32, 33, 34, 35, 36, 0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};

casadi_int shooting_spjac_eq_z_n_in(void) { return 2; }
casadi_int shooting_spjac_eq_z_n_out(void) { return 1; }
const char* shooting_spjac_eq_z_name_in(casadi_int i) {
  switch (i) {
    case 0: return "z";
    case 1: return "u";
    default: return 0;
  }
}
const char* shooting_spjac_eq_z_name_out(casadi_int i) {
  switch (i) {
    case 0: return "spjac_eq_z";
    default: return 0;
  }
}
casadi_real shooting_spjac_eq_z_default_in(casadi_int i) { (void)i; return 0; }
const casadi_int* shooting_spjac_eq_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return shooting_spjac_eq_z_sin0;
    case 1: return shooting_spjac_eq_z_sin1;
    default: return 0;
  }
}
const casadi_int* shooting_spjac_eq_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return shooting_spjac_eq_z_sout0;
    default: return 0;
  }
}
int shooting_spjac_eq_z_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 36;
  return 0;
}
int shooting_spjac_eq_z_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w) {
  if (sz_arg) *sz_arg = 2 * sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1 * sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 36 * sizeof(casadi_real);
  return 0;
}
int shooting_spjac_eq_z_checkout(void) { return 0; }
void shooting_spjac_eq_z_release(int mem) { (void)mem; }
void shooting_spjac_eq_z_incref(void) {}
void shooting_spjac_eq_z_decref(void) {}

#ifdef __cplusplus
}
#endif
