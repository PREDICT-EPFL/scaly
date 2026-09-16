#pragma once

#ifndef SCALY_SUCCESS
#define SCALY_SUCCESS 0
#endif
#ifndef SCALY_ERR_NULL_ABI
#define SCALY_ERR_NULL_ABI 1
#endif
#ifndef SCALY_ERR_NULL_WORK
#define SCALY_ERR_NULL_WORK 2
#endif
#ifndef SCALY_ERR_NULL_RESULT
#define SCALY_ERR_NULL_RESULT 3
#endif
#ifndef SCALY_ERR_NULL_INPUT
#define SCALY_ERR_NULL_INPUT 4
#endif

#define shooting_spjac_eq_z_SZ_ARG 2
#define shooting_spjac_eq_z_SZ_RES 1
#define shooting_spjac_eq_z_SZ_IW 0
#define shooting_spjac_eq_z_SZ_W 0

// Universal CasADi-style ABI for shooting_spjac_eq_z.
#ifdef __cplusplus
extern "C" {
#endif
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, void* mem);
int shooting_spjac_eq_z_sz_arg(void);
int shooting_spjac_eq_z_sz_res(void);
int shooting_spjac_eq_z_sz_iw(void);
int shooting_spjac_eq_z_sz_w(void);
void* shooting_spjac_eq_z_alloc_mem(void);
int shooting_spjac_eq_z_init_mem(void* mem);
void shooting_spjac_eq_z_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[16]; } shooting_spjac_eq_z_z_in;
typedef struct { double data[6]; } shooting_spjac_eq_z_u_in;
typedef struct { double data[36]; } shooting_spjac_eq_z_spjac_eq_z_out;
#ifdef __cplusplus
static_assert(sizeof(shooting_spjac_eq_z_z_in) == sizeof(double) * 16, "shooting_spjac_eq_z_z_in size mismatch");
static_assert(sizeof(shooting_spjac_eq_z_u_in) == sizeof(double) * 6, "shooting_spjac_eq_z_u_in size mismatch");
static_assert(sizeof(shooting_spjac_eq_z_spjac_eq_z_out) == sizeof(double) * 36, "shooting_spjac_eq_z_spjac_eq_z_out size mismatch");
static inline int shooting_spjac_eq_z_call(const shooting_spjac_eq_z_z_in& in_z, const shooting_spjac_eq_z_u_in& in_u, shooting_spjac_eq_z_spjac_eq_z_out& out_spjac_eq_z) {
  double w[shooting_spjac_eq_z_SZ_W > 0 ? shooting_spjac_eq_z_SZ_W : 1];
  const double* arg[shooting_spjac_eq_z_SZ_ARG > 0 ? shooting_spjac_eq_z_SZ_ARG : 1] = {in_z.data, in_u.data};
  double* res[shooting_spjac_eq_z_SZ_RES > 0 ? shooting_spjac_eq_z_SZ_RES : 1] = {out_spjac_eq_z.data};
  return shooting_spjac_eq_z(arg, res, nullptr, shooting_spjac_eq_z_SZ_W ? w : nullptr, nullptr);
}
#endif

// Sparse output metadata for compact derivative buffers.
#define shooting_spjac_eq_z_spjac_eq_z_NNZ 36
#define shooting_spjac_eq_z_spjac_eq_z_NROW 12
#define shooting_spjac_eq_z_spjac_eq_z_NCOL 16
static const int shooting_spjac_eq_z_spjac_eq_z_rows[36] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11};
static const int shooting_spjac_eq_z_spjac_eq_z_cols[36] = {0, 2, 4, 1, 3, 5, 2, 3, 6, 2, 3, 7, 4, 6, 8, 5, 7, 9, 6, 7, 10, 6, 7, 11, 8, 10, 12, 9, 11, 13, 10, 11, 14, 10, 11, 15};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_row_ptr[13] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_col_ind[36] = {0, 2, 4, 1, 3, 5, 2, 3, 6, 2, 3, 7, 4, 6, 8, 5, 7, 9, 6, 7, 10, 6, 7, 11, 8, 10, 12, 9, 11, 13, 10, 11, 14, 10, 11, 15};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_val_perm[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_col_ptr[17] = {0, 1, 2, 5, 8, 10, 12, 16, 20, 22, 24, 28, 32, 33, 34, 35, 36};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_row_ind[36] = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_val_perm[36] = {0, 3, 1, 6, 9, 4, 7, 10, 2, 12, 5, 15, 8, 13, 18, 21, 11, 16, 19, 22, 14, 24, 17, 27, 20, 25, 30, 33, 23, 28, 31, 34, 26, 29, 32, 35};
