#pragma once

#include <stddef.h>

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

#ifndef casadi_real
#define casadi_real double
#endif
#ifndef casadi_int
#define casadi_int long long int
#endif

#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#else
#define SCALY_ALIGNAS(n) _Alignas(n)
#endif
#endif

#define shooting_spjac_eq_z_SZ_ARG 2
#define shooting_spjac_eq_z_SZ_RES 1
#define shooting_spjac_eq_z_SZ_IW 0
#define shooting_spjac_eq_z_SZ_W 36

// The pointer ABI for shooting_spjac_eq_z.
#ifdef __cplusplus
extern "C" {
#endif
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem);
casadi_int shooting_spjac_eq_z_n_in(void);
casadi_int shooting_spjac_eq_z_n_out(void);
const char* shooting_spjac_eq_z_name_in(casadi_int i);
const char* shooting_spjac_eq_z_name_out(casadi_int i);
casadi_real shooting_spjac_eq_z_default_in(casadi_int i);
const casadi_int* shooting_spjac_eq_z_sparsity_in(casadi_int i);
const casadi_int* shooting_spjac_eq_z_sparsity_out(casadi_int i);
int shooting_spjac_eq_z_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
int shooting_spjac_eq_z_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
int shooting_spjac_eq_z_checkout(void);
void shooting_spjac_eq_z_release(int mem);
void shooting_spjac_eq_z_incref(void);
void shooting_spjac_eq_z_decref(void);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[16]; } shooting_spjac_eq_z_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[6]; } shooting_spjac_eq_z_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[36]; } shooting_spjac_eq_z_spjac_eq_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[shooting_spjac_eq_z_SZ_W > 0 ? shooting_spjac_eq_z_SZ_W : 1]; } shooting_spjac_eq_z_workspace_t;
static inline int shooting_spjac_eq_z_call(const shooting_spjac_eq_z_z_t* z, const shooting_spjac_eq_z_u_t* u, shooting_spjac_eq_z_spjac_eq_z_t* spjac_eq_z, shooting_spjac_eq_z_workspace_t* workspace) {
  const double* arg[shooting_spjac_eq_z_SZ_ARG > 0 ? shooting_spjac_eq_z_SZ_ARG : 1] = {z->data, u->data};
  double* res[shooting_spjac_eq_z_SZ_RES > 0 ? shooting_spjac_eq_z_SZ_RES : 1] = {spjac_eq_z->data};
  return shooting_spjac_eq_z(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}

// Sparse output metadata for compact derivative buffers.
#define shooting_spjac_eq_z_spjac_eq_z_NNZ 36
#define shooting_spjac_eq_z_spjac_eq_z_NROW 12
#define shooting_spjac_eq_z_spjac_eq_z_NCOL 16
static const int shooting_spjac_eq_z_spjac_eq_z_rows[36] = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
static const int shooting_spjac_eq_z_spjac_eq_z_cols[36] = {0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 14, 15};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_row_ptr[13] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_col_ind[36] = {0, 2, 4, 1, 3, 5, 2, 3, 6, 2, 3, 7, 4, 6, 8, 5, 7, 9, 6, 7, 10, 6, 7, 11, 8, 10, 12, 9, 11, 13, 10, 11, 14, 10, 11, 15};
static const int shooting_spjac_eq_z_spjac_eq_z_csr_val_perm[36] = {0, 2, 8, 1, 5, 10, 3, 6, 12, 4, 7, 16, 9, 13, 20, 11, 17, 22, 14, 18, 24, 15, 19, 28, 21, 25, 32, 23, 29, 33, 26, 30, 34, 27, 31, 35};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_col_ptr[17] = {0, 1, 2, 5, 8, 10, 12, 16, 20, 22, 24, 28, 32, 33, 34, 35, 36};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_row_ind[36] = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_val_perm[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
