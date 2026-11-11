/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c shooting_spjac_eq_z.c
 * clang -O3 -fno-math-errno -c shooting_spjac_eq_z.c
 * Link with: -lm
 */
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

#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SCALY_ALIGNAS(n) _Alignas(n)
#else
#define SCALY_ALIGNAS(n)
#endif
#endif

#define shooting_spjac_eq_z_SZ_ARG 2
#define shooting_spjac_eq_z_SZ_RES 1
#define shooting_spjac_eq_z_SZ_IW 0
#define shooting_spjac_eq_z_SZ_W 0
#define shooting_spjac_eq_z_FRAME_BYTES 4768

// The pointer ABI for shooting_spjac_eq_z.
#ifdef __cplusplus
extern "C" {
#endif
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem);
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

// Sparse leaf metadata: one local pattern shared by all batch entries.
#define shooting_spjac_eq_z_spjac_eq_z_NROW 12
#define shooting_spjac_eq_z_spjac_eq_z_NCOL 16
#define shooting_spjac_eq_z_spjac_eq_z_NNZ 36
#define shooting_spjac_eq_z_spjac_eq_z_NDIM 2
#define shooting_spjac_eq_z_spjac_eq_z_STORAGE_SIZE 36
static const int shooting_spjac_eq_z_spjac_eq_z_shape[2] = {12, 16};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_col_ptr[17] = {0, 1, 2, 5, 8, 10, 12, 16, 20, 22, 24, 28, 32, 33, 34, 35, 36};
static const int shooting_spjac_eq_z_spjac_eq_z_csc_row_ind[36] = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
