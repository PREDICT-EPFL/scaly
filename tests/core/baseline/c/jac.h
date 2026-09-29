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
#else
#define SCALY_ALIGNAS(n) _Alignas(n)
#endif
#endif

#define dynamics_jac_znext_z_SZ_ARG 2
#define dynamics_jac_znext_z_SZ_RES 1
#define dynamics_jac_znext_z_SZ_IW 0
#define dynamics_jac_znext_z_SZ_W 0

// The pointer ABI for dynamics_jac_znext_z.
#ifdef __cplusplus
extern "C" {
#endif
int dynamics_jac_znext_z(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_jac_znext_z_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[2]; } dynamics_jac_znext_z_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[16]; } dynamics_jac_znext_z_jac_znext_z_t;  // 4 x 4, row-major (C order)
typedef struct { SCALY_ALIGNAS(16) double data[dynamics_jac_znext_z_SZ_W > 0 ? dynamics_jac_znext_z_SZ_W : 1]; } dynamics_jac_znext_z_workspace_t;
static inline int dynamics_jac_znext_z_call(const dynamics_jac_znext_z_z_t* z, const dynamics_jac_znext_z_u_t* u, dynamics_jac_znext_z_jac_znext_z_t* jac_znext_z, dynamics_jac_znext_z_workspace_t* workspace) {
  const double* arg[dynamics_jac_znext_z_SZ_ARG > 0 ? dynamics_jac_znext_z_SZ_ARG : 1] = {z->data, u->data};
  double* res[dynamics_jac_znext_z_SZ_RES > 0 ? dynamics_jac_znext_z_SZ_RES : 1] = {jac_znext_z->data};
  return dynamics_jac_znext_z(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
