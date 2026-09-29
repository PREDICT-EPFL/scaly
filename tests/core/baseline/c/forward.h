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

#define dynamics_SZ_ARG 2
#define dynamics_SZ_RES 1
#define dynamics_SZ_IW 0
#define dynamics_SZ_W 0

// The pointer ABI for dynamics.
#ifdef __cplusplus
extern "C" {
#endif
int dynamics(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[2]; } dynamics_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_znext_t;
typedef struct { SCALY_ALIGNAS(16) double data[dynamics_SZ_W > 0 ? dynamics_SZ_W : 1]; } dynamics_workspace_t;
static inline int dynamics_call(const dynamics_z_t* z, const dynamics_u_t* u, dynamics_znext_t* znext, dynamics_workspace_t* workspace) {
  const double* arg[dynamics_SZ_ARG > 0 ? dynamics_SZ_ARG : 1] = {z->data, u->data};
  double* res[dynamics_SZ_RES > 0 ? dynamics_SZ_RES : 1] = {znext->data};
  return dynamics(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
