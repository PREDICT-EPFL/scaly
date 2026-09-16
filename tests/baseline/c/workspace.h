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

#define workspace_SZ_ARG 1
#define workspace_SZ_RES 2
#define workspace_SZ_IW 0
#define workspace_SZ_W 2048

// Universal CasADi-style ABI for workspace.
#ifdef __cplusplus
extern "C" {
#endif
int workspace(const double** arg, double** res, int* iw, double* w, void* mem);
int workspace_sz_arg(void);
int workspace_sz_res(void);
int workspace_sz_iw(void);
int workspace_sz_w(void);
void* workspace_alloc_mem(void);
int workspace_init_mem(void* mem);
void workspace_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[2048]; } workspace_x_in;
typedef struct { double data[1]; } workspace_sum_out;
typedef struct { double data[1]; } workspace_sumsqr_out;
#ifdef __cplusplus
static_assert(sizeof(workspace_x_in) == sizeof(double) * 2048, "workspace_x_in size mismatch");
static_assert(sizeof(workspace_sum_out) == sizeof(double) * 1, "workspace_sum_out size mismatch");
static_assert(sizeof(workspace_sumsqr_out) == sizeof(double) * 1, "workspace_sumsqr_out size mismatch");
static inline int workspace_call(const workspace_x_in& in_x, workspace_sum_out& out_sum, workspace_sumsqr_out& out_sumsqr) {
  double w[workspace_SZ_W > 0 ? workspace_SZ_W : 1];
  const double* arg[workspace_SZ_ARG > 0 ? workspace_SZ_ARG : 1] = {in_x.data};
  double* res[workspace_SZ_RES > 0 ? workspace_SZ_RES : 1] = {out_sum.data, out_sumsqr.data};
  return workspace(arg, res, nullptr, workspace_SZ_W ? w : nullptr, nullptr);
}
#endif
