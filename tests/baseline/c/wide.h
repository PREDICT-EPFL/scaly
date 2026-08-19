#pragma once

#ifndef ALLOY_SUCCESS
#define ALLOY_SUCCESS 0
#endif
#ifndef ALLOY_ERR_NULL_ABI
#define ALLOY_ERR_NULL_ABI 1
#endif
#ifndef ALLOY_ERR_NULL_WORK
#define ALLOY_ERR_NULL_WORK 2
#endif
#ifndef ALLOY_ERR_NULL_RESULT
#define ALLOY_ERR_NULL_RESULT 3
#endif
#ifndef ALLOY_ERR_NULL_INPUT
#define ALLOY_ERR_NULL_INPUT 4
#endif

#define wide_SZ_ARG 2
#define wide_SZ_RES 2
#define wide_SZ_IW 0
#define wide_SZ_W 1600

// Universal CasADi-style ABI for wide.
#ifdef __cplusplus
extern "C" {
#endif
int wide(const double** arg, double** res, int* iw, double* w, void* mem);
int wide_sz_arg(void);
int wide_sz_res(void);
int wide_sz_iw(void);
int wide_sz_w(void);
void* wide_alloc_mem(void);
int wide_init_mem(void* mem);
void wide_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[40]; } wide_x_in;
typedef struct { double data[40]; } wide_y_in;
typedef struct { double data[40]; } wide_z_out;
typedef struct { double data[4]; } wide_tail_out;
#ifdef __cplusplus
static_assert(sizeof(wide_x_in) == sizeof(double) * 40, "wide_x_in size mismatch");
static_assert(sizeof(wide_y_in) == sizeof(double) * 40, "wide_y_in size mismatch");
static_assert(sizeof(wide_z_out) == sizeof(double) * 40, "wide_z_out size mismatch");
static_assert(sizeof(wide_tail_out) == sizeof(double) * 4, "wide_tail_out size mismatch");
static inline int wide_call(const wide_x_in& in_x, const wide_y_in& in_y, wide_z_out& out_z, wide_tail_out& out_tail) {
  double w[wide_SZ_W > 0 ? wide_SZ_W : 1];
  const double* arg[wide_SZ_ARG > 0 ? wide_SZ_ARG : 1] = {in_x.data, in_y.data};
  double* res[wide_SZ_RES > 0 ? wide_SZ_RES : 1] = {out_z.data, out_tail.data};
  return wide(arg, res, nullptr, wide_SZ_W ? w : nullptr, nullptr);
}
#endif
