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

#define dynamics_jac_znext_z_SZ_ARG 2
#define dynamics_jac_znext_z_SZ_RES 1
#define dynamics_jac_znext_z_SZ_IW 0
#define dynamics_jac_znext_z_SZ_W 0

// Universal CasADi-style ABI for dynamics_jac_znext_z.
#ifdef __cplusplus
extern "C" {
#endif
int dynamics_jac_znext_z(const double** arg, double** res, int* iw, double* w, void* mem);
int dynamics_jac_znext_z_sz_arg(void);
int dynamics_jac_znext_z_sz_res(void);
int dynamics_jac_znext_z_sz_iw(void);
int dynamics_jac_znext_z_sz_w(void);
void* dynamics_jac_znext_z_alloc_mem(void);
int dynamics_jac_znext_z_init_mem(void* mem);
void dynamics_jac_znext_z_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[4]; } dynamics_jac_znext_z_z_in;
typedef struct { double data[2]; } dynamics_jac_znext_z_u_in;
typedef struct { double data[16]; } dynamics_jac_znext_z_jac_znext_z_out;
#ifdef __cplusplus
static_assert(sizeof(dynamics_jac_znext_z_z_in) == sizeof(double) * 4, "dynamics_jac_znext_z_z_in size mismatch");
static_assert(sizeof(dynamics_jac_znext_z_u_in) == sizeof(double) * 2, "dynamics_jac_znext_z_u_in size mismatch");
static_assert(sizeof(dynamics_jac_znext_z_jac_znext_z_out) == sizeof(double) * 16, "dynamics_jac_znext_z_jac_znext_z_out size mismatch");
static inline int dynamics_jac_znext_z_call(const dynamics_jac_znext_z_z_in& in_z, const dynamics_jac_znext_z_u_in& in_u, dynamics_jac_znext_z_jac_znext_z_out& out_jac_znext_z) {
  double w[dynamics_jac_znext_z_SZ_W > 0 ? dynamics_jac_znext_z_SZ_W : 1];
  const double* arg[dynamics_jac_znext_z_SZ_ARG > 0 ? dynamics_jac_znext_z_SZ_ARG : 1] = {in_z.data, in_u.data};
  double* res[dynamics_jac_znext_z_SZ_RES > 0 ? dynamics_jac_znext_z_SZ_RES : 1] = {out_jac_znext_z.data};
  return dynamics_jac_znext_z(arg, res, nullptr, dynamics_jac_znext_z_SZ_W ? w : nullptr, nullptr);
}
#endif
