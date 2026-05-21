#include <math.h>
#include <stddef.h>

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

#ifdef __cplusplus
extern "C" {
#endif

static void vec_dot_raw(const double* in0, const double* in1, double* out0, double* w) {
  (void)w;
  double s0[5];
  double s1[1];
  const double* v0 = in0;
  const double* v1;
  const double* v2 = in1;
  const double* v3;
  v1 = v0;
  v3 = v2;
  s0[0] = v1[0] * v3[0];
  s0[1] = v1[1] * v3[1];
  s0[2] = v1[2] * v3[2];
  s0[3] = v1[3] * v3[3];
  s0[4] = v1[4] * v3[4];
  s1[0] = 0.0;
  s1[0] += s0[0];
  s1[0] += s0[1];
  s1[0] += s0[2];
  s1[0] += s0[3];
  s1[0] += s0[4];
  out0[0] = s1[0];
}

int vec_dot_sz_arg(void) { return 2; }
int vec_dot_sz_res(void) { return 1; }
int vec_dot_sz_iw(void) { return 0; }
int vec_dot_sz_w(void) { return 0; }
void* vec_dot_alloc_mem(void) { return NULL; }
int vec_dot_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void vec_dot_free_mem(void* mem) { (void)mem; }

int vec_dot(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!arg[1]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  vec_dot_raw(arg[0], arg[1], res[0], w);
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
