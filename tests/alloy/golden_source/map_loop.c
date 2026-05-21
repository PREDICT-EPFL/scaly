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

static void stage_raw(const double* in0, double* out0, double* w) {
  (void)w;
  double s0[2];
  double s1[1];
  const double* v0 = in0;
  s0[0] = sin(v0[0]);
  s0[1] = sin(v0[1]);
  s1[0] = 0.0;
  s1[0] += s0[0];
  s1[0] += s0[1];
  out0[0] = s1[0];
}

static void mapped_raw(const double* in0, double* out0, double* w) {
  (void)w;
  double s0[4];
  const double* v0 = in0;
  for (int it = 0; it < 4; ++it) {
    stage_raw(v0 + it * 2, s0 + it, NULL);
  }
  out0[0] = s0[0];
  out0[1] = s0[1];
  out0[2] = s0[2];
  out0[3] = s0[3];
}

int mapped_sz_arg(void) { return 1; }
int mapped_sz_res(void) { return 1; }
int mapped_sz_iw(void) { return 0; }
int mapped_sz_w(void) { return 0; }
void* mapped_alloc_mem(void) { return NULL; }
int mapped_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void mapped_free_mem(void* mem) { (void)mem; }

int mapped(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  mapped_raw(arg[0], res[0], w);
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
