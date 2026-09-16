#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

int workspace_sz_arg(void) { return 1; }
int workspace_sz_res(void) { return 2; }
int workspace_sz_iw(void) { return 0; }
int workspace_sz_w(void) { return 2048; }
void* workspace_alloc_mem(void) { return NULL; }
int workspace_init_mem(void* mem) { (void)mem; return SCALY_SUCCESS; }
void workspace_free_mem(void* mem) { (void)mem; }

int workspace(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  double* s0 = w + 0;
  for (long long i_t0 = 0; i_t0 < 2048; ++i_t0) {
    s0[i_t0] = sin(arg[0][i_t0]);
  }
  res[0][0] = 0.0;
  for (long long i_sum = 0; i_sum < 2048; ++i_sum) {
    res[0][0] = (res[0][0] + s0[i_sum]);
  }
  res[1][0] = 0.0;
  for (long long i_sumsqr = 0; i_sumsqr < 2048; ++i_sumsqr) {
    double v0 = s0[i_sumsqr];
    res[1][0] = (res[1][0] + (v0 * v0));
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
