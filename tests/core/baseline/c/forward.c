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

int dynamics(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double v0 = arg[0][2];
  double v1 = arg[0][3];
  double v2 = (0.10000000000000001 * ((v0 * v0) + (v1 * v1)));
  *(double2*)(res[0]) = (double2){(arg[0][0] + (0.050000000000000003 * v0)), (arg[0][1] + (0.050000000000000003 * v1))};
  *(double2*)(res[0] + 2) = (double2){(v0 + (0.050000000000000003 * (arg[1][0] - (v2 * v0)))), (v1 + (0.050000000000000003 * (arg[1][1] - (v2 * v1))))};
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
