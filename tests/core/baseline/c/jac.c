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

int dynamics_jac_znext_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double v0 = arg[0][2];
  double v1 = (0.10000000000000001 * (2.0 * v0));
  double v2 = arg[0][3];
  double v3 = (0.10000000000000001 * ((v0 * v0) + (v2 * v2)));
  double v4 = (0.10000000000000001 * (2.0 * v2));
  *(double2*)(res[0]) = (double2){1.0, 0.0};
  *(double2*)(res[0] + 2) = (double2){0.050000000000000003, 0.0};
  *(double2*)(res[0] + 4) = (double2){0.0, 1.0};
  *(double2*)(res[0] + 6) = (double2){0.0, 0.050000000000000003};
  *(double2*)(res[0] + 8) = (double2){0.0, 0.0};
  *(double2*)(res[0] + 10) = (double2){(1.0 - (0.050000000000000003 * ((v1 * v0) + v3))), (-(0.050000000000000003 * (v4 * v0)))};
  *(double2*)(res[0] + 12) = (double2){0.0, 0.0};
  *(double2*)(res[0] + 14) = (double2){(-(0.050000000000000003 * (v1 * v2))), (1.0 - (0.050000000000000003 * ((v4 * v2) + v3)))};
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
