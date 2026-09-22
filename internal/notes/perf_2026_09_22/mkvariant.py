import re, sys
src = open('race_car_closed_loop_N200_hess_lower_fast.c').read()
# hoist prologue (arch independent)
hoist = src[src.index('static void race_car_eq_interstage'):src.index('#if defined(__AVX512F__)')]
k8 = src[src.index('static inline __attribute__((always_inline)) void dyn_hess_v8('):src.index('#define VT vd8')]
k8 = k8.replace('vd8', 'VT').replace('_ZGVeN8v_sin', 'VSIN').replace('_ZGVeN8v_cos', 'VCOS').replace('_ZGVeN8v_tanh', 'VTANH')
k8 = re.sub(r'\(VT\)\{\} \+ \((.*)\);$', r'VBC(\1);', k8, flags=re.M).replace('dyn_hess_v8', 'DYN')
main = src[src.index('int race_car_closed_loop_N200_hess_lower('):]
W = int(sys.argv[1]); mode = sys.argv[2]  # mode: lane (per-lane scalar libm), orig (original scalar kernel)
hdr = '#include <math.h>\n#include <stddef.h>\n#include <string.h>\n#include "race_car_closed_loop_N200_hess_lower.h"\n'
hdr += 'typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));\n#define N 200\n'
if W == 1:
  hdr += '#define VT double\n#define VW 1\n#define VBC(x) (x)\n#define VSIN sin\n#define VCOS cos\n#define VTANH tanh\n'
else:
  hdr += f'typedef double VT __attribute__((vector_size({8*W})));\n#define VW {W}\n#define VBC(x) ((VT){{}} + (x))\n'
  for f in ('sin','cos','tanh'):
    hdr += f'static inline VT V{f.upper()}(VT x) {{ VT r; for (int i = 0; i < VW; ++i) r[i] = {f}(x[i]); return r; }}\n'
out = hdr + hoist
if mode == 'orig':
  assert W == 1
  out += 'void race_car_eq_interstage_adj0_0_1_fwd4cb455f2851e_adj_eq_z_hoisted_1_raw(const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, double*, double*);\n'
  out += '''static inline void DYN(double z2, double z3, double z4, double z5, double p1, double p2, double p3, double p4, double p5, double p6, double l0, double l1, double l2, double l3, double t2, double t28, double t69, double t7, double t895_2, double t895_3, double t93, double* o) {
  double zz[6] = {0, 0, z2, z3, z4, z5}; double ll[4] = {l0, l1, l2, l3}; double P[7] = {0, p1, p2, p3, p4, p5, p6}; double t895[4] = {0, 0, t895_2, t895_3};
  race_car_eq_interstage_adj0_0_1_fwd4cb455f2851e_adj_eq_z_hoisted_1_raw(zz, P, ll, &t2, &t28, &t69, &t7, t895, &t93, o, NULL);
}
'''
else:
  out += k8
out += main
open(f'variant_w{W}_{mode}.c', 'w').write(out)
