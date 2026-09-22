#!/bin/bash
# C-81: build and time every 2026-09-22 variant on the x86 reference machine. Run from a directory
# holding the generated race_car_closed_loop_N200_hess_lower.{c,h} (gen_hess_kernel.py), the
# hand-written race_car_closed_loop_N200_hess_lower_fast.c and the mkvariant.py outputs
# variant_w{1,2,4,8}_lane.c and variant_w1_orig.c. Usage: x86_variants.sh <repo> [core]
set -u
REPO=$1; CORE=${2:-3}; R=$REPO/internal/notes/perf_2026_09_22
F="-O3 -march=native -fno-math-errno -I. -DFN=race_car_closed_loop_N200_hess_lower"
zig() { uvx --from ziglang python-zig cc "$@"; }
try() { CC=$1; label=$2; shift 2
  $CC $F -o bench_$label "$@" $R/bench_hess.c -lm 2>err_$label.txt || { echo "$label: FAIL"; head -3 err_$label.txt; return; }
  printf "%-28s %-6s " $label $CC; taskset -c $CORE ./bench_$label 0 out_$label.bin | tr -d '\n'
  printf "  %-45s " "$(nm -u bench_$label | grep -o '_ZGV[a-zA-Z0-9_]*' | sort -u | tr '\n' ' ')"
  [ -f out_baseline_gcc.bin ] && uv run --project $REPO python -c "
import sys, numpy as np; a=np.fromfile('out_baseline_gcc.bin'); b=np.fromfile('out_$label.bin'); d=np.abs(a-b)
print(f'maxabs={d.max():.1e} at {d.argmax()}')" 2>/dev/null || echo; }
# original stage kernel out of line, for `mode orig` and kern_only.c
{ printf '#include <math.h>\ntypedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));\n'
  awk '/hoisted_1_raw\(const double\* z/,/^}/' race_car_closed_loop_N200_hess_lower.c | sed '1s/^static __attribute__((noinline)) //'; } > orig_kernel.c
sed 's/__attribute__((noinline)) //' race_car_closed_loop_N200_hess_lower.c > baseline_inl.c
# vector libm variants: explicit glibc prototypes (any compiler), clang builtins, gcc simd attribute
for w in 4 8; do p=$([ $w = 4 ] && echo d || echo e)
  sed -e "/^static inline VT VSIN/,/^static inline VT VTANH.*$/d" -e "s|^#define VW $w\$|#define VW $w\nVT _ZGV${p}N${w}v_sin(VT); VT _ZGV${p}N${w}v_cos(VT); VT _ZGV${p}N${w}v_tanh(VT);\n#define VSIN _ZGV${p}N${w}v_sin\n#define VCOS _ZGV${p}N${w}v_cos\n#define VTANH _ZGV${p}N${w}v_tanh|" variant_w${w}_lane.c > variant_w${w}_mvec.c
  sed -e "/^static inline VT VSIN/,/^static inline VT VTANH.*$/d" -e "s|^#define VW $w\$|#define VW $w\n#define VSIN __builtin_elementwise_sin\n#define VCOS __builtin_elementwise_cos\n#define VTANH __builtin_elementwise_tanh|" variant_w${w}_lane.c > variant_w${w}_ew.c
done
sed 's|^#include <math.h>|#include <math.h>\n__attribute__((__simd__("notinbranch"))) double sin(double);\n__attribute__((__simd__("notinbranch"))) double cos(double);\n__attribute__((__simd__("notinbranch"))) double tanh(double);|' variant_w1_lane.c > variant_w1_simddecl.c
for CC in gcc clang zig; do
  try $CC baseline_$CC race_car_closed_loop_N200_hess_lower.c
  try $CC baseline_inl_$CC baseline_inl.c
  try $CC fused_origkern_$CC variant_w1_orig.c orig_kernel.c
  for w in 1 2 4 8; do try $CC w${w}_lane_$CC variant_w${w}_lane.c; done
  try $CC w1_lane_fastmath_$CC variant_w1_lane.c -ffast-math
  for w in 4 8; do try $CC w${w}_mvec_$CC variant_w${w}_mvec.c -lmvec; done
  try $CC w8_mvec_fastmath_$CC variant_w8_mvec.c -ffast-math -lmvec
  try $CC hand_fast_$CC race_car_closed_loop_N200_hess_lower_fast.c -lmvec
done
try gcc w1_simddecl_gcc variant_w1_simddecl.c
try gcc w1_simddecl_notrap_gcc variant_w1_simddecl.c -fno-trapping-math
try clang w1_simddecl_clang variant_w1_simddecl.c
for CC in clang zig; do
  for w in 4 8; do try $CC w${w}_ew_veclib_$CC variant_w${w}_ew.c -fveclib=libmvec -lmvec; try $CC w${w}_ew_$CC variant_w${w}_ew.c; done
  try $CC w1_veclib_$CC variant_w1_lane.c -fveclib=libmvec -lmvec
  try $CC w1_veclib_256_$CC variant_w1_lane.c -fveclib=libmvec -mprefer-vector-width=256 -lmvec
done
for CC in gcc clang; do
  $CC -O3 -march=native -fno-math-errno -o kern_$CC $R/kern_only.c orig_kernel.c -lm && printf "$CC: " && taskset -c $CORE ./kern_$CC
  $CC -O3 -march=native -fno-math-errno -o trig_$CC $R/trigcost.c -lm && printf "$CC: " && taskset -c $CORE ./trig_$CC
done
ldd bench_hand_fast_zig | grep mvec
