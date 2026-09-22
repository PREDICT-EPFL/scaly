#include <math.h>
typedef double v2 __attribute__((vector_size(16)));
typedef double v4 __attribute__((vector_size(32)));
v2 ew_sin2(v2 x)  { return __builtin_elementwise_sin(x); }
v2 ew_exp2(v2 x)  { return __builtin_elementwise_exp(x); }
v2 ew_fma2(v2 a, v2 b, v2 c) { return __builtin_elementwise_fma(a, b, c); }
v2 ew_max2(v2 a, v2 b) { return __builtin_elementwise_max(a, b); }
v2 ew_sqrt2(v2 x) { return __builtin_elementwise_sqrt(x); }
v4 ew_sin4(v4 x)  { return __builtin_elementwise_sin(x); }
v4 ew_arith4(v4 a, v4 b) { return a * b + a / b; }
