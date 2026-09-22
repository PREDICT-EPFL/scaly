typedef double v2 __attribute__((vector_size(16)));
typedef double v4 __attribute__((vector_size(32)));
typedef long long i2 __attribute__((vector_size(16)));
v2 f_shuf(v4 x) { return __builtin_shufflevector(x, x, 1, 3); }
v2 f_cvt(i2 i) { return __builtin_convertvector(i, v2); }
v2 f_idx(v2 x) { x[0] = x[1] * 2.0; return x; }
v2 f_sel(v2 a, v2 b) { return a > b ? a : b; }
