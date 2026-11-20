#ifndef SCALY_EMBEDDED_MATH_H
#define SCALY_EMBEDDED_MATH_H

float sinf(float);
float cosf(float);
float sqrtf(float);
float fabsf(float);

/* A macro of C99's <math.h>, which a board toolchain's header defines through the compiler, as here. */
#ifndef isfinite
#define isfinite(x) __builtin_isfinite(x)
#endif

#endif
