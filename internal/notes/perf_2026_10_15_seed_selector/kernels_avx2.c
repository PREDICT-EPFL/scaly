/* Hand-written AVX2 bit-mask selector on padded rows: one byte per 8 entries, bits -> lane masks -> AND. */
#include <immintrin.h>
#include <stdint.h>

static inline __m256d lanes(uint8_t byte, __m256i bits, __m256d cand) {
  __m256i hit = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(byte), bits), bits);  /* all-ones where the bit is set */
  return _mm256_and_pd(_mm256_castsi256_pd(hit), cand);
}

__attribute__((noinline)) void sel_J_avx2_bits(const uint8_t* restrict k0, const uint8_t* restrict k1, const uint8_t* restrict k2,
                                               const uint8_t* restrict k3, const double* restrict c0, const double* restrict c1,
                                               const double* restrict c2, const double* restrict c3, double* restrict out, long nd, long m) {
  const __m256i lo = _mm256_setr_epi64x(1, 2, 4, 8), hi = _mm256_setr_epi64x(16, 32, 64, 128);
  long row = (m + 7) / 8, full = m / 8;
  for (long a = 0; a < nd; ++a) {
    double* o = out + a * m;
    const uint8_t *b0 = k0 + a * row, *b1 = k1 + a * row, *b2 = k2 + a * row, *b3 = k3 + a * row;
    for (long blk = 0; blk < full; ++blk) {
      long j = blk * 8;
      for (int half = 0; half < 2; ++half, j += 4) {
        __m256i bits = half ? hi : lo;
        __m256d s = _mm256_add_pd(_mm256_add_pd(lanes(b0[blk], bits, _mm256_loadu_pd(c0 + j)), lanes(b1[blk], bits, _mm256_loadu_pd(c1 + j))),
                                  _mm256_add_pd(lanes(b2[blk], bits, _mm256_loadu_pd(c2 + j)), lanes(b3[blk], bits, _mm256_loadu_pd(c3 + j))));
        _mm256_storeu_pd(o + j, s);
      }
    }
    for (long j = full * 8; j < m; ++j) {
      int l = j & 7;
      o[j] = (((b0[full] >> l) & 1) ? c0[j] : 0.0) + (((b1[full] >> l) & 1) ? c1[j] : 0.0) + (((b2[full] >> l) & 1) ? c2[j] : 0.0) +
             (((b3[full] >> l) & 1) ? c3[j] : 0.0);
    }
  }
}
