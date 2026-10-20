/* The npmpc selector loop in each mask format: out[a][j] = sum_l mask_l[a][j] * cand_l[j]. */
#include <stdint.h>

#define RESTRICT __restrict
#define BIT(k, i) (((k)[(i) >> 3] >> ((i) & 7)) & 1)

/* As generated today: one flat loop, index rebuilt with / and %, no restrict. */
__attribute__((noinline)) void sel_flat(const double* k0, const double* k1, const double* k2, const double* k3, const double* c0, const double* c1,
                                        const double* c2, const double* c3, double* out, long nd, long stages) {
  for (long i = 0; i < nd * stages * 9; ++i) {
    long v = ((i / 9) % stages) * 9 + i % 9;
    out[i] = k0[i] * c0[v] + k1[i] * c1[v] + k2[i] * c2[v] + k3[i] * c3[v];
  }
}

#define CONTIG(name, MASK_T, TERM)                                                                                                            \
  __attribute__((noinline)) void name(const MASK_T* RESTRICT k0, const MASK_T* RESTRICT k1, const MASK_T* RESTRICT k2, const MASK_T* RESTRICT k3, \
                                      const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2, const double* RESTRICT c3, \
                                      double* RESTRICT out, long nd, long m) {                                                                 \
    for (long a = 0; a < nd; ++a)                                                                                                              \
      for (long j = 0; j < m; ++j) {                                                                                                           \
        long i = a * m + j;                                                                                                                    \
        out[i] = TERM(k0, c0) + TERM(k1, c1) + TERM(k2, c2) + TERM(k3, c3);                                                                    \
      }                                                                                                                                        \
  }

#define MUL_DOUBLE(k, c) ((k)[i] * (c)[j])
#define SEL_BYTE(k, c) ((k)[i] ? (c)[j] : 0.0)
#define SEL_BIT(k, c) (BIT(k, i) ? (c)[j] : 0.0)
#define CAST_BYTE(k, c) ((double)(k)[i] * (c)[j])
#define CAST_BIT(k, c) ((double)BIT(k, i) * (c)[j])

CONTIG(sel_A_double, double, MUL_DOUBLE)
CONTIG(sel_B_byte_select, uint8_t, SEL_BYTE)
CONTIG(sel_C_bit_select, uint8_t, SEL_BIT)
CONTIG(sel_D_byte_cast, uint8_t, CAST_BYTE)
CONTIG(sel_E_bit_cast, uint8_t, CAST_BIT)

/* The same bit variants with vectorization forced. */
#define CONTIG_FORCED(name, MASK_T, TERM)                                                                                                     \
  __attribute__((noinline)) void name(const MASK_T* RESTRICT k0, const MASK_T* RESTRICT k1, const MASK_T* RESTRICT k2, const MASK_T* RESTRICT k3, \
                                      const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2, const double* RESTRICT c3, \
                                      double* RESTRICT out, long nd, long m) {                                                                 \
    for (long a = 0; a < nd; ++a) {                                                                                                            \
      _Pragma("clang loop vectorize(enable)") for (long j = 0; j < m; ++j) {                                                                    \
        long i = a * m + j;                                                                                                                    \
        out[i] = TERM(k0, c0) + TERM(k1, c1) + TERM(k2, c2) + TERM(k3, c3);                                                                    \
      }                                                                                                                                        \
    }                                                                                                                                          \
  }
CONTIG_FORCED(sel_C2_bit_select_forced, uint8_t, SEL_BIT)
CONTIG_FORCED(sel_E2_bit_cast_forced, uint8_t, CAST_BIT)

/* Bits with each direction's row padded to whole bytes: entries 8b..8b+7 of a row share byte b, bit = lane. */
#define PADDED(name, TERM)                                                                                                                    \
  __attribute__((noinline)) void name(const uint8_t* RESTRICT k0, const uint8_t* RESTRICT k1, const uint8_t* RESTRICT k2, const uint8_t* RESTRICT k3, \
                                      const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2, const double* RESTRICT c3, \
                                      double* RESTRICT out, long nd, long m) {                                                                 \
    long row = (m + 7) / 8;                                                                                                                    \
    for (long a = 0; a < nd; ++a) {                                                                                                            \
      const uint8_t *b0 = k0 + a * row, *b1 = k1 + a * row, *b2 = k2 + a * row, *b3 = k3 + a * row;                                           \
      double* o = out + a * m;                                                                                                                 \
      _Pragma("clang loop vectorize(enable)") for (long j = 0; j < m; ++j) o[j] = TERM(b0, c0) + TERM(b1, c1) + TERM(b2, c2) + TERM(b3, c3);   \
    }                                                                                                                                          \
  }
#define PBIT(b) (((b)[j >> 3] >> (j & 7)) & 1)
#define SEL_PBIT(b, c) (PBIT(b) ? (c)[j] : 0.0)
#define CAST_PBIT(b, c) ((double)PBIT(b) * (c)[j])
PADDED(sel_F_padded_bit_select, SEL_PBIT)
PADDED(sel_G_padded_bit_cast, CAST_PBIT)

/* Padded bits in explicit blocks of 8: one byte load per block, the bit index is the constant lane number. */
#define BLOCKED(name, LANE)                                                                                                                    \
  __attribute__((noinline)) void name(const uint8_t* RESTRICT k0, const uint8_t* RESTRICT k1, const uint8_t* RESTRICT k2, const uint8_t* RESTRICT k3, \
                                      const double* RESTRICT c0, const double* RESTRICT c1, const double* RESTRICT c2, const double* RESTRICT c3, \
                                      double* RESTRICT out, long nd, long m) {                                                                 \
    long row = (m + 7) / 8, full = m / 8;                                                                                                      \
    for (long a = 0; a < nd; ++a) {                                                                                                            \
      double* o = out + a * m;                                                                                                                 \
      for (long blk = 0; blk < full; ++blk) {                                                                                                  \
        unsigned m0 = k0[a * row + blk], m1 = k1[a * row + blk], m2 = k2[a * row + blk], m3 = k3[a * row + blk];                             \
        long j0 = blk * 8;                                                                                                                     \
        for (int l = 0; l < 8; ++l) o[j0 + l] = LANE(m0, c0) + LANE(m1, c1) + LANE(m2, c2) + LANE(m3, c3);                                     \
      }                                                                                                                                        \
      for (long j = full * 8; j < m; ++j) {                                                                                                    \
        unsigned l = j & 7, m0 = k0[a * row + full], m1 = k1[a * row + full], m2 = k2[a * row + full], m3 = k3[a * row + full];             \
        long j0 = full * 8;                                                                                                                    \
        o[j] = LANE(m0, c0) + LANE(m1, c1) + LANE(m2, c2) + LANE(m3, c3);                                                                      \
      }                                                                                                                                        \
    }                                                                                                                                          \
  }
#define LANE_SEL(mk, c) (((mk) >> l) & 1 ? (c)[j0 + l] : 0.0)
#define LANE_CAST(mk, c) ((double)(((mk) >> l) & 1) * (c)[j0 + l])
BLOCKED(sel_H_blocked_bit_select, LANE_SEL)
BLOCKED(sel_I_blocked_bit_cast, LANE_CAST)

/* Bits through portable vector types, the form Scaly's lane code already uses: broadcast the mask byte,
   AND with each lane's bit, compare to get an all-ones/all-zeros lane, AND it with the candidate. */
#ifndef KW
#define KW 4 /* lanes per vector */
#endif
typedef int64_t vi64 __attribute__((vector_size(8 * KW)));
typedef double vf64 __attribute__((vector_size(8 * KW)));
typedef double vf64u __attribute__((vector_size(8 * KW), aligned(8)));

static inline vf64 keep(uint8_t byte, vi64 bits, const double* c) {
  vi64 hit = ((vi64)(byte + (vi64){}) & bits) == bits;
  return (vf64)((vi64)(*(const vf64u*)c) & hit);
}

__attribute__((noinline)) void sel_K_vector_bits(const uint8_t* RESTRICT k0, const uint8_t* RESTRICT k1, const uint8_t* RESTRICT k2,
                                                 const uint8_t* RESTRICT k3, const double* RESTRICT c0, const double* RESTRICT c1,
                                                 const double* RESTRICT c2, const double* RESTRICT c3, double* RESTRICT out, long nd, long m) {
  vi64 bits[8 / KW];
  for (int h = 0; h < 8 / KW; ++h)
    for (int l = 0; l < KW; ++l) bits[h][l] = (int64_t)1 << (h * KW + l);
  long row = (m + 7) / 8, full = m / 8;
  for (long a = 0; a < nd; ++a) {
    double* o = out + a * m;
    const uint8_t *b0 = k0 + a * row, *b1 = k1 + a * row, *b2 = k2 + a * row, *b3 = k3 + a * row;
    for (long blk = 0; blk < full; ++blk)
      for (int h = 0; h < 8 / KW; ++h) {
        long j = blk * 8 + h * KW;
        vf64 s = keep(b0[blk], bits[h], c0 + j) + keep(b1[blk], bits[h], c1 + j) + keep(b2[blk], bits[h], c2 + j) + keep(b3[blk], bits[h], c3 + j);
        *(vf64u*)(o + j) = s;
      }
    for (long j = full * 8; j < m; ++j) {
      int l = j & 7;
      o[j] = (((b0[full] >> l) & 1) ? c0[j] : 0.0) + (((b1[full] >> l) & 1) ? c1[j] : 0.0) + (((b2[full] >> l) & 1) ? c2[j] : 0.0) +
             (((b3[full] >> l) & 1) ? c3[j] : 0.0);
    }
  }
}
