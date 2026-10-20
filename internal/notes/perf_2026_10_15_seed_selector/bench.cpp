// Times each selector variant on real npmpc masks; every variant must reproduce sel_A_double bit for bit first.
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
void sel_flat(const double*, const double*, const double*, const double*, const double*, const double*, const double*, const double*, double*, long, long);
#define DECL(name, T) void name(const T*, const T*, const T*, const T*, const double*, const double*, const double*, const double*, double*, long, long);
DECL(sel_A_double, double)
DECL(sel_B_byte_select, uint8_t)
DECL(sel_C_bit_select, uint8_t)
DECL(sel_D_byte_cast, uint8_t)
DECL(sel_E_bit_cast, uint8_t)
DECL(sel_C2_bit_select_forced, uint8_t)
DECL(sel_E2_bit_cast_forced, uint8_t)
DECL(sel_F_padded_bit_select, uint8_t)
DECL(sel_G_padded_bit_cast, uint8_t)
DECL(sel_H_blocked_bit_select, uint8_t)
DECL(sel_I_blocked_bit_cast, uint8_t)
DECL(sel_J_avx2_bits, uint8_t)
DECL(sel_K_vector_bits, uint8_t)
}

struct Case {
  long nd = 7, stages, m;
  std::vector<double> kd[4], c[4], out;
  std::vector<uint8_t> kb[4], kbits[4], kpad[4];
};

static Case load(long stages) {
  Case s;
  s.stages = stages;
  s.m = stages * 9;
  long n = s.nd * s.m;
  std::FILE* f = std::fopen(("masks_N" + std::to_string(stages) + ".bin").c_str(), "rb");
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> u(-1, 1);
  for (int l = 0; l < 4; ++l) {
    s.kd[l].resize(n);
    if (std::fread(s.kd[l].data(), sizeof(double), n, f) != (size_t)n) std::abort();
    s.kb[l].resize(n);
    s.kbits[l].assign((n + 7) / 8, 0);
    for (long i = 0; i < n; ++i) {
      s.kb[l][i] = (uint8_t)s.kd[l][i];
      s.kbits[l][i >> 3] |= (uint8_t)(s.kb[l][i] << (i & 7));
    }
    long row = (s.m + 7) / 8;
    s.kpad[l].assign(s.nd * row, 0);
    for (long a = 0; a < s.nd; ++a)
      for (long j = 0; j < s.m; ++j) s.kpad[l][a * row + (j >> 3)] |= (uint8_t)(s.kb[l][a * s.m + j] << (j & 7));
    s.c[l].resize(s.m);
    for (auto& x : s.c[l]) x = u(rng);
  }
  std::fclose(f);
  s.out.assign(n, 0);
  return s;
}

template <class F>
static void run(benchmark::State& state, long stages, F call) {
  static Case n50 = load(50), n200 = load(200);
  Case& s = stages == 50 ? n50 : n200;
  std::vector<double> ref(s.out.size());
  sel_A_double(s.kd[0].data(), s.kd[1].data(), s.kd[2].data(), s.kd[3].data(), s.c[0].data(), s.c[1].data(), s.c[2].data(), s.c[3].data(), ref.data(), s.nd, s.m);
  std::fill(s.out.begin(), s.out.end(), -7.0);
  call(s);
  // Equal as numbers: a masked-out 0 * c is -0.0 for negative c where a select gives +0.0.
  long bitwise = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!(ref[i] == s.out[i])) {
      state.SkipWithError("output differs from sel_A_double");
      return;
    }
    bitwise += std::memcmp(&ref[i], &s.out[i], sizeof(double)) != 0;
  }
  state.counters["signed_zero_diffs"] = double(bitwise);
  for (auto _ : state) {
    call(s);
    benchmark::DoNotOptimize(s.out.data());
    benchmark::ClobberMemory();
  }
  state.counters["ns/elem"] = benchmark::Counter(double(s.out.size()), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
}

#define ARGS(s) s.c[0].data(), s.c[1].data(), s.c[2].data(), s.c[3].data(), s.out.data(), s.nd
#define REG(label, bytes_per_elem, body)                                                                                         \
  for (long stages : {50L, 200L})                                                                                                  \
    benchmark::RegisterBenchmark((std::string(label) + "/N" + std::to_string(stages)).c_str(), [=](benchmark::State& st) {       \
      run(st, stages, [](Case& s) { body; });                                                                                     \
      st.counters["mask_kB"] = 4 * 7 * stages * 9 * (bytes_per_elem) / 1000.0;                                                    \
    })->MinTime(0.5)->Repetitions(5)->ReportAggregatesOnly(true);

int main(int argc, char** argv) {
  REG("flat_as_generated", 8.0, sel_flat(s.kd[0].data(), s.kd[1].data(), s.kd[2].data(), s.kd[3].data(), ARGS(s) / s.nd * 0 + s.nd, s.stages))
  REG("A_double", 8.0, sel_A_double(s.kd[0].data(), s.kd[1].data(), s.kd[2].data(), s.kd[3].data(), ARGS(s), s.m))
  REG("B_byte_select", 1.0, sel_B_byte_select(s.kb[0].data(), s.kb[1].data(), s.kb[2].data(), s.kb[3].data(), ARGS(s), s.m))
  REG("C_bit_select", 0.125, sel_C_bit_select(s.kbits[0].data(), s.kbits[1].data(), s.kbits[2].data(), s.kbits[3].data(), ARGS(s), s.m))
  REG("D_byte_cast", 1.0, sel_D_byte_cast(s.kb[0].data(), s.kb[1].data(), s.kb[2].data(), s.kb[3].data(), ARGS(s), s.m))
  REG("E_bit_cast", 0.125, sel_E_bit_cast(s.kbits[0].data(), s.kbits[1].data(), s.kbits[2].data(), s.kbits[3].data(), ARGS(s), s.m))
  REG("C2_bit_select_forced", 0.125, sel_C2_bit_select_forced(s.kbits[0].data(), s.kbits[1].data(), s.kbits[2].data(), s.kbits[3].data(), ARGS(s), s.m))
  REG("E2_bit_cast_forced", 0.125, sel_E2_bit_cast_forced(s.kbits[0].data(), s.kbits[1].data(), s.kbits[2].data(), s.kbits[3].data(), ARGS(s), s.m))
  REG("F_padded_bit_select", 0.125, sel_F_padded_bit_select(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  REG("G_padded_bit_cast", 0.125, sel_G_padded_bit_cast(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  REG("H_blocked_bit_select", 0.125, sel_H_blocked_bit_select(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  REG("I_blocked_bit_cast", 0.125, sel_I_blocked_bit_cast(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  REG("J_avx2_intrinsics_bits", 0.125, sel_J_avx2_bits(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  REG("K_vector_type_bits", 0.125, sel_K_vector_bits(s.kpad[0].data(), s.kpad[1].data(), s.kpad[2].data(), s.kpad[3].data(), ARGS(s), s.m))
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
