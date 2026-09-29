#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#ifndef SCALY_SUCCESS
#define SCALY_SUCCESS 0
#endif
#ifndef SCALY_ERR_NULL_ABI
#define SCALY_ERR_NULL_ABI 1
#endif
#ifndef SCALY_ERR_NULL_WORK
#define SCALY_ERR_NULL_WORK 2
#endif
#ifndef SCALY_ERR_NULL_RESULT
#define SCALY_ERR_NULL_RESULT 3
#endif
#ifndef SCALY_ERR_NULL_INPUT
#define SCALY_ERR_NULL_INPUT 4
#endif

#define dynamics_SZ_ARG 2
#define dynamics_SZ_RES 1
#define dynamics_SZ_IW 0
#define dynamics_SZ_W 0

#ifndef SCALY_BUFFER_HPP
#define SCALY_BUFFER_HPP
// A fixed-shape, row-major, 16-byte aligned buffer with inline storage: an aggregate to place on
// the stack, in a static, or behind make_unique. shape is the Expr shape; data is its flattening.
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  template <typename... I> T& operator()(I... idx) { return data[offset(idx...)]; }
  template <typename... I> const T& operator()(I... idx) const { return data[offset(idx...)]; }
  template <typename... I> static constexpr std::size_t offset(I... idx) {
    static_assert(sizeof...(I) == ndim, "Buffer: one index per dimension");
    const std::size_t index[] = {static_cast<std::size_t>(idx)..., 0};
    std::size_t off = 0;
    for (std::size_t k = 0; k < ndim; ++k) {
      assert(index[k] < shape[k]);
      off = off * shape[k] + index[k];
    }
    return off;
  }
};
#endif

namespace dynamics {
// The pointer ABI for dynamics; the same C symbols the C header declares.
extern "C" int dynamics(const double** arg, double** res, int* iw, double* w, int mem);

using z_t = Buffer<double, 4>;
using u_t = Buffer<double, 2>;
using znext_t = Buffer<double, 4>;
using workspace_t = Buffer<double, dynamics_SZ_W>;
constexpr int sz_arg = dynamics_SZ_ARG;
constexpr int sz_res = dynamics_SZ_RES;
constexpr int sz_iw = dynamics_SZ_IW;
constexpr int sz_w = dynamics_SZ_W;

inline int call(const z_t& z, const u_t& u, znext_t& znext, workspace_t& workspace) {
  const double* arg[sz_arg > 0 ? sz_arg : 1] = {z.ptr(), u.ptr()};
  double* res[sz_res > 0 ? sz_res : 1] = {znext.ptr()};
  return dynamics(arg, res, nullptr, sz_w ? workspace.ptr() : nullptr, 0);
}
}  // namespace dynamics
