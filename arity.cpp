#include <array>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

template <typename T, std::size_t... Ns> class Shaped {
public:
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * 1);
  static constexpr bool is_scalar = ndim == 0;
};

template <typename T, std::size_t... Ns>
class Buffer : public Shaped<T, Ns...> {
public:
  Buffer() = default;

  static Buffer alloc() {
    if constexpr (Shaped<T, Ns...>::size == 0)
      return Buffer{nullptr};
    void *raw = std::aligned_alloc(alignment,
                                   (nbytes + alignment - 1) & ~(alignment - 1));
    if (!raw)
      throw std::bad_alloc();
    return Buffer{static_cast<T *>(raw)};
  }

  static void free(Buffer &buffer) {
    std::free(buffer.data);
    buffer.data = nullptr;
  }

private:
  explicit Buffer(T *data) : data(data) {}

  T *data = nullptr;
  static constexpr std::size_t alignment = 16;
  static constexpr std::size_t nbytes = Shaped<T, Ns...>::size * sizeof(T);
};

template <typename T, std::size_t... Ns>
class Expr : public Shaped<T, Ns...> {};

template <typename T> struct is_expr : std::false_type {};

template <typename T, std::size_t... Ns>
struct is_expr<Expr<T, Ns...>> : std::true_type {};

template <typename T>
inline constexpr bool is_expr_v = is_expr<std::remove_cvref_t<T>>::value;

template <typename T> struct buffer_of;

template <typename T, std::size_t... Ns> struct buffer_of<Expr<T, Ns...>> {
  using type = Buffer<T, Ns...>;
};

template <typename T>
using buffer_of_t = typename buffer_of<std::remove_cvref_t<T>>::type;

template <typename Signature> class Function;

template <typename Result, typename... Inputs>
  requires is_expr_v<Result> && (is_expr_v<Inputs> && ...)
class Function<Result(Inputs...)> {
public:
  using SymbolicFunction = std::function<Result(Inputs...)>;
  using NumericalFunction =
      std::function<buffer_of_t<Result>(buffer_of_t<Inputs>...)>;

  template <typename Fn>
    requires std::is_invocable_r_v<Result, Fn &, Inputs...>
  explicit Function(Fn fn) {}

  Result symbolic_call(Inputs... inputs) const {
    return symbolic_fn(std::move(inputs)...);
  }

  buffer_of_t<Result> numerical_call(buffer_of_t<Inputs>... inputs) const {
    return numerical_fn(std::move(inputs)...);
  }

private:
  SymbolicFunction symbolic_fn;
  NumericalFunction numerical_fn;
};

int main() {
  using X = Expr<double, 3, 4>;
  using Scale = Expr<double>;

  Function<X(X, Scale)> first_arg{
      [](X x, Scale s) { return x; },
  };

  static_assert(
      std::same_as<decltype(first_arg.symbolic_call(X{}, Scale{})), X>);
  static_assert(std::same_as<decltype(first_arg.numerical_call(
                                 Buffer<double, 3, 4>{}, Buffer<double>{})),
                             Buffer<double, 3, 4>>);
}
