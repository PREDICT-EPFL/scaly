"""C identifier spelling and scoped allocation for generated C and C++ names."""

from __future__ import annotations

from collections.abc import Iterable
import re

_KEYWORDS = set(
  """
  alignas alignof asm auto bool break case catch char char8_t char16_t char32_t class const constexpr
  consteval constinit const_cast continue co_await co_return co_yield decltype default delete do double
  dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable
  namespace new noexcept nullptr operator private protected public register reinterpret_cast requires
  restrict return short signed sizeof static static_assert static_cast struct switch template this
  thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t
  while _Alignas _Alignof _Atomic _BitInt _Bool _Complex _Decimal32 _Decimal64 _Decimal128 _Generic
  _Imaginary _Noreturn _Static_assert _Thread_local and and_eq bitand bitor compl concept not not_eq or
  or_eq typeof typeof_unqual xor xor_eq
""".split()
)
_MATH = set(
  """
  acos acosh asin asinh atan atanh atan2 cbrt ceil copysign cos cosh erf erfc exp exp2 expm1 fabs fdim
  floor fma fmax fmin fmod frexp hypot ilogb ldexp lgamma llrint llround log log10 log1p log2 logb lrint
  lround modf nan nearbyint nextafter nexttoward pow remainder remquo rint round scalbln scalbn sin sinh
  sqrt tan tanh tgamma trunc isfinite isinf isnan isnormal signbit fpclassify isgreater isgreaterequal
  isless islessequal islessgreater isunordered
""".split()
)
_RESERVED = (
  _KEYWORDS
  | _MATH
  | {n + suffix for n in _MATH for suffix in ("f", "l")}
  | set(
    """
  arg res iw w mem solver_options double2 NULL NAN INFINITY HUGE_VAL HUGE_VALF HUGE_VALL
  size_t ptrdiff_t int64_t uint64_t memcpy memmove memset memcmp malloc calloc realloc free abort exit
  printf fprintf sprintf snprintf vprintf vfprintf vsprintf vsnprintf scanf fscanf sscanf getchar
  putchar puts fputs fgets fread fwrite fopen fclose fflush fseek ftell rewind fgetpos fsetpos feof
  ferror clearerr perror remove rename tmpfile tmpnam setbuf setvbuf ungetc getc putc stdin stdout stderr
  clock time difftime mktime localtime gmtime strftime asctime ctime clock_gettime timespec
  strlen strcpy strncpy strcat strncat strcmp strncmp strchr strrchr strstr strspn strcspn strpbrk
  strtok strerror strcoll strxfrm abs labs llabs div ldiv lldiv atof atoi atol atoll strtod strtof
  strtold strtol strtoll strtoul strtoull rand srand qsort bsearch getenv atexit quick_exit _Exit
  aligned_alloc max_align_t offsetof scaly_solver_option scaly_solver_stats
""".split()
  )
)


def c_ident(name: str) -> str:
  """Spell a Scaly name as an ASCII C identifier. NameScope owns reservations."""
  ident = re.sub(r"[^a-zA-Z0-9_]", "_", name) or "_"
  if ident != name:
    ident = re.sub(r"_+", "_", ident)
  return f"_{ident}" if ident[:1].isdigit() else ident


class NameScope:
  """Allocate deterministic identifiers against language names and enclosing declarations."""

  def __init__(self, occupied: Iterable[str] = (), *, parent: NameScope | None = None, header: bool = False) -> None:
    self.occupied = set(occupied)
    self.parent = parent
    self.header = header

  def child(self, occupied: Iterable[str] = ()) -> NameScope:
    return NameScope(occupied, parent=self, header=self.header)

  def contains(self, name: str) -> bool:
    return name in self.occupied or bool(self.parent and self.parent.contains(name))

  def reserved(self, name: str) -> bool:
    if self.header:
      return name in _KEYWORDS
    return name in _RESERVED or "__" in name or bool(re.match(r"_[A-Z]", name)) or name.startswith("SCALY_") or bool(re.fullmatch(r"k[0-9]+", name))

  def claim(self, name: str) -> str:
    """Keep an exported symbol unchanged, or report a declaration clash to the user."""
    symbol = c_ident(name)
    if self.reserved(name) or self.reserved(symbol) or self.contains(symbol):
      raise ValueError(f"exported C identifier {symbol!r} clashes with another declaration; give the function a distinct name")
    self.occupied.add(symbol)
    return symbol

  def allocate(self, base: str, *, generated: bool = False, suffixes: tuple[str, ...] = ()) -> str:
    """Reserve a spelling and its derived suffixes. Generated macros may use SCALY_."""
    base = c_ident(base)
    base = re.sub(r"_+", "_", base)
    if re.match(r"_[A-Z]", base):
      base = "scaly" + base
    if not generated and base.startswith("SCALY_"):
      base = "scaly_" + base[6:]
    name = base
    suffix = 2
    if not generated and self.reserved(name):
      name = base = base + ("_2" if self.header else "_")
    while (
      self.contains(name)
      or (not generated and self.reserved(name))
      or any(self.contains(name + suffix) or (not generated and self.reserved(name + suffix)) for suffix in suffixes)
    ):
      name = f"{base}{'' if base.endswith('_') else '_'}{suffix}"
      suffix += 1
    self.occupied.update((name, *(name + suffix for suffix in suffixes)))
    return name

  def procedure(self, base: str) -> str:
    """Reserve both a program procedure name and the renderer's raw symbol."""
    return self.allocate(base.rstrip("_") or "scaly", suffixes=("_raw",))
