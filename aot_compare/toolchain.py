"""Third-party pieces the benchmarks link, each fetched or built once into build/third_party: Google Benchmark, IREE's runtime, XLA's AOT runtime and Abseil."""

import importlib.util
import re
import shutil
import subprocess
import tarfile
import urllib.request
from functools import cache
from pathlib import Path

THIRD_PARTY = Path(__file__).parent / "build" / "third_party"
# The flags Scaly's own benchmark cells compile with, used for everything the harness compiles.
FLAGS = ("-O3", "-march=native", "-fno-math-errno", "-fveclib=libmvec")
# IREE publishes its runtime headers and libraries only in this prebuilt tarball; the PyPI runtime
# wheel holds the same code sealed in its Python extension. Its build stamp, 3.12.0rc20260917,
# matches the iree-base-compiler 3.12.0 wheel that compiles the modules.
IREE_DIST = "https://github.com/iree-org/iree/releases/download/v3.12.0/iree-dist-3.12.0rc20260917-linux-x86_64.tar.xz"


def run(cmd: list, cwd: Path | None = None) -> None:
  subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def _cmake_install(src: Path, prefix: Path, *options: str, cxx_flags: str = "-O3") -> None:
  build = THIRD_PARTY / f"{prefix.name}-build"
  run(
    [
      "cmake",
      "-S",
      src,
      "-B",
      build,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_C_COMPILER=clang",
      "-DCMAKE_CXX_COMPILER=clang++",
      f"-DCMAKE_CXX_FLAGS={cxx_flags}",
      f"-DCMAKE_INSTALL_PREFIX={prefix}",
      "-DCMAKE_INSTALL_LIBDIR=lib",
      *options,
    ]
  )
  run(["cmake", "--build", build, "--parallel"])
  run(["cmake", "--install", build])
  shutil.rmtree(build)


def _clone(repo: str, tag: str) -> Path:
  src = THIRD_PARTY / f"{Path(repo).stem}-src"
  shutil.rmtree(src, ignore_errors=True)
  run(["git", "clone", "--quiet", "--depth=1", "--branch", tag, repo, src])
  return src


@cache
def gbench() -> Path:
  """Google Benchmark at the tag Scaly's harness pins."""
  prefix = THIRD_PARTY / "gbench"
  if not (prefix / "lib" / "libbenchmark.a").exists():
    src = _clone("https://github.com/google/benchmark.git", "v1.9.5")
    _cmake_install(src, prefix, "-DBUILD_SHARED_LIBS=OFF", "-DBENCHMARK_ENABLE_TESTING=OFF", "-DBENCHMARK_ENABLE_WERROR=OFF")
    shutil.rmtree(src)
  return prefix


# In link order: the loader for dispatches linked into the binary, the runtime as one archive, and
# the flatbuffer reader the runtime uses for module metadata.
IREE_LIBRARIES = ("iree_hal_local_loaders_static_library_loader", "iree_runtime_unified", "flatcc_parsing")


@cache
def iree_runtime() -> Path:
  """IREE's runtime headers and static libraries, extracted from the release's prebuilt distribution."""
  prefix = THIRD_PARTY / "iree-runtime"
  if not (prefix / "lib" / "libiree_runtime_unified.a").exists():
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    archive = THIRD_PARTY / Path(IREE_DIST).name
    urllib.request.urlretrieve(IREE_DIST, archive)
    with tarfile.open(archive) as tar:
      keep = [f"lib/lib{name}.a" for name in IREE_LIBRARIES]
      members = [m for m in tar if m.name in keep or (m.name.startswith("include/iree/") and not m.name.startswith("include/iree/compiler/"))]
      tar.extractall(prefix, members=members, filter="data")
    archive.unlink()
  return prefix


def tensorflow_dir() -> Path:
  spec = importlib.util.find_spec("tensorflow")
  assert spec is not None and spec.submodule_search_locations is not None
  return Path(spec.submodule_search_locations[0])


@cache
def xla_aot_runtime() -> tuple[Path, Path]:
  """The runtime TensorFlow ships as source for its AOT-compiled classes: one native object, then the archive.

  TensorFlow's own CMake file builds it at plain -O3, and some of its Eigen convolution sources do not
  compile once -march=native enables AVX-512. Of all of it, the generated schedule calls only the
  f64 matrix product, so that one source is rebuilt with the harness flags and linked ahead of the
  archive, which then supplies everything else.
  """
  prefix = THIRD_PARTY / "xla-aot-runtime"
  archive, native = prefix / "libtf_xla_runtime.a", prefix / "dot_lib_f64.o"
  if not (archive.exists() and native.exists()):
    src = tensorflow_dir() / "xla_aot_runtime_src"
    build = THIRD_PARTY / "xla-aot-runtime-build"
    run(
      [
        "cmake",
        "-S",
        src,
        "-B",
        build,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=clang++",
        "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CXX_FLAGS=-O3 -D_GLIBCXX_USE_CXX11_ABI=1",
      ]
    )
    run(["cmake", "--build", build, "--parallel"])
    prefix.mkdir(parents=True, exist_ok=True)
    shutil.copy(build / archive.name, archive)
    shutil.rmtree(build)
    dot_lib = src / "xla" / "backends" / "cpu" / "runtime" / "dot_lib_f64.cc"
    run(
      [
        "clang++",
        *FLAGS,
        "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-Wno-ignored-attributes",
        "-I",
        tensorflow_dir() / "include",
        "-c",
        dot_lib,
        "-o",
        native,
      ]
    )
  return native, archive


@cache
def abseil() -> tuple[Path, ...]:
  """Abseil at the release TensorFlow bundles headers for, which the XLA AOT classes call into."""
  prefix = THIRD_PARTY / "abseil"
  if not (prefix / "lib").exists():
    config = (tensorflow_dir() / "include" / "absl" / "base" / "config.h").read_text()
    version = re.search(r"#define ABSL_LTS_RELEASE_VERSION (\d+)", config)
    patch = re.search(r"#define ABSL_LTS_RELEASE_PATCH_LEVEL (\d+)", config)
    assert version and patch, "TensorFlow's Abseil headers name no LTS release"
    src = _clone("https://github.com/abseil/abseil-cpp.git", f"{version[1]}.{patch[1]}")
    _cmake_install(
      src, prefix, "-DABSL_BUILD_TESTING=OFF", "-DABSL_PROPAGATE_CXX_STD=ON", "-DCMAKE_CXX_STANDARD=17", cxx_flags="-O3 -D_GLIBCXX_USE_CXX11_ABI=1"
    )
    shutil.rmtree(src)
  return tuple(sorted((prefix / "lib").glob("libabsl_*.a")))


@cache
def host_cpu() -> str:
  """The CPU name -march=native resolves to, which XLA's AOT compiler needs spelled out."""
  out = subprocess.run(["clang", "-march=native", "-###", "-x", "c", "-c", "/dev/null"], capture_output=True, text=True, check=True).stderr
  match = re.search(r'"-target-cpu" "([^"]+)"', out)
  assert match, "clang did not report a target CPU"
  return match[1]
