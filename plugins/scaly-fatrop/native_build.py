"""Build and bundle Fatrop with a private static BLASFEO and the pinned license texts."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path

_DEFAULT_CC, _DEFAULT_CXX = "cc", ("c++", "g++", "clang++")
if platform.system() == "Windows":
  _DEFAULT_CC, _DEFAULT_CXX = "zig cc -mcpu=haswell", ("zig c++ -mcpu=haswell",)
_BUILD_CONFIG = json.loads((Path(__file__).parent / "src/scaly_fatrop/build_config.json").read_text())


def _compiler_command(value: str) -> list[str]:
  if path := shutil.which(value):
    return [path]
  command = [part.strip('"') for part in shlex.split(value, posix=os.name != "nt")]
  if command and (path := shutil.which(command[0])):
    return [path, *command[1:]]
  return []


def _blasfeo_target(system: str, machine: str) -> str:
  if machine in {"x86_64", "amd64"}:
    return "X64_INTEL_HASWELL"
  if system == "Darwin" and machine in {"arm64", "aarch64"}:
    return "ARMV8A_APPLE_M1"
  if system == "Linux" and machine in {"arm64", "aarch64"}:
    return "ARMV8A_ARM_CORTEX_A76"
  raise RuntimeError(f"Unsupported Fatrop platform: {system} {machine}")


def _build_fatrop(hook, third_party_dir: Path, out_dir: Path) -> None:
  system = platform.system()
  if system not in {"Linux", "Darwin", "Windows"}:
    raise RuntimeError(f"Unsupported Fatrop platform: {system}")
  target = _blasfeo_target(system, platform.machine().lower())
  configure = ["-G", os.environ.get("CMAKE_GENERATOR", "Ninja")] if system == "Windows" else []
  command = _compiler_command(os.environ.get("CC", _DEFAULT_CC))
  if len(command) >= 2 and Path(command[0]).stem == "zig" and command[1] == "cc":
    # CMake needs Zig's subcommands for assembly and archive tools, including in BLASFEO's probes.
    toolchain = third_party_dir / "zig.cmake"
    cxx = _compiler_command(os.environ.get("CXX", "")) or [command[0], "c++", *command[2:]]
    lines = [
      f"set(CMAKE_{language}_COMPILER {' '.join(json.dumps(part.replace(chr(92), '/')) for part in args)})"
      for language, args in (("C", command), ("CXX", cxx), ("ASM", command))
    ]
    zig = json.dumps(command[0].replace("\\", "/"))
    lines += [f"set(CMAKE_AR {zig})", f"set(CMAKE_RANLIB {zig})", "set(CMAKE_ASM_COMPILER_ID Clang)"]
    rules = third_party_dir / "zig-rules.cmake"
    lines.append(f"set(CMAKE_USER_MAKE_RULES_OVERRIDE {json.dumps(rules.as_posix())})")
    archive_rules = []
    for language in ("C", "CXX", "ASM"):
      archive_rules += [
        f'set(CMAKE_{language}_ARCHIVE_CREATE "<CMAKE_AR> ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")',
        f'set(CMAKE_{language}_ARCHIVE_APPEND "<CMAKE_AR> ar q <TARGET> <LINK_FLAGS> <OBJECTS>")',
        f'set(CMAKE_{language}_ARCHIVE_FINISH "<CMAKE_RANLIB> ranlib <TARGET>")',
      ]
    rules.write_text("\n".join(archive_rules) + "\n")
    toolchain.write_text("\n".join(lines) + "\n")
    configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}")
  sources = {}
  for name, url in (("fatrop", "https://github.com/meco-group/fatrop.git"), ("blasfeo", "https://github.com/giaf/blasfeo.git")):
    source = third_party_dir / name
    tag = _BUILD_CONFIG[name]["tag"]
    hook.app.display_info(f"Cloning {url}@{tag}")
    subprocess.run(["git", "clone", "--depth=1", "--branch", tag, url, str(source)], check=True)
    sources[name] = source

  blasfeo_install = third_party_dir / "blasfeo-install"
  blasfeo_build = third_party_dir / "blasfeo-build"
  subprocess.run(
    [
      "cmake",
      *configure,
      "-S",
      str(sources["blasfeo"]),
      "-B",
      str(blasfeo_build),
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
      f"-DTARGET={target}",
      "-DBUILD_SHARED_LIBS=OFF",
      "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
      "-DCMAKE_C_VISIBILITY_PRESET=hidden",
      "-DBLAS_API=OFF",
      "-DBLASFEO_EXAMPLES=OFF",
      "-DBLASFEO_HEADERS_INSTALLATION_DIRECTORY=include",
      f"-DCMAKE_INSTALL_PREFIX={blasfeo_install}",
    ],
    check=True,
  )
  subprocess.run(["cmake", "--build", str(blasfeo_build), "--parallel"], check=True)
  subprocess.run(["cmake", "--install", str(blasfeo_build)], check=True)

  # BLASFEO's assembly exports are not affected by C visibility flags. Restrict the final
  # library to Fatrop's C interface so neither its C nor assembly kernels can bind to PIQP's.
  exports = third_party_dir / "fatrop.exports"
  if system == "Linux":
    exports.write_text("{ global: fatrop_ocp_c_*; local: *; };\n")
    link_flags = f"-Wl,--version-script={exports}"
  elif system == "Darwin":
    exports.write_text("_fatrop_ocp_c_*\n")
    link_flags = f"-Wl,-exported_symbols_list,{exports}"
  else:
    # PE uses the C interface's explicit __declspec(dllexport) annotations.
    link_flags = ""
    cmake = sources["fatrop"] / "src/CMakeLists.txt"
    cmake.write_text(cmake.read_text() + '\nset_target_properties(fatrop PROPERTIES PREFIX "")\n')
  build = third_party_dir / "fatrop-build"
  subprocess.run(
    [
      "cmake",
      *configure,
      "-S",
      str(sources["fatrop"]),
      "-B",
      str(build),
      "-DCMAKE_BUILD_TYPE=Release",
      "-DBUILD_TESTS=OFF",
      "-DBUILD_EXAMPLES=OFF",
      "-DBUILD_WITH_C_INTERFACE=ON",
      "-DBUILD_SHARED_LIBS=ON",
      "-DWITH_BUILD_BLASFEO=OFF",
      f"-Dblasfeo_DIR={blasfeo_install / 'share/cmake/blasfeo'}",
      "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
      "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
      f"-DCMAKE_CXX_FLAGS={os.environ.get('CXXFLAGS', '')} -DGCC_HASCLASSVISIBILITY",
      f"-DCMAKE_SHARED_LINKER_FLAGS={os.environ.get('LDFLAGS', '')} {link_flags}",
      "-DCMAKE_INSTALL_LIBDIR=lib",
      "-DCMAKE_INSTALL_BINDIR=lib",
      f"-DCMAKE_INSTALL_PREFIX={out_dir}",
      "-DCMAKE_INSTALL_NAME_DIR=@rpath",
      "-DCMAKE_INSTALL_RPATH=@loader_path" if system == "Darwin" else "-DCMAKE_INSTALL_RPATH=$ORIGIN",
    ],
    check=True,
  )
  subprocess.run(["cmake", "--build", str(build), "--parallel"], check=True)
  subprocess.run(["cmake", "--install", str(build)], check=True)
  shutil.rmtree(out_dir / "lib/cmake")
  shutil.copytree(blasfeo_install / "include", out_dir / "include", dirs_exist_ok=True)

  # Changing Mach-O load commands invalidates the arm64 signature. Sign the installed image
  # after CMake rewrites its install name and rpath, before it reaches the shared cache.
  if system == "Darwin":
    for library in (out_dir / "lib").glob("*.dylib"):
      if not library.is_symlink():
        subprocess.run(["codesign", "--force", "--sign", "-", str(library)], check=True)

  licenses = out_dir / "licenses"
  notices = ["# Third-party notices for scaly-fatrop", "", "| Component | Version | License | Upstream | Texts |", "|---|---|---|---|---|"]
  for name, license_id, files in (
    ("fatrop", "BSD-2-Clause AND EPL-2.0", ("LICENSE.txt", "LICENSE-EPL-2.0.txt")),
    ("blasfeo", "BSD-2-Clause", ("LICENSE.txt",)),
  ):
    directory = licenses / name
    directory.mkdir(parents=True)
    for filename in files:
      shutil.copy2(sources[name] / filename, directory / filename)
    url = "https://github.com/meco-group/fatrop" if name == "fatrop" else "https://github.com/giaf/blasfeo"
    notices.append(f"| {name} | {_BUILD_CONFIG[name]['version']} | {license_id} | {url}/tree/{_BUILD_CONFIG[name]['tag']} | `{name}/` |")
  (licenses / "THIRD_PARTY_NOTICES.md").write_text("\n".join(notices) + "\n")
