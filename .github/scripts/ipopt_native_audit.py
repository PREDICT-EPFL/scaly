"""Audit frozen native dependency closure, loader paths, machine types and operating-system floors."""

from __future__ import annotations

import importlib.resources
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess


def pe_imports(pe):
  return [entry.dll.decode() for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT") for entry in getattr(pe, table, [])]


def system_import(system, name):
  if system == "Windows":
    return name == PurePosixPath(name).name and (
      name.lower().startswith(("api-ms-win-", "ext-ms-win-")) or (Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / name).is_file()
    )
  if system == "Darwin":
    return name.startswith(("/usr/lib/", "/System/Library/Frameworks/"))
  return re.fullmatch(r"(?:ld-linux[^/]*|lib(?:c|m|pthread|dl|rt|util|resolv|z|gcc_s|stdc\+\+)\.so[^/]*)", name) is not None


def resolve_import(system, name, origin, files, loader_paths):
  if system_import(system, name):
    return None
  parent = str(PurePosixPath(origin).parent)
  if system == "Windows":
    candidates = [f"{parent}/{name}"] if "/" not in name and "\\" not in name else []
  elif system == "Darwin":
    candidates = [name.replace("@loader_path", parent)] if name.startswith("@loader_path/") else []
  else:
    candidates = [path.replace("${ORIGIN}", parent).replace("$ORIGIN", parent) + "/" + name for path in loader_paths] if "/" not in name else []
  indexed = {key.lower(): key for key in files} if system == "Windows" else files
  for candidate in candidates:
    normalized = posixpath.normpath(candidate)
    key = normalized.lower() if system == "Windows" else normalized
    if key in indexed:
      return indexed[key] if system == "Windows" else normalized
  raise RuntimeError(f"Unresolved or non-relocatable import: {origin} imports {name} via {loader_paths}")


def check_versions(versions, ceilings):
  for version in versions:
    prefix, _, value = version.partition("_")
    if prefix in ceilings and (not re.fullmatch(r"\d+(?:\.\d+)*", value) or tuple(map(int, value.split("."))) > ceilings[prefix]):
      raise RuntimeError(f"Compatibility floor exceeded: {version}")


def elf(path, arch):
  from elftools.elf.elffile import ELFFile
  from elftools.elf.dynamic import DynamicSection
  from elftools.elf.gnuversions import GNUVerNeedSection
  from elftools.elf.sections import SymbolTableSection

  with path.open("rb") as stream:
    binary = ELFFile(stream)
    expected = "EM_X86_64" if arch == "x86_64" else "EM_AARCH64"
    if binary["e_machine"] != expected or binary.elfclass != 64:
      raise RuntimeError(f"Wrong ELF machine: {path}")
    dynamic = binary.get_section_by_name(".dynamic")
    assert isinstance(dynamic, DynamicSection)
    imports, paths = [], []
    for tag in dynamic.iter_tags():
      if tag.entry.d_tag == "DT_NEEDED":
        imports.append(getattr(tag, "needed"))
      if tag.entry.d_tag in ("DT_RPATH", "DT_RUNPATH"):
        paths.extend(getattr(tag, "rpath", getattr(tag, "runpath", "")).split(":"))
    versions = {}
    section = binary.get_section_by_name(".gnu.version_r")
    if isinstance(section, GNUVerNeedSection):
      versions = {version.name: [entry.name for entry in entries] for version, entries in section.iter_versions()}
    symbols = binary.get_section_by_name(".dynsym")
    assert isinstance(symbols, SymbolTableSection)
    undefined = [symbol.name for symbol in symbols.iter_symbols() if symbol.entry.st_shndx == "SHN_UNDEF"]
    return {"imports": imports, "loader_paths": paths, "versions": versions, "undefined": undefined}


def macho(path, arch):
  machine = subprocess.check_output(["lipo", "-archs", str(path)], text=True).strip()
  if machine != arch:
    raise RuntimeError(f"Wrong Mach-O architecture {machine}: {path}")
  commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
  imports, paths, minima = [], [], []
  for command in commands.split("Load command")[1:]:
    if re.search(r"cmd LC_(?:LOAD|LOAD_WEAK|REEXPORT|LOAD_UPWARD)_DYLIB\b", command):
      imports.append(re.findall(r"\n\s+name (.*?) \(offset", command)[0])
    if "cmd LC_RPATH\n" in command:
      paths.append(re.findall(r"\n\s+path (.*?) \(offset", command)[0])
    if "cmd LC_BUILD_VERSION" in command or "cmd LC_VERSION_MIN_MACOSX" in command:
      version = re.search(r"\n\s+(?:minos|version) (\d+)\.(\d+)", command)
      assert version is not None
      minima.append(tuple(map(int, version.groups())))
  if not minima or max(minima) > (15, 0):
    raise RuntimeError(f"macOS 15.0 floor exceeded: {path}: {minima}")
  subprocess.run(["codesign", "--verify", "--strict", str(path)], check=True)
  return {"imports": imports, "loader_paths": paths, "minima": minima, "signature": "verified"}


def audit(root, manifest):
  system, arch = manifest["target"]["system"], manifest["target"]["arch"]
  expected_tag = {"Windows": "win_amd64", "Linux": f"manylinux_2_27_{arch}", "Darwin": f"macosx_15_0_{arch}"}[system]
  if manifest["target"]["platform_tag"] != expected_tag:
    raise RuntimeError("Candidate changed the established compatibility floor")
  policy = None
  if system == "Linux":
    policies = json.loads(importlib.resources.files("auditwheel").joinpath("policy/manylinux-policy.json").read_text())
    policy = next(value for value in policies if value["name"] == "manylinux_2_27")
  results = {}
  for name in manifest["dependencies"]:
    path = root / name
    if system == "Windows":
      import pefile

      pe = pefile.PE(str(path))
      if pe.FILE_HEADER.Machine != 0x8664:
        raise RuntimeError(f"Wrong PE machine: {name}")
      data = {
        "imports": pe_imports(pe),
        "loader_paths": [],
        "delay_imports": [entry.dll.decode() for entry in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])],
      }
      pe.close()
    elif system == "Darwin":
      data = macho(path, arch)
    else:
      assert policy is not None
      data = elf(path, arch)
      for dependency, versions in data["versions"].items():
        if system_import(system, dependency):
          allowed = policy["symbol_versions"][arch]
          for version in versions:
            prefix, _, number = version.partition("_")
            if number not in allowed.get(prefix, []):
              raise RuntimeError(f"{name}: {version} is outside manylinux_2_27")
      for library, symbols in policy["blacklist"].items():
        if library in data["imports"] and set(symbols) & set(data["undefined"]):
          raise RuntimeError(f"{name}: blacklisted manylinux symbol")
      check_versions([v for versions in data["versions"].values() for v in versions], {"GLIBC": (2, 27)})
      notes = subprocess.check_output(["readelf", "-n", str(path)], text=True)
      if re.search(r"x86 ISA needed:.*x86-64-v[234]", notes):
        raise RuntimeError(f"{name}: ELF requires a newer x86 CPU baseline")
      data["elf_notes"] = notes
    for entry in data["loader_paths"]:
      if not entry.startswith(("$ORIGIN", "${ORIGIN}", "@loader_path")):
        raise RuntimeError(f"Stale or external loader path: {name}: {entry}")
    resolved = [resolve_import(system, dependency, name, manifest["files"], data["loader_paths"]) for dependency in data["imports"]]
    bundled = sorted({value for value in resolved if value is not None})
    if bundled != sorted(manifest["dependencies"][name]["bundled"]):
      raise RuntimeError(f"Manifest dependency graph disagrees with binary imports, including delay imports: {name}")
    data["resolved_bundled"] = bundled
    results[name] = data
  headers = (root / "scaly_ipopt/include/coin-or/IpoptConfig.h").read_text()
  if re.search(r"^\s*#\s*define\s+IPOPT_(?:SINGLE|INT64)\b", headers, re.MULTILINE):
    raise RuntimeError("Wrong precision/index interface")
  return results
