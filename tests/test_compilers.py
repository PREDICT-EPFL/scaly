"""Check test compiler selection against each platform's supported toolchains."""

from types import SimpleNamespace

import pytest

from tests import compilers
from scaly.codegen.toolchain import Compiler


@pytest.mark.parametrize("system,name", [(system, name) for system in ("win32", "linux", "darwin") for name in ("gcc", "clang", "zig")])
def test_named_compiler_respects_supported_platforms(system, name, monkeypatch):
  monkeypatch.setattr(compilers, "sys", SimpleNamespace(platform=system))
  found = []
  monkeypatch.setattr(compilers.shutil, "which", lambda name: found.append(name) or f"/tools/{name}")
  monkeypatch.setattr(compilers, "find_c_compiler", lambda: Compiler(("/env/zig", "cc"), "zig-bin"))
  if system == "win32" and name == "clang":
    with pytest.raises(pytest.skip.Exception, match="Windows.*Zig"):
      compilers.named_compiler(name)
    assert found == []
  else:
    assert compilers.named_compiler(name) == (("/env/zig", "cc") if name == "zig" else (f"/tools/{name}",))
