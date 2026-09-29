from __future__ import annotations

import os
import subprocess
import sys


from scaly.codegen import aot
from scaly.utils import env


def test_aot_cli_writes_the_module_pair(tmp_path, monkeypatch, capsys) -> None:
  (tmp_path / "scaly_aot_cli_target.py").write_text(
    "import scaly as sc\n\n\ndef build():\n  x = sc.sym('x', 2)\n  return sc.Function.from_exprs('aot_cli', [x], [x * x], ['x'], ['y'])\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  out = tmp_path / "generated"
  aot.main(["scaly_aot_cli_target:build", "-o", str(out)])
  assert "#define aot_cli_SZ_W 0" in (out / "aot_cli.h").read_text()
  assert (out / "aot_cli.c").read_text().startswith('#include "aot_cli.h"')
  assert "sz_w: 0" in capsys.readouterr().out

  # The documented invocation goes through codegen/__main__.py. Running aot itself would re-execute
  # an already-imported module (RuntimeWarning, two copies of its render-observer list), so -W error
  # is what pins the shim in place.
  env_vars = {**os.environ, "PYTHONPATH": str(tmp_path)}
  argv = [sys.executable, "-W", "error::RuntimeWarning", "-m", "scaly.codegen", "scaly_aot_cli_target:build", "-o", str(tmp_path / "cli")]
  proc = subprocess.run(argv, check=False, capture_output=True, text=True, env=env_vars)
  assert proc.returncode == 0, proc.stderr
  assert (tmp_path / "cli" / "aot_cli.h").read_text() == (out / "aot_cli.h").read_text()


def test_env_registry_mentions_native_build_controls() -> None:
  names = {v.name for v in env.ENV_VARS}  # the compiler's own; each package adds its through an entry point
  assert "SCALY_CACHE_DIR" in names
  assert "SCALY_CC" in names
