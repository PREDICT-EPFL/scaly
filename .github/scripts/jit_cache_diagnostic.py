"""Count use of the restored nightly JIT cache in the diagnostic pytest job for issue #440."""

from collections import Counter

import pytest

from scaly.codegen import jit
from scaly.codegen.toolchain import cache_root
from scaly.utils.env import shared_lib_ext


_snapshot = {}
_counts = Counter()
_patch = pytest.MonkeyPatch()


def pytest_configure(config):
  global _snapshot
  root = cache_root()
  if hasattr(config, "workerinput"):
    _snapshot = config.workerinput["scaly_restored_cache"]
  else:
    _snapshot = {
      "index": [str(path) for path in (root / "index").glob("*.json")],
      "libraries": [str(path) for path in root.glob(f"*/lib*{shared_lib_ext()}")],
    }
  restored_index = set(_snapshot["index"])
  restored_libraries = set(_snapshot["libraries"])
  read = jit._index_read
  build = jit._build_artifact

  def index_read(index):
    artifact = read(index)
    path = jit._index_path(index)
    if path.parent.parent == root:
      kind = (
        "index misses"
        if artifact is None or not artifact.lib_path.exists()
        else "restored index hits"
        if str(path) in restored_index
        else "new index hits"
      )
      _counts[kind] += 1
    return artifact

  def build_artifact(fun):
    artifact = build(fun)
    if artifact.lib_path.parent.parent == root:
      _counts["restored library uses" if str(artifact.lib_path) in restored_libraries else "other library uses"] += 1
    return artifact

  _patch.setattr(jit, "_index_read", index_read)
  _patch.setattr(jit, "_build_artifact", build_artifact)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
  node.workerinput["scaly_restored_cache"] = _snapshot


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
  _counts.update(node.workeroutput.get("scaly_cache_counts", {}))


def pytest_sessionfinish(session):
  if hasattr(session.config, "workeroutput"):
    session.config.workeroutput["scaly_cache_counts"] = dict(_counts)


def pytest_terminal_summary(terminalreporter):
  terminalreporter.section("nightly JIT cache diagnostic")
  terminalreporter.write_line(f"Restored snapshot: {len(_snapshot['index'])} index entries, {len(_snapshot['libraries'])} libraries")
  terminalreporter.write_line("Counts cover pytest workers using SCALY_CACHE_DIR; subprocesses and isolated test caches are excluded.")
  for kind in ("restored index hits", "new index hits", "index misses", "restored library uses", "other library uses"):
    terminalreporter.write_line(f"{kind}: {_counts[kind]}")


def pytest_unconfigure(config):
  _patch.undo()
