from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict
from typing import Any

import numpy as np

from alloy.utils import load_torch_state_dict


def test_load_torch_state_dict_reads_modern_torch_zip_without_torch(tmp_path) -> None:
  torch = types.ModuleType("torch")
  torch_utils = types.ModuleType("torch._utils")

  class FloatStorage:
    pass

  FloatStorage.__module__ = "torch"
  FloatStorage.__qualname__ = "FloatStorage"
  setattr(torch, "FloatStorage", FloatStorage)

  def _rebuild_tensor_v2(*args: Any) -> None:
    del args
    raise RuntimeError("pickle should only record this global")

  _rebuild_tensor_v2.__module__ = "torch._utils"
  _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
  setattr(torch_utils, "_rebuild_tensor_v2", _rebuild_tensor_v2)

  old_torch = sys.modules.get("torch")
  old_utils = sys.modules.get("torch._utils")
  sys.modules["torch"] = torch
  sys.modules["torch._utils"] = torch_utils
  try:
    storage = np.arange(12, dtype=np.float32)
    matrix = storage.reshape(3, 4)
    vector = storage[2:7]

    class StorageRef:
      pass

    storage_ref = StorageRef()

    class FakeTensor:
      def __init__(self, arr: np.ndarray):
        self.arr = arr

      def __reduce__(self):
        offset = int((self.arr.ctypes.data - storage.ctypes.data) // storage.itemsize)
        stride = tuple(int(x // storage.itemsize) for x in self.arr.strides)
        return (_rebuild_tensor_v2, (storage_ref, offset, self.arr.shape, stride, False, OrderedDict()))

    class TorchPickler(pickle.Pickler):
      def persistent_id(self, obj: Any) -> tuple[Any, ...] | None:
        return ("storage", FloatStorage, "0", "cpu", storage.size) if obj is storage_ref else None

    data = io.BytesIO()
    pickler = TorchPickler(data, protocol=2)
    pickler.dump(OrderedDict([("matrix", FakeTensor(matrix)), ("vector", FakeTensor(vector))]))
  finally:
    if old_torch is None:
      sys.modules.pop("torch", None)
    else:
      sys.modules["torch"] = old_torch
    if old_utils is None:
      sys.modules.pop("torch._utils", None)
    else:
      sys.modules["torch._utils"] = old_utils

  path = tmp_path / "state.pth"
  with zipfile.ZipFile(path, "w") as zf:
    zf.writestr("state/data.pkl", data.getvalue())
    zf.writestr("state/data/0", storage.tobytes())

  state = load_torch_state_dict(path)
  np.testing.assert_array_equal(state["matrix"], matrix)
  np.testing.assert_array_equal(state["vector"], vector)
