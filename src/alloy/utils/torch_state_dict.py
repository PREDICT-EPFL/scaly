from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_TORCH_STORAGE_DTYPES = {
  "DoubleStorage": np.dtype(np.float64),
  "FloatStorage": np.dtype(np.float32),
  "HalfStorage": np.dtype(np.float16),
  "LongStorage": np.dtype(np.int64),
  "IntStorage": np.dtype(np.int32),
  "ShortStorage": np.dtype(np.int16),
  "CharStorage": np.dtype(np.int8),
  "ByteStorage": np.dtype(np.uint8),
  "BoolStorage": np.dtype(np.bool_),
  "UntypedStorage": np.dtype(np.uint8),
}

_TORCH_DTYPES = {
  "float64": np.dtype(np.float64),
  "double": np.dtype(np.float64),
  "float32": np.dtype(np.float32),
  "float": np.dtype(np.float32),
  "float16": np.dtype(np.float16),
  "half": np.dtype(np.float16),
  "int64": np.dtype(np.int64),
  "long": np.dtype(np.int64),
  "int32": np.dtype(np.int32),
  "int": np.dtype(np.int32),
  "int16": np.dtype(np.int16),
  "short": np.dtype(np.int16),
  "int8": np.dtype(np.int8),
  "uint8": np.dtype(np.uint8),
  "bool": np.dtype(np.bool_),
}


def _torch_dtype(x: Any) -> np.dtype:
  if isinstance(x, np.dtype):
    return x
  if hasattr(x, "dtype"):
    return np.dtype(x.dtype)
  return np.dtype(x)


def load_torch_state_dict(path: str | Path) -> dict[str, np.ndarray]:
  """Load a simple PyTorch ``state_dict`` zip checkpoint without importing torch.

  Supports the modern ``torch.save(state_dict, ...)`` zip layout: ``data.pkl``
  stores tensor metadata and ``data/<n>`` stores raw CPU storage bytes. This is
  intentionally narrow; unsupported pickle globals fail loudly instead of being
  materialized.
  """
  p = Path(path)
  storage_blobs: dict[str, bytes] = {}

  def rebuild_tensor(storage: tuple[Any, ...], storage_offset: int, size: tuple[int, ...], stride: tuple[int, ...], *args: Any) -> np.ndarray:
    del args
    if len(storage) < 5 or storage[0] != "storage":
      raise TypeError(f"unsupported torch storage persistent id: {storage!r}")
    dtype = _torch_dtype(storage[1])
    key, numel = str(storage[2]), int(storage[4])
    raw = storage_blobs[key]
    base = np.frombuffer(raw, dtype=dtype, count=numel)
    shape = tuple(int(x) for x in size)
    strides = tuple(int(x) * dtype.itemsize for x in stride)
    if not shape:
      return np.asarray(base[int(storage_offset)], dtype=dtype)
    return np.lib.stride_tricks.as_strided(base[int(storage_offset) :], shape=shape, strides=strides).copy()

  def rebuild_parameter(data: np.ndarray, *args: Any) -> np.ndarray:
    del args
    return data

  class Parameter:
    tensor: np.ndarray

    def __setstate__(self, state: tuple[np.ndarray, ...]) -> None:
      self.tensor = state[0]

  class TorchPickle(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
      if module == "torch" and name in _TORCH_STORAGE_DTYPES:
        return _TORCH_STORAGE_DTYPES[name]
      if module == "torch" and name in _TORCH_DTYPES:
        return _TORCH_DTYPES[name]
      if module in {"torch._utils", "torch._tensor"} and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
        return rebuild_tensor
      if module in {"torch._utils", "torch.nn.parameter"} and name in {"_rebuild_parameter", "_rebuild_parameter_with_state"}:
        return rebuild_parameter
      if module == "torch.nn.parameter" and name == "Parameter":
        return Parameter
      if module in {"collections", "numpy", "numpy.core.multiarray", "_codecs", "builtins"}:
        return super().find_class(module, name)
      raise pickle.UnpicklingError(f"unsupported global in torch checkpoint: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
      return pid

  if not zipfile.is_zipfile(p):
    raise ValueError(f"unsupported legacy non-zip PyTorch checkpoint: {p}")
  with zipfile.ZipFile(p) as zf:
    names = zf.namelist()
    base = names[0].split("/", 1)[0]
    storage_blobs = {name.rsplit("/", 1)[-1]: zf.read(name) for name in names if name.startswith(f"{base}/data/")}
    with zf.open(f"{base}/data.pkl") as data:
      state = TorchPickle(data).load()
  if not isinstance(state, dict):
    raise TypeError(f"expected a state_dict in {p}, got {type(state).__name__}")
  return {k: v.tensor if isinstance(v, Parameter) else v for k, v in state.items()}


__all__ = ["load_torch_state_dict"]
