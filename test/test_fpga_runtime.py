import os
from pathlib import Path

import numpy as np
import pytest

cache_dir = Path(".pytest_cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("CACHEDB", str(cache_dir / "tinygrad_cache.db"))
os.environ.setdefault("PYTHON", "1")

from tinygrad import dtypes

from extra.fpga.runtime import FPGARuntime


def test_fpga_runtime_buffers_roundtrip_and_tensor_wrap():
  rt = FPGARuntime()
  buf = rt.alloc(16, alignment=8)
  payload = bytes(range(16))
  rt.copyin(buf.handle, payload)
  out = rt.copyout(buf.handle)
  assert out == payload
  tensor = buf.as_tensor((16,), dtype=dtypes.uint8)
  np.testing.assert_array_equal(tensor.numpy(), np.frombuffer(payload, dtype=np.uint8))
  rt.launch(1, global_size=(1, 1, 1), local_size=(1, 1, 1), params={"handle": buf.handle})
  assert rt.mailbox.commands[-1].opcode == "LAUNCH"


def test_fpga_runtime_validates_alignment_and_shape():
  rt = FPGARuntime()
  with pytest.raises(ValueError):
    rt.alloc(10, alignment=16)
  buf = rt.alloc(32, alignment=32)
  with pytest.raises(ValueError):
    buf.as_tensor((), dtype=dtypes.uint8)
  with pytest.raises(TypeError):
    buf.as_tensor((32,), dtype="uint8")  # type: ignore[arg-type]
