import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

cache_dir = Path(".pytest_cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("CACHEDB", str(cache_dir / "tinygrad_cache.db"))
os.environ.setdefault("PYTHON", "1")

from tinygrad import Tensor, dtypes

from extra.fpga.capture import CaptureMetadata, capture_from_tensor, wrap_dma_buffer


def test_wrap_dma_buffer_reads_host_memory():
  raw = (ctypes.c_uint8 * 16)(*range(16))
  ptr = ctypes.addressof(raw)
  metadata = CaptureMetadata(protocol="pcie", lanes=4, sample_rate_hz=8e9)

  capture = wrap_dma_buffer(ptr, (4, 4), dtype=dtypes.uint8, metadata=metadata)

  np.testing.assert_array_equal(capture.tensor.numpy(), np.arange(16, dtype=np.uint8).reshape(4, 4))
  raw[0] = 99
  np.testing.assert_array_equal(capture.tensor.numpy()[0, 0], np.uint8(99))


@pytest.mark.parametrize("overlap", [0, 2])
def test_capture_chunker_alignment(overlap):
  tensor = Tensor.arange(0, 24, dtype=dtypes.int32).reshape(2, 12)
  capture = capture_from_tensor(tensor, metadata=CaptureMetadata(protocol="ethernet", lanes=2))
  ref = tensor.numpy()

  chunks = list(capture.chunk(symbols_per_chunk=5, axis=1, overlap=overlap))
  step = 5 - overlap if overlap else 5
  start = 0
  for chunk in chunks:
    width = chunk.tensor.shape[1]
    np.testing.assert_array_equal(chunk.tensor.numpy(), ref[:, start:start+width])
    start += step


def test_select_lanes_supports_axis_argument():
  tensor = Tensor.arange(0, 48, dtype=dtypes.int32).reshape(3, 4, 4)
  capture = capture_from_tensor(tensor, metadata=CaptureMetadata(protocol="pcie", lanes=4))
  subset = capture.select_lanes([1, 3], axis=1)
  np.testing.assert_array_equal(subset.tensor.numpy(), tensor.numpy()[:, [1, 3], :])
