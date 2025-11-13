import os
from pathlib import Path

import numpy as np

cache_dir = Path(".pytest_cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("CACHEDB", str(cache_dir / "tinygrad_cache.db"))
os.environ.setdefault("PYTHON", "1")

from tinygrad import Tensor, dtypes

from extra.fpga.protocols import (
  TMDS_CONTROL_CODES,
  K28_5_POS,
  encode_tmds_data,
  ethernet_64b66b_decode,
  hdmi_tmds_decode,
  pcie_128b130b_decode,
  pcie_ordered_set_mask,
)


def test_pcie_128b130b_decode_basic():
  sync = [1, 0]
  payload = ([0, 1] * 64)[:128]
  block = sync + payload
  tensor = Tensor(block, dtype=dtypes.uint8).reshape(1, 1, 130)

  decoded = pcie_128b130b_decode(tensor)

  np.testing.assert_array_equal(decoded.sync.numpy().reshape(-1), np.array(sync, dtype=np.uint8))
  np.testing.assert_array_equal(decoded.payload.numpy()[0, 0], np.array(payload, dtype=np.uint8))
  expected_disp = np.cumsum(np.array(payload, dtype=np.int32) * 2 - 1)
  np.testing.assert_array_equal(decoded.disparity.numpy()[0, 0], expected_disp)


def test_pcie_ordered_set_mask_detects_com():
  symbols = Tensor([0, K28_5_POS, 0], dtype=dtypes.uint16)
  mask = pcie_ordered_set_mask(symbols)
  assert mask[1].item() is True
  assert mask[0].item() is False


def test_ethernet_64b66b_decode_returns_bytes():
  sync = [1, 1]
  payload_bits = [int(i % 2 == 0) for i in range(64)]
  frame = Tensor(sync + payload_bits, dtype=dtypes.uint8).reshape(1, 66)
  decoded = ethernet_64b66b_decode(frame)
  np.testing.assert_array_equal(decoded.sync.numpy().reshape(-1), np.array(sync, dtype=np.uint8))
  assert decoded.payload.shape[-1] == 8


def test_hdmi_tmds_decode_control_code():
  code = next(iter(TMDS_CONTROL_CODES.keys()))
  symbols = Tensor([code], dtype=dtypes.uint16)
  decoded = hdmi_tmds_decode(symbols)
  assert decoded.control_mask[0].item() is True


def test_hdmi_tmds_decode_roundtrip_random_data():
  data = Tensor.randint(12, high=256, dtype=dtypes.int32)
  symbols = encode_tmds_data(data.cast(dtypes.uint8))
  decoded = hdmi_tmds_decode(symbols)
  np.testing.assert_array_equal(decoded.data_bytes.numpy(), data.cast(dtypes.uint8).numpy())
  assert bool(decoded.control_mask.max().item()) is False
