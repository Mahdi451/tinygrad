import os
from pathlib import Path

import numpy as np

cache_dir = Path(".pytest_cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("CACHEDB", str(cache_dir / "tinygrad_cache.db"))
os.environ.setdefault("PYTHON", "1")

from tinygrad import Tensor, dtypes

from extra.fpga.compliance import (
  ComplianceSuite,
  bit_error_rate_check,
  jitter_peak_to_peak_check,
  lane_skew_check,
)


def test_bit_error_rate_check_passes_when_under_threshold():
  bits = Tensor([0, 1, 0, 1], dtype=dtypes.uint8)
  ref = Tensor([0, 1, 1, 1], dtype=dtypes.uint8)
  result = bit_error_rate_check(bits, ref, threshold=0.5)
  assert result.passed.item() is True
  np.testing.assert_allclose(result.measurement.numpy(), 0.25)


def test_jitter_pp_and_skew_checks_feed_suite():
  edges = Tensor([[0.0, 100.0, 200.5, 300.0]], dtype=dtypes.float32)
  jitter_result = jitter_peak_to_peak_check(edges, ui_ps=100.0, limit_ui=0.5)
  skew_result = lane_skew_check(Tensor([[0.0, 20.0, 40.0]], dtype=dtypes.float32), limit_ps=50.0)
  suite = ComplianceSuite()
  suite.add(jitter_result)
  suite.add(skew_result)
  assert suite.overall_pass() is True
