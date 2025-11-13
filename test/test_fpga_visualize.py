import os
from pathlib import Path

import numpy as np

cache_dir = Path(".pytest_cache")
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("CACHEDB", str(cache_dir / "tinygrad_cache.db"))
os.environ.setdefault("PYTHON", "1")

from tinygrad import Tensor, dtypes

from extra.fpga.visualize import phase_error_trace, tensor_histogram


def test_tensor_histogram_counts_match_samples():
  values = Tensor.randn(1024, dtype=dtypes.float32)
  hist = tensor_histogram(values, bins=16, value_range=(Tensor(-2.0), Tensor(2.0)))
  assert hist["edges"].shape[0] == 17
  np.testing.assert_equal(int(hist["counts"].sum()), 1024)


def test_phase_error_trace_returns_numpy_payload():
  edges = Tensor([[0.0, 100.0, 205.0, 300.0]], dtype=dtypes.float32)
  trace = phase_error_trace(edges, ui_ps=100.0)
  assert "phase_error_ps" in trace
  assert isinstance(trace["phase_error_ps"], np.ndarray)
