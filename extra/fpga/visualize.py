from __future__ import annotations

import numpy as np

from tinygrad import Tensor, dtypes


def _scalar_from_range(value) -> float:
  if isinstance(value, Tensor): return float(value.item())
  return float(value)


def tensor_histogram(values:Tensor, bins:int=64, value_range:tuple[float | Tensor, float | Tensor]|None=None) -> dict[str, np.ndarray]:
  vals = values.cast(dtypes.float32).realize()
  if value_range is not None:
    lo, hi = (_scalar_from_range(value_range[0]), _scalar_from_range(value_range[1]))
  else:
    lo, hi = float(vals.min().item()), float(vals.max().item())
  if lo == hi: hi = lo + 1.0
  edges = Tensor.linspace(lo, hi, bins+1, device=vals.device, dtype=vals.dtype)
  norm = (vals - lo) / (hi - lo)
  buckets = norm.mul(bins).floor().clip(0, bins-1).cast(dtypes.int32).reshape(-1, 1)
  counts = buckets._one_hot_along_dim(bins).sum(0).reshape(bins)
  return {"edges": edges.numpy(), "counts": counts.numpy()}


def phase_error_trace(edge_times_ps:Tensor, ui_ps:float) -> dict[str, np.ndarray]:
  idx = Tensor.arange(edge_times_ps.shape[-1], dtype=edge_times_ps.dtype, device=edge_times_ps.device)
  reference = edge_times_ps[..., :1] + idx * ui_ps
  phase = (edge_times_ps - reference).reshape(-1)
  return {"phase_error_ps": phase.numpy()}
