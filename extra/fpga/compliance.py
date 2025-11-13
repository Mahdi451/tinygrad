from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from tinygrad import Tensor, dtypes


@dataclass(slots=True)
class ComplianceResult:
  name: str
  measurement: Tensor
  limit: float
  margin: Tensor
  passed: Tensor

  def to_dict(self) -> dict:
    return {
      "name": self.name,
      "measurement": self.measurement.numpy(),
      "limit": self.limit,
      "margin": self.margin.numpy(),
      "passed": bool(self.passed.min().item()),
    }


def _limit_result(name:str, measurement:Tensor, limit:float, *, polarity:str) -> ComplianceResult:
  limit_tensor = Tensor(limit, device=measurement.device, dtype=measurement.dtype)
  margin = (limit_tensor - measurement) if polarity == "max" else (measurement - limit_tensor)
  passed = margin >= 0
  return ComplianceResult(name, measurement, limit, margin, passed)


def bit_error_rate_check(bits:Tensor, reference:Tensor, *, threshold:float) -> ComplianceResult:
  if bits.shape != reference.shape: raise ValueError("bit and reference shapes must match")
  errors = (bits != reference).cast(dtypes.float32)
  ber = errors.mean()
  return _limit_result("bit_error_rate", ber, threshold, polarity="max")


def jitter_peak_to_peak_check(edge_times_ps:Tensor, *, ui_ps:float, limit_ui:float) -> ComplianceResult:
  if edge_times_ps.ndim == 0: raise ValueError("edge_times_ps must describe at least one edge")
  idx = Tensor.arange(edge_times_ps.shape[-1], dtype=edge_times_ps.dtype, device=edge_times_ps.device)
  reference = edge_times_ps[..., :1] + idx * ui_ps
  phase_err = edge_times_ps - reference
  peak_to_peak = phase_err.max(-1) - phase_err.min(-1)
  limit = ui_ps * limit_ui
  return _limit_result("jitter_pp", peak_to_peak, limit, polarity="max")


def lane_skew_check(arrival_times_ps:Tensor, *, limit_ps:float) -> ComplianceResult:
  span = arrival_times_ps.max(-1) - arrival_times_ps.min(-1)
  return _limit_result("lane_skew", span, limit_ps, polarity="max")


@dataclass(slots=True)
class ComplianceSuite:
  results: list[ComplianceResult] = field(default_factory=list)

  def add(self, result:ComplianceResult) -> ComplianceResult:
    self.results.append(result)
    return result

  def overall_pass(self) -> bool:
    return all(bool(res.passed.min().item()) for res in self.results)

  def margins(self) -> dict[str, Tensor]:
    return {res.name: res.margin for res in self.results}
