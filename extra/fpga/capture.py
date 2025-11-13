from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Sequence

from tinygrad import Tensor, dtypes

ProtocolName = Literal["pcie", "ethernet", "hdmi"]


@dataclass(slots=True)
class CaptureMetadata:
  """Describes how a captured tensor should be interpreted."""

  protocol: ProtocolName
  lanes: int
  sample_rate_hz: float | None = None
  timestamp_ns: int | None = None
  description: str = ""
  extras: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(slots=True)
class Capture:
  """Small container coupling a Tensor capture with its metadata."""

  tensor: Tensor
  metadata: CaptureMetadata
  symbols_per_word: int = 1

  def chunk(self, symbols_per_chunk:int, *, axis:int=-1, overlap:int=0) -> "CaptureChunker":
    """Return an iterator that yields overlapping protocol-aligned slices."""
    return CaptureChunker(self, symbols_per_chunk, axis=axis, overlap=overlap)

  def select_lanes(self, lanes:Sequence[int], axis:int=0) -> "Capture":
    """Return a view that only keeps the requested electrical lanes along `axis`."""
    if len(lanes) == 0: raise ValueError("lanes must be non-empty")
    ndim = self.tensor.ndim
    axis = axis if axis >= 0 else ndim + axis
    if not 0 <= axis < ndim: raise ValueError(f"axis {axis} out of range for shape {self.tensor.shape}")
    idx = Tensor(lanes, dtype=dtypes.int32, device=self.tensor.device)
    target = self.tensor if axis == 0 else self.tensor.transpose(axis, 0)
    expand_shape = [len(lanes), *target.shape[1:]]
    gather_index = idx.reshape(len(lanes), *([1] * (target.ndim - 1))).expand(*expand_shape)
    lane_view = target.gather(0, gather_index)
    lane_view = lane_view if axis == 0 else lane_view.transpose(0, axis)
    return Capture(lane_view, self.metadata, self.symbols_per_word)


class CaptureChunker:
  """Iterates over a Capture and yields sub-captures without copying data."""

  def __init__(self, capture:Capture, symbols_per_chunk:int, *, axis:int=-1, overlap:int=0):
    if symbols_per_chunk <= 0: raise ValueError("symbols_per_chunk must be > 0")
    if overlap < 0: raise ValueError("overlap must be >= 0")
    self.capture, self.symbols_per_chunk = capture, symbols_per_chunk
    self.axis = axis if axis >= 0 else len(capture.tensor.shape) + axis
    if not 0 <= self.axis < len(capture.tensor.shape):
      raise ValueError(f"axis {axis} out of range for shape {capture.tensor.shape}")
    if overlap >= symbols_per_chunk:
      raise ValueError("overlap must be smaller than symbols_per_chunk")
    self.overlap = overlap

  def __iter__(self) -> Iterator[Capture]:
    total_symbols = self.capture.tensor.shape[self.axis]
    step = self.symbols_per_chunk - self.overlap
    if step <= 0: step = self.symbols_per_chunk
    start = 0
    while start < total_symbols:
      end = min(start + self.symbols_per_chunk, total_symbols)
      slices = [slice(None)] * len(self.capture.tensor.shape)
      slices[self.axis] = slice(start, end)
      chunk_tensor = self.capture.tensor[tuple(slices)]
      yield Capture(chunk_tensor, self.capture.metadata, self.capture.symbols_per_word)
      start += step


def wrap_dma_buffer(ptr:int, shape:tuple[int, ...], *, dtype=dtypes.uint8, device:str="CPU",
    metadata:CaptureMetadata, symbols_per_word:int=1) -> Capture:
  """
  Wrap an FPGA DMA buffer (specified via host pointer) as a Tensor-backed capture.

  The caller must ensure the underlying memory remains valid for the lifetime of the Capture.
  """
  if ptr == 0: raise ValueError("dma pointer must be non-zero")
  if len(shape) == 0: raise ValueError("shape must describe at least one dimension")
  tensor = Tensor.from_blob(ptr, shape, dtype=dtype, device=device)
  return Capture(tensor, metadata, symbols_per_word)


def capture_from_tensor(tensor:Tensor, *, metadata:CaptureMetadata, symbols_per_word:int=1) -> Capture:
  """Helper for unit tests or file-backed captures that already reside in a Tensor."""
  return Capture(tensor, metadata, symbols_per_word)
