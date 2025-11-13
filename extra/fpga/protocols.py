from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tinygrad import Tensor, dtypes

K28_5_POS = 0b0011111010
K28_5_NEG = 0b1100000101
TMDS_CONTROL_CODES = {
  0b1101010100: (0, 0),
  0b0010101011: (0, 1),
  0b0101010100: (1, 0),
  0b1010101011: (1, 1),
}


def _ensure_bits(symbols:Tensor, width:int) -> Tensor:
  if symbols.shape[-1] == width: return symbols.cast(dtypes.bool)
  if not dtypes.is_int(symbols.dtype): raise ValueError(f"expected integer tensor or bit tensor, got {symbols.dtype}")
  mask = Tensor([1 << i for i in range(width)], device=symbols.device, dtype=dtypes.int64).reshape(*([1] * symbols.ndim), width)
  return symbols.unsqueeze(-1).bitwise_and(mask).ne(0)


def _bits_to_int(bits:Tensor) -> Tensor:
  weights = Tensor([1 << i for i in range(bits.shape[-1])], device=bits.device, dtype=dtypes.int64).reshape(
    *([1] * (bits.ndim - 1)), bits.shape[-1])
  return (bits.cast(dtypes.int64) * weights).sum(-1)


def _bits_to_bytes(bits:Tensor) -> Tensor:
  reshaped = bits.reshape(*bits.shape[:-1], -1, 8).cast(dtypes.int32)
  weights = Tensor([1 << i for i in range(8)], device=bits.device, dtype=dtypes.int32).reshape(
    *([1] * (reshaped.ndim - 1)), 8)
  return (reshaped * weights).sum(-1).cast(dtypes.uint8)


def _running_disparity(bits:Tensor, axis:int=-1) -> Tensor:
  signed = bits.cast(dtypes.int32).mul(2).sub(1)
  return signed.cumsum(axis=axis)


@dataclass(slots=True)
class FramedPayload:
  payload: Tensor
  sync: Tensor
  disparity: Tensor


def pcie_128b130b_decode(blocks:Tensor) -> FramedPayload:
  bits = _ensure_bits(blocks, 130)
  sync = bits[..., :2].cast(dtypes.uint8)
  payload = bits[..., 2:]
  disparity = _running_disparity(payload, axis=-1)
  return FramedPayload(payload, sync, disparity)


def ethernet_64b66b_decode(blocks:Tensor) -> FramedPayload:
  bits = _ensure_bits(blocks, 66)
  sync = bits[..., :2].cast(dtypes.uint8)
  payload_bits = bits[..., 2:]
  disparity = _running_disparity(payload_bits, axis=-1)
  payload_bytes = _bits_to_bytes(payload_bits)
  return FramedPayload(payload_bytes, sync, disparity)


def pcie_ordered_set_mask(symbols:Tensor) -> Tensor:
  ints = _bits_to_int(_ensure_bits(symbols, 10)) if symbols.shape[-1] != 10 else symbols
  return (ints == K28_5_POS) | (ints == K28_5_NEG)


def pcie_8b10b_running_disparity(symbols:Tensor) -> Tensor:
  bits = _ensure_bits(symbols, 10)
  return _running_disparity(bits, axis=-1)


@dataclass(slots=True)
class TMDSDecode:
  data_bytes: Tensor
  control_mask: Tensor
  hsync: Tensor
  vsync: Tensor


def _decode_tmds_data(symbols:Tensor) -> Tensor:
  bits = _ensure_bits(symbols, 10).cast(dtypes.int32)
  invert_flag = bits[..., 9:10].cast(dtypes.bool)
  use_xnor = bits[..., 8:9].logical_not()
  payload = bits[..., :8]
  inv_mask = invert_flag.cast(dtypes.int32).expand(*payload.shape)
  qm = payload.bitwise_xor(inv_mask)
  decoded = [qm[..., 0:1]]
  prev_q = qm[..., 0:1]
  for i in range(1, 8):
    cur_q = qm[..., i:i+1]
    xor = prev_q.bitwise_xor(cur_q)
    xnor = xor.logical_not().cast(dtypes.int32)
    next_bit = use_xnor.where(xnor, xor)
    decoded.append(next_bit)
    prev_q = cur_q
  decoded_bits = decoded[0].cat(*decoded[1:], dim=-1)
  decoded_bytes = _bits_to_bytes(decoded_bits.cast(dtypes.uint8)).reshape(symbols.shape)
  return decoded_bytes.cast(dtypes.uint8)


def encode_tmds_data(data_bytes:Tensor) -> Tensor:
  bits = _ensure_bits(data_bytes, 8).cast(dtypes.int32)
  ones = bits.sum(-1, keepdim=True)
  use_xnor = (ones > 4) | ((ones == 4) & (bits[..., :1] == 0))
  qm = [bits[..., 0:1]]
  prev = qm[0]
  for i in range(1, 8):
    cur = bits[..., i:i+1]
    xor = prev.bitwise_xor(cur)
    xnor = xor.logical_not().cast(dtypes.int32)
    nxt = use_xnor.where(xnor, xor)
    qm.append(nxt)
    prev = nxt
  qm_bits = qm[0].cat(*qm[1:], dim=-1)
  qm_flag = use_xnor.logical_not().cast(dtypes.int32)
  invert = (qm_bits.sum(-1, keepdim=True) > 4).cast(dtypes.int32)
  payload = qm_bits.bitwise_xor(invert.expand(*qm_bits.shape))
  payload_int = _bits_to_int(payload.cast(dtypes.uint16))
  return payload_int | (qm_flag.squeeze(-1).cast(dtypes.uint16) << 8) | (invert.squeeze(-1).cast(dtypes.uint16) << 9)


def hdmi_tmds_decode(symbols:Tensor) -> TMDSDecode:
  ints = _bits_to_int(_ensure_bits(symbols, 10)) if symbols.shape[-1] != 10 else symbols
  ints = ints.cast(dtypes.uint16)
  control_mask = Tensor.zeros(*ints.shape, dtype=dtypes.bool, device=ints.device)
  hsync = Tensor.zeros(*ints.shape, dtype=dtypes.bool, device=ints.device)
  vsync = Tensor.zeros(*ints.shape, dtype=dtypes.bool, device=ints.device)
  for code, (hs, vs) in TMDS_CONTROL_CODES.items():
    mask = ints == code
    control_mask = control_mask | mask
    if hs: hsync = hsync | mask
    if vs: vsync = vsync | mask
  data_symbols = control_mask.where(Tensor.zeros_like(ints, dtype=ints.dtype), ints)
  data_bytes = _decode_tmds_data(data_symbols)
  return TMDSDecode(data_bytes, control_mask, hsync, vsync)
