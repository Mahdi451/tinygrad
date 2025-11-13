from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from math import prod
from typing import Any, List

from tinygrad import Tensor, dtypes


@dataclass(slots=True)
class PCIeCommand:
  opcode: str
  addr: int | None = None
  length: int | None = None
  args: dict[str, Any] = field(default_factory=dict)


class MailboxInterface:
  """Tiny mailbox shim that records PCIe commands for debugging."""

  def __init__(self):
    self.commands: List[PCIeCommand] = []

  def send(self, opcode:str, *, addr:int|None=None, length:int|None=None, **kwargs):
    self.commands.append(PCIeCommand(opcode, addr, length, kwargs))


class FPGABuffer:
  """Host buffer that mirrors an FPGA BAR mapping and can be wrapped as a Tensor."""

  def __init__(self, size:int, handle:int, alignment:int=64):
    if size % alignment != 0: raise ValueError(f"size {size} must be a multiple of alignment {alignment}")
    self.size, self.handle, self.alignment = size, handle, alignment
    self._storage = ctypes.create_string_buffer(size)
    self.addr = ctypes.addressof(self._storage)

  def copyin(self, mv:memoryview):
    if len(mv) != self.size: raise ValueError("size mismatch for copyin")
    data = bytes(mv)
    ctypes.memmove(self.addr, data, self.size)

  def copyout(self) -> bytes:
    return bytes(self._storage.raw)

  def as_tensor(self, shape:tuple[int, ...], dtype=dtypes.uint8) -> Tensor:
    if not isinstance(shape, tuple): shape = tuple(shape)
    if len(shape) == 0 or any((not isinstance(dim, int)) or dim <= 0 for dim in shape):
      raise ValueError("shape must be a tuple of positive integers")
    if not hasattr(dtype, "itemsize"):
      raise TypeError("dtype must be a tinygrad dtype")
    if prod(shape) * dtype.itemsize != self.size:
      raise ValueError("shape and dtype must match buffer size")
    return Tensor.from_blob(self.addr, shape, dtype=dtype, device="CPU")


class FPGARuntime:
  def __init__(self, mailbox:MailboxInterface|None=None):
    self.mailbox = mailbox or MailboxInterface()
    self._handle = 0
    self.buffers: dict[int, FPGABuffer] = {}

  def alloc(self, size:int, *, alignment:int=64) -> FPGABuffer:
    self._handle += 1
    buf = FPGABuffer(size, self._handle, alignment=alignment)
    self.buffers[self._handle] = buf
    self.mailbox.send("ALLOC", addr=buf.addr, length=size, handle=self._handle, alignment=alignment)
    return buf

  def free(self, handle:int):
    buf = self.buffers.pop(handle)
    self.mailbox.send("FREE", addr=buf.addr, handle=handle)

  def copyin(self, handle:int, data:bytes):
    buf = self.buffers[handle]
    buf.copyin(memoryview(data))
    self.mailbox.send("WRITE", addr=buf.addr, length=len(data), handle=handle)

  def copyout(self, handle:int) -> bytes:
    buf = self.buffers[handle]
    self.mailbox.send("READ", addr=buf.addr, length=buf.size, handle=handle)
    return buf.copyout()

  def launch(self, program_id:int, *, global_size:tuple[int,int,int], local_size:tuple[int,int,int], params:dict[str, Any]|None=None):
    self.mailbox.send("LAUNCH", addr=None, length=None, program=program_id, global_size=global_size, local_size=local_size, params=params or {})
