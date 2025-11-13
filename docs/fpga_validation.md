# FPGA Validation Architecture

This document outlines the modules we are adding to help electrical validation engineers analyze PCIe, Ethernet, and HDMI traffic with tinygrad.

## Capture Loader

* Location: `extra/fpga/capture.py`
* Goals:
  * Wrap DMA buffers that arrive from FPGA testers without copying (`Tensor.from_blob`).
  * Track per-capture metadata (`CaptureMetadata`) containing timestamp domains, lane count, encoding (PCIe 128b/130b, 64b/66b, TMDS), and sampling rate.
  * Provide chunked realizations so extremely long captures can be sliced into protocol-aligned batches via `CaptureChunker`, yielding lazily realized Tensors.

## Protocol Analysis Kernels

* Location: `extra/fpga/protocols.py`
* Goals:
  * PCIe utilities for block alignment, scrambling descramble, disparity, and ordered-set detection over tensors shaped `(lanes, symbols)`.
  * Ethernet helpers for 64b/66b sync header extraction, block decoding, and PCS BER counters.
  * HDMI TMDS decoder that maps 10-bit symbols to pixel/control codes and surfaces lane skew via Tensor reductions.
* All kernels stay pure tinygrad graphs so they can be JITed and profiled with the usual `DEBUG` tooling.

## Compliance Harness

* Location: `extra/fpga/compliance.py`
* Goals:
  * Define threshold assertions (timing, BER, jitter, skew) as Tensor ops returning boolean masks plus scalar margins.
  * Optional autograd pass (using `.backward()`) lets us correlate parameter sensitivities (e.g., equalization settings) with failed checks.
  * Provide presets for PCIe, Ethernet, and HDMI referencing IEEE tables so validation suites stay self-documenting.

## Visualization Hooks

* Location: `extra/fpga/visualize.py`
* Goals:
  * Convert Tensor summaries (histograms, eye approximations, skew traces) into small NumPy payloads that plotting libraries can consume.
  * Keep heavy computation in tinygrad; only final samples are materialized to NumPy to minimize host memory use.

## FPGA Runtime Shim

* Location: `extra/fpga/runtime.py`
* Goals:
  * Provide a minimal `TinyDevice` abstraction backed by PCIe BAR or mailbox commands so the FPGA can serve as a tinygrad accelerator.
  * Implement the ~25 required ops incrementally (alloc, copyin/out, kernel launch placeholder) and expose a `FPGARuntime` hook for experiments.

With this layout we can implement each layer independently while sharing metadata and chunking utilities across all validation workflows.
