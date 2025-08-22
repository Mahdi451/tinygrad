# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tinygrad is a minimalist deep learning framework that follows a "RISC-like" approach compared to PyTorch's "CISC-like" complexity. It implements lazy evaluation where operations build computation graphs that get optimized and fused before execution.

## Development Commands

### Installation and Setup
```bash
# Install from source (recommended)
python3 -m pip install -e .

# Install with testing dependencies
python3 -m pip install -e '.[testing]'

# Install with linting dependencies
python3 -m pip install -e '.[linting]'
```

### Testing
```bash
# Run specific test files
python3 test/test_ops.py
python3 test/test_tensor.py

# Run full test suite
python3 -m pytest test/

# Run with parallel execution
python3 -m pytest test/ -n auto

# Run unit tests only
python3 -m pytest test/unit/

# Run external tests (longer running)
python3 -m pytest test/external/
```

### Linting and Type Checking
```bash
# Run ruff linter
ruff check .

# Run mypy type checker
mypy tinygrad/

# Run pre-commit hooks
pre-commit run --all-files

# Install pre-commit hooks (run once)
pre-commit install
```

### Debugging and Development
```bash
# Debug tensor operations with kernel visualization
DEBUG=3 python3 -c "from tinygrad import Tensor; ..."

# Show generated code
DEBUG=4 python3 -c "from tinygrad import Tensor; ..."

# Check default accelerator
python3 -c "from tinygrad import Device; print(Device.DEFAULT)"

# Profile and benchmark
python3 -m pytest test/test_speed_v_torch.py

# Run examples
python3 examples/beautiful_mnist.py
python3 examples/stable_diffusion.py
```

## Architecture Overview

Tinygrad follows a 4-layer abstraction model:
```
Tensor (Frontend) → UOp (Computation Graph) → Schedule/Kernelize → Runtime (Backend)
```

### Core Components

**Tensor Layer** (`tinygrad/tensor.py`)
- High-level user interface with autograd support
- Lazy evaluation - operations create UOp graphs without immediate execution
- Shape tracking via ShapeTracker for zero-copy movement operations

**UOp System** (`tinygrad/uop/`)
- Low-level computation representation with ~70 primitive operations
- Immutable computation graphs with pattern matching and rewriting
- `UOp`: Core computation node, `Ops`: primitive operations enum

**Engine** (`tinygrad/engine/`)
- `schedule.py`: Orders operations and manages dependencies
- `realize.py`: Converts scheduled operations to executable programs
- `memory.py`: Manages buffer allocation and memory planning
- `jit.py`: Just-in-time compilation and caching

**Shape Management** (`tinygrad/shape/`)
- `ShapeTracker`: Manages tensor reshaping without data movement
- `View`: Represents memory layout transformations
- Movement operations (reshape, permute, pad) are "free" - only modify indexing

**Kernelize** (`tinygrad/kernelize/`)
- Groups UOps into executable kernels with fusion
- Memory optimization to minimize buffer allocations
- Dependency analysis for correct execution order

**Code Generation** (`tinygrad/codegen/`)
- `linearize.py`: Converts UOp graphs to linear instruction sequences
- Loop optimization, memory coalescing, vectorization
- BEAM search for optimal kernel configurations

**Runtime/Backends** (`tinygrad/runtime/`)
- Unified interface across accelerators: CPU, CUDA, Metal, AMD, OpenCL, WebGPU
- Device-specific optimizations handled transparently
- `ops_*.py` files implement backend-specific operations

### Execution Flow

1. Tensor operations create UOp computation graphs (lazy evaluation)
2. When data is needed (`.numpy()`, `.item()`), graphs are realized
3. UOps are grouped into kernels with fusion optimization
4. Operations are scheduled respecting dependencies
5. Kernels are rendered to target-specific code and compiled
6. Programs execute on target devices with optimized memory management

## Development Guidelines

### Code Style
- Line length: 150 characters (ruff.toml)
- Use 2-space indentation
- Follow existing patterns in the codebase
- Core library focuses on simplicity and readability over line count

### Testing Requirements
- All new features must have regression tests
- Bug fixes require tests that would have caught the bug
- Use `@unittest.expectedFailure` for known broken functionality
- Avoid brittle tests that break with minor implementation changes

### Performance Considerations
- Any claimed "speedup" must be benchmarked
- Consider simplicity vs performance tradeoffs
- Use process replay tests for refactors: include [pr] in PR title
- BEAM search finds optimal kernel configurations automatically

### Backend Development
- Adding new accelerators requires implementing ~25 low-level ops
- Follow existing backend patterns in `tinygrad/runtime/ops_*.py`
- Ensure device-specific optimizations are transparent to users

### Common Patterns
- **Lazy Evaluation**: Build graphs, defer execution until needed
- **Zero-Copy Moves**: Use ShapeTracker for reshaping without data copies
- **Kernel Fusion**: Combine operations to reduce memory bandwidth
- **UOp Rewriting**: Use pattern matching for algebraic simplifications

## Key Files and Directories

**Core Framework**
- `tinygrad/tensor.py`: Main user-facing Tensor class
- `tinygrad/device.py`: Device management and buffer abstraction
- `tinygrad/dtype.py`: Data type definitions and conversions

**Computation**
- `tinygrad/uop/ops.py`: Primitive operation definitions
- `tinygrad/uop/symbolic.py`: Symbolic mathematics system
- `tinygrad/shape/shapetracker.py`: Shape and memory layout tracking

**Execution Engine**
- `tinygrad/engine/schedule.py`: Operation scheduling
- `tinygrad/engine/realize.py`: Graph realization
- `tinygrad/kernelize/kernelize.py`: Kernel grouping and fusion

**Neural Networks**
- `tinygrad/nn/`: Neural network layers and utilities
- `tinygrad/nn/optim.py`: Optimizers (Adam, SGD, etc.)

**Examples and Tests**
- `examples/`: Working examples including LLaMA, Stable Diffusion
- `test/`: Comprehensive test suite
- `test/external/`: Longer-running external tests

This architecture enables tinygrad to be both extremely simple (~10k lines) and highly efficient, achieving competitive performance with much larger frameworks.