# Day 1: Tinygrad Architecture Deep Dive & Development Environment

## Overview
Today we'll master the fundamental 4-layer architecture of tinygrad, set up a robust development environment, and understand the execution flow from Tensor operations to hardware execution. By the end, you'll have hands-on experience with debugging techniques and make your first contribution.

## Learning Objectives
- ✅ Understand tinygrad's 4-layer architecture in detail
- ✅ Master debugging techniques with DEBUG flags
- ✅ Trace execution from Tensor → UOp → Schedule → Runtime
- ✅ Set up development workflow with testing
- ✅ Make first meaningful contribution to codebase

---

## Part 1: Architecture Foundation (90 minutes)

### The 4-Layer Architecture

Tinygrad follows a clean separation of concerns through 4 distinct layers:

```
┌─────────────────────────────────────┐
│ 1. Tensor (Frontend) - User API    │ ← High-level operations, autograd
├─────────────────────────────────────┤
│ 2. UOp (Computation Graph)         │ ← Primitive operations, optimization
├─────────────────────────────────────┤
│ 3. Engine (Schedule/Kernelize)     │ ← Execution planning, memory management
├─────────────────────────────────────┤
│ 4. Runtime (Backend)               │ ← Device-specific execution
└─────────────────────────────────────┘
```

### Layer 1: Tensor Frontend (`tinygrad/tensor.py`)

The Tensor class is the main user interface, implementing lazy evaluation and autograd.

**Key Concepts:**
- **Lazy Evaluation**: Operations create computation graphs without immediate execution
- **Autograd**: Automatic gradient computation through backpropagation  
- **Shape Tracking**: Zero-copy operations through ShapeTracker

**Deep Dive Code Example:**

```python
#!/usr/bin/env python3
"""
Day 1 Exercise: Understanding Tensor Layer and Lazy Evaluation
"""

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import DEBUG, getenv
import os

def explore_tensor_fundamentals():
    """
    Understanding how Tensor operations work internally
    """
    print("=== Tensor Layer Deep Dive ===")
    
    # Enable detailed debugging
    os.environ['DEBUG'] = '2'
    
    # Create tensors - notice no computation happens yet!
    print("\\n1. Creating tensors (lazy evaluation):")
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor a device: {a.device}")
    print(f"Tensor a dtype: {a.dtype}")
    
    # Look at the underlying UOp - this is the computation graph node
    print(f"\\nTensor a UOp: {a.uop}")
    print(f"UOp operation: {a.uop.op}")
    print(f"UOp args: {a.uop.arg}")
    
    # Chain operations - still lazy!
    print("\\n2. Chaining operations (still lazy):")
    c = a + b  # Creates UOp graph, doesn't execute
    d = c * 2.0
    e = d.sum()
    
    print(f"Final tensor e UOp: {e.uop}")
    print(f"UOp source chain length: {len(list(e.uop.toposort()))}")
    
    # Force realization - this triggers execution
    print("\\n3. Forcing realization (.numpy() triggers execution):")
    result = e.numpy()  # THIS is when computation actually happens
    print(f"Final result: {result}")
    
    return result

def trace_execution_flow():
    """
    Detailed tracing of how operations flow through the system
    """
    print("\\n=== Execution Flow Tracing ===")
    
    # Set maximum debug level to see everything
    os.environ['DEBUG'] = '4'
    
    print("\\n1. Simple operation with full tracing:")
    x = Tensor([1.0, 2.0, 3.0])
    y = x * 2.0 + 1.0
    
    print("\\nBefore realization - examining UOp graph:")
    print(f"Number of operations in graph: {len(list(y.uop.toposort()))}")
    
    # Print the computation graph
    for i, uop in enumerate(y.uop.toposort()):
        print(f"  UOp {i}: {uop.op} {uop.arg}")
    
    print("\\nTriggering realization...")
    result = y.numpy()
    print(f"Result: {result}")
    
    return result

def understand_lazy_evaluation():
    """
    Compare lazy vs eager evaluation performance and behavior
    """
    print("\\n=== Lazy Evaluation Deep Dive ===")
    
    import time
    
    # Create large tensors for timing comparison
    print("\\n1. Building complex computation graph:")
    start_time = time.time()
    
    # This is all lazy - no actual computation!
    x = Tensor.randn(1000, 1000)
    for i in range(10):
        x = x * 2.0 + 1.0
        x = x.relu()
        x = x.sum(axis=1, keepdim=True)
    
    graph_build_time = time.time() - start_time
    print(f"Graph construction time: {graph_build_time:.6f} seconds")
    print(f"Operations in final graph: {len(list(x.uop.toposort()))}")
    
    # NOW the computation happens
    print("\\n2. Executing computation graph:")
    start_time = time.time()
    result = x.numpy()
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    print(f"Result shape: {result.shape}")
    
    return result

def explore_autograd_system():
    """
    Understanding automatic differentiation in tinygrad
    """
    print("\\n=== Autograd System Exploration ===")
    
    # Enable gradient computation
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    
    print(f"x requires_grad: {x.requires_grad}")
    print(f"y requires_grad: {y.requires_grad}")
    
    # Forward pass
    z = (x * y).sum()
    print(f"\\nForward result: {z.numpy()}")
    
    # Examine the computation graph for gradients
    print(f"Computation graph size: {len(list(z.uop.toposort()))}")
    
    # Backward pass
    z.backward()
    
    print(f"\\nGradient of x: {x.grad.numpy()}")
    print(f"Gradient of y: {y.grad.numpy()}")
    
    # Verify gradients manually
    print("\\nManual gradient verification:")
    print(f"∂z/∂x should equal y: {y.numpy()}")
    print(f"∂z/∂y should equal x: {x.numpy()}")
    
    return x.grad.numpy(), y.grad.numpy()

if __name__ == "__main__":
    # Run all exploration functions
    explore_tensor_fundamentals()
    trace_execution_flow()
    understand_lazy_evaluation()
    explore_autograd_system()
```

### Layer 2: UOp System (`tinygrad/uop/`)

UOps (Universal Operations) are the primitive building blocks of computation.

**Key Files:**
- `ops.py` - Operation definitions and UOp class
- `spec.py` - Type specifications and validation
- `symbolic.py` - Symbolic mathematics

**Deep Dive Code Example:**

```python
#!/usr/bin/env python3
"""
Day 1 Exercise: Understanding UOp System
"""

from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops
import os

def explore_uop_fundamentals():
    """
    Understanding Universal Operations (UOps)
    """
    print("=== UOp System Deep Dive ===")
    
    # Enable UOp-level debugging  
    os.environ['DEBUG'] = '3'
    
    print("\\n1. Examining UOp creation from Tensor operations:")
    
    # Simple operation to see UOp creation
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    
    # Examine the UOp graph
    print(f"\\nResulting UOp: {c.uop}")
    print(f"UOp operation type: {c.uop.op}")
    print(f"UOp dtype: {c.uop.dtype}")
    print(f"UOp shape: {c.uop.shape}")
    
    # Walk through the computation graph
    print("\\n2. Complete UOp graph topology:")
    for i, uop in enumerate(c.uop.toposort()):
        print(f"  {i}: {uop.op.name:<12} | dtype: {uop.dtype} | shape: {uop.shape}")
        if hasattr(uop, 'arg') and uop.arg is not None:
            print(f"     arg: {uop.arg}")
    
    return c

def examine_uop_operations():
    """
    Study the different types of UOp operations
    """
    print("\\n=== UOp Operation Categories ===")
    
    # Import all operation types
    from tinygrad.uop.ops import Ops
    
    # Categorize operations
    unary_ops = [op for op in dir(Ops) if not op.startswith('_')]
    print(f"\\nTotal operations available: {len(unary_ops)}")
    
    # Test different operation types
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    
    print("\\n1. Unary operations:")
    ops_to_test = [
        ("SQRT", lambda t: t.sqrt()),
        ("EXP", lambda t: t.exp()), 
        ("LOG", lambda t: t.log()),
        ("RELU", lambda t: t.relu()),
    ]
    
    for name, op_func in ops_to_test:
        try:
            result = op_func(x)
            uop_graph = list(result.uop.toposort())
            print(f"  {name:<8}: {len(uop_graph)} UOps in graph")
            
            # Find the specific operation UOp
            for uop in uop_graph:
                if hasattr(Ops, name) and uop.op == getattr(Ops, name):
                    print(f"    Found {name} UOp: {uop}")
                    break
        except Exception as e:
            print(f"  {name:<8}: Error - {e}")
    
    print("\\n2. Binary operations:")
    y = Tensor([[5.0, 6.0], [7.0, 8.0]])
    
    binary_ops = [
        ("ADD", lambda a, b: a + b),
        ("MUL", lambda a, b: a * b),
        ("MAX", lambda a, b: a.maximum(b)),
        ("CMPEQ", lambda a, b: a == b),
    ]
    
    for name, op_func in binary_ops:
        try:
            result = op_func(x, y)
            uop_graph = list(result.uop.toposort())
            print(f"  {name:<8}: {len(uop_graph)} UOps in graph")
        except Exception as e:
            print(f"  {name:<8}: Error - {e}")
    
    print("\\n3. Reduction operations:")
    reduction_ops = [
        ("SUM", lambda t: t.sum()),
        ("MAX", lambda t: t.max()),
        ("MEAN", lambda t: t.mean()),
    ]
    
    for name, op_func in reduction_ops:
        try:
            result = op_func(x)
            uop_graph = list(result.uop.toposort())
            print(f"  {name:<8}: {len(uop_graph)} UOps, result shape: {result.shape}")
        except Exception as e:
            print(f"  {name:<8}: Error - {e}")

def analyze_uop_optimization():
    """
    Understanding UOp graph optimization and rewriting
    """
    print("\\n=== UOp Optimization Analysis ===")
    
    # Create a computation that should be optimizable
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    
    # Redundant operations that should be optimized
    y = x + 0.0  # Adding zero
    z = y * 1.0  # Multiplying by one
    w = z + z    # Can be optimized to z * 2
    
    print("\\n1. Before optimization:")
    pre_opt_graph = list(w.uop.toposort())
    print(f"UOp graph size: {len(pre_opt_graph)}")
    for i, uop in enumerate(pre_opt_graph):
        print(f"  {i}: {uop.op.name}")
    
    # Force realization to see optimized graph
    result = w.numpy()
    
    print(f"\\nFinal result: {result}")
    print("\\nNote: Optimizations happen during kernelization/rendering phases")

if __name__ == "__main__":
    explore_uop_fundamentals()
    examine_uop_operations() 
    analyze_uop_optimization()
```

---

## Part 2: Development Environment Setup (60 minutes)

### Advanced Environment Configuration

```bash
#!/bin/bash
# advanced_setup.sh - Complete development environment setup

echo "=== Tinygrad Advanced Development Setup ==="

# 1. Clone and install with all dependencies
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad

# Install in development mode with all extras
python3 -m pip install -e '.[testing,linting,docs]'

# 2. Set up pre-commit hooks for code quality
pip install pre-commit
pre-commit install

# 3. Install additional development tools
pip install pytest-cov pytest-xdist ipdb jupyter notebook

# 4. Set up environment variables for development
cat >> ~/.bashrc << 'EOF'
# Tinygrad development environment
export TINYGRAD_DEBUG=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Common debug aliases
alias tg-debug1='DEBUG=1 python3'
alias tg-debug2='DEBUG=2 python3' 
alias tg-debug3='DEBUG=3 python3'
alias tg-debug4='DEBUG=4 python3'

# Testing shortcuts
alias tg-test='python3 -m pytest test/'
alias tg-test-fast='python3 -m pytest test/unit/ -x'
alias tg-test-cov='python3 -m pytest test/ --cov=tinygrad --cov-report=html'
EOF

# 5. Verify installation
echo "\\n=== Verification ==="
python3 -c "
from tinygrad import Tensor, Device
print(f'✅ Tinygrad imported successfully')
print(f'✅ Default device: {Device.DEFAULT}')
print(f'✅ Basic operation: {(Tensor([1,2,3]) + 1).numpy()}')
"

echo "\\n=== Setup complete! ==="
echo "Restart your shell or run: source ~/.bashrc"
```

### Development Tools Configuration

```python
#!/usr/bin/env python3
"""
development_tools.py - Essential development and debugging tools
"""

import os
import sys
import time
import traceback
from typing import Any, Dict, List
from pathlib import Path

class TinygradDeveloper:
    """
    Development utilities for tinygrad exploration and debugging
    """
    
    def __init__(self):
        self.debug_levels = {
            0: "No debug output",
            1: "Basic execution info", 
            2: "Detailed tensor operations",
            3: "UOp graph construction",
            4: "Full code generation and execution"
        }
    
    def set_debug_level(self, level: int):
        """Set DEBUG environment variable"""
        os.environ['DEBUG'] = str(level)
        print(f"Debug level set to {level}: {self.debug_levels.get(level, 'Unknown')}")
    
    def trace_tensor_operation(self, operation_name: str, tensor_func, *args, **kwargs):
        """
        Trace a tensor operation with detailed debugging
        """
        print(f"\\n=== Tracing: {operation_name} ===")
        
        # Time the operation
        start_time = time.time()
        
        try:
            # Execute with full debugging
            old_debug = os.environ.get('DEBUG', '0')
            self.set_debug_level(3)
            
            result = tensor_func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            print(f"\\nOperation completed in {execution_time:.6f} seconds")
            print(f"Result shape: {result.shape}")
            print(f"Result dtype: {result.dtype}")
            
            # Restore previous debug level
            os.environ['DEBUG'] = old_debug
            
            return result
            
        except Exception as e:
            print(f"\\nError in {operation_name}: {e}")
            print(f"Traceback:\\n{traceback.format_exc()}")
            os.environ['DEBUG'] = old_debug
            return None
    
    def analyze_uop_graph(self, tensor):
        """
        Detailed analysis of UOp computation graph
        """
        print(f"\\n=== UOp Graph Analysis ===")
        
        uop_graph = list(tensor.uop.toposort())
        print(f"Total UOps: {len(uop_graph)}")
        
        # Categorize operations
        op_counts = {}
        for uop in uop_graph:
            op_name = uop.op.name
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
        
        print("\\nOperation breakdown:")
        for op, count in sorted(op_counts.items()):
            print(f"  {op:<15}: {count}")
        
        print("\\nDetailed graph:")
        for i, uop in enumerate(uop_graph):
            print(f"  {i:2d}: {uop.op.name:<12} | {uop.dtype} | {uop.shape}")
            if hasattr(uop, 'arg') and uop.arg is not None:
                print(f"      arg: {uop.arg}")
        
        return uop_graph
    
    def benchmark_operation(self, name: str, operation, iterations: int = 100):
        """
        Benchmark an operation with statistics
        """
        print(f"\\n=== Benchmarking: {name} ===")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            result = operation()
            end_time = time.time()
            times.append(end_time - start_time)
        
        import statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        print(f"Iterations: {iterations}")
        print(f"Mean time: {mean_time:.6f} seconds")
        print(f"Median time: {median_time:.6f} seconds") 
        print(f"Std dev: {std_dev:.6f} seconds")
        print(f"Min time: {min(times):.6f} seconds")
        print(f"Max time: {max(times):.6f} seconds")
        
        return {
            'mean': mean_time,
            'median': median_time,
            'std_dev': std_dev,
            'min': min(times),
            'max': max(times),
            'all_times': times
        }
    
    def compare_with_numpy(self, tinygrad_op, numpy_op, test_data, tolerance=1e-6):
        """
        Compare tinygrad operation with numpy equivalent
        """
        print(f"\\n=== Tinygrad vs NumPy Comparison ===")
        
        # Tinygrad operation
        start_time = time.time()
        tg_result = tinygrad_op(test_data)
        if hasattr(tg_result, 'numpy'):
            tg_result = tg_result.numpy()
        tg_time = time.time() - start_time
        
        # NumPy operation  
        if hasattr(test_data, 'numpy'):
            np_data = test_data.numpy()
        else:
            np_data = test_data
            
        start_time = time.time()
        np_result = numpy_op(np_data)
        np_time = time.time() - start_time
        
        # Compare results
        import numpy as np
        diff = np.abs(tg_result - np_result)
        max_diff = np.max(diff)
        
        print(f"Tinygrad time: {tg_time:.6f} seconds")
        print(f"NumPy time: {np_time:.6f} seconds")
        print(f"Speedup: {np_time/tg_time:.2f}x")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Within tolerance ({tolerance}): {max_diff < tolerance}")
        
        return {
            'tinygrad_time': tg_time,
            'numpy_time': np_time,
            'speedup': np_time/tg_time,
            'max_diff': max_diff,
            'within_tolerance': max_diff < tolerance
        }

# Usage examples
if __name__ == "__main__":
    from tinygrad import Tensor
    import numpy as np
    
    dev = TinygradDeveloper()
    
    # Example 1: Trace a complex operation
    def complex_operation():
        x = Tensor.randn(100, 100)
        y = x.relu().sum(axis=1).exp().mean()
        return y
    
    result = dev.trace_tensor_operation("Complex Operation", complex_operation)
    
    # Example 2: Analyze UOp graph
    x = Tensor([1., 2., 3., 4.])
    y = (x * 2 + 1).relu().sum()
    dev.analyze_uop_graph(y)
    
    # Example 3: Benchmark operation
    def matmul_operation():
        a = Tensor.randn(128, 128)
        b = Tensor.randn(128, 128)
        return (a @ b).numpy()
    
    benchmark_results = dev.benchmark_operation("Matrix Multiplication", matmul_operation, 10)
    
    # Example 4: Compare with NumPy
    test_tensor = Tensor.randn(1000, 1000)
    comparison = dev.compare_with_numpy(
        lambda x: x.sum(axis=1).mean(),
        lambda x: np.mean(np.sum(x, axis=1)),
        test_tensor
    )
```

---

## Part 3: Debugging Mastery (45 minutes)

### DEBUG Flags Deep Dive

Tinygrad provides comprehensive debugging through DEBUG environment variable:

```python
#!/usr/bin/env python3
"""
debugging_mastery.py - Master all DEBUG levels and techniques
"""

import os
import subprocess
from tinygrad import Tensor

def demonstrate_debug_levels():
    """
    Show what each DEBUG level reveals
    """
    print("=== DEBUG Levels Demonstration ===\\n")
    
    # Test operation for all debug levels
    def test_operation():
        x = Tensor([[1., 2.], [3., 4.]])
        y = x.relu().sum()
        return y.numpy()
    
    debug_levels = {
        0: "No debug output (production mode)",
        1: "Basic execution flow and timing",
        2: "Tensor operations and shape changes", 
        3: "UOp graph construction and optimization",
        4: "Full code generation, kernel compilation, and execution",
        5: "Even more detailed kernel information",
        6: "Everything including memory management"
    }
    
    for level, description in debug_levels.items():
        print(f"\\n{'='*60}")
        print(f"DEBUG={level}: {description}")
        print('='*60)
        
        # Set debug level
        os.environ['DEBUG'] = str(level)
        
        print(f"\\nRunning test operation with DEBUG={level}:")
        try:
            result = test_operation()
            print(f"\\nResult: {result}")
        except Exception as e:
            print(f"Error: {e}")
        
        input("\\nPress Enter to continue to next debug level...")

def advanced_debugging_techniques():
    """
    Advanced debugging and introspection techniques
    """
    print("\\n=== Advanced Debugging Techniques ===")
    
    # Technique 1: Manual UOp inspection
    print("\\n1. Manual UOp Graph Inspection:")
    x = Tensor([1., 2., 3.])
    y = x * 2 + 1
    
    print("Before realization:")
    for i, uop in enumerate(y.uop.toposort()):
        print(f"  UOp {i}: {uop}")
        print(f"    Operation: {uop.op}")
        print(f"    Shape: {uop.shape}")
        print(f"    DType: {uop.dtype}")
        if hasattr(uop, 'src') and uop.src:
            print(f"    Sources: {len(uop.src)} UOps")
    
    # Technique 2: Selective debugging with context
    print("\\n2. Selective Debugging with Context:")
    
    def debug_context(debug_level, operation_name):
        """Context manager for temporary debug level changes"""
        class DebugContext:
            def __init__(self, level, name):
                self.level = str(level)
                self.name = name
                self.old_level = os.environ.get('DEBUG', '0')
            
            def __enter__(self):
                os.environ['DEBUG'] = self.level
                print(f"\\n[DEBUG CONTEXT: {self.name} at level {self.level}]")
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                os.environ['DEBUG'] = self.old_level
                print(f"[END DEBUG CONTEXT: {self.name}]\\n")
        
        return DebugContext(debug_level, operation_name)
    
    # Use debug context for specific operations
    with debug_context(2, "Matrix multiplication"):
        a = Tensor.randn(4, 4)
        b = Tensor.randn(4, 4)
        c = a @ b
        result = c.numpy()
    
    # Technique 3: Custom debugging with hooks
    print("\\n3. Custom Operation Tracing:")
    
    class OperationTracer:
        def __init__(self):
            self.operations = []
        
        def trace_tensor_op(self, tensor, op_name):
            """Trace tensor operation details"""
            info = {
                'operation': op_name,
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'device': tensor.device,
                'uop_graph_size': len(list(tensor.uop.toposort())),
                'requires_grad': getattr(tensor, 'requires_grad', False)
            }
            self.operations.append(info)
            print(f"TRACE: {op_name} -> {info}")
            return tensor
    
    tracer = OperationTracer()
    
    # Trace a sequence of operations
    x = Tensor([[1., 2.], [3., 4.]])
    x = tracer.trace_tensor_op(x, "initial")
    x = tracer.trace_tensor_op(x.relu(), "relu")
    x = tracer.trace_tensor_op(x.sum(), "sum")
    x = tracer.trace_tensor_op(x.exp(), "exp")
    
    print(f"\\nTotal operations traced: {len(tracer.operations)}")

def debug_common_issues():
    """
    Debug common development issues with techniques
    """
    print("\\n=== Debugging Common Issues ===")
    
    # Issue 1: Shape mismatches
    print("\\n1. Debugging Shape Mismatches:")
    try:
        a = Tensor([[1., 2.], [3., 4.]])  # 2x2
        b = Tensor([1., 2., 3.])          # 3,
        c = a + b  # This should fail
        result = c.numpy()
    except Exception as e:
        print(f"Shape mismatch error: {e}")
        print("Debug technique: Check tensor shapes before operations")
        print(f"  a.shape = {a.shape}")
        print(f"  b.shape = {b.shape}")
        print("  Solution: Reshape or broadcast properly")
    
    # Issue 2: Memory issues
    print("\\n2. Debugging Memory Issues:")
    os.environ['DEBUG'] = '1'  # Show memory allocation
    
    try:
        # This might cause memory issues on small systems
        # large_tensor = Tensor.randn(10000, 10000)  # Commented to avoid OOM
        print("For memory debugging:")
        print("  - Use DEBUG=1 to see memory allocations")
        print("  - Monitor tensor.device and buffer usage")
        print("  - Check for memory leaks with repeated operations")
    except Exception as e:
        print(f"Memory error: {e}")
    
    # Issue 3: Device issues
    print("\\n3. Debugging Device Issues:")
    from tinygrad import Device
    print(f"Available devices: {Device.DEFAULT}")
    
    # Test device operations
    x = Tensor([1., 2., 3.])
    print(f"Tensor device: {x.device}")
    
    # Issue 4: Gradient computation issues
    print("\\n4. Debugging Gradient Issues:")
    x = Tensor([1., 2., 3.], requires_grad=True)
    y = (x ** 2).sum()
    
    print(f"Before backward: x.grad = {x.grad}")
    y.backward()
    print(f"After backward: x.grad = {x.grad.numpy()}")
    print(f"Expected gradient: {2 * x.numpy()}")

if __name__ == "__main__":
    print("Tinygrad Debugging Mastery")
    print("=" * 50)
    
    choice = input("\\nChoose demonstration:\\n1. DEBUG levels\\n2. Advanced techniques\\n3. Common issues\\nEnter choice (1-3): ")
    
    if choice == '1':
        demonstrate_debug_levels()
    elif choice == '2':
        advanced_debugging_techniques()
    elif choice == '3':
        debug_common_issues()
    else:
        print("Running all demonstrations...")
        # Run a subset to avoid overwhelming output
        advanced_debugging_techniques()
        debug_common_issues()
```

---

## Part 4: First Contribution (45 minutes)

### Contributing to Test Coverage

Let's make a meaningful contribution by improving test coverage:

```python
#!/usr/bin/env python3
"""
first_contribution.py - Your first meaningful contribution to tinygrad
"""

import numpy as np
import pytest
from tinygrad import Tensor

class TestTensorEdgeCases:
    """
    New test cases for tensor edge cases - your first contribution!
    """
    
    def test_empty_tensor_operations(self):
        """Test operations on empty tensors"""
        # Test empty tensor creation
        empty = Tensor([])
        assert empty.shape == (0,), f"Expected shape (0,), got {empty.shape}"
        assert empty.numel() == 0, f"Expected 0 elements, got {empty.numel()}"
        
        # Test operations on empty tensors
        empty_2d = Tensor(np.array([]).reshape(0, 5))
        assert empty_2d.shape == (0, 5), f"Expected shape (0, 5), got {empty_2d.shape}"
        
        # Test that operations don't crash
        result = empty_2d.sum()
        assert result.shape == (), f"Sum of empty tensor should be scalar, got shape {result.shape}"
    
    def test_single_element_tensor_operations(self):
        """Test operations on single-element tensors"""
        single = Tensor([42.0])
        
        # Test basic operations
        assert (single + 1).numpy() == [43.0]
        assert (single * 2).numpy() == [84.0]
        assert single.sum().numpy() == 42.0
        assert single.mean().numpy() == 42.0
        
        # Test reduction operations
        assert single.max().numpy() == 42.0
        assert single.min().numpy() == 42.0
    
    def test_zero_dimensional_tensor_operations(self):
        """Test scalar (0-D) tensor operations"""
        scalar = Tensor(5.0)
        assert scalar.shape == (), f"Expected scalar shape (), got {scalar.shape}"
        assert scalar.ndim == 0, f"Expected 0 dimensions, got {scalar.ndim}"
        
        # Test operations with scalars
        result = scalar + 3.0
        assert result.shape == (), f"Scalar + scalar should be scalar, got shape {result.shape}"
        assert result.numpy() == 8.0
    
    def test_large_tensor_shapes(self):
        """Test tensors with large dimensions"""
        # Test tensor with many dimensions
        shape = (2, 1, 3, 1, 4, 1)
        multi_dim = Tensor.ones(shape)
        assert multi_dim.shape == shape, f"Expected shape {shape}, got {multi_dim.shape}"
        assert multi_dim.ndim == len(shape), f"Expected {len(shape)} dimensions, got {multi_dim.ndim}"
        
        # Test squeeze operation
        squeezed = multi_dim.squeeze()
        expected_squeezed_shape = (2, 3, 4)
        assert squeezed.shape == expected_squeezed_shape, f"Expected squeezed shape {expected_squeezed_shape}, got {squeezed.shape}"
    
    def test_tensor_dtype_consistency(self):
        """Test dtype consistency across operations"""
        # Test integer tensors
        int_tensor = Tensor([1, 2, 3], dtype=Tensor.int32)
        assert int_tensor.dtype == Tensor.int32, f"Expected int32, got {int_tensor.dtype}"
        
        # Test float tensors
        float_tensor = Tensor([1.0, 2.0, 3.0], dtype=Tensor.float32)
        assert float_tensor.dtype == Tensor.float32, f"Expected float32, got {float_tensor.dtype}"
        
        # Test mixed operations (should upcast appropriately)
        mixed_result = int_tensor + float_tensor
        # Note: Check tinygrad's specific upcasting rules
        assert mixed_result.dtype in [Tensor.float32, Tensor.float64], f"Expected float dtype, got {mixed_result.dtype}"
    
    def test_tensor_device_consistency(self):
        """Test device consistency across operations"""
        from tinygrad import Device
        
        tensor = Tensor([1., 2., 3.])
        assert tensor.device == Device.DEFAULT, f"Expected device {Device.DEFAULT}, got {tensor.device}"
        
        # Test that operations maintain device consistency
        result = tensor * 2 + 1
        assert result.device == tensor.device, f"Device should be consistent across operations"
    
    def test_gradient_edge_cases(self):
        """Test gradient computation edge cases"""
        # Test gradient with zero
        x = Tensor([0.0], requires_grad=True)
        y = x * x
        y.backward()
        expected_grad = 2 * x.numpy()
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-6)
        
        # Test gradient with negative values
        x_neg = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y_neg = x_neg.relu()
        y_neg.sum().backward()
        expected_grad_neg = (x_neg.numpy() > 0).astype(float)
        np.testing.assert_allclose(x_neg.grad.numpy(), expected_grad_neg, rtol=1e-6)
    
    def test_numerical_stability(self):
        """Test numerical stability of operations"""
        # Test with very small numbers
        small = Tensor([1e-10, 1e-20, 1e-30])
        result_small = small.log()
        assert not np.any(np.isnan(result_small.numpy())), "Log of small numbers should not be NaN"
        
        # Test with very large numbers
        large = Tensor([1e10, 1e20])
        result_large = large.exp()
        # Note: This might overflow, but should be handled gracefully
        assert np.all(np.isfinite(result_large.numpy()) | np.isinf(result_large.numpy())), "Exp should handle large numbers"
    
    def test_broadcasting_edge_cases(self):
        """Test edge cases in broadcasting"""
        # Test broadcasting with 1-element tensors
        a = Tensor([[1, 2, 3]])  # Shape: (1, 3)
        b = Tensor([[1], [2]])   # Shape: (2, 1)
        
        result = a + b
        expected_shape = (2, 3)
        assert result.shape == expected_shape, f"Expected broadcast shape {expected_shape}, got {result.shape}"
        
        # Verify the actual values
        expected_values = np.array([[2, 3, 4], [3, 4, 5]])
        np.testing.assert_allclose(result.numpy(), expected_values)
    
    def test_memory_efficiency(self):
        """Test memory-efficient operations"""
        # Test in-place-like operations (tinygrad doesn't have true in-place)
        x = Tensor([1., 2., 3., 4.])
        original_data_ptr = id(x.uop)  # Track UOp identity
        
        # Operations should create new tensors, not modify in-place
        y = x.relu()
        assert id(y.uop) != original_data_ptr, "Operations should create new UOps"
        
        # Original tensor should be unchanged
        np.testing.assert_allclose(x.numpy(), [1., 2., 3., 4.])

# Validation testing framework
class TinygradValidator:
    """
    Validation framework for testing tinygrad operations
    """
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.failures = []
    
    def validate_operation(self, name: str, tinygrad_fn, numpy_fn, test_inputs, tolerance=1e-6):
        """
        Validate a tinygrad operation against numpy equivalent
        """
        self.tests_run += 1
        
        try:
            # Run tinygrad operation
            if isinstance(test_inputs, (list, tuple)):
                tg_inputs = [Tensor(inp) if not isinstance(inp, Tensor) else inp for inp in test_inputs]
                tg_result = tinygrad_fn(*tg_inputs)
            else:
                tg_input = Tensor(test_inputs) if not isinstance(test_inputs, Tensor) else test_inputs
                tg_result = tinygrad_fn(tg_input)
            
            if hasattr(tg_result, 'numpy'):
                tg_result = tg_result.numpy()
            
            # Run numpy operation
            if isinstance(test_inputs, (list, tuple)):
                np_inputs = [inp.numpy() if hasattr(inp, 'numpy') else np.array(inp) for inp in test_inputs]
                np_result = numpy_fn(*np_inputs)
            else:
                np_input = test_inputs.numpy() if hasattr(test_inputs, 'numpy') else np.array(test_inputs)
                np_result = numpy_fn(np_input)
            
            # Compare results
            if np.allclose(tg_result, np_result, rtol=tolerance, atol=tolerance):
                self.tests_passed += 1
                print(f"✅ {name}: PASSED")
                return True
            else:
                max_diff = np.max(np.abs(tg_result - np_result))
                error_msg = f"Results differ by {max_diff:.2e} (tolerance: {tolerance:.2e})"
                self.failures.append(f"{name}: {error_msg}")
                print(f"❌ {name}: FAILED - {error_msg}")
                return False
                
        except Exception as e:
            self.failures.append(f"{name}: Exception - {str(e)}")
            print(f"❌ {name}: ERROR - {str(e)}")
            return False
    
    def run_comprehensive_validation(self):
        """
        Run comprehensive validation of tinygrad operations
        """
        print("=== Comprehensive Tinygrad Validation ===\\n")
        
        # Test data
        test_matrices = [
            np.array([[1., 2.], [3., 4.]]),
            np.array([[1., -2., 3.], [-4., 5., -6.]]),
            np.random.randn(5, 5),
        ]
        
        test_vectors = [
            np.array([1., 2., 3.]),
            np.array([-1., 0., 1.]),
            np.random.randn(10),
        ]
        
        # Unary operations
        unary_tests = [
            ("relu", lambda x: x.relu(), lambda x: np.maximum(x, 0)),
            ("exp", lambda x: x.exp(), lambda x: np.exp(x)),
            ("log", lambda x: x.log(), lambda x: np.log(np.abs(x) + 1e-8)),  # Add small epsilon for stability
            ("sqrt", lambda x: x.sqrt(), lambda x: np.sqrt(np.abs(x))),
            ("sum", lambda x: x.sum(), lambda x: np.sum(x)),
            ("mean", lambda x: x.mean(), lambda x: np.mean(x)),
        ]
        
        print("Testing unary operations:")
        for name, tg_op, np_op in unary_tests:
            for i, test_data in enumerate(test_vectors + test_matrices):
                test_name = f"{name}_test_{i}"
                self.validate_operation(test_name, tg_op, np_op, test_data)
        
        # Binary operations
        binary_tests = [
            ("add", lambda x, y: x + y, lambda x, y: x + y),
            ("mul", lambda x, y: x * y, lambda x, y: x * y),
            ("matmul", lambda x, y: x @ y, lambda x, y: x @ y),
        ]
        
        print("\\nTesting binary operations:")
        for name, tg_op, np_op in binary_tests:
            for i, (a, b) in enumerate(zip(test_matrices, test_matrices)):
                if name == "matmul" and a.shape[-1] != b.shape[0]:
                    b = b.T  # Transpose for compatible matmul
                test_name = f"{name}_test_{i}"
                self.validate_operation(test_name, tg_op, np_op, [a, b])
        
        # Print summary
        print(f"\\n=== Validation Summary ===")
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Success rate: {self.tests_passed/self.tests_run*100:.1f}%")
        
        if self.failures:
            print(f"\\nFailures ({len(self.failures)}):")
            for failure in self.failures:
                print(f"  - {failure}")
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    print("=== Your First Tinygrad Contribution ===\\n")
    
    # Run the new test cases
    print("1. Running edge case tests...")
    test_instance = TestTensorEdgeCases()
    
    # Run each test method
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for method_name in test_methods:
        try:
            print(f"\\nRunning {method_name}...")
            getattr(test_instance, method_name)()
            print(f"✅ {method_name} passed")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    # Run comprehensive validation
    print("\\n2. Running comprehensive validation...")
    validator = TinygradValidator()
    success = validator.run_comprehensive_validation()
    
    if success:
        print("\\n🎉 All validations passed! Ready to contribute to tinygrad!")
    else:
        print("\\n⚠️  Some validations failed. These could be areas for improvement!")
```

### Create Your First Pull Request

```bash
#!/bin/bash
# first_contribution.sh - Script to create your first contribution

echo "=== Creating Your First Tinygrad Contribution ==="

# 1. Create a new branch for your contribution
git checkout -b feature/improve-tensor-edge-case-tests

# 2. Add your new test file
cp first_contribution.py test/test_tensor_edge_cases.py

# 3. Run the tests to make sure they work
echo "Running new tests..."
python3 -m pytest test/test_tensor_edge_cases.py -v

# 4. Run existing tests to make sure nothing is broken
echo "Running existing tests to check for regressions..."
python3 -m pytest test/test_tensor.py -v

# 5. Run linting
echo "Running code quality checks..."
ruff check test/test_tensor_edge_cases.py
mypy test/test_tensor_edge_cases.py

# 6. Add files to git
git add test/test_tensor_edge_cases.py

# 7. Commit your changes
git commit -m "Add comprehensive edge case tests for Tensor class

- Add tests for empty tensor operations
- Add tests for single-element and zero-dimensional tensors  
- Add tests for large tensor shapes and dtype consistency
- Add tests for gradient edge cases and numerical stability
- Add tests for broadcasting edge cases and memory efficiency
- Add comprehensive validation framework for tinygrad operations

These tests improve coverage for edge cases that weren't previously tested
and provide a framework for ongoing validation of tinygrad operations."

echo "✅ Contribution committed! Next steps:"
echo "1. Push branch: git push origin feature/improve-tensor-edge-case-tests"
echo "2. Create pull request on GitHub"
echo "3. Address any review feedback"
```

---

## Day 1 Wrap-up & Assignments

### What You've Accomplished Today

1. ✅ **Deep Architecture Understanding**: Mastered the 4-layer tinygrad architecture
2. ✅ **Development Environment**: Set up advanced debugging and development tools
3. ✅ **Debugging Mastery**: Learned to use DEBUG flags and introspection techniques
4. ✅ **Real Contribution**: Created comprehensive test cases for edge scenarios
5. ✅ **Validation Framework**: Built tools for ongoing tinygrad validation

### Tomorrow's Preview: UOp System Deep Dive

Day 2 will focus on:
- **UOp Internal Architecture**: Pattern matching, graph rewriting, optimization passes
- **Custom UOp Creation**: Building new primitive operations
- **Performance Analysis**: UOp-level optimization and profiling
- **Advanced Debugging**: UOp graph visualization and manipulation

### Homework Assignments

1. **Explore More Operations**: Use the debugging tools to trace at least 5 different tensor operations
2. **Find Edge Cases**: Try to break tinygrad with unusual inputs and document what happens
3. **Performance Comparison**: Benchmark tinygrad vs numpy on operations relevant to your work
4. **Read Source Code**: Study `tinygrad/tensor.py` lines 100-200 to understand tensor internals

### Self-Assessment Checklist

- [ ] Can I explain tinygrad's 4-layer architecture to someone else?
- [ ] Can I use DEBUG flags to understand what tinygrad is doing?
- [ ] Can I trace a tensor operation from creation to execution?
- [ ] Can I write and run tests for tinygrad functionality?
- [ ] Can I identify the UOp graph for a given tensor operation?

**Ready for Day 2? The UOp system awaits! 🚀**