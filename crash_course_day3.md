# Day 3: Engine Deep Dive - Scheduling, Memory & Execution

## Overview
Today we master tinygrad's execution engine - the system that transforms UOp graphs into efficient, executable schedules. You'll understand how operations are ordered, memory is planned, and kernels are fused for optimal performance. By the end, you'll be able to analyze and optimize execution schedules and implement custom scheduling strategies.

## Learning Objectives
- ✅ Master the scheduling system and ScheduleItem creation
- ✅ Understand memory planning and optimization strategies
- ✅ Learn kernel fusion and execution ordering
- ✅ Build custom schedulers and memory planners
- ✅ Create advanced execution analysis tools
- ✅ Implement performance monitoring for the engine

---

## Part 1: Scheduling System Deep Dive (120 minutes)

### Understanding ScheduleItem and Schedule Creation

The scheduler converts UOp graphs into ordered execution plans with optimized memory usage:

```python
#!/usr/bin/env python3
"""
Day 3 Exercise: Deep dive into tinygrad's scheduling system
"""

import os
import time
import traceback
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field

from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars
from tinygrad.engine.realize import run_schedule  
from tinygrad.engine.memory import memory_planner
from tinygrad.device import Device, Buffer
from tinygrad.uop.ops import UOp, Ops, Variable
from tinygrad import Tensor
from tinygrad.helpers import DEBUG, Metadata

@dataclass
class ScheduleAnalysis:
    """Analysis results for a schedule"""
    total_items: int
    memory_operations: int
    compute_operations: int
    dependency_chains: List[List[int]]
    parallelizable_groups: List[List[int]]
    memory_usage_peak: int
    execution_time_estimate: float
    optimization_opportunities: List[str]

class ScheduleExplorer:
    """
    Advanced scheduler analysis and exploration tool
    """
    
    def __init__(self):
        self.schedule_cache = {}
        self.analysis_history = []
        self.custom_schedulers = {}
    
    def understand_schedule_creation(self, tensor_operation):
        """
        Deep dive into how tensor operations become execution schedules
        """
        print("=== Schedule Creation Deep Dive ===\\n")
        
        # Create a tensor operation to analyze
        print("1. Creating tensor operation for analysis:")
        result = tensor_operation()
        print(f"Operation result shape: {result.shape}")
        print(f"Operation result device: {result.device}")
        
        # Examine the UOp graph before scheduling
        print("\\n2. UOp graph before scheduling:")
        uop_graph = list(result.uop.toposort())
        print(f"Total UOps in graph: {len(uop_graph)}")
        
        for i, uop in enumerate(uop_graph):
            print(f"  {i:2d}: {uop.op.name:<12} | {uop.dtype} | {uop.shape}")
            if hasattr(uop, 'arg') and uop.arg is not None:
                arg_str = str(uop.arg)[:50] + "..." if len(str(uop.arg)) > 50 else str(uop.arg)
                print(f"      arg: {arg_str}")
        
        # Force realization to see the scheduling process
        print("\\n3. Forcing realization to trigger scheduling:")
        
        # Enable detailed debugging for scheduling
        os.environ['DEBUG'] = '3'
        start_time = time.time()
        
        # This will trigger the scheduling process
        numpy_result = result.numpy()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\\nExecution completed in {execution_time:.6f} seconds")
        print(f"Result: {numpy_result}")
        
        return numpy_result, execution_time
    
    def analyze_schedule_items(self, tensor_operation) -> ScheduleAnalysis:
        """
        Analyze the ScheduleItems created for a tensor operation
        """
        print("\\n=== Schedule Items Analysis ===\\n")
        
        # Create the operation
        result = tensor_operation()
        
        # We'll analyze what scheduling would produce
        # Note: Direct access to schedule creation is complex,
        # so we'll analyze the UOp structure that feeds into it
        uop_graph = list(result.uop.toposort())
        
        # Simulate schedule analysis
        print("1. UOp Graph Analysis for Scheduling:")
        
        memory_ops = sum(1 for u in uop_graph if u.op in [Ops.BUFFER, Ops.VIEW, Ops.RESHAPE])
        compute_ops = len(uop_graph) - memory_ops
        
        print(f"Total UOps: {len(uop_graph)}")
        print(f"Memory operations: {memory_ops}")
        print(f"Compute operations: {compute_ops}")
        
        # Analyze dependencies
        dependency_chains = self._find_dependency_chains(uop_graph)
        parallelizable_groups = self._find_parallelizable_groups(uop_graph)
        
        print(f"\\n2. Dependency Analysis:")
        print(f"Dependency chains found: {len(dependency_chains)}")
        print(f"Parallelizable groups: {len(parallelizable_groups)}")
        
        # Estimate memory usage
        memory_estimate = self._estimate_memory_usage(uop_graph)
        print(f"\\n3. Memory Analysis:")
        print(f"Estimated peak memory: {memory_estimate / (1024*1024):.2f} MB")
        
        # Find optimization opportunities
        optimizations = self._find_schedule_optimizations(uop_graph)
        print(f"\\n4. Optimization Opportunities:")
        for opt in optimizations:
            print(f"  - {opt}")
        
        analysis = ScheduleAnalysis(
            total_items=len(uop_graph),
            memory_operations=memory_ops,
            compute_operations=compute_ops,
            dependency_chains=dependency_chains,
            parallelizable_groups=parallelizable_groups,
            memory_usage_peak=memory_estimate,
            execution_time_estimate=0.0,  # Would need actual timing
            optimization_opportunities=optimizations
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _find_dependency_chains(self, uop_graph: List[UOp]) -> List[List[int]]:
        """Find dependency chains in the UOp graph"""
        chains = []
        visited = set()
        
        def dfs_chain(uop_idx, current_chain):
            if uop_idx in visited:
                return
            
            visited.add(uop_idx)
            current_chain.append(uop_idx)
            
            uop = uop_graph[uop_idx]
            if uop.src:
                # Find the indices of source UOps
                for src in uop.src:
                    try:
                        src_idx = uop_graph.index(src)
                        dfs_chain(src_idx, current_chain[:])  # Continue chain
                    except ValueError:
                        pass  # Source not in current graph
            else:
                # End of chain
                chains.append(current_chain[:])
        
        # Start DFS from leaf nodes (no dependencies)
        for i, uop in enumerate(uop_graph):
            if not uop.src and i not in visited:
                dfs_chain(i, [])
        
        return chains
    
    def _find_parallelizable_groups(self, uop_graph: List[UOp]) -> List[List[int]]:
        """Find groups of operations that can run in parallel"""
        # Build dependency levels
        levels = defaultdict(list)
        
        def get_level(uop_idx):
            uop = uop_graph[uop_idx]
            if not uop.src:
                return 0
            
            max_src_level = 0
            for src in uop.src:
                try:
                    src_idx = uop_graph.index(src)
                    max_src_level = max(max_src_level, get_level(src_idx))
                except ValueError:
                    pass
            
            return max_src_level + 1
        
        # Group UOps by their dependency level
        for i, uop in enumerate(uop_graph):
            level = get_level(i)
            levels[level].append(i)
        
        # Return groups with more than one operation (parallelizable)
        return [group for group in levels.values() if len(group) > 1]
    
    def _estimate_memory_usage(self, uop_graph: List[UOp]) -> int:
        """Estimate memory usage for the UOp graph"""
        total_bytes = 0
        
        for uop in uop_graph:
            if hasattr(uop, 'shape') and uop.shape:
                elements = 1
                for dim in uop.shape:
                    if isinstance(dim, int):
                        elements *= dim
                
                # Estimate 4 bytes per element (float32)
                dtype_size = 4
                total_bytes += elements * dtype_size
        
        return total_bytes
    
    def _find_schedule_optimizations(self, uop_graph: List[UOp]) -> List[str]:
        """Identify optimization opportunities in the schedule"""
        optimizations = []
        
        # Look for fusion opportunities
        fusion_candidates = []
        for i, uop in enumerate(uop_graph):
            if uop.op in [Ops.ADD, Ops.MUL, Ops.RELU]:
                fusion_candidates.append(i)
        
        if len(fusion_candidates) > 1:
            optimizations.append(f"Kernel fusion opportunity: {len(fusion_candidates)} fusible operations")
        
        # Look for memory access patterns
        memory_intensive_ops = [i for i, u in enumerate(uop_graph) if u.op in [Ops.BUFFER, Ops.VIEW]]
        if len(memory_intensive_ops) > 3:
            optimizations.append(f"Memory access optimization: {len(memory_intensive_ops)} memory operations")
        
        # Look for redundant computations
        op_signatures = {}
        for i, uop in enumerate(uop_graph):
            sig = (uop.op, uop.dtype, str(uop.arg) if uop.arg else None)
            if sig in op_signatures:
                optimizations.append(f"Potential redundant computation at UOp {i}")
            op_signatures[sig] = i
        
        return optimizations
    
    def compare_scheduling_strategies(self, operations: List[Tuple[str, callable]]) -> Dict:
        """
        Compare different operations to understand scheduling differences
        """
        print("\\n=== Scheduling Strategy Comparison ===\\n")
        
        results = {}
        
        for name, operation in operations:
            print(f"Analyzing: {name}")
            print("-" * 40)
            
            analysis = self.analyze_schedule_items(operation)
            
            # Create summary
            summary = {
                'total_operations': analysis.total_items,
                'memory_compute_ratio': analysis.memory_operations / max(analysis.compute_operations, 1),
                'parallelism_potential': len(analysis.parallelizable_groups),
                'dependency_complexity': len(analysis.dependency_chains),
                'memory_efficiency': analysis.memory_usage_peak / max(analysis.total_items, 1),
                'optimization_count': len(analysis.optimization_opportunities)
            }
            
            results[name] = summary
            
            print(f"Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            print()
        
        # Compare results
        print("Comparison Matrix:")
        print(f"{'Operation':<20} | {'Ops':<5} | {'Mem/Comp':<8} | {'Parallel':<8} | {'Memory (MB)':<12}")
        print("-" * 70)
        
        for name, summary in results.items():
            memory_mb = summary['memory_efficiency'] / (1024 * 1024)
            print(f"{name:<20} | {summary['total_operations']:<5} | {summary['memory_compute_ratio']:<8.2f} | "
                  f"{summary['parallelism_potential']:<8} | {memory_mb:<12.2f}")
        
        return results

def explore_schedule_optimization():
    """
    Deep exploration of schedule optimization techniques
    """
    print("=== Schedule Optimization Techniques ===\\n")
    
    explorer = ScheduleExplorer()
    
    # Test different operation types to see scheduling differences
    test_operations = [
        ("Element-wise Add", lambda: Tensor([1., 2., 3., 4.]) + Tensor([5., 6., 7., 8.])),
        ("Matrix Multiplication", lambda: Tensor([[1., 2.], [3., 4.]]) @ Tensor([[5., 6.], [7., 8.]])),
        ("Reduction Operation", lambda: Tensor.randn(100, 100).sum(axis=1)),
        ("Complex Chain", lambda: (Tensor.randn(50, 50).relu().sum(axis=1).exp().mean())),
        ("Broadcasting", lambda: Tensor([[1., 2.]]) + Tensor([[3.], [4.], [5.]])),
        ("Multiple Operations", lambda: Tensor.randn(20, 20) @ Tensor.randn(20, 20) + Tensor.randn(20, 20)),
    ]
    
    print("1. Individual Operation Analysis:")
    for name, operation in test_operations:
        print(f"\\n{name}:")
        print("-" * 30)
        result, exec_time = explorer.understand_schedule_creation(operation)
        
    print("\\n" + "="*60)
    print("2. Comparative Scheduling Analysis:")
    comparison_results = explorer.compare_scheduling_strategies(test_operations)
    
    return comparison_results

def understand_variable_binding():
    """
    Understanding how Variables are bound and resolved in scheduling
    """
    print("\\n=== Variable Binding in Scheduling ===\\n")
    
    # Variables are used for dynamic shapes and values
    print("1. Creating operations with symbolic variables:")
    
    # Create tensors with dynamic shapes (simulated)
    x = Tensor.randn(10, 20)
    y = Tensor.randn(20, 30)
    
    # Matrix multiplication creates dependencies that need scheduling
    result = x @ y
    
    print(f"Input shapes: {x.shape} @ {y.shape} = {result.shape}")
    
    # Examine the UOp graph
    uop_graph = list(result.uop.toposort())
    
    print("\\n2. UOp graph with Variables:")
    variables_found = []
    
    for i, uop in enumerate(uop_graph):
        print(f"  {i}: {uop.op.name} | shape: {uop.shape}")
        
        # Look for Variables in the UOp
        if hasattr(uop, 'arg') and isinstance(uop.arg, Variable):
            variables_found.append((i, uop.arg))
            print(f"     -> Contains Variable: {uop.arg}")
        
        # Check shape for Variables
        if hasattr(uop, 'shape') and uop.shape:
            for dim in uop.shape:
                if isinstance(dim, Variable):
                    variables_found.append((i, dim))
                    print(f"     -> Shape Variable: {dim}")
    
    print(f"\\nTotal Variables found: {len(variables_found)}")
    
    # Force realization to see variable resolution
    print("\\n3. Variable resolution during realization:")
    os.environ['DEBUG'] = '2'
    
    start_time = time.time()
    numpy_result = result.numpy()
    end_time = time.time()
    
    print(f"Result shape: {numpy_result.shape}")
    print(f"Execution time: {end_time - start_time:.6f}s")
    
    return variables_found

if __name__ == "__main__":
    print("Day 3: Engine Deep Dive - Scheduling System")
    print("=" * 50)
    
    # Run all explorations
    comparison_results = explore_schedule_optimization()
    variables = understand_variable_binding()
    
    print("\\n" + "="*50)
    print("Schedule Exploration Complete!")
```

### Memory Planning Deep Dive

```python
#!/usr/bin/env python3
"""
memory_planning.py - Deep dive into tinygrad's memory planning system
"""

import os
import time
import gc
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from tinygrad.device import Device, Buffer
from tinygrad.engine.memory import memory_planner
from tinygrad.engine.schedule import ScheduleItem
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import DEBUG

@dataclass
class MemoryAnalysis:
    """Memory usage analysis results"""
    peak_usage: int
    total_allocations: int
    reuse_opportunities: int
    fragmentation_estimate: float
    allocation_timeline: List[Tuple[float, str, int]]  # time, action, size
    memory_efficiency: float

class MemoryPlannerExplorer:
    """
    Advanced memory planning analysis and optimization
    """
    
    def __init__(self):
        self.allocation_history = []
        self.memory_snapshots = []
        self.custom_planners = {}
    
    def understand_memory_planning(self):
        """
        Deep dive into how tinygrad plans and manages memory
        """
        print("=== Memory Planning Deep Dive ===\\n")
        
        print("1. Memory allocation patterns:")
        
        # Create operations with different memory patterns
        operations = [
            ("Small tensor", lambda: Tensor.randn(10, 10)),
            ("Large tensor", lambda: Tensor.randn(1000, 1000)),
            ("Multiple small", lambda: [Tensor.randn(5, 5) for _ in range(10)]),
            ("Chain operations", lambda: Tensor.randn(100, 100).relu().sum().exp()),
        ]
        
        for name, operation in operations:
            print(f"\\n{name}:")
            print("-" * 30)
            
            # Monitor memory before
            gc.collect()  # Force garbage collection
            initial_tensors = len([obj for obj in gc.get_objects() if isinstance(obj, Tensor)])
            
            start_time = time.time()
            result = operation()
            
            # Force realization
            if isinstance(result, list):
                numpy_results = [t.numpy() for t in result]
            else:
                numpy_result = result.numpy()
            
            end_time = time.time()
            
            # Monitor memory after
            final_tensors = len([obj for obj in gc.get_objects() if isinstance(obj, Tensor)])
            
            print(f"  Execution time: {(end_time - start_time)*1000:.2f}ms")
            print(f"  Tensor objects: {initial_tensors} -> {final_tensors}")
            
            if isinstance(result, list):
                total_memory = sum(t.nbytes for t in result if hasattr(t, 'nbytes'))
                print(f"  Total memory: {total_memory / (1024*1024):.2f} MB for {len(result)} tensors")
            else:
                if hasattr(result, 'nbytes'):
                    print(f"  Memory usage: {result.nbytes / (1024*1024):.2f} MB")
    
    def analyze_memory_reuse_patterns(self, tensor_operation):
        """
        Analyze memory reuse opportunities in tensor operations
        """
        print("\\n=== Memory Reuse Analysis ===\\n")
        
        # Create operation and analyze its memory pattern
        print("1. Creating operation for memory analysis:")
        result = tensor_operation()
        
        # Get UOp graph to understand memory operations
        uop_graph = list(result.uop.toposort())
        
        # Find memory-related operations
        memory_ops = []
        for i, uop in enumerate(uop_graph):
            if uop.op in [Ops.BUFFER, Ops.VIEW, Ops.RESHAPE]:
                memory_ops.append((i, uop))
        
        print(f"Memory operations found: {len(memory_ops)}")
        
        # Analyze buffer lifetimes
        buffer_lifetimes = {}
        for i, (idx, uop) in enumerate(memory_ops):
            # Estimate when buffer is created and when it's last used
            first_use = idx
            last_use = idx
            
            # Find last usage by looking for dependencies
            for j, later_uop in enumerate(uop_graph[idx+1:], idx+1):
                if later_uop.src and uop in later_uop.src:
                    last_use = j
            
            buffer_lifetimes[i] = (first_use, last_use, last_use - first_use)
        
        print("\\n2. Buffer lifetime analysis:")
        for buf_id, (first, last, duration) in buffer_lifetimes.items():
            print(f"  Buffer {buf_id}: ops {first}-{last} (lifetime: {duration})")
        
        # Find reuse opportunities
        reuse_opportunities = []
        for buf1_id, (f1, l1, d1) in buffer_lifetimes.items():
            for buf2_id, (f2, l2, d2) in buffer_lifetimes.items():
                if buf1_id != buf2_id and l1 < f2:  # buf1 ends before buf2 starts
                    reuse_opportunities.append((buf1_id, buf2_id))
        
        print(f"\\n3. Memory reuse opportunities: {len(reuse_opportunities)}")
        for buf1, buf2 in reuse_opportunities:
            print(f"  Buffer {buf1} -> Buffer {buf2}")
        
        return buffer_lifetimes, reuse_opportunities
    
    def implement_custom_memory_planner(self):
        """
        Implement and test custom memory planning strategies
        """
        print("\\n=== Custom Memory Planning Strategies ===\\n")
        
        def analyze_memory_pressure(tensors: List[Tensor]) -> Dict[str, Any]:
            """Analyze memory pressure for a list of tensors"""
            total_memory = sum(t.nbytes for t in tensors if hasattr(t, 'nbytes'))
            max_single = max((t.nbytes for t in tensors if hasattr(t, 'nbytes')), default=0)
            
            return {
                'total_memory': total_memory,
                'max_single_tensor': max_single,
                'tensor_count': len(tensors),
                'avg_tensor_size': total_memory / len(tensors) if tensors else 0,
                'memory_pressure': 'high' if total_memory > 100 * 1024 * 1024 else 'low'  # 100MB threshold
            }
        
        def greedy_memory_planner(operations: List[callable]) -> Dict[str, Any]:
            """Simple greedy memory planning strategy"""
            print("Greedy memory planning strategy:")
            
            allocated_tensors = []
            peak_memory = 0
            current_memory = 0
            allocation_events = []
            
            for i, operation in enumerate(operations):
                print(f"  Step {i+1}: Executing operation")
                
                # Execute operation
                start_time = time.time()
                result = operation()
                
                # Track allocation
                if hasattr(result, 'nbytes'):
                    current_memory += result.nbytes
                    allocated_tensors.append(result)
                    allocation_events.append((time.time(), 'alloc', result.nbytes))
                
                peak_memory = max(peak_memory, current_memory)
                
                # Simple cleanup: remove old tensors if memory pressure is high
                if current_memory > 50 * 1024 * 1024:  # 50MB threshold
                    if allocated_tensors:
                        old_tensor = allocated_tensors.pop(0)
                        if hasattr(old_tensor, 'nbytes'):
                            current_memory -= old_tensor.nbytes
                            allocation_events.append((time.time(), 'free', old_tensor.nbytes))
                        del old_tensor
                
                print(f"    Current memory: {current_memory / (1024*1024):.2f} MB")
                print(f"    Peak memory: {peak_memory / (1024*1024):.2f} MB")
            
            return {
                'peak_memory': peak_memory,
                'final_memory': current_memory,
                'allocation_events': allocation_events,
                'efficiency': current_memory / peak_memory if peak_memory > 0 else 1.0
            }
        
        def optimal_memory_planner(operations: List[callable]) -> Dict[str, Any]:
            """More sophisticated memory planning with lifetime analysis"""
            print("\\nOptimal memory planning strategy:")
            
            # Pre-analyze all operations to determine lifetimes
            operation_results = []
            operation_sizes = []
            
            print("  Phase 1: Analyzing operation requirements")
            for i, operation in enumerate(operations):
                # Dry run to determine memory requirements
                try:
                    result = operation()
                    size = getattr(result, 'nbytes', 0)
                    operation_results.append(result)
                    operation_sizes.append(size)
                    print(f"    Operation {i+1}: {size / (1024*1024):.2f} MB")
                except Exception as e:
                    print(f"    Operation {i+1}: Failed - {e}")
                    operation_sizes.append(0)
            
            # Plan optimal allocation order
            print("  Phase 2: Optimal allocation planning")
            total_size = sum(operation_sizes)
            peak_required = max(operation_sizes) if operation_sizes else 0
            
            # Sort by size for optimal packing
            size_order = sorted(enumerate(operation_sizes), key=lambda x: x[1], reverse=True)
            
            print(f"    Total memory needed: {total_size / (1024*1024):.2f} MB")
            print(f"    Peak single allocation: {peak_required / (1024*1024):.2f} MB")
            print(f"    Optimal allocation order: {[i for i, _ in size_order]}")
            
            return {
                'total_memory': total_size,
                'peak_required': peak_required,
                'operation_count': len(operations),
                'memory_efficiency': peak_required / total_size if total_size > 0 else 1.0,
                'allocation_order': [i for i, _ in size_order]
            }
        
        # Test both strategies
        test_operations = [
            lambda: Tensor.randn(100, 100),
            lambda: Tensor.randn(50, 50),
            lambda: Tensor.randn(200, 200),
            lambda: Tensor.randn(25, 25),
            lambda: Tensor.randn(150, 150),
        ]
        
        print("Testing memory planning strategies:")
        print("=" * 40)
        
        greedy_results = greedy_memory_planner(test_operations)
        optimal_results = optimal_memory_planner(test_operations)
        
        print("\\nStrategy Comparison:")
        print(f"Greedy - Peak memory: {greedy_results['peak_memory'] / (1024*1024):.2f} MB")
        print(f"Greedy - Efficiency: {greedy_results['efficiency']:.2%}")
        print(f"Optimal - Peak required: {optimal_results['peak_required'] / (1024*1024):.2f} MB")
        print(f"Optimal - Efficiency: {optimal_results['memory_efficiency']:.2%}")
        
        return greedy_results, optimal_results
    
    def monitor_memory_usage_patterns(self, operations: List[Tuple[str, callable]]) -> MemoryAnalysis:
        """
        Monitor and analyze memory usage patterns across different operations
        """
        print("\\n=== Memory Usage Pattern Monitoring ===\\n")
        
        allocation_timeline = []
        peak_usage = 0
        total_allocations = 0
        
        for name, operation in operations:
            print(f"Monitoring: {name}")
            
            # Record memory before
            gc.collect()
            start_time = time.time()
            
            # Execute operation
            result = operation()
            
            # Force realization and measure
            if hasattr(result, 'numpy'):
                numpy_result = result.numpy()
                memory_used = getattr(result, 'nbytes', 0)
            else:
                memory_used = 0
            
            end_time = time.time()
            
            # Record allocation event
            allocation_timeline.append((start_time, f"alloc_{name}", memory_used))
            allocation_timeline.append((end_time, f"free_{name}", -memory_used))
            
            peak_usage = max(peak_usage, memory_used)
            total_allocations += 1 if memory_used > 0 else 0
            
            print(f"  Memory used: {memory_used / (1024*1024):.2f} MB")
            print(f"  Duration: {(end_time - start_time)*1000:.2f}ms")
        
        # Calculate efficiency metrics
        total_memory_moved = sum(abs(size) for _, _, size in allocation_timeline)
        memory_efficiency = peak_usage / (total_memory_moved / 2) if total_memory_moved > 0 else 1.0
        
        analysis = MemoryAnalysis(
            peak_usage=peak_usage,
            total_allocations=total_allocations,
            reuse_opportunities=0,  # Would need deeper analysis
            fragmentation_estimate=1.0 - memory_efficiency,
            allocation_timeline=allocation_timeline,
            memory_efficiency=memory_efficiency
        )
        
        print(f"\\nMemory Analysis Summary:")
        print(f"  Peak usage: {peak_usage / (1024*1024):.2f} MB")
        print(f"  Total allocations: {total_allocations}")
        print(f"  Memory efficiency: {memory_efficiency:.2%}")
        print(f"  Fragmentation estimate: {analysis.fragmentation_estimate:.2%}")
        
        return analysis

def demonstrate_memory_planning():
    """
    Comprehensive demonstration of memory planning concepts
    """
    print("=== Memory Planning Demonstration ===\\n")
    
    explorer = MemoryPlannerExplorer()
    
    # Basic memory planning understanding
    explorer.understand_memory_planning()
    
    # Memory reuse analysis
    def complex_operation():
        x = Tensor.randn(100, 100)
        y = x @ x.T
        z = y.relu()
        return z.sum()
    
    lifetimes, reuse_ops = explorer.analyze_memory_reuse_patterns(complex_operation)
    
    # Custom memory planners
    greedy_results, optimal_results = explorer.implement_custom_memory_planner()
    
    # Pattern monitoring
    test_operations = [
        ("Small Matrix", lambda: Tensor.randn(50, 50)),
        ("Large Matrix", lambda: Tensor.randn(500, 500)),
        ("Chain Ops", lambda: Tensor.randn(100, 100).relu().sum()),
        ("Broadcast", lambda: Tensor.randn(100, 1) + Tensor.randn(1, 100)),
    ]
    
    memory_analysis = explorer.monitor_memory_usage_patterns(test_operations)
    
    return {
        'lifetimes': lifetimes,
        'reuse_opportunities': reuse_ops,
        'greedy_results': greedy_results,
        'optimal_results': optimal_results,
        'memory_analysis': memory_analysis
    }

if __name__ == "__main__":
    print("Day 3: Memory Planning Deep Dive")
    print("=" * 40)
    
    results = demonstrate_memory_planning()
    
    print("\\n" + "="*40)
    print("Memory Planning Analysis Complete!")
```

---

## Part 2: Kernel Fusion and Optimization (90 minutes)

### Understanding Kernel Fusion Strategy

```python
#!/usr/bin/env python3
"""
kernel_fusion.py - Deep dive into kernel fusion and optimization strategies
"""

import os
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, Ops
from tinygrad import Tensor
from tinygrad.helpers import DEBUG

@dataclass
class FusionAnalysis:
    """Analysis of fusion opportunities and results"""
    total_operations: int
    fusible_groups: List[List[int]]
    fusion_savings: int
    memory_bandwidth_reduction: float
    execution_time_estimate: float
    fusion_efficiency: float

class KernelFusionExplorer:
    """
    Advanced kernel fusion analysis and optimization
    """
    
    def __init__(self):
        self.fusion_history = []
        self.fusion_patterns = {}
        self.custom_fusion_rules = []
    
    def understand_fusion_opportunities(self, tensor_operation):
        """
        Identify and analyze kernel fusion opportunities
        """
        print("=== Kernel Fusion Opportunities Analysis ===\\n")
        
        # Create the operation and get UOp graph
        result = tensor_operation()
        uop_graph = list(result.uop.toposort())
        
        print(f"1. UOp Graph Analysis:")
        print(f"Total operations: {len(uop_graph)}")
        
        # Categorize operations by fusion potential
        fusible_ops = []
        memory_ops = []
        control_ops = []
        
        fusible_op_types = {Ops.ADD, Ops.MUL, Ops.RELU, Ops.EXP, Ops.LOG, Ops.SQRT}
        memory_op_types = {Ops.BUFFER, Ops.VIEW, Ops.RESHAPE}
        
        for i, uop in enumerate(uop_graph):
            if uop.op in fusible_op_types:
                fusible_ops.append((i, uop))
            elif uop.op in memory_op_types:
                memory_ops.append((i, uop))
            else:
                control_ops.append((i, uop))
        
        print(f"\\n2. Operation Categories:")
        print(f"  Fusible operations: {len(fusible_ops)}")
        print(f"  Memory operations: {len(memory_ops)}")
        print(f"  Control operations: {len(control_ops)}")
        
        # Find fusion groups
        fusion_groups = self._find_fusion_groups(uop_graph, fusible_ops)
        
        print(f"\\n3. Fusion Groups Found: {len(fusion_groups)}")
        for i, group in enumerate(fusion_groups):
            ops_in_group = [uop_graph[idx].op.name for idx in group]
            print(f"  Group {i+1}: {ops_in_group}")
        
        # Estimate fusion benefits
        fusion_benefits = self._estimate_fusion_benefits(uop_graph, fusion_groups)
        
        print(f"\\n4. Fusion Benefits:")
        print(f"  Memory bandwidth reduction: {fusion_benefits['bandwidth_reduction']:.1%}")
        print(f"  Kernel launch overhead reduction: {fusion_benefits['launch_reduction']:.1%}")
        print(f"  Estimated speedup: {fusion_benefits['estimated_speedup']:.2f}x")
        
        return fusion_groups, fusion_benefits
    
    def _find_fusion_groups(self, uop_graph: List[UOp], fusible_ops: List[Tuple[int, UOp]]) -> List[List[int]]:
        """Find groups of operations that can be fused together"""
        fusion_groups = []
        visited = set()
        
        def can_fuse(op1_idx: int, op2_idx: int) -> bool:
            """Check if two operations can be fused"""
            op1, op2 = uop_graph[op1_idx], uop_graph[op2_idx]
            
            # Basic fusion rules
            # 1. Operations must be compatible types
            fusible_types = {Ops.ADD, Ops.MUL, Ops.RELU, Ops.EXP, Ops.LOG}
            if op1.op not in fusible_types or op2.op not in fusible_types:
                return False
            
            # 2. Must have compatible shapes
            if hasattr(op1, 'shape') and hasattr(op2, 'shape'):
                if op1.shape != op2.shape and op1.shape != () and op2.shape != ():
                    return False
            
            # 3. Check data dependency
            if op2 in op1.src or op1 in op2.src:
                return True  # Direct dependency can be fused
            
            # 4. Check if they share inputs
            if op1.src and op2.src:
                shared_inputs = set(op1.src) & set(op2.src)
                if shared_inputs:
                    return True
            
            return False
        
        def build_fusion_group(start_idx: int) -> List[int]:
            """Build a fusion group starting from a given operation"""
            group = [start_idx]
            queue = [start_idx]
            
            while queue:
                current_idx = queue.pop(0)
                
                # Look for fusible neighbors
                for candidate_idx, _ in fusible_ops:
                    if candidate_idx not in visited and candidate_idx != current_idx:
                        if can_fuse(current_idx, candidate_idx):
                            group.append(candidate_idx)
                            queue.append(candidate_idx)
                            visited.add(candidate_idx)
            
            return group
        
        # Build fusion groups
        for op_idx, _ in fusible_ops:
            if op_idx not in visited:
                group = build_fusion_group(op_idx)
                if len(group) > 1:  # Only keep groups with multiple operations
                    fusion_groups.append(group)
                visited.update(group)
        
        return fusion_groups
    
    def _estimate_fusion_benefits(self, uop_graph: List[UOp], fusion_groups: List[List[int]]) -> Dict[str, float]:
        """Estimate the benefits of kernel fusion"""
        total_ops = len(uop_graph)
        fusible_ops = sum(len(group) for group in fusion_groups)
        
        # Estimate bandwidth reduction (fewer intermediate results)
        bandwidth_reduction = 0.0
        if fusible_ops > 0:
            # Each fused operation saves one intermediate write/read
            intermediate_saves = fusible_ops - len(fusion_groups)
            bandwidth_reduction = intermediate_saves / fusible_ops
        
        # Estimate kernel launch reduction
        launch_reduction = 0.0
        if total_ops > 0:
            original_kernels = fusible_ops  # Assume each op was a kernel
            fused_kernels = len(fusion_groups)
            launch_reduction = (original_kernels - fused_kernels) / original_kernels if original_kernels > 0 else 0
        
        # Estimate overall speedup
        # Speedup comes from reduced memory bandwidth and fewer kernel launches
        estimated_speedup = 1.0 + (bandwidth_reduction * 0.3) + (launch_reduction * 0.2)
        
        return {
            'bandwidth_reduction': bandwidth_reduction,
            'launch_reduction': launch_reduction,
            'estimated_speedup': estimated_speedup,
            'fusion_efficiency': (bandwidth_reduction + launch_reduction) / 2
        }
    
    def implement_custom_fusion_strategy(self):
        """
        Implement and test custom kernel fusion strategies
        """
        print("\\n=== Custom Fusion Strategy Implementation ===\\n")
        
        def aggressive_fusion_strategy(uop_graph: List[UOp]) -> List[List[int]]:
            """Aggressive fusion: try to fuse as many operations as possible"""
            print("Aggressive Fusion Strategy:")
            
            # Group all compatible operations together
            element_wise_ops = []
            reduction_ops = []
            memory_ops = []
            
            for i, uop in enumerate(uop_graph):
                if uop.op in {Ops.ADD, Ops.MUL, Ops.RELU, Ops.EXP}:
                    element_wise_ops.append(i)
                elif uop.op in {Ops.SUM, Ops.MAX, Ops.MEAN}:
                    reduction_ops.append(i)
                elif uop.op in {Ops.BUFFER, Ops.VIEW}:
                    memory_ops.append(i)
            
            fusion_groups = []
            if len(element_wise_ops) > 1:
                fusion_groups.append(element_wise_ops)
            if len(reduction_ops) > 1:
                fusion_groups.append(reduction_ops)
            
            print(f"  Element-wise group: {element_wise_ops}")
            print(f"  Reduction group: {reduction_ops}")
            print(f"  Total groups: {len(fusion_groups)}")
            
            return fusion_groups
        
        def conservative_fusion_strategy(uop_graph: List[UOp]) -> List[List[int]]:
            """Conservative fusion: only fuse operations with direct dependencies"""
            print("\\nConservative Fusion Strategy:")
            
            fusion_groups = []
            processed = set()
            
            for i, uop in enumerate(uop_graph):
                if i in processed:
                    continue
                
                # Look for direct dependencies that can be fused
                group = [i]
                for j, other_uop in enumerate(uop_graph[i+1:], i+1):
                    if j in processed:
                        continue
                    
                    # Check if other_uop directly depends on uop
                    if uop in other_uop.src and uop.op in {Ops.ADD, Ops.MUL, Ops.RELU}:
                        group.append(j)
                        processed.add(j)
                
                if len(group) > 1:
                    fusion_groups.append(group)
                processed.add(i)
            
            print(f"  Direct dependency groups: {len(fusion_groups)}")
            for i, group in enumerate(fusion_groups):
                ops = [uop_graph[idx].op.name for idx in group]
                print(f"    Group {i+1}: {ops}")
            
            return fusion_groups
        
        def adaptive_fusion_strategy(uop_graph: List[UOp]) -> List[List[int]]:
            """Adaptive fusion: balance fusion benefits with complexity"""
            print("\\nAdaptive Fusion Strategy:")
            
            # Score each potential fusion based on benefits vs complexity
            fusion_candidates = []
            
            for i, uop1 in enumerate(uop_graph):
                for j, uop2 in enumerate(uop_graph[i+1:], i+1):
                    # Calculate fusion score
                    compatibility_score = self._calculate_fusion_compatibility(uop1, uop2)
                    benefit_score = self._calculate_fusion_benefit(uop1, uop2)
                    complexity_penalty = self._calculate_fusion_complexity(uop1, uop2)
                    
                    total_score = compatibility_score * benefit_score - complexity_penalty
                    
                    if total_score > 0.5:  # Threshold for fusion
                        fusion_candidates.append(((i, j), total_score))
            
            # Sort by score and build non-overlapping groups
            fusion_candidates.sort(key=lambda x: x[1], reverse=True)
            
            fusion_groups = []
            used_ops = set()
            
            for (op1_idx, op2_idx), score in fusion_candidates:
                if op1_idx not in used_ops and op2_idx not in used_ops:
                    fusion_groups.append([op1_idx, op2_idx])
                    used_ops.update([op1_idx, op2_idx])
            
            print(f"  Adaptive fusion groups: {len(fusion_groups)}")
            print(f"  Average fusion score: {sum(score for _, score in fusion_candidates[:len(fusion_groups)]) / max(len(fusion_groups), 1):.2f}")
            
            return fusion_groups
        
        # Test all strategies
        test_expr = (Tensor.randn(100, 100).relu() + 1.0).exp().sum()
        uop_graph = list(test_expr.uop.toposort())
        
        print("Testing fusion strategies on complex expression:")
        print(f"Expression: (x.relu() + 1.0).exp().sum()")
        print(f"Total UOps: {len(uop_graph)}")
        print("=" * 50)
        
        aggressive_groups = aggressive_fusion_strategy(uop_graph)
        conservative_groups = conservative_fusion_strategy(uop_graph)
        adaptive_groups = adaptive_fusion_strategy(uop_graph)
        
        return {
            'aggressive': aggressive_groups,
            'conservative': conservative_groups,
            'adaptive': adaptive_groups
        }
    
    def _calculate_fusion_compatibility(self, uop1: UOp, uop2: UOp) -> float:
        """Calculate compatibility score for fusing two operations"""
        score = 0.0
        
        # Same operation type bonus
        if uop1.op == uop2.op:
            score += 0.5
        
        # Compatible operation types
        element_wise = {Ops.ADD, Ops.MUL, Ops.RELU, Ops.EXP}
        if uop1.op in element_wise and uop2.op in element_wise:
            score += 0.3
        
        # Shape compatibility
        if hasattr(uop1, 'shape') and hasattr(uop2, 'shape'):
            if uop1.shape == uop2.shape:
                score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_fusion_benefit(self, uop1: UOp, uop2: UOp) -> float:
        """Calculate potential benefit of fusing two operations"""
        benefit = 0.0
        
        # Memory bandwidth savings
        if hasattr(uop1, 'shape') and uop1.shape:
            elements = 1
            for dim in uop1.shape:
                if isinstance(dim, int):
                    elements *= dim
            # Benefit proportional to data size
            benefit += min(elements / 10000, 0.5)
        
        # Kernel launch overhead savings
        benefit += 0.3
        
        return min(benefit, 1.0)
    
    def _calculate_fusion_complexity(self, uop1: UOp, uop2: UOp) -> float:
        """Calculate complexity penalty for fusing two operations"""
        complexity = 0.0
        
        # Different operation types increase complexity
        if uop1.op != uop2.op:
            complexity += 0.2
        
        # Complex operations increase complexity
        complex_ops = {Ops.CONV, Ops.MATMUL}
        if uop1.op in complex_ops or uop2.op in complex_ops:
            complexity += 0.3
        
        return complexity
    
    def benchmark_fusion_strategies(self, test_operations: List[Tuple[str, callable]]):
        """
        Benchmark different fusion strategies on various operations
        """
        print("\\n=== Fusion Strategy Benchmarking ===\\n")
        
        results = {}
        
        for name, operation in test_operations:
            print(f"Benchmarking: {name}")
            print("-" * 40)
            
            # Create operation
            result = operation()
            uop_graph = list(result.uop.toposort())
            
            # Test fusion opportunities
            fusion_groups, benefits = self.understand_fusion_opportunities(operation)
            
            # Time the original operation
            start_time = time.time()
            numpy_result = result.numpy()
            original_time = time.time() - start_time
            
            # Estimate fused performance (simulation)
            estimated_fused_time = original_time / benefits['estimated_speedup']
            
            results[name] = {
                'original_time': original_time,
                'estimated_fused_time': estimated_fused_time,
                'speedup': benefits['estimated_speedup'],
                'fusion_groups': len(fusion_groups),
                'fusion_efficiency': benefits['fusion_efficiency']
            }
            
            print(f"  Original time: {original_time*1000:.2f}ms")
            print(f"  Estimated fused time: {estimated_fused_time*1000:.2f}ms")
            print(f"  Estimated speedup: {benefits['estimated_speedup']:.2f}x")
            print(f"  Fusion groups: {len(fusion_groups)}")
            print()
        
        # Summary
        print("Benchmarking Summary:")
        print(f"{'Operation':<20} | {'Original (ms)':<12} | {'Fused (ms)':<10} | {'Speedup':<8} | {'Groups':<6}")
        print("-" * 70)
        
        for name, data in results.items():
            print(f"{name:<20} | {data['original_time']*1000:<12.2f} | {data['estimated_fused_time']*1000:<10.2f} | "
                  f"{data['speedup']:<8.2f} | {data['fusion_groups']:<6}")
        
        return results

def demonstrate_kernel_fusion():
    """
    Comprehensive demonstration of kernel fusion concepts
    """
    print("=== Kernel Fusion Demonstration ===\\n")
    
    explorer = KernelFusionExplorer()
    
    # Test different operation patterns
    test_operations = [
        ("Element-wise Chain", lambda: (Tensor.randn(100, 100) + 1.0).relu().exp()),
        ("Matrix + Element-wise", lambda: (Tensor.randn(50, 50) @ Tensor.randn(50, 50)).relu()),
        ("Broadcast + Ops", lambda: Tensor.randn(100, 1) + Tensor.randn(1, 100).exp()),
        ("Reduction Chain", lambda: Tensor.randn(200, 200).sum(axis=1).exp().mean()),
        ("Complex Expression", lambda: ((Tensor.randn(75, 75) * 2).relu() + 1).log().sum()),
    ]
    
    print("1. Fusion Opportunity Analysis:")
    print("=" * 40)
    
    fusion_results = {}
    for name, operation in test_operations:
        print(f"\\n{name}:")
        groups, benefits = explorer.understand_fusion_opportunities(operation)
        fusion_results[name] = (groups, benefits)
    
    print("\\n" + "="*60)
    print("2. Custom Fusion Strategy Testing:")
    strategy_results = explorer.implement_custom_fusion_strategy()
    
    print("\\n" + "="*60)
    print("3. Fusion Strategy Benchmarking:")
    benchmark_results = explorer.benchmark_fusion_strategies(test_operations)
    
    return {
        'fusion_analysis': fusion_results,
        'strategy_comparison': strategy_results,
        'benchmark_results': benchmark_results
    }

if __name__ == "__main__":
    print("Day 3: Kernel Fusion Deep Dive")
    print("=" * 40)
    
    results = demonstrate_kernel_fusion()
    
    print("\\n" + "="*40)
    print("Kernel Fusion Analysis Complete!")
```

---

## Part 3: Execution Pipeline and Performance Analysis (60 minutes)

### Complete Execution Pipeline Analysis

```python
#!/usr/bin/env python3
"""
execution_pipeline.py - Complete analysis of tinygrad's execution pipeline
"""

import os
import time
import traceback
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import statistics

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.helpers import DEBUG, GlobalCounters
from tinygrad.uop.ops import UOp, Ops

@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics"""
    graph_construction_time: float
    scheduling_time: float
    memory_planning_time: float
    kernel_fusion_time: float
    code_generation_time: float
    execution_time: float
    total_time: float
    memory_peak: int
    kernel_count: int
    fusion_efficiency: float

class ExecutionPipelineAnalyzer:
    """
    Comprehensive analysis of the complete execution pipeline
    """
    
    def __init__(self):
        self.execution_history = []
        self.performance_profiles = {}
        self.optimization_metrics = {}
    
    def analyze_complete_pipeline(self, tensor_operation, operation_name: str = "unknown"):
        """
        Analyze the complete execution pipeline from tensor operation to result
        """
        print(f"=== Complete Pipeline Analysis: {operation_name} ===\\n")
        
        metrics = ExecutionMetrics(
            graph_construction_time=0.0,
            scheduling_time=0.0,
            memory_planning_time=0.0,
            kernel_fusion_time=0.0,
            code_generation_time=0.0,
            execution_time=0.0,
            total_time=0.0,
            memory_peak=0,
            kernel_count=0,
            fusion_efficiency=0.0
        )
        
        total_start = time.time()
        
        # Phase 1: Graph Construction
        print("1. Graph Construction Phase:")
        construction_start = time.time()
        
        result = tensor_operation()
        
        construction_end = time.time()
        metrics.graph_construction_time = construction_end - construction_start
        
        # Analyze the constructed graph
        uop_graph = list(result.uop.toposort())
        print(f"  UOp graph size: {len(uop_graph)} operations")
        print(f"  Construction time: {metrics.graph_construction_time*1000:.2f}ms")
        
        # Categorize operations
        op_categories = defaultdict(int)
        for uop in uop_graph:
            op_categories[uop.op.name] += 1
        
        print(f"  Operation breakdown:")
        for op_name, count in sorted(op_categories.items()):
            print(f"    {op_name}: {count}")
        
        # Phase 2: Pipeline Analysis (simulated timing)
        print("\\n2. Pipeline Phases Analysis:")
        
        # Scheduling phase (simulated)
        scheduling_start = time.time()
        self._simulate_scheduling_phase(uop_graph)
        scheduling_end = time.time()
        metrics.scheduling_time = scheduling_end - scheduling_start
        print(f"  Scheduling: {metrics.scheduling_time*1000:.2f}ms")
        
        # Memory planning phase (simulated)
        memory_start = time.time()
        memory_usage = self._simulate_memory_planning(uop_graph)
        memory_end = time.time()
        metrics.memory_planning_time = memory_end - memory_start
        metrics.memory_peak = memory_usage
        print(f"  Memory planning: {metrics.memory_planning_time*1000:.2f}ms")
        print(f"  Peak memory estimate: {memory_usage / (1024*1024):.2f}MB")
        
        # Kernel fusion phase (simulated)
        fusion_start = time.time()
        fusion_stats = self._simulate_kernel_fusion(uop_graph)
        fusion_end = time.time()
        metrics.kernel_fusion_time = fusion_end - fusion_start
        metrics.kernel_count = fusion_stats['kernel_count']
        metrics.fusion_efficiency = fusion_stats['fusion_efficiency']
        print(f"  Kernel fusion: {metrics.kernel_fusion_time*1000:.2f}ms")
        print(f"  Kernels after fusion: {metrics.kernel_count}")
        print(f"  Fusion efficiency: {metrics.fusion_efficiency:.1%}")
        
        # Phase 3: Actual Execution
        print("\\n3. Execution Phase:")
        execution_start = time.time()
        
        # Enable detailed debugging for execution
        old_debug = os.environ.get('DEBUG', '0')
        os.environ['DEBUG'] = '2'
        
        try:
            numpy_result = result.numpy()
            execution_end = time.time()
            metrics.execution_time = execution_end - execution_start
            
            print(f"  Execution time: {metrics.execution_time*1000:.2f}ms")
            print(f"  Result shape: {numpy_result.shape}")
            
        finally:
            os.environ['DEBUG'] = old_debug
        
        # Calculate total time
        total_end = time.time()
        metrics.total_time = total_end - total_start
        
        print(f"\\n4. Pipeline Summary:")
        print(f"  Total time: {metrics.total_time*1000:.2f}ms")
        
        # Time breakdown
        overhead_time = (metrics.graph_construction_time + metrics.scheduling_time + 
                        metrics.memory_planning_time + metrics.kernel_fusion_time)
        
        print(f"  Breakdown:")
        print(f"    Graph construction: {metrics.graph_construction_time/metrics.total_time*100:.1f}%")
        print(f"    Pipeline overhead: {overhead_time/metrics.total_time*100:.1f}%")
        print(f"    Actual execution: {metrics.execution_time/metrics.total_time*100:.1f}%")
        
        self.execution_history.append((operation_name, metrics))
        return metrics, numpy_result
    
    def _simulate_scheduling_phase(self, uop_graph: List[UOp]):
        """Simulate the scheduling phase analysis"""
        # Analyze dependencies and ordering
        dependency_analysis = defaultdict(int)
        
        for uop in uop_graph:
            if uop.src:
                dependency_analysis['has_dependencies'] += 1
            else:
                dependency_analysis['leaf_nodes'] += 1
        
        # Simulate some processing time
        time.sleep(0.001)  # 1ms simulation
    
    def _simulate_memory_planning(self, uop_graph: List[UOp]) -> int:
        """Simulate memory planning and return estimated peak usage"""
        total_memory = 0
        
        for uop in uop_graph:
            if hasattr(uop, 'shape') and uop.shape:
                elements = 1
                for dim in uop.shape:
                    if isinstance(dim, int):
                        elements *= dim
                
                # Estimate 4 bytes per float32 element
                total_memory += elements * 4
        
        # Simulate processing time
        time.sleep(0.0005)  # 0.5ms simulation
        
        return total_memory
    
    def _simulate_kernel_fusion(self, uop_graph: List[UOp]) -> Dict[str, Any]:
        """Simulate kernel fusion analysis"""
        fusible_ops = sum(1 for uop in uop_graph if uop.op in {Ops.ADD, Ops.MUL, Ops.RELU, Ops.EXP})
        
        # Simulate fusion grouping
        estimated_kernels = max(1, fusible_ops // 3)  # Assume fusion groups of ~3
        fusion_efficiency = (fusible_ops - estimated_kernels) / max(fusible_ops, 1)
        
        # Simulate processing time
        time.sleep(0.0008)  # 0.8ms simulation
        
        return {
            'kernel_count': estimated_kernels,
            'fusion_efficiency': fusion_efficiency,
            'fusible_operations': fusible_ops
        }
    
    def compare_execution_strategies(self, operations: List[Tuple[str, callable]]):
        """
        Compare execution strategies across different operation types
        """
        print("\\n=== Execution Strategy Comparison ===\\n")
        
        results = {}
        
        for name, operation in operations:
            print(f"Analyzing: {name}")
            print("-" * 30)
            
            metrics, _ = self.analyze_complete_pipeline(operation, name)
            results[name] = metrics
        
        # Comparative analysis
        print("\\n" + "="*60)
        print("Execution Strategy Comparison:")
        print(f"{'Operation':<20} | {'Total (ms)':<10} | {'Exec (ms)':<9} | {'Overhead':<9} | {'Kernels':<7}")
        print("-" * 70)
        
        for name, metrics in results.items():
            overhead_pct = (1 - metrics.execution_time / metrics.total_time) * 100
            print(f"{name:<20} | {metrics.total_time*1000:<10.2f} | {metrics.execution_time*1000:<9.2f} | "
                  f"{overhead_pct:<9.1f}% | {metrics.kernel_count:<7}")
        
        return results
    
    def performance_regression_testing(self, operation_factory, iterations: int = 10):
        """
        Performance regression testing for consistent execution analysis
        """
        print(f"\\n=== Performance Regression Testing ({iterations} iterations) ===\\n")
        
        execution_times = []
        total_times = []
        memory_peaks = []
        
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")
            
            # Create fresh operation each time
            operation = operation_factory()
            
            # Measure execution
            start_time = time.time()
            result = operation.numpy()
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            total_times.append(execution_time)  # Simplified for this test
            
            # Estimate memory
            if hasattr(operation, 'nbytes'):
                memory_peaks.append(operation.nbytes)
            else:
                memory_peaks.append(0)
            
            print(f"  Execution time: {execution_time*1000:.2f}ms")
        
        # Statistical analysis
        print(f"\\nRegression Analysis:")
        print(f"  Mean execution time: {statistics.mean(execution_times)*1000:.2f}ms")
        print(f"  Std deviation: {statistics.stdev(execution_times)*1000:.2f}ms" if len(execution_times) > 1 else "  Std deviation: 0.00ms")
        print(f"  Min time: {min(execution_times)*1000:.2f}ms")
        print(f"  Max time: {max(execution_times)*1000:.2f}ms")
        print(f"  Coefficient of variation: {(statistics.stdev(execution_times)/statistics.mean(execution_times)*100):.1f}%" if len(execution_times) > 1 else "  Coefficient of variation: 0.0%")
        
        # Performance stability analysis
        if len(execution_times) > 1:
            cv = statistics.stdev(execution_times) / statistics.mean(execution_times)
            if cv < 0.05:
                print(f"  Performance: STABLE (CV: {cv:.3f})")
            elif cv < 0.15:
                print(f"  Performance: ACCEPTABLE (CV: {cv:.3f})")
            else:
                print(f"  Performance: UNSTABLE (CV: {cv:.3f})")
        
        return {
            'execution_times': execution_times,
            'mean_time': statistics.mean(execution_times),
            'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'stability': 'stable' if len(execution_times) > 1 and statistics.stdev(execution_times)/statistics.mean(execution_times) < 0.05 else 'unstable'
        }
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report
        """
        report = []
        report.append("=" * 60)
        report.append("TINYGRAD EXECUTION PIPELINE PERFORMANCE REPORT")
        report.append("=" * 60)
        
        if not self.execution_history:
            report.append("No execution data available.")
            return "\\n".join(report)
        
        # Summary statistics
        total_operations = len(self.execution_history)
        all_metrics = [metrics for _, metrics in self.execution_history]
        
        avg_total_time = statistics.mean(m.total_time for m in all_metrics)
        avg_execution_time = statistics.mean(m.execution_time for m in all_metrics)
        avg_overhead = avg_total_time - avg_execution_time
        
        report.append(f"\\nSUMMARY:")
        report.append(f"  Operations analyzed: {total_operations}")
        report.append(f"  Average total time: {avg_total_time*1000:.2f}ms")
        report.append(f"  Average execution time: {avg_execution_time*1000:.2f}ms")
        report.append(f"  Average overhead: {avg_overhead*1000:.2f}ms ({avg_overhead/avg_total_time*100:.1f}%)")
        
        # Detailed breakdown
        report.append(f"\\nDETAILED ANALYSIS:")
        report.append(f"{'Operation':<20} | {'Total':<8} | {'Exec':<8} | {'Overhead':<8} | {'Kernels':<7} | {'Memory':<8}")
        report.append("-" * 75)
        
        for name, metrics in self.execution_history:
            overhead = metrics.total_time - metrics.execution_time
            memory_mb = metrics.memory_peak / (1024 * 1024)
            report.append(f"{name:<20} | {metrics.total_time*1000:<8.1f} | {metrics.execution_time*1000:<8.1f} | "
                         f"{overhead*1000:<8.1f} | {metrics.kernel_count:<7} | {memory_mb:<8.1f}")
        
        # Performance insights
        report.append(f"\\nPERFORMANCE INSIGHTS:")
        
        # Find bottlenecks
        bottleneck_ops = [(name, metrics) for name, metrics in self.execution_history 
                         if (metrics.total_time - metrics.execution_time) / metrics.total_time > 0.5]
        
        if bottleneck_ops:
            report.append(f"  High overhead operations ({len(bottleneck_ops)}):")
            for name, metrics in bottleneck_ops:
                overhead_pct = (metrics.total_time - metrics.execution_time) / metrics.total_time * 100
                report.append(f"    - {name}: {overhead_pct:.1f}% overhead")
        
        # Memory usage analysis
        high_memory_ops = [(name, metrics) for name, metrics in self.execution_history 
                          if metrics.memory_peak > 100 * 1024 * 1024]  # > 100MB
        
        if high_memory_ops:
            report.append(f"  High memory operations ({len(high_memory_ops)}):")
            for name, metrics in high_memory_ops:
                memory_mb = metrics.memory_peak / (1024 * 1024)
                report.append(f"    - {name}: {memory_mb:.1f}MB")
        
        report.append("\\n" + "=" * 60)
        
        return "\\n".join(report)

def demonstrate_execution_pipeline():
    """
    Comprehensive demonstration of execution pipeline analysis
    """
    print("=== Execution Pipeline Demonstration ===\\n")
    
    analyzer = ExecutionPipelineAnalyzer()
    
    # Test operations of varying complexity
    test_operations = [
        ("Simple Add", lambda: Tensor([1., 2., 3.]) + Tensor([4., 5., 6.])),
        ("Matrix Multiply", lambda: Tensor.randn(100, 100) @ Tensor.randn(100, 100)),
        ("Element-wise Chain", lambda: (Tensor.randn(200, 200) + 1.0).relu().exp()),
        ("Reduction", lambda: Tensor.randn(500, 500).sum(axis=1)),
        ("Complex Expression", lambda: ((Tensor.randn(150, 150) @ Tensor.randn(150, 150)).relu() + 1.0).log().mean()),
    ]
    
    print("1. Individual Pipeline Analysis:")
    print("=" * 50)
    
    # Analyze each operation
    for name, operation in test_operations:
        analyzer.analyze_complete_pipeline(operation, name)
        print("\\n" + "-"*50 + "\\n")
    
    print("2. Comparative Analysis:")
    print("=" * 50)
    comparison_results = analyzer.compare_execution_strategies(test_operations)
    
    print("\\n3. Performance Regression Testing:")
    print("=" * 50)
    
    # Test a specific operation for stability
    def stable_operation():
        return Tensor.randn(100, 100).sum()
    
    regression_results = analyzer.performance_regression_testing(stable_operation, iterations=5)
    
    print("\\n4. Performance Report:")
    print("=" * 50)
    report = analyzer.generate_performance_report()
    print(report)
    
    return {
        'comparison_results': comparison_results,
        'regression_results': regression_results,
        'performance_report': report
    }

if __name__ == "__main__":
    # Set up environment
    os.environ['DEBUG'] = '1'
    
    print("Day 3: Complete Execution Pipeline Analysis")
    print("=" * 50)
    
    results = demonstrate_execution_pipeline()
    
    print("\\n" + "="*50)
    print("Execution Pipeline Analysis Complete!")
```

---

## Day 3 Wrap-up & Advanced Challenges

### What You've Mastered Today

1. ✅ **Scheduling System**: Deep understanding of how UOp graphs become execution schedules
2. ✅ **Memory Planning**: Advanced memory optimization and allocation strategies
3. ✅ **Kernel Fusion**: Custom fusion strategies and optimization techniques
4. ✅ **Pipeline Analysis**: Complete execution pipeline profiling and optimization
5. ✅ **Performance Engineering**: Advanced performance analysis and regression testing

### Tomorrow's Preview: Runtime Backends & Device Programming

Day 4 will focus on:
- **Backend Architecture**: Understanding device-specific implementations
- **Runtime Systems**: How kernels execute on different hardware
- **Custom Backend Development**: Building new device support
- **Performance Optimization**: Device-specific optimization techniques

### Advanced Homework Assignments

1. **Custom Scheduler**: Implement a scheduler optimized for your electrical testing workload
2. **Memory Profiler**: Build a real-time memory usage monitor for tinygrad
3. **Fusion Optimizer**: Create domain-specific kernel fusion rules
4. **Performance Benchmark**: Develop a comprehensive performance testing suite

### Self-Assessment Checklist

- [ ] Can I analyze and optimize execution schedules?
- [ ] Can I implement custom memory planning strategies?
- [ ] Can I design kernel fusion optimizations?
- [ ] Can I profile and debug the complete execution pipeline?
- [ ] Can I identify and resolve performance bottlenecks?

### Practical Challenge: Build Your Custom Engine

```python
# Challenge: Build an engine optimized for electrical validation
class ElectricalValidationEngine:
    def __init__(self):
        self.scheduler = CustomScheduler()
        self.memory_planner = MemoryPlanner()
        self.fusion_engine = FusionEngine()
    
    def optimize_for_real_time_monitoring(self, operations):
        """Optimize for minimal latency electrical monitoring"""
        # TODO: Implement optimizations for:
        # - Streaming data processing
        # - Minimal memory allocation
        # - Predictable execution times
        pass
    
    def optimize_for_batch_validation(self, operations):
        """Optimize for throughput in batch testing scenarios"""
        # TODO: Implement optimizations for:
        # - Maximum kernel fusion
        # - Optimal memory reuse
        # - Parallel execution
        pass
```

**Ready for Day 4? Backend development and device programming await! 🚀**