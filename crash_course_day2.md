# Day 2: UOp System Mastery & Graph Optimization

## Overview
Today we dive deep into the Universal Operations (UOp) system - the computational heart of tinygrad. You'll understand how operations are represented, optimized, and transformed through graph rewriting. By the end, you'll be able to create custom UOps, visualize computation graphs, and understand the optimization passes that make tinygrad efficient.

## Learning Objectives
- ✅ Master the UOp class and operation hierarchy
- ✅ Understand graph construction and pattern matching
- ✅ Learn graph rewriting and optimization techniques
- ✅ Build custom UOp operations and transformations
- ✅ Create advanced debugging and visualization tools
- ✅ Implement performance profiling for UOp graphs

---

## Part 1: UOp Architecture Deep Dive (120 minutes)

### Understanding UOp Internal Structure

The UOp system is tinygrad's internal representation language for computation. Every tensor operation ultimately becomes a graph of UOps.

```python
#!/usr/bin/env python3
"""
Day 2 Exercise: UOp System Deep Architecture Exploration
"""

import os
import time
import weakref
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, Counter

from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite
from tinygrad.uop.spec import type_verify
from tinygrad import Tensor
from tinygrad.dtype import dtypes

class UOpExplorer:
    """
    Advanced UOp system explorer and analyzer
    """
    
    def __init__(self):
        self.uop_cache = {}
        self.analysis_history = []
    
    def analyze_uop_structure(self, uop: UOp) -> Dict:
        """
        Comprehensive analysis of UOp structure and properties
        """
        analysis = {
            'uop_id': id(uop),
            'operation': uop.op.name,
            'dtype': str(uop.dtype),
            'shape': uop.shape,
            'device': getattr(uop, 'device', 'unknown'),
            'arg': uop.arg,
            'source_count': len(uop.src) if uop.src else 0,
            'hash': hash(uop),
        }
        
        # Analyze sources (inputs)
        if uop.src:
            analysis['sources'] = [self.analyze_uop_structure(src) for src in uop.src]
        
        # Check for special properties
        analysis['is_leaf'] = len(uop.src) == 0 if uop.src else True
        analysis['is_const'] = uop.op == Ops.CONST
        analysis['is_buffer'] = uop.op == Ops.BUFFER
        analysis['is_view'] = uop.op == Ops.VIEW
        
        return analysis
    
    def trace_uop_graph_construction(self, tensor_operation):
        """
        Trace how tensor operations build UOp graphs step by step
        """
        print("=== UOp Graph Construction Tracing ===\\n")
        
        # Monkey patch UOp creation to trace
        original_new = UOp.__new__
        creation_log = []
        
        def traced_new(cls, op, dtype, src=None, arg=None):
            uop = original_new(cls)
            creation_info = {
                'id': id(uop),
                'op': op.name,
                'dtype': str(dtype),
                'src_count': len(src) if src else 0,
                'arg': str(arg)[:50] + '...' if arg and len(str(arg)) > 50 else str(arg),
                'timestamp': time.time()
            }
            creation_log.append(creation_info)
            print(f"UOp Created: {op.name} | dtype: {dtype} | sources: {len(src) if src else 0}")
            return uop
        
        # Apply tracing
        UOp.__new__ = traced_new
        
        try:
            # Execute the operation
            result = tensor_operation()
            
            print(f"\\nTotal UOps created: {len(creation_log)}")
            print("\\nCreation timeline:")
            for i, info in enumerate(creation_log):
                print(f"  {i:2d}: {info['op']:<12} | {info['dtype']:<10} | sources: {info['src_count']}")
            
            return result, creation_log
            
        finally:
            # Restore original method
            UOp.__new__ = original_new
    
    def analyze_operation_patterns(self, uop: UOp) -> Dict:
        """
        Analyze patterns in UOp graph for optimization opportunities
        """
        all_uops = list(uop.toposort())
        
        # Count operation types
        op_counts = Counter(u.op for u in all_uops)
        
        # Find repeated patterns
        patterns = defaultdict(int)
        for u in all_uops:
            if u.src:
                pattern = (u.op, tuple(src.op for src in u.src))
                patterns[pattern] += 1
        
        # Find potential optimization opportunities
        optimizations = []
        
        # Look for identity operations (x + 0, x * 1, etc.)
        for u in all_uops:
            if u.op == Ops.ADD and u.src:
                for src in u.src:
                    if src.op == Ops.CONST and src.arg == 0:
                        optimizations.append(f"ADD with zero at UOp {id(u)}")
            
            if u.op == Ops.MUL and u.src:
                for src in u.src:
                    if src.op == Ops.CONST and src.arg == 1:
                        optimizations.append(f"MUL with one at UOp {id(u)}")
        
        # Look for redundant operations
        redundant = []
        seen_patterns = {}
        for u in all_uops:
            if u.src:
                signature = (u.op, tuple(id(src) for src in u.src), u.arg)
                if signature in seen_patterns:
                    redundant.append(f"Redundant {u.op.name} at UOp {id(u)}")
                seen_patterns[signature] = u
        
        return {
            'total_uops': len(all_uops),
            'operation_counts': dict(op_counts),
            'common_patterns': dict(patterns),
            'optimization_opportunities': optimizations,
            'redundant_operations': redundant,
            'graph_depth': self._calculate_graph_depth(uop),
            'parallelism_opportunities': self._find_parallelism(all_uops)
        }
    
    def _calculate_graph_depth(self, uop: UOp) -> int:
        """Calculate the depth of the UOp graph"""
        if not uop.src:
            return 0
        return 1 + max(self._calculate_graph_depth(src) for src in uop.src)
    
    def _find_parallelism(self, all_uops: List[UOp]) -> List[List[UOp]]:
        """Find UOps that can be executed in parallel"""
        # Build dependency graph
        dependencies = defaultdict(set)
        dependents = defaultdict(set)
        
        for uop in all_uops:
            if uop.src:
                for src in uop.src:
                    dependencies[uop].add(src)
                    dependents[src].add(uop)
        
        # Find groups that can run in parallel
        parallel_groups = []
        processed = set()
        
        for uop in all_uops:
            if uop in processed:
                continue
                
            # Find all UOps at the same "level"
            same_level = []
            queue = [uop]
            
            while queue:
                current = queue.pop(0)
                if current in processed:
                    continue
                    
                # Check if all dependencies are satisfied
                deps_satisfied = all(dep in processed for dep in dependencies[current])
                if deps_satisfied:
                    same_level.append(current)
                    processed.add(current)
                else:
                    queue.append(current)
            
            if len(same_level) > 1:
                parallel_groups.append(same_level)
        
        return parallel_groups

def explore_uop_operations():
    """
    Deep exploration of different UOp operation types and their properties
    """
    print("=== UOp Operations Deep Dive ===\\n")
    
    explorer = UOpExplorer()
    
    # Test different operation categories
    test_operations = [
        ("Simple Addition", lambda: Tensor([1., 2., 3.]) + Tensor([4., 5., 6.])),
        ("Matrix Multiplication", lambda: Tensor([[1., 2.], [3., 4.]]) @ Tensor([[5., 6.], [7., 8.]])),
        ("Complex Expression", lambda: (Tensor([1., 2., 3.]) * 2 + 1).relu().sum()),
        ("Reduction Chain", lambda: Tensor.randn(4, 4).sum(axis=1).mean()),
        ("Broadcasting", lambda: Tensor([[1., 2.]]) + Tensor([[3.], [4.]])),
    ]
    
    for name, operation in test_operations:
        print(f"\\n{name}:")
        print("-" * 40)
        
        # Trace construction
        result, creation_log = explorer.trace_uop_graph_construction(operation)
        
        # Analyze patterns
        if hasattr(result, 'uop'):
            analysis = explorer.analyze_operation_patterns(result.uop)
            
            print(f"\\nGraph Analysis:")
            print(f"  Total UOps: {analysis['total_uops']}")
            print(f"  Graph depth: {analysis['graph_depth']}")
            print(f"  Operation types: {analysis['operation_counts']}")
            
            if analysis['optimization_opportunities']:
                print(f"  Optimizations found: {len(analysis['optimization_opportunities'])}")
                for opt in analysis['optimization_opportunities'][:3]:  # Show first 3
                    print(f"    - {opt}")
            
            if analysis['parallelism_opportunities']:
                print(f"  Parallel groups: {len(analysis['parallelism_opportunities'])}")

def understand_uop_creation_and_caching():
    """
    Understand how UOps are created, cached, and managed
    """
    print("\\n=== UOp Creation and Caching ===\\n")
    
    # UOps are heavily cached for efficiency
    print("1. UOp Caching Demonstration:")
    
    # Create identical UOps - they should be the same object
    uop1 = UOp(Ops.CONST, dtypes.float32, arg=3.14)
    uop2 = UOp(Ops.CONST, dtypes.float32, arg=3.14)
    
    print(f"UOp1 id: {id(uop1)}")
    print(f"UOp2 id: {id(uop2)}")
    print(f"Same object: {uop1 is uop2}")
    print(f"Equal: {uop1 == uop2}")
    
    # Different args should be different objects
    uop3 = UOp(Ops.CONST, dtypes.float32, arg=2.71)
    print(f"UOp3 id: {id(uop3)}")
    print(f"UOp1 is UOp3: {uop1 is uop3}")
    
    print("\\n2. UOp Construction with Sources:")
    
    # Build a small graph
    a = UOp(Ops.CONST, dtypes.float32, arg=1.0)
    b = UOp(Ops.CONST, dtypes.float32, arg=2.0)
    c = UOp(Ops.ADD, dtypes.float32, (a, b))
    
    print(f"Constant A: {a}")
    print(f"Constant B: {b}")
    print(f"Add C: {c}")
    print(f"C sources: {c.src}")
    print(f"C sources count: {len(c.src)}")
    
    print("\\n3. UOp Graph Traversal:")
    
    # Traverse the graph
    all_uops = list(c.toposort())
    print(f"Topological order ({len(all_uops)} UOps):")
    for i, uop in enumerate(all_uops):
        print(f"  {i}: {uop.op.name} | {uop.dtype} | arg: {uop.arg}")
    
    print("\\n4. UOp Memory Management:")
    
    # UOps use weak references for memory efficiency
    import gc
    import weakref
    
    # Create a temporary UOp and check cleanup
    temp_uop = UOp(Ops.CONST, dtypes.float32, arg=42.0)
    weak_ref = weakref.ref(temp_uop)
    print(f"Weak reference alive: {weak_ref() is not None}")
    
    del temp_uop
    gc.collect()
    print(f"After deletion and GC: {weak_ref() is not None}")

if __name__ == "__main__":
    # Enable detailed debugging
    os.environ['DEBUG'] = '2'
    
    print("Day 2: UOp System Deep Dive")
    print("=" * 50)
    
    explore_uop_operations()
    understand_uop_creation_and_caching()
```

### UOp Graph Visualization and Analysis

```python
#!/usr/bin/env python3
"""
uop_visualization.py - Advanced UOp graph visualization and analysis tools
"""

import graphviz
import json
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from tinygrad.uop.ops import UOp, Ops
from tinygrad import Tensor

class UOpGraphVisualizer:
    """
    Create visual representations of UOp graphs for better understanding
    """
    
    def __init__(self):
        self.color_map = {
            Ops.CONST: '#FFB6C1',      # Light pink
            Ops.BUFFER: '#98FB98',      # Pale green  
            Ops.VIEW: '#87CEEB',        # Sky blue
            Ops.ADD: '#FFA07A',         # Light salmon
            Ops.MUL: '#DDA0DD',         # Plum
            Ops.RELU: '#F0E68C',        # Khaki
            Ops.SUM: '#20B2AA',         # Light sea green
            Ops.RESHAPE: '#D3D3D3',     # Light gray
            Ops.PERMUTE: '#B0C4DE',     # Light steel blue
        }
        self.default_color = '#FFFFFF'  # White
    
    def visualize_uop_graph(self, uop: UOp, filename: str = "uop_graph", format: str = "svg") -> str:
        """
        Create a visual representation of UOp graph using Graphviz
        """
        try:
            import graphviz
        except ImportError:
            print("graphviz not installed. Install with: pip install graphviz")
            return self._create_text_visualization(uop)
        
        dot = graphviz.Digraph(comment='UOp Graph')
        dot.attr(rankdir='TB', size='12,8')
        dot.attr('node', shape='box', style='filled,rounded')
        
        # Get all UOps in topological order
        all_uops = list(uop.toposort())
        uop_to_id = {u: f"uop_{i}" for i, u in enumerate(all_uops)}
        
        # Add nodes
        for u in all_uops:
            node_id = uop_to_id[u]
            color = self.color_map.get(u.op, self.default_color)
            
            # Create label with operation info
            label = f"{u.op.name}\\n"
            label += f"dtype: {u.dtype}\\n"
            if u.shape:
                label += f"shape: {u.shape}\\n"
            if u.arg is not None:
                arg_str = str(u.arg)
                if len(arg_str) > 20:
                    arg_str = arg_str[:17] + "..."
                label += f"arg: {arg_str}"
            
            dot.node(node_id, label, fillcolor=color)
        
        # Add edges
        for u in all_uops:
            if u.src:
                for src in u.src:
                    dot.edge(uop_to_id[src], uop_to_id[u])
        
        # Render the graph
        output_path = dot.render(filename, format=format, cleanup=True)
        print(f"UOp graph saved to: {output_path}")
        return output_path
    
    def _create_text_visualization(self, uop: UOp) -> str:
        """
        Create a text-based visualization when graphviz is not available
        """
        result = []
        result.append("UOp Graph (Text Visualization)")
        result.append("=" * 40)
        
        all_uops = list(uop.toposort())
        uop_to_id = {u: i for i, u in enumerate(all_uops)}
        
        for i, u in enumerate(all_uops):
            indent = "  " * self._get_depth_level(u, all_uops)
            sources = f" ← [{', '.join(str(uop_to_id[src]) for src in u.src)}]" if u.src else ""
            
            result.append(f"{indent}{i:2d}: {u.op.name:<12} | {u.dtype} | {u.shape}{sources}")
            if u.arg is not None:
                result.append(f"{indent}    arg: {u.arg}")
        
        text_viz = "\\n".join(result)
        print(text_viz)
        return text_viz
    
    def _get_depth_level(self, target_uop: UOp, all_uops: List[UOp]) -> int:
        """Calculate the depth level of a UOp for indentation"""
        if not target_uop.src:
            return 0
        return 1 + max(self._get_depth_level(src, all_uops) for src in target_uop.src)
    
    def analyze_graph_complexity(self, uop: UOp) -> Dict:
        """
        Analyze the complexity characteristics of a UOp graph
        """
        all_uops = list(uop.toposort())
        
        # Basic statistics
        stats = {
            'total_nodes': len(all_uops),
            'max_depth': self._calculate_max_depth(uop),
            'avg_fanout': sum(len(u.src) if u.src else 0 for u in all_uops) / len(all_uops),
            'operation_diversity': len(set(u.op for u in all_uops)),
        }
        
        # Find critical path (longest dependency chain)
        stats['critical_path_length'] = self._find_critical_path_length(uop)
        
        # Calculate parallelism potential
        stats['max_parallel_width'] = self._calculate_max_parallel_width(all_uops)
        
        # Memory usage estimation
        stats['estimated_memory_ops'] = sum(1 for u in all_uops if u.op in [Ops.BUFFER, Ops.VIEW])
        
        return stats
    
    def _calculate_max_depth(self, uop: UOp) -> int:
        """Calculate maximum depth of the graph"""
        if not uop.src:
            return 1
        return 1 + max(self._calculate_max_depth(src) for src in uop.src)
    
    def _find_critical_path_length(self, uop: UOp) -> int:
        """Find the length of the critical path (longest dependency chain)"""
        memo = {}
        
        def dfs(node):
            if node in memo:
                return memo[node]
            
            if not node.src:
                memo[node] = 1
                return 1
            
            max_path = max(dfs(src) for src in node.src)
            memo[node] = max_path + 1
            return memo[node]
        
        return dfs(uop)
    
    def _calculate_max_parallel_width(self, all_uops: List[UOp]) -> int:
        """Calculate maximum number of operations that could run in parallel"""
        # Build dependency levels
        levels = defaultdict(list)
        
        def get_level(uop):
            if not uop.src:
                return 0
            return 1 + max(get_level(src) for src in uop.src)
        
        for uop in all_uops:
            level = get_level(uop)
            levels[level].append(uop)
        
        return max(len(ops) for ops in levels.values()) if levels else 0
    
    def compare_graphs(self, uop1: UOp, uop2: UOp, name1: str = "Graph 1", name2: str = "Graph 2"):
        """
        Compare two UOp graphs and highlight differences
        """
        print(f"\\n=== Graph Comparison: {name1} vs {name2} ===")
        
        stats1 = self.analyze_graph_complexity(uop1)
        stats2 = self.analyze_graph_complexity(uop2)
        
        print(f"\\n{name1:<20} | {name2:<20} | Difference")
        print("-" * 60)
        
        for key in stats1:
            val1 = stats1[key]
            val2 = stats2[key]
            diff = val2 - val1 if isinstance(val1, (int, float)) else "N/A"
            print(f"{key:<20}: {val1:<8} | {val2:<8} | {diff}")
        
        # Operation type comparison
        ops1 = Counter(u.op for u in uop1.toposort())
        ops2 = Counter(u.op for u in uop2.toposort())
        
        all_ops = set(ops1.keys()) | set(ops2.keys())
        print(f"\\nOperation counts:")
        for op in sorted(all_ops, key=lambda x: x.name):
            count1 = ops1.get(op, 0)
            count2 = ops2.get(op, 0)
            diff = count2 - count1
            print(f"  {op.name:<15}: {count1:<3} | {count2:<3} | {diff:+d}")

# Example usage and testing
def demonstrate_visualization():
    """
    Demonstrate UOp graph visualization capabilities
    """
    print("=== UOp Graph Visualization Demo ===\\n")
    
    visualizer = UOpGraphVisualizer()
    
    # Create test graphs of increasing complexity
    test_cases = [
        ("Simple Addition", lambda: Tensor([1., 2., 3.]) + Tensor([4., 5., 6.])),
        ("Matrix Multiplication", lambda: Tensor([[1., 2.], [3., 4.]]) @ Tensor([[5., 6.], [7., 8.]])),
        ("Complex Expression", lambda: (Tensor([1., 2., 3.]) * 2 + 1).relu().sum().exp()),
        ("Neural Network Layer", lambda: Tensor.randn(10, 784) @ Tensor.randn(784, 128) + Tensor.randn(128)),
    ]
    
    for name, operation in test_cases:
        print(f"\\n{name}:")
        print("-" * 40)
        
        # Create the computation
        result = operation()
        
        # Visualize
        viz_path = visualizer.visualize_uop_graph(result.uop, f"uop_{name.lower().replace(' ', '_')}")
        
        # Analyze complexity
        complexity = visualizer.analyze_graph_complexity(result.uop)
        print(f"Graph complexity analysis:")
        for key, value in complexity.items():
            print(f"  {key}: {value}")
    
    # Demonstrate graph comparison
    print("\\n=== Graph Optimization Comparison ===")
    
    # Create two equivalent but differently structured computations
    simple = Tensor([1., 2., 3.]) + 0.0  # Should be optimizable
    optimized = Tensor([1., 2., 3.])      # Already optimal
    
    visualizer.compare_graphs(simple.uop, optimized.uop, "With +0", "Direct")

if __name__ == "__main__":
    from collections import Counter
    demonstrate_visualization()
```

---

## Part 2: Pattern Matching and Graph Rewriting (90 minutes)

### Understanding Pattern Matching System

Tinygrad uses a sophisticated pattern matching system to optimize UOp graphs:

```python
#!/usr/bin/env python3
"""
pattern_matching.py - Deep dive into tinygrad's pattern matching and rewriting system
"""

from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite
from tinygrad.uop.upat import UPat
from tinygrad import Tensor
from tinygrad.dtype import dtypes
import time
from typing import List, Dict, Callable, Optional

class PatternMatchingExplorer:
    """
    Explore and understand tinygrad's pattern matching system
    """
    
    def __init__(self):
        self.custom_patterns = []
        self.rewrite_stats = {}
    
    def understand_upat_system(self):
        """
        Understanding UPat (UOp Pattern) system for matching computation graphs
        """
        print("=== UPat Pattern Matching System ===\\n")
        
        # UPat is used to match specific patterns in UOp graphs
        print("1. Basic UPat Patterns:")
        
        # Simple pattern matching
        const_pattern = UPat(Ops.CONST)
        print(f"Constant pattern: {const_pattern}")
        
        # Pattern with dtype constraint
        float_const_pattern = UPat(Ops.CONST, dtype=dtypes.float32)
        print(f"Float constant pattern: {float_const_pattern}")
        
        # Pattern with value constraint
        zero_pattern = UPat(Ops.CONST, arg=0.0)
        print(f"Zero constant pattern: {zero_pattern}")
        
        print("\\n2. Testing Pattern Matches:")
        
        # Create test UOps
        test_uops = [
            UOp(Ops.CONST, dtypes.float32, arg=0.0),    # Should match zero_pattern
            UOp(Ops.CONST, dtypes.float32, arg=1.0),    # Should match float_const_pattern but not zero_pattern
            UOp(Ops.CONST, dtypes.int32, arg=0),        # Should match const_pattern but not float_const_pattern
            UOp(Ops.ADD, dtypes.float32, ()),           # Should not match any constant patterns
        ]
        
        patterns = [
            ("const_pattern", const_pattern),
            ("float_const_pattern", float_const_pattern), 
            ("zero_pattern", zero_pattern)
        ]
        
        for i, uop in enumerate(test_uops):
            print(f"\\nUOp {i}: {uop.op.name}, dtype: {uop.dtype}, arg: {uop.arg}")
            for pattern_name, pattern in patterns:
                try:
                    match = pattern.match(uop)
                    print(f"  {pattern_name}: {'✓' if match else '✗'}")
                except Exception as e:
                    print(f"  {pattern_name}: Error - {e}")
    
    def explore_complex_patterns(self):
        """
        Explore complex patterns with source matching and named captures
        """
        print("\\n=== Complex Pattern Matching ===\\n")
        
        print("1. Patterns with Source Constraints:")
        
        # Pattern for addition with constant zero (x + 0)
        add_zero_pattern = UPat(Ops.ADD, src=(
            UPat(name="x"),  # Any UOp, captured as "x"
            UPat(Ops.CONST, arg=0.0, name="zero")  # Constant zero, captured as "zero"
        ))
        
        print(f"Add-zero pattern: {add_zero_pattern}")
        
        # Create test graph: x + 0
        x = UOp(Ops.CONST, dtypes.float32, arg=5.0)
        zero = UOp(Ops.CONST, dtypes.float32, arg=0.0)
        add_zero = UOp(Ops.ADD, dtypes.float32, (x, zero))
        
        print(f"\\nTest graph: {add_zero}")
        match_result = add_zero_pattern.match(add_zero)
        print(f"Pattern match result: {match_result}")
        
        if match_result:
            print("Captured variables:")
            for name, uop in match_result.items():
                print(f"  {name}: {uop}")
        
        print("\\n2. Patterns for Optimization Rules:")
        
        # Common optimization patterns
        optimization_patterns = [
            # x + 0 = x
            ("add_zero", UPat(Ops.ADD, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0)))),
            # x * 1 = x  
            ("mul_one", UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=1.0)))),
            # x * 0 = 0
            ("mul_zero", UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0)))),
            # max(x, x) = x
            ("max_same", UPat(Ops.MAX, src=(UPat(name="x"), UPat(name="x")))),
        ]
        
        # Test these patterns
        test_graphs = [
            ("x + 0", UOp(Ops.ADD, dtypes.float32, (
                UOp(Ops.CONST, dtypes.float32, arg=42.0),
                UOp(Ops.CONST, dtypes.float32, arg=0.0)
            ))),
            ("x * 1", UOp(Ops.MUL, dtypes.float32, (
                UOp(Ops.CONST, dtypes.float32, arg=42.0),
                UOp(Ops.CONST, dtypes.float32, arg=1.0)
            ))),
            ("x * 0", UOp(Ops.MUL, dtypes.float32, (
                UOp(Ops.CONST, dtypes.float32, arg=42.0),
                UOp(Ops.CONST, dtypes.float32, arg=0.0)
            ))),
        ]
        
        for graph_name, graph_uop in test_graphs:
            print(f"\\nTesting graph: {graph_name}")
            for pattern_name, pattern in optimization_patterns:
                match = pattern.match(graph_uop)
                print(f"  {pattern_name}: {'✓' if match else '✗'}")
                if match:
                    print(f"    Captured: {match}")
    
    def implement_custom_rewrite_rules(self):
        """
        Implement and test custom graph rewrite rules
        """
        print("\\n=== Custom Rewrite Rules ===\\n")
        
        def add_zero_elimination(x: UOp, zero: UOp) -> UOp:
            """Eliminate addition with zero: x + 0 → x"""
            print(f"Applying add-zero elimination: {x} + {zero} → {x}")
            return x
        
        def mul_one_elimination(x: UOp, one: UOp) -> UOp:
            """Eliminate multiplication by one: x * 1 → x"""
            print(f"Applying mul-one elimination: {x} * {one} → {x}")
            return x
        
        def mul_zero_to_zero(x: UOp, zero: UOp) -> UOp:
            """Replace multiplication by zero with zero: x * 0 → 0"""
            print(f"Applying mul-zero to zero: {x} * {zero} → {zero}")
            return zero
        
        # Define custom rewrite rules
        custom_rules = [
            # x + 0 → x
            (UPat(Ops.ADD, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0, name="zero"))), add_zero_elimination),
            # x * 1 → x
            (UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=1.0, name="one"))), mul_one_elimination),
            # x * 0 → 0
            (UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0, name="zero"))), mul_zero_to_zero),
        ]
        
        # Create a PatternMatcher with custom rules
        custom_matcher = PatternMatcher(custom_rules)
        
        print("1. Testing Custom Rewrite Rules:")
        
        # Create test expressions that should be optimized
        test_expressions = [
            ("x + 0", UOp(Ops.ADD, dtypes.float32, (
                UOp(Ops.CONST, dtypes.float32, arg=42.0),
                UOp(Ops.CONST, dtypes.float32, arg=0.0)
            ))),
            ("x * 1", UOp(Ops.MUL, dtypes.float32, (
                UOp(Ops.CONST, dtypes.float32, arg=42.0),
                UOp(Ops.CONST, dtypes.float32, arg=1.0)
            ))),
            ("(x + 0) * 1", UOp(Ops.MUL, dtypes.float32, (
                UOp(Ops.ADD, dtypes.float32, (
                    UOp(Ops.CONST, dtypes.float32, arg=42.0),
                    UOp(Ops.CONST, dtypes.float32, arg=0.0)
                )),
                UOp(Ops.CONST, dtypes.float32, arg=1.0)
            ))),
        ]
        
        for name, expr in test_expressions:
            print(f"\\nOriginal expression: {name}")
            print(f"  Before: {expr}")
            
            # Apply custom rewrites
            optimized = graph_rewrite(expr, custom_matcher)
            print(f"  After:  {optimized}")
            
            # Check if optimization occurred
            if optimized != expr:
                print("  ✓ Optimization applied!")
            else:
                print("  - No optimization applied")
    
    def analyze_existing_rewrite_system(self):
        """
        Analyze tinygrad's existing rewrite system and optimization passes
        """
        print("\\n=== Existing Rewrite System Analysis ===\\n")
        
        # Create a complex expression to see what optimizations are applied
        print("1. Creating Complex Expression for Analysis:")
        
        # Build: ((x + 0) * 1 + (y * 0)) * 2
        x = Tensor([1., 2., 3.])
        y = Tensor([4., 5., 6.])
        
        # Force some inefficient operations
        expr = ((x + 0.0) * 1.0 + (y * 0.0)) * 2.0
        
        print(f"Original expression UOp graph size: {len(list(expr.uop.toposort()))}")
        
        # Analyze the graph before realization
        print("\\nUOp graph before realization:")
        for i, uop in enumerate(expr.uop.toposort()):
            print(f"  {i}: {uop.op.name} | arg: {uop.arg}")
        
        # Force realization to trigger optimizations
        print("\\nForcing realization to trigger optimizations...")
        result = expr.numpy()
        print(f"Result: {result}")
        
        print("\\n2. Optimization Statistics:")
        # The actual optimization happens during the realization process
        # in the schedule and kernelize phases
        
        print("Note: Full optimization analysis requires diving into:")
        print("  - tinygrad.engine.schedule optimization passes")
        print("  - tinygrad.kernelize graph fusion rules") 
        print("  - Backend-specific optimization in codegen")

def demonstrate_advanced_pattern_matching():
    """
    Advanced demonstration of pattern matching capabilities
    """
    print("=== Advanced Pattern Matching Demo ===\\n")
    
    explorer = PatternMatchingExplorer()
    
    # Run all exploration functions
    explorer.understand_upat_system()
    explorer.explore_complex_patterns()
    explorer.implement_custom_rewrite_rules()
    explorer.analyze_existing_rewrite_system()
    
    print("\\n=== Pattern Matching Performance Test ===")
    
    # Test performance of pattern matching
    large_expr = Tensor.randn(100, 100)
    for _ in range(10):
        large_expr = large_expr * 1.0 + 0.0  # Add inefficient operations
    
    start_time = time.time()
    graph_size_before = len(list(large_expr.uop.toposort()))
    
    # Force realization
    result = large_expr.numpy()
    
    end_time = time.time()
    
    print(f"Complex expression with {graph_size_before} UOps")
    print(f"Realization time: {end_time - start_time:.4f} seconds")
    print(f"Result shape: {result.shape}")

if __name__ == "__main__":
    demonstrate_advanced_pattern_matching()
```

### Custom Graph Optimization Implementation

```python
#!/usr/bin/env python3
"""
custom_optimization.py - Implement custom graph optimization passes
"""

from typing import Dict, List, Callable, Tuple, Set
from collections import defaultdict
import time
from tinygrad.uop.ops import UOp, Ops, UPat, PatternMatcher, graph_rewrite
from tinygrad import Tensor
from tinygrad.dtype import dtypes

class CustomOptimizer:
    """
    Custom optimization pass implementation for UOp graphs
    """
    
    def __init__(self):
        self.optimization_stats = defaultdict(int)
        self.patterns = []
        self.rewrite_functions = {}
    
    def register_optimization(self, name: str, pattern: UPat, rewrite_fn: Callable):
        """Register a new optimization rule"""
        self.patterns.append((pattern, rewrite_fn))
        self.rewrite_functions[name] = rewrite_fn
        print(f"Registered optimization: {name}")
    
    def algebraic_simplifications(self):
        """
        Implement algebraic simplification rules
        """
        print("=== Implementing Algebraic Simplifications ===\\n")
        
        # Define algebraic rules
        def add_zero_rule(x: UOp, zero: UOp) -> UOp:
            """x + 0 → x"""
            self.optimization_stats['add_zero'] += 1
            return x
        
        def mul_one_rule(x: UOp, one: UOp) -> UOp:
            """x * 1 → x"""
            self.optimization_stats['mul_one'] += 1
            return x
        
        def mul_zero_rule(x: UOp, zero: UOp) -> UOp:
            """x * 0 → 0"""
            self.optimization_stats['mul_zero'] += 1
            return zero
        
        def add_commutative_rule(x: UOp, y: UOp) -> UOp:
            """Canonicalize addition: ensure constants come last"""
            if x.op == Ops.CONST and y.op != Ops.CONST:
                self.optimization_stats['add_commutative'] += 1
                return UOp(Ops.ADD, x.dtype, (y, x))
            return UOp(Ops.ADD, x.dtype, (x, y))
        
        def double_negative_rule(x: UOp) -> UOp:
            """--x → x"""
            self.optimization_stats['double_negative'] += 1
            return x.src[0] if x.src else x
        
        # Register the rules
        self.register_optimization(
            "add_zero",
            UPat(Ops.ADD, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0, name="zero"))),
            add_zero_rule
        )
        
        self.register_optimization(
            "mul_one", 
            UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=1.0, name="one"))),
            mul_one_rule
        )
        
        self.register_optimization(
            "mul_zero",
            UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=0.0, name="zero"))),
            mul_zero_rule
        )
        
        print(f"Registered {len(self.patterns)} algebraic simplification rules")
    
    def strength_reduction_optimizations(self):
        """
        Implement strength reduction optimizations (replace expensive ops with cheaper ones)
        """
        print("\\n=== Implementing Strength Reduction ===\\n")
        
        def mul_by_two_rule(x: UOp, two: UOp) -> UOp:
            """x * 2 → x + x (addition might be faster than multiplication)"""
            self.optimization_stats['mul_by_two'] += 1
            return UOp(Ops.ADD, x.dtype, (x, x))
        
        def div_by_two_rule(x: UOp, two: UOp) -> UOp:
            """x / 2 → x * 0.5 (multiplication might be faster than division)"""
            self.optimization_stats['div_by_two'] += 1
            half = UOp(Ops.CONST, x.dtype, arg=0.5)
            return UOp(Ops.MUL, x.dtype, (x, half))
        
        def power_of_two_rule(x: UOp, exp: UOp) -> UOp:
            """x^2 → x * x"""
            self.optimization_stats['power_of_two'] += 1
            return UOp(Ops.MUL, x.dtype, (x, x))
        
        # Register strength reduction rules
        self.register_optimization(
            "mul_by_two",
            UPat(Ops.MUL, src=(UPat(name="x"), UPat(Ops.CONST, arg=2.0, name="two"))),
            mul_by_two_rule
        )
        
        # Note: POW operation might not be available in basic UOp set
        # This is an example of what could be implemented
        
        print(f"Registered strength reduction optimizations")
    
    def redundancy_elimination(self, uop: UOp) -> UOp:
        """
        Eliminate redundant computations (common subexpression elimination)
        """
        print("\\n=== Redundancy Elimination ===\\n")
        
        # Build a map of equivalent computations
        all_uops = list(uop.toposort())
        signature_to_uop = {}
        redundant_count = 0
        
        def get_signature(u: UOp) -> tuple:
            """Create a signature for a UOp to identify equivalent computations"""
            if u.src:
                src_sigs = tuple(id(src) for src in u.src)
                return (u.op, u.dtype, src_sigs, u.arg)
            else:
                return (u.op, u.dtype, u.arg)
        
        # Find redundant computations
        for u in all_uops:
            sig = get_signature(u)
            if sig in signature_to_uop:
                redundant_count += 1
                print(f"Found redundant computation: {u.op.name}")
            else:
                signature_to_uop[sig] = u
        
        print(f"Found {redundant_count} redundant computations out of {len(all_uops)} total")
        self.optimization_stats['redundant_eliminated'] = redundant_count
        
        # Note: Actual elimination would require graph reconstruction
        return uop
    
    def apply_optimizations(self, uop: UOp) -> UOp:
        """
        Apply all registered optimizations to a UOp graph
        """
        print("\\n=== Applying All Optimizations ===\\n")
        
        # Create pattern matcher with all registered patterns
        matcher = PatternMatcher(self.patterns)
        
        # Apply optimizations iteratively until no more changes
        current_uop = uop
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            print(f"Optimization iteration {iteration + 1}")
            
            # Count UOps before optimization
            uops_before = len(list(current_uop.toposort()))
            
            # Apply pattern-based optimizations
            optimized_uop = graph_rewrite(current_uop, matcher)
            
            # Apply redundancy elimination
            optimized_uop = self.redundancy_elimination(optimized_uop)
            
            # Count UOps after optimization
            uops_after = len(list(optimized_uop.toposort()))
            
            print(f"  UOps: {uops_before} → {uops_after}")
            
            # Check if any changes were made
            if optimized_uop == current_uop:
                print("  No more optimizations possible")
                break
            
            current_uop = optimized_uop
            iteration += 1
        
        # Print optimization statistics
        print(f"\\nOptimization Summary:")
        print(f"  Iterations: {iteration}")
        for opt_name, count in self.optimization_stats.items():
            if count > 0:
                print(f"  {opt_name}: {count} applications")
        
        return current_uop
    
    def benchmark_optimization_impact(self, tensor_operation: Callable) -> Dict:
        """
        Benchmark the impact of optimizations on performance
        """
        print("\\n=== Optimization Performance Benchmark ===\\n")
        
        # Measure without optimizations (create fresh expression each time)
        unoptimized_times = []
        for _ in range(5):
            start_time = time.time()
            result_unopt = tensor_operation()
            if hasattr(result_unopt, 'numpy'):
                _ = result_unopt.numpy()  # Force realization
            unoptimized_times.append(time.time() - start_time)
        
        # Get the UOp graph for optimization
        test_expr = tensor_operation()
        original_graph_size = len(list(test_expr.uop.toposort()))
        
        # Apply optimizations
        optimized_uop = self.apply_optimizations(test_expr.uop)
        optimized_graph_size = len(list(optimized_uop.toposort()))
        
        # Note: Actually using the optimized graph would require 
        # reconstructing the tensor, which is complex
        # This is a demonstration of the analysis approach
        
        avg_unoptimized_time = sum(unoptimized_times) / len(unoptimized_times)
        
        results = {
            'original_graph_size': original_graph_size,
            'optimized_graph_size': optimized_graph_size,
            'graph_size_reduction': original_graph_size - optimized_graph_size,
            'avg_execution_time': avg_unoptimized_time,
            'optimization_stats': dict(self.optimization_stats)
        }
        
        print(f"Benchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return results

def demonstrate_custom_optimization():
    """
    Demonstrate the custom optimization system
    """
    print("=== Custom Optimization System Demo ===\\n")
    
    optimizer = CustomOptimizer()
    
    # Set up optimization rules
    optimizer.algebraic_simplifications()
    optimizer.strength_reduction_optimizations()
    
    # Create test expressions with optimization opportunities
    test_expressions = [
        # Simple algebraic optimizations
        lambda: Tensor([1., 2., 3.]) + 0.0,
        lambda: Tensor([1., 2., 3.]) * 1.0,
        lambda: Tensor([1., 2., 3.]) * 0.0,
        
        # More complex expressions
        lambda: (Tensor([1., 2., 3.]) + 0.0) * 1.0 + (Tensor([4., 5., 6.]) * 0.0),
        lambda: Tensor([1., 2., 3.]) * 2.0,  # Strength reduction candidate
        
        # Very complex expression with multiple optimization opportunities
        lambda: ((Tensor.randn(10, 10) + 0.0) * 1.0 + (Tensor.randn(10, 10) * 0.0)) * 2.0 + 0.0,
    ]
    
    for i, expr_fn in enumerate(test_expressions):
        print(f"\\nTest Expression {i + 1}:")
        print("-" * 40)
        
        # Create the expression
        expr = expr_fn()
        
        print(f"Original graph size: {len(list(expr.uop.toposort()))} UOps")
        
        # Apply optimizations
        optimizer.optimization_stats.clear()  # Reset stats for this expression
        optimized_uop = optimizer.apply_optimizations(expr.uop)
        
        print(f"Optimized graph size: {len(list(optimized_uop.toposort()))} UOps")
        
        # Benchmark if it's not too complex
        if i < 4:  # Only benchmark simpler expressions
            results = optimizer.benchmark_optimization_impact(expr_fn)

if __name__ == "__main__":
    demonstrate_custom_optimization()
```

---

## Part 3: Advanced UOp Debugging and Profiling (60 minutes)

### UOp-Level Performance Analysis

```python
#!/usr/bin/env python3
"""
uop_profiling.py - Advanced UOp-level performance analysis and debugging
"""

import time
import traceback
import functools
import statistics
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, Ops
from tinygrad import Tensor
import os

@dataclass
class UOpProfileData:
    """Data structure for UOp profiling information"""
    operation: str
    count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_estimate: int
    dependency_depth: int

class UOpProfiler:
    """
    Advanced profiler for UOp-level performance analysis
    """
    
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.operation_counts = Counter()
        self.memory_usage = defaultdict(int)
        self.dependency_graph = {}
        self.profiling_enabled = False
    
    def enable_profiling(self):
        """Enable UOp-level profiling"""
        self.profiling_enabled = True
        print("UOp profiling enabled")
    
    def disable_profiling(self):
        """Disable UOp-level profiling"""
        self.profiling_enabled = False
        print("UOp profiling disabled")
    
    def profile_uop_graph(self, uop: UOp, operation_name: str = "unknown") -> UOpProfileData:
        """
        Profile a complete UOp graph for performance characteristics
        """
        if not self.profiling_enabled:
            self.enable_profiling()
        
        print(f"\\n=== Profiling UOp Graph: {operation_name} ===")
        
        # Get all UOps in topological order
        all_uops = list(uop.toposort())
        
        # Basic statistics
        op_counts = Counter(u.op for u in all_uops)
        total_uops = len(all_uops)
        max_depth = self._calculate_depth(uop)
        
        # Estimate memory usage (rough approximation)
        memory_estimate = self._estimate_memory_usage(all_uops)
        
        # Analyze operation patterns
        patterns = self._analyze_patterns(all_uops)
        
        # Time the graph construction (simulation)
        start_time = time.time()
        # Simulate traversing the graph
        for u in all_uops:
            self.operation_counts[u.op] += 1
            # Simulate some work
            pass
        construction_time = time.time() - start_time
        
        profile_data = UOpProfileData(
            operation=operation_name,
            count=total_uops,
            total_time=construction_time,
            avg_time=construction_time / total_uops if total_uops > 0 else 0,
            min_time=0,  # Would need actual execution timing
            max_time=construction_time,
            memory_estimate=memory_estimate,
            dependency_depth=max_depth
        )
        
        print(f"Graph Analysis:")
        print(f"  Total UOps: {total_uops}")
        print(f"  Max depth: {max_depth}")
        print(f"  Memory estimate: {memory_estimate} bytes")
        print(f"  Construction time: {construction_time:.6f}s")
        
        print(f"\\nOperation breakdown:")
        for op, count in op_counts.most_common():
            print(f"  {op.name:<15}: {count:4d} ({count/total_uops*100:.1f}%)")
        
        print(f"\\nPattern analysis:")
        for pattern, count in patterns.items():
            if count > 1:
                print(f"  {pattern}: {count} occurrences")
        
        return profile_data
    
    def _calculate_depth(self, uop: UOp) -> int:
        """Calculate the depth of the UOp graph"""
        if not uop.src:
            return 0
        return 1 + max(self._calculate_depth(src) for src in uop.src)
    
    def _estimate_memory_usage(self, all_uops: List[UOp]) -> int:
        """Rough estimation of memory usage"""
        total_bytes = 0
        for uop in all_uops:
            if hasattr(uop, 'shape') and uop.shape:
                # Estimate based on shape and dtype
                elements = 1
                for dim in uop.shape:
                    if isinstance(dim, int):
                        elements *= dim
                dtype_size = 4  # Assume 32-bit floats
                total_bytes += elements * dtype_size
        return total_bytes
    
    def _analyze_patterns(self, all_uops: List[UOp]) -> Dict[str, int]:
        """Analyze common patterns in the UOp graph"""
        patterns = defaultdict(int)
        
        for uop in all_uops:
            if uop.src:
                # Pattern: operation with its source types
                src_ops = tuple(src.op.name for src in uop.src)
                pattern = f"{uop.op.name}({', '.join(src_ops)})"
                patterns[pattern] += 1
                
                # Look for specific optimization opportunities
                if uop.op == Ops.ADD and len(uop.src) == 2:
                    if any(src.op == Ops.CONST and src.arg == 0 for src in uop.src):
                        patterns["add_with_zero"] += 1
                
                if uop.op == Ops.MUL and len(uop.src) == 2:
                    if any(src.op == Ops.CONST and src.arg == 1 for src in uop.src):
                        patterns["mul_with_one"] += 1
                    if any(src.op == Ops.CONST and src.arg == 0 for src in uop.src):
                        patterns["mul_with_zero"] += 1
        
        return dict(patterns)
    
    def compare_uop_graphs(self, graphs: List[Tuple[str, UOp]]) -> Dict:
        """
        Compare multiple UOp graphs for performance analysis
        """
        print("\\n=== UOp Graph Comparison ===\\n")
        
        profiles = []
        for name, uop in graphs:
            profile = self.profile_uop_graph(uop, name)
            profiles.append((name, profile))
        
        # Create comparison table
        print(f"\\nComparison Results:")
        print(f"{'Graph':<20} | {'UOps':<6} | {'Depth':<6} | {'Memory (KB)':<12} | {'Time (ms)':<10}")
        print("-" * 70)
        
        for name, profile in profiles:
            memory_kb = profile.memory_estimate / 1024
            time_ms = profile.total_time * 1000
            print(f"{name:<20} | {profile.count:<6} | {profile.dependency_depth:<6} | {memory_kb:<12.1f} | {time_ms:<10.3f}")
        
        # Find the best and worst performers
        best_uops = min(profiles, key=lambda x: x[1].count)
        worst_uops = max(profiles, key=lambda x: x[1].count)
        
        print(f"\\nBest (fewest UOps): {best_uops[0]} with {best_uops[1].count} UOps")
        print(f"Worst (most UOps): {worst_uops[0]} with {worst_uops[1].count} UOps")
        
        return {
            'profiles': profiles,
            'best': best_uops,
            'worst': worst_uops
        }
    
    def profile_tensor_operation_lifecycle(self, operation_fn: Callable, iterations: int = 10):
        """
        Profile the complete lifecycle of a tensor operation
        """
        print(f"\\n=== Tensor Operation Lifecycle Profiling ===\\n")
        
        lifecycle_times = {
            'graph_construction': [],
            'realization': [],
            'total': []
        }
        
        graph_sizes = []
        
        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations}")
            
            # Time graph construction
            start_total = time.time()
            start_construction = time.time()
            
            # Create the tensor operation (builds UOp graph)
            tensor_result = operation_fn()
            
            construction_time = time.time() - start_construction
            
            # Analyze the graph
            if hasattr(tensor_result, 'uop'):
                graph_size = len(list(tensor_result.uop.toposort()))
                graph_sizes.append(graph_size)
            
            # Time realization
            start_realization = time.time()
            
            # Force realization
            if hasattr(tensor_result, 'numpy'):
                result = tensor_result.numpy()
            else:
                result = tensor_result
            
            realization_time = time.time() - start_realization
            total_time = time.time() - start_total
            
            lifecycle_times['graph_construction'].append(construction_time)
            lifecycle_times['realization'].append(realization_time)
            lifecycle_times['total'].append(total_time)
        
        # Calculate statistics
        stats = {}
        for phase, times in lifecycle_times.items():
            stats[phase] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0,
                'min': min(times),
                'max': max(times)
            }
        
        avg_graph_size = statistics.mean(graph_sizes) if graph_sizes else 0
        
        print(f"\\nLifecycle Profiling Results ({iterations} iterations):")
        print(f"Average graph size: {avg_graph_size:.1f} UOps")
        print()
        
        for phase, phase_stats in stats.items():
            print(f"{phase.replace('_', ' ').title()}:")
            print(f"  Mean: {phase_stats['mean']*1000:.3f}ms")
            print(f"  Median: {phase_stats['median']*1000:.3f}ms")
            print(f"  Std dev: {phase_stats['stdev']*1000:.3f}ms")
            print(f"  Range: {phase_stats['min']*1000:.3f}ms - {phase_stats['max']*1000:.3f}ms")
            print()
        
        # Calculate ratios
        if stats['total']['mean'] > 0:
            construction_ratio = stats['graph_construction']['mean'] / stats['total']['mean']
            realization_ratio = stats['realization']['mean'] / stats['total']['mean']
            
            print(f"Time distribution:")
            print(f"  Graph construction: {construction_ratio*100:.1f}%")
            print(f"  Realization: {realization_ratio*100:.1f}%")
        
        return {
            'stats': stats,
            'average_graph_size': avg_graph_size,
            'raw_times': lifecycle_times
        }

def demonstrate_uop_profiling():
    """
    Demonstrate advanced UOp profiling capabilities
    """
    print("=== UOp Profiling Demonstration ===\\n")
    
    profiler = UOpProfiler()
    
    # Test different complexity levels
    test_operations = [
        ("Simple Add", lambda: Tensor([1., 2., 3.]) + Tensor([4., 5., 6.])),
        ("Matrix Mul", lambda: Tensor([[1., 2.], [3., 4.]]) @ Tensor([[5., 6.], [7., 8.]])),
        ("Complex Chain", lambda: (Tensor.randn(10, 10).relu().sum(axis=1).exp().mean())),
        ("Inefficient Expr", lambda: (Tensor([1., 2., 3.]) + 0.0) * 1.0 + (Tensor([4., 5., 6.]) * 0.0)),
    ]
    
    # Profile individual operations
    print("1. Individual Operation Profiling:")
    profiles = []
    for name, operation in test_operations:
        print(f"\\n{name}:")
        print("-" * 30)
        result = operation()
        if hasattr(result, 'uop'):
            profile = profiler.profile_uop_graph(result.uop, name)
            profiles.append((name, result.uop))
    
    # Compare operations
    print("\\n" + "="*50)
    print("2. Operation Comparison:")
    comparison = profiler.compare_uop_graphs(profiles)
    
    # Lifecycle profiling
    print("\\n" + "="*50)
    print("3. Lifecycle Profiling:")
    
    def complex_operation():
        x = Tensor.randn(50, 50)
        y = x @ x.T
        z = y.relu().sum(axis=1)
        return z.exp().mean()
    
    lifecycle_results = profiler.profile_tensor_operation_lifecycle(complex_operation, iterations=5)
    
    # Memory usage analysis
    print("\\n" + "="*50)
    print("4. Memory Usage Analysis:")
    
    memory_test_ops = [
        ("Small tensor", lambda: Tensor.randn(10, 10).sum()),
        ("Medium tensor", lambda: Tensor.randn(100, 100).sum()),
        ("Large tensor", lambda: Tensor.randn(500, 500).sum()),
    ]
    
    print(f"\\n{'Operation':<15} | {'Estimated Memory':<20} | {'UOp Count':<12}")
    print("-" * 50)
    
    for name, operation in memory_test_ops:
        result = operation()
        if hasattr(result, 'uop'):
            profile = profiler.profile_uop_graph(result.uop, name)
            memory_mb = profile.memory_estimate / (1024 * 1024)
            print(f"{name:<15} | {memory_mb:<20.2f}MB | {profile.count:<12}")

if __name__ == "__main__":
    # Set up debugging
    os.environ['DEBUG'] = '1'
    
    demonstrate_uop_profiling()
```

---

## Day 2 Wrap-up & Advanced Exercises

### What You've Mastered Today

1. ✅ **UOp Architecture**: Deep understanding of Universal Operations and graph structure
2. ✅ **Pattern Matching**: Master pattern matching with UPat and custom rewrite rules
3. ✅ **Graph Optimization**: Implemented custom optimization passes and analysis
4. ✅ **Performance Profiling**: Built advanced UOp-level performance analysis tools
5. ✅ **Debugging Mastery**: Advanced debugging techniques for UOp graphs

### Tomorrow's Preview: Engine & Scheduling System

Day 3 will focus on:
- **Schedule Creation**: Understanding how UOp graphs become execution schedules
- **Memory Planning**: How tinygrad manages memory allocation and optimization
- **Kernel Fusion**: How operations are grouped into efficient kernels
- **Backend Integration**: How schedules are executed on different devices

### Advanced Homework Assignments

1. **Custom Optimization Pass**: Implement a complete optimization pass for a specific domain (e.g., neural networks)
2. **Performance Analysis**: Profile real neural network operations and identify optimization opportunities
3. **Pattern Discovery**: Analyze existing tinygrad models to find new optimization patterns
4. **Visualization Tool**: Extend the UOp visualizer to show optimization transformations

### Self-Assessment Checklist

- [ ] Can I create and analyze UOp graphs programmatically?
- [ ] Can I implement custom pattern matching rules?
- [ ] Can I write optimization passes that improve performance?
- [ ] Can I profile UOp-level performance bottlenecks?
- [ ] Can I visualize and understand complex computation graphs?

### Practical Exercise: Build Your Own Optimizer

```python
# Challenge: Build an optimizer for your electrical testing domain
class ElectricalValidationOptimizer(CustomOptimizer):
    def __init__(self):
        super().__init__()
        self.register_electrical_patterns()
    
    def register_electrical_patterns(self):
        """Register patterns specific to electrical validation"""
        # Example: Optimize signal processing chains
        # TODO: Implement patterns for:
        # - FFT optimizations (when available)
        # - Filter chain simplifications 
        # - Statistical computation optimizations
        pass
    
    def optimize_for_real_time(self, uop: UOp) -> UOp:
        """Optimize specifically for real-time electrical monitoring"""
        # TODO: Implement optimizations for:
        # - Minimal latency
        # - Streaming computation
        # - Memory efficiency
        pass
```

**Ready for Day 3? The execution engine awaits! 🚀**