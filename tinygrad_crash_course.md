# Tinygrad 7-Day Crash Course for Contributors
*Automation, Validation & Documentation Focus*

## Course Overview

This crash course is designed for newcomers who want to become effective tinygrad contributors, with emphasis on **automation**, **validation**, and **documentation**. Perfect for those who need to teach others afterward.

### Learning Pathways (Choose Your Style)

**🚀 Pathway A: Project-Based Learning** *(Learning by Implementing)*
- Start with real projects and work backwards to understand concepts
- Build working tools while learning the architecture
- Perfect for hands-on learners who prefer immediate results

**🔧 Pathway B: Bottom-Up Foundation** *(Systematic Learning)*
- Master fundamentals first, then build complexity
- Understand each layer before moving to the next
- Ideal for those teaching others or needing deep understanding

---

## Pre-Course Setup (30 minutes)

### Environment Setup
```bash
# Clone and install tinygrad
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e '.[testing,linting]'

# Verify installation
python3 -c "from tinygrad import Tensor; print('✅ Tinygrad installed successfully')"

# Check available devices
python3 -c "from tinygrad import Device; print(f'Default device: {Device.DEFAULT}')"

# Install additional tools for automation/validation
pip install pytest matplotlib pandas jupyter numpy
```

### Repository Structure Overview
```
tinygrad/
├── tinygrad/           # Core framework
│   ├── tensor.py       # Main user interface
│   ├── device.py       # Device management
│   ├── engine/         # Execution engine
│   ├── uop/           # Low-level operations  
│   ├── shape/         # Shape management
│   ├── runtime/       # Backend implementations
│   └── nn/            # Neural network layers
├── test/              # Test suite (your focus!)
├── examples/          # Working examples
└── docs/             # Documentation
```

---

## Day 1: Architecture Overview & First Contributions

### Morning: Choose Your Pathway

#### 🚀 Pathway A: Project-Based Start
**Goal**: Build a simple validation tool while learning architecture

**Project**: Create a tensor operation validator
```python
# Project: tinygrad_validator.py
from tinygrad import Tensor
import numpy as np

def validate_tensor_operation(operation_name, tinygrad_fn, numpy_fn, test_data):
    """Validate tinygrad operations against numpy baseline"""
    # Your implementation here - we'll build this together
    pass

# Test cases to implement
test_basic_ops()      # +, -, *, /
test_reductions()     # sum, mean, max
test_shapes()         # reshape, transpose
```

**Learning Through Implementation:**
- Understand Tensor API by testing it
- Learn debugging with `DEBUG=1,2,3,4`
- Discover execution flow by watching operations

#### 🔧 Pathway B: Foundation First
**Goal**: Understand the 4-layer architecture systematically

**Study Path**:
1. **Tensor Layer** (`tinygrad/tensor.py:1-100`)
   - Understand lazy evaluation
   - Explore autograd system
   - Learn shape tracking

2. **UOp System** (`tinygrad/uop/ops.py`)
   - Study primitive operations (~70 ops)
   - Understand computation graphs
   - Learn pattern matching

3. **Engine** (`tinygrad/engine/schedule.py:1-50`)
   - Execution ordering
   - Dependency management
   - Memory planning

### Afternoon: First Real Contribution (Both Pathways)

**Validation Focus**: Improve test coverage
```bash
# Find gaps in test coverage
python3 -m pytest test/ --cov=tinygrad --cov-report=html

# Pick an area with low coverage and add tests
# Good starter areas:
# - test/test_tensor.py (add edge cases)
# - test/unit/test_helpers.py (add utility tests)
# - test/external/ (add new benchmark)
```

**Documentation Focus**: Create beginner guides
```markdown
# Example: Create tinygrad/docs/newcomer_guide.md
- Common gotchas for new contributors
- Debugging techniques with DEBUG flags
- How to read tensor execution traces
```

### Day 1 Deliverables
- ✅ Working development environment
- ✅ First test contribution or documentation fix
- ✅ Understanding of execution flow (Pathway A) OR architecture layers (Pathway B)

---

## Day 2: Deep Dive into Testing & Validation

### Morning: Master the Test Suite

**Understanding Current Tests**:
```bash
# Explore test categories
ls test/                  # Main test directory
ls test/unit/            # Unit tests (focus here!)
ls test/external/        # Integration tests
ls test/models/          # Model validation

# Run specific test categories
python3 -m pytest test/unit/ -v
python3 -m pytest test/test_tensor.py::TestTensor::test_add -v
```

**Test Architecture Study**:
- `test/helpers.py` - Common test utilities
- `test/unit/test_*.py` - Individual component tests
- `test/external/` - Long-running validation tests

### 🚀 Pathway A: Build Advanced Validator

**Project Extension**: Multi-device tensor validator
```python
# Advanced validator with device testing
class TinygradDeviceValidator:
    def __init__(self):
        self.devices = self.discover_available_devices()
    
    def validate_cross_device(self, operation, test_data):
        """Test operation consistency across devices"""
        # Implementation builds on Day 1 work
        pass
    
    def benchmark_operation(self, operation, sizes):
        """Performance validation"""
        pass
    
    def generate_test_report(self):
        """Auto-generate test documentation"""
        pass
```

### 🔧 Pathway B: Test Framework Deep Dive

**Study Core Testing Patterns**:
1. **Tensor Testing** (`test/test_tensor.py:1-200`)
   - How operations are validated
   - Numerical precision handling
   - Shape testing patterns

2. **UOp Testing** (`test/test_uops.py:1-100`) 
   - Low-level operation validation
   - Graph construction testing
   - Pattern matching validation

3. **Device Testing** (`test/unit/test_device.py`)
   - Backend-specific testing
   - Memory management validation
   - Performance benchmarking

### Afternoon: Contribute Real Test Improvements

**Automation Focus**: Create test automation tools
```python
# automation/test_generator.py
def generate_tensor_tests(operations_list, data_types, shapes):
    """Auto-generate comprehensive test suites"""
    pass

def create_regression_tests(bug_reports):
    """Convert GitHub issues to test cases"""
    pass
```

**Validation Focus**: Improve existing tests
- Add edge cases to `test/test_ops.py`
- Enhance error message validation
- Create device-specific test suites

### Day 2 Deliverables
- ✅ Advanced validation tool (Pathway A) OR deep test framework knowledge (Pathway B)
- ✅ Meaningful test contribution to tinygrad
- ✅ Test automation scripts

---

## Day 3: UOp System & Low-Level Operations

### Morning: Understanding UOps (Universal Operations)

**Key Files to Study**:
- `tinygrad/uop/ops.py` - All primitive operations
- `tinygrad/uop/spec.py` - Operation specifications  
- `test/test_uops.py` - UOp testing patterns

### 🚀 Pathway A: Build UOp Debugging Tool

**Project**: UOp graph visualization and validation
```python
# uop_debugger.py
class UOpDebugger:
    def trace_execution(self, tensor_operation):
        """Trace tensor operation to UOp graph"""
        pass
    
    def validate_uop_graph(self, graph):
        """Validate UOp graph correctness"""
        pass
    
    def visualize_execution(self, graph):
        """Create execution flow diagrams"""
        pass
    
    def benchmark_uops(self, operation_list):
        """Performance analysis of UOp sequences"""
        pass
```

### 🔧 Pathway B: UOp System Mastery

**Study Path**:
1. **Operation Categories** (`tinygrad/uop/ops.py:1-100`)
   - Unary ops (SQRT, EXP, LOG, etc.)
   - Binary ops (ADD, MUL, MAX, etc.)
   - Movement ops (RESHAPE, PERMUTE, etc.)
   - Reduce ops (SUM, MAX, etc.)

2. **UOp Execution** (`tinygrad/uop/__init__.py`)
   - Graph construction
   - Pattern matching and rewriting
   - Optimization passes

3. **Testing Patterns** (`test/test_uops.py:1-150`)
   - How UOps are validated
   - Graph correctness testing
   - Performance validation

### Afternoon: UOp Contributions

**Validation Improvements**:
```python
# Contribute to test/test_uops.py
def test_uop_memory_usage():
    """Validate UOp memory efficiency"""
    pass

def test_uop_fusion_correctness():
    """Validate operation fusion produces correct results"""
    pass

def test_uop_device_compatibility():
    """Ensure UOps work across all devices"""
    pass
```

**Documentation Contributions**:
- Document UOp debugging techniques
- Create UOp operation reference
- Write troubleshooting guides

### Day 3 Deliverables
- ✅ UOp debugging tool (Pathway A) OR deep UOp system understanding (Pathway B)
- ✅ UOp-related test contributions
- ✅ UOp documentation improvements

---

## Day 4: Device Backends & Runtime Systems

### Morning: Backend Architecture

**Key Areas**:
- `tinygrad/runtime/ops_*.py` - Device-specific implementations
- `tinygrad/device.py` - Device abstraction layer
- `test/unit/test_device.py` - Device testing

### 🚀 Pathway A: Multi-Device Automation Tool

**Project**: Automated device testing framework
```python
# device_automation.py
class DeviceTestAutomation:
    def discover_devices(self):
        """Auto-detect available devices"""
        pass
    
    def run_compatibility_suite(self):
        """Test operation compatibility across devices"""
        pass
    
    def benchmark_devices(self, operations):
        """Performance comparison across devices"""
        pass
    
    def generate_device_report(self):
        """Create device capability documentation"""
        pass
```

### 🔧 Pathway B: Backend Deep Dive

**Study Focus**:
1. **Device Abstraction** (`tinygrad/device.py:1-200`)
   - Buffer management
   - Device selection
   - Memory allocation

2. **CPU Backend** (`tinygrad/runtime/ops_cpu.py`)
   - Simplest backend to understand
   - NumPy integration patterns
   - Error handling

3. **GPU Backends** (`tinygrad/runtime/ops_cuda.py`, `ops_metal.py`)
   - GPU-specific optimizations
   - Memory management
   - Kernel execution

### Afternoon: Backend Contributions

**For Edge Device Future**: Study lightweight backends
- `tinygrad/runtime/ops_python.py` - Pure Python fallback
- `tinygrad/runtime/ops_disk.py` - Persistent storage
- `tinygrad/runtime/ops_remote.py` - Distributed computing

**Validation Contributions**:
```python
# Improve device testing
def test_device_memory_limits():
    """Test behavior at memory boundaries"""
    pass

def test_device_error_handling():
    """Validate proper error propagation"""
    pass

def test_cross_device_operations():
    """Test operations spanning multiple devices"""
    pass
```

### Day 4 Deliverables
- ✅ Device automation tool (Pathway A) OR backend expertise (Pathway B)
- ✅ Device-related test improvements
- ✅ Edge device preparation knowledge

---

## Day 5: Execution Engine & Optimization

### Morning: Scheduling & Memory Management

**Core Files**:
- `tinygrad/engine/schedule.py` - Operation scheduling
- `tinygrad/engine/memory.py` - Memory planning
- `tinygrad/engine/realize.py` - Graph execution

### 🚀 Pathway A: Performance Analysis Tool

**Project**: Execution profiler and optimizer
```python
# performance_analyzer.py
class TinygradProfiler:
    def profile_execution(self, model_or_operations):
        """Profile memory usage and execution time"""
        pass
    
    def find_bottlenecks(self, profile_data):
        """Identify performance bottlenecks"""
        pass
    
    def suggest_optimizations(self, analysis):
        """Recommend optimization strategies"""
        pass
    
    def validate_optimizations(self, before, after):
        """Validate optimization improvements"""
        pass
```

### 🔧 Pathway B: Engine Mastery

**Study Focus**:
1. **Scheduling** (`tinygrad/engine/schedule.py:1-200`)
   - Dependency resolution
   - Execution ordering
   - Memory planning

2. **Realization** (`tinygrad/engine/realize.py:1-150`)
   - Lazy evaluation triggers
   - Graph compilation
   - Device execution

3. **Memory Management** (`tinygrad/engine/memory.py`)
   - Buffer allocation
   - Memory optimization
   - Garbage collection

### Afternoon: Engine Contributions

**Performance Testing**:
```python
# Add to test/external/
def test_memory_efficiency():
    """Validate memory usage patterns"""
    pass

def test_scheduling_performance():
    """Benchmark scheduling algorithms"""
    pass

def test_large_model_execution():
    """Test execution of large models"""
    pass
```

**Validation Tools**:
- Memory leak detection
- Performance regression testing
- Optimization validation

### Day 5 Deliverables
- ✅ Performance analysis tool (Pathway A) OR engine expertise (Pathway B)
- ✅ Performance test contributions
- ✅ Optimization validation methods

---

## Day 6: Neural Networks & High-Level APIs

### Morning: NN Layer Implementation

**Key Files**:
- `tinygrad/nn/__init__.py` - Neural network layers
- `tinygrad/nn/optim.py` - Optimizers
- `test/test_nn.py` - NN testing patterns

### 🚀 Pathway A: NN Validation Framework

**Project**: Comprehensive neural network testing
```python
# nn_validator.py
class NeuralNetworkValidator:
    def validate_layer_correctness(self, layer_type, configurations):
        """Test NN layer implementations"""
        pass
    
    def test_training_stability(self, model, datasets):
        """Validate training convergence"""
        pass
    
    def benchmark_inference(self, models):
        """Performance testing for inference"""
        pass
    
    def validate_gradient_flow(self, model):
        """Test backpropagation correctness"""
        pass
```

### 🔧 Pathway B: NN Architecture Deep Dive

**Study Areas**:
1. **Core Layers** (`tinygrad/nn/__init__.py:1-300`)
   - Linear, Conv2d, BatchNorm
   - Activation functions
   - Dropout, Embedding

2. **Optimizers** (`tinygrad/nn/optim.py`)
   - SGD, Adam, AdamW
   - Learning rate scheduling
   - Parameter updates

3. **Training Patterns** (`examples/beautiful_mnist.py`)
   - Training loops
   - Loss computation
   - Gradient updates

### Afternoon: NN Contributions

**Enhanced Testing**:
```python
# Improve test/test_nn.py
def test_layer_parameter_counts():
    """Validate parameter counting"""
    pass

def test_layer_memory_usage():
    """Test memory efficiency of layers"""
    pass

def test_optimizer_convergence():
    """Validate optimizer behavior"""
    pass
```

**Documentation**:
- NN layer usage examples
- Training best practices
- Common debugging techniques

### Day 6 Deliverables
- ✅ NN validation framework (Pathway A) OR NN expertise (Pathway B)
- ✅ Neural network test improvements
- ✅ NN documentation contributions

---

## Day 7: Integration, Documentation & Future Contributions

### Morning: Comprehensive Integration

**Combine All Learning**: Create end-to-end validation system

### 🚀 Pathway A: Complete Automation Suite
**Final Project**: Integrated tinygrad validation platform
```python
# tinygrad_test_platform.py
class TinygradTestPlatform:
    def __init__(self):
        self.tensor_validator = TensorValidator()
        self.device_tester = DeviceTestAutomation() 
        self.performance_profiler = TinygradProfiler()
        self.nn_validator = NeuralNetworkValidator()
    
    def run_full_validation(self, target="all"):
        """Complete tinygrad validation suite"""
        pass
    
    def generate_comprehensive_report(self):
        """Auto-generate validation documentation"""
        pass
    
    def continuous_integration_setup(self):
        """Set up automated testing pipeline"""
        pass
```

### 🔧 Pathway B: Expert Knowledge Synthesis
**Final Integration**: Comprehensive understanding documentation
- Create architectural decision records
- Document debugging workflows
- Build troubleshooting guides
- Design contribution pathways for newcomers

### Afternoon: Documentation & Teaching Preparation

**Create Teaching Materials**:
1. **Quick Start Guides** for different contributor types
2. **Common Pitfalls** documentation
3. **Debugging Handbook** with real examples
4. **Contribution Pathways** for different skill levels

**Future Contribution Planning**:
- Identify areas needing improvement
- Plan ongoing contributions
- Set up development workflows
- Create team onboarding process

### Day 7 Deliverables
- ✅ Complete validation platform (Pathway A) OR comprehensive documentation (Pathway B)
- ✅ Teaching materials for future team members
- ✅ Ongoing contribution plan
- ✅ Team development workflow

---

## Success Metrics

### By End of Week, You Should Be Able To:

**For Everyone**:
- ✅ Understand tinygrad's 4-layer architecture
- ✅ Debug issues using DEBUG flags and tools
- ✅ Write effective tests for any component
- ✅ Navigate the codebase confidently
- ✅ Make meaningful contributions to tests/docs
- ✅ Teach others the fundamentals

**Pathway A Graduates**:
- ✅ Built working automation and validation tools
- ✅ Can profile and optimize tinygrad performance
- ✅ Created comprehensive testing frameworks
- ✅ Ready to build advanced validation systems

**Pathway B Graduates**:
- ✅ Deep understanding of each architectural layer
- ✅ Can explain design decisions and tradeoffs
- ✅ Expert-level debugging and troubleshooting skills
- ✅ Ready to contribute to core framework development

---

## Ongoing Contribution Areas (Post-Course)

### High-Impact Areas for Your Focus:

**Automation & Testing**:
- Continuous integration improvements
- Performance regression detection
- Cross-device compatibility testing
- Automated benchmark generation

**Validation & Quality**:
- Edge case testing
- Numerical precision validation
- Memory leak detection
- Error handling improvements

**Documentation**:
- Beginner tutorials and guides
- API documentation improvements
- Debugging and troubleshooting guides
- Architecture decision records

### Team Development:
- Onboarding automation for new contributors
- Knowledge sharing sessions
- Code review guidelines
- Contribution workflow optimization

---

## Daily Time Commitment

**Recommended Schedule** (adjust based on availability):
- **Morning (2-3 hours)**: Core learning/implementation
- **Afternoon (2-3 hours)**: Hands-on contributions
- **Evening (30-60 minutes)**: Documentation/reflection

**Total**: ~5-7 hours/day for intensive learning
**Alternative**: Spread over 2 weeks at 3-4 hours/day

Ready to start? Pick your pathway and let's begin Day 1! 🚀