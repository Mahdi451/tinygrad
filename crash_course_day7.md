# Tinygrad Crash Course - Day 7: Production & Contributions

## Learning Objectives
- Master the tinygrad contribution workflow
- Understand production deployment at scale
- Implement advanced debugging and profiling techniques
- Set up team development processes
- Create automated testing and CI/CD pipelines
- Contribute meaningfully to the open source project

## Prerequisites
- Completed Days 1-6
- Understanding of tinygrad architecture and optimization techniques
- Experience with JIT compilation and quantization

---

## Part 1: Contributing to Tinygrad

### Understanding the Contribution Workflow

```python
# examples/day7_contribution_workflow.py
from tinygrad import Tensor, Device
from tinygrad.uop.uops import UOp, Ops
from tinygrad.engine.schedule import create_schedule
import subprocess
import os

class ContributionWorkflowExplorer:
    """Guide for contributing to tinygrad"""
    
    def __init__(self):
        self.project_root = self.find_tinygrad_root()
        print(f"Tinygrad project root: {self.project_root}")
    
    def find_tinygrad_root(self):
        """Find tinygrad project root"""
        current = os.getcwd()
        while current != '/':
            if os.path.exists(os.path.join(current, 'tinygrad')):
                return current
            current = os.path.dirname(current)
        return os.getcwd()
    
    def setup_development_environment(self):
        """Set up development environment for contributions"""
        print("\n=== Development Environment Setup ===")
        
        setup_commands = [
            "# Clone the repository",
            "git clone https://github.com/tinygrad/tinygrad.git",
            "cd tinygrad",
            "",
            "# Install in development mode",
            "python3 -m pip install -e '.[testing,linting]'",
            "",
            "# Install pre-commit hooks", 
            "pre-commit install",
            "",
            "# Verify installation",
            "python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]))'",
        ]
        
        for cmd in setup_commands:
            print(cmd)
        
        print("\n✓ Development environment ready")
        print("✓ Pre-commit hooks installed")
        print("✓ Testing and linting dependencies available")
    
    def demonstrate_testing_workflow(self):
        """Show comprehensive testing workflow"""
        print("\n=== Testing Workflow ===")
        
        # Test categories in tinygrad
        test_categories = {
            "Unit Tests": {
                "path": "test/",
                "files": ["test_ops.py", "test_tensor.py", "test_dtype.py"],
                "purpose": "Core functionality testing",
                "command": "python3 -m pytest test/test_ops.py -v"
            },
            "External Tests": {
                "path": "test/external/", 
                "files": ["test_onnx.py", "test_efficientnet.py"],
                "purpose": "Integration with external systems",
                "command": "python3 -m pytest test/external/ -v --tb=short"
            },
            "Speed Tests": {
                "path": "test/",
                "files": ["test_speed_v_torch.py"],
                "purpose": "Performance benchmarking",
                "command": "python3 -m pytest test/test_speed_v_torch.py"
            },
            "Process Replay": {
                "path": "test/",
                "files": ["test_process_replay.py"],
                "purpose": "Regression testing for refactors",
                "command": "PROCESS_REPLAY_CAPTURE=1 python3 test.py"
            }
        }
        
        for category, info in test_categories.items():
            print(f"\n{category}:")
            print(f"  Path: {info['path']}")
            print(f"  Purpose: {info['purpose']}")
            print(f"  Command: {info['command']}")
            print(f"  Example files: {', '.join(info['files'])}")
        
        # Run a sample test
        print(f"\nRunning sample test...")
        try:
            # Basic tensor operation test
            x = Tensor([1, 2, 3])
            y = Tensor([4, 5, 6])
            z = x + y
            assert z.numpy().tolist() == [5, 7, 9]
            print("✓ Basic tensor test passed")
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    def explore_code_structure(self):
        """Understand tinygrad codebase structure for contributions"""
        print("\n=== Codebase Structure ===")
        
        structure = {
            "tinygrad/": {
                "tensor.py": "Main Tensor class - high-level API",
                "device.py": "Device abstraction and buffer management", 
                "dtype.py": "Data type definitions and conversions"
            },
            "tinygrad/uop/": {
                "ops.py": "Primitive operation definitions (Ops enum)",
                "uops.py": "UOp class and computation graph nodes",
                "graph.py": "Graph utilities and visualization",
                "symbolic.py": "Symbolic mathematics for shape inference"
            },
            "tinygrad/engine/": {
                "schedule.py": "Operation scheduling and dependency resolution",
                "realize.py": "Graph realization and execution",
                "memory.py": "Memory planning and buffer allocation",
                "jit.py": "Just-in-time compilation and caching"
            },
            "tinygrad/runtime/": {
                "ops_cpu.py": "CPU backend implementation",
                "ops_cuda.py": "CUDA/GPU backend implementation", 
                "ops_metal.py": "Apple Metal backend",
                "ops_opencl.py": "OpenCL backend for various devices"
            },
            "tinygrad/shape/": {
                "shapetracker.py": "Shape and memory layout tracking",
                "view.py": "View transformations (reshape, permute, etc)"
            },
            "tinygrad/codegen/": {
                "linearize.py": "Convert UOp graphs to linear code",
                "kernel.py": "Kernel generation and optimization"
            }
        }
        
        print("Key directories and files for contributions:")
        for directory, files in structure.items():
            print(f"\n{directory}")
            for file, description in files.items():
                print(f"  {file}: {description}")
        
        # Common contribution areas
        print(f"\n=== Common Contribution Areas ===")
        contribution_areas = [
            "🔧 Backend Development: Add support for new accelerators",
            "⚡ Optimization: Improve kernel fusion and scheduling",
            "🧪 Testing: Add regression tests and edge case coverage",
            "📚 Documentation: Improve code comments and examples",
            "🐛 Bug Fixes: Address issues in GitHub issue tracker",
            "🚀 Features: Implement new operations or neural network layers"
        ]
        
        for area in contribution_areas:
            print(f"  {area}")

class BugFixContribution:
    """Example of contributing a bug fix"""
    
    def __init__(self):
        print("=== Bug Fix Contribution Example ===")
    
    def identify_bug(self):
        """Step 1: Identify and reproduce a bug"""
        print("\n1. Bug Identification")
        print("   Found in GitHub issues or through testing")
        
        # Example: Hypothetical bug in tensor reshaping
        print("\n   Example bug report:")
        print("   'Tensor.reshape() fails for certain shape combinations'")
        
        # Reproduce the bug
        try:
            x = Tensor.randn(6, 4)
            # This might fail in certain edge cases
            y = x.reshape(3, 8)  
            y.realize()
            print("   ✓ Bug reproduction successful")
        except Exception as e:
            print(f"   ✗ Bug reproduced: {e}")
    
    def create_regression_test(self):
        """Step 2: Create a regression test"""
        print("\n2. Regression Test Creation")
        
        test_code = '''
def test_tensor_reshape_edge_case():
    """Test that would have caught the bug"""
    import numpy as np
    from tinygrad import Tensor
    
    # Test various reshape combinations
    test_cases = [
        ((6, 4), (3, 8)),
        ((12, 1), (4, 3)),
        ((2, 2, 3), (12,)),
        ((24,), (2, 3, 4))
    ]
    
    for original_shape, target_shape in test_cases:
        # Test with tinygrad
        x_tinygrad = Tensor.randn(*original_shape)
        y_tinygrad = x_tinygrad.reshape(*target_shape)
        
        # Test with numpy for comparison
        x_numpy = np.random.randn(*original_shape)
        y_numpy = x_numpy.reshape(*target_shape)
        
        # Shapes should match
        assert y_tinygrad.shape == y_numpy.shape
        assert y_tinygrad.numel() == y_numpy.size
        
    print("✓ All reshape tests passed")
'''
        
        print("   Regression test template:")
        print(test_code)
    
    def implement_fix(self):
        """Step 3: Implement the fix"""
        print("\n3. Fix Implementation")
        
        # Example fix in tensor.py (simplified)
        fix_code = '''
# In tinygrad/tensor.py
def reshape(self, *shape):
    """Fixed reshape implementation"""
    # Validate shape compatibility
    new_shape = tuple(int(x) for x in shape)
    
    # Check that total elements match
    if self.numel() != np.prod(new_shape):
        raise ValueError(f"Cannot reshape {self.shape} to {new_shape}: "
                        f"incompatible sizes {self.numel()} vs {np.prod(new_shape)}")
    
    # Handle edge cases properly
    if any(dim < 0 for dim in new_shape):
        # Implement proper -1 dimension inference
        negative_dims = sum(1 for dim in new_shape if dim == -1)
        if negative_dims > 1:
            raise ValueError("Only one dimension can be -1")
        
        if negative_dims == 1:
            known_size = np.prod([dim for dim in new_shape if dim != -1])
            inferred_size = self.numel() // known_size
            new_shape = tuple(inferred_size if dim == -1 else dim for dim in new_shape)
    
    # Use existing reshape logic with proper validation
    return self._reshape(new_shape)
'''
        
        print("   Example fix:")
        print(fix_code)
        print("\n   ✓ Fix addresses edge cases and improves error handling")
    
    def submit_pull_request(self):
        """Step 4: Submit pull request"""
        print("\n4. Pull Request Submission")
        
        pr_checklist = [
            "✓ Write clear commit message describing the fix",
            "✓ Add regression test that would catch the bug",
            "✓ Verify all existing tests still pass",
            "✓ Run linting (ruff check .)",  
            "✓ Run type checking (mypy tinygrad/)",
            "✓ Update documentation if needed",
            "✓ Reference the GitHub issue in PR description"
        ]
        
        for item in pr_checklist:
            print(f"   {item}")
        
        print(f"\n   PR Template:")
        pr_template = '''
## Description
Fix tensor reshape edge case handling

## Issue
Fixes #1234 - Tensor.reshape() fails for certain shape combinations

## Changes
- Added proper shape validation in Tensor.reshape()
- Improved error messages for invalid reshapes  
- Added regression tests for edge cases

## Testing
- [x] Added new test cases
- [x] All existing tests pass
- [x] Manually verified fix works for reported cases

## Checklist
- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Added tests for new functionality
- [x] Documentation updated if needed
'''
        print(pr_template)

## Part 2: Advanced Production Deployment

class ProductionScaleDeployment:
    """Large-scale production deployment strategies"""
    
    def __init__(self):
        print("=== Production Scale Deployment ===")
    
    def implement_model_versioning(self):
        """Model versioning and A/B testing system"""
        print("\n=== Model Versioning System ===")
        
        class ModelVersionManager:
            """Manage multiple model versions in production"""
            
            def __init__(self):
                self.models = {}
                self.traffic_split = {}
                self.performance_metrics = {}
            
            def register_model(self, version, model_path, weight=0.0):
                """Register a new model version"""
                print(f"Registering model version: {version}")
                
                # Load model (simplified)
                model_config = {
                    'path': model_path,
                    'loaded_at': time.time(),
                    'warmup_completed': False
                }
                
                self.models[version] = model_config
                self.traffic_split[version] = weight
                self.performance_metrics[version] = {
                    'requests': 0,
                    'latency': [],
                    'errors': 0
                }
                
                print(f"✓ Model {version} registered with {weight*100:.1f}% traffic")
            
            def route_request(self, request_id):
                """Route request to appropriate model version"""
                import random
                
                # Weighted random selection based on traffic split
                rand = random.random()
                cumulative = 0.0
                
                for version, weight in self.traffic_split.items():
                    cumulative += weight
                    if rand <= cumulative:
                        self.performance_metrics[version]['requests'] += 1
                        return version
                
                # Fallback to first model
                return list(self.models.keys())[0]
            
            def update_traffic_split(self, version, new_weight):
                """Update traffic distribution"""
                print(f"Updating {version} traffic to {new_weight*100:.1f}%")
                
                # Normalize weights
                total_weight = sum(self.traffic_split.values()) - self.traffic_split[version] + new_weight
                
                for v in self.traffic_split:
                    if v == version:
                        self.traffic_split[v] = new_weight
                    else:
                        self.traffic_split[v] = self.traffic_split[v] / total_weight * (1.0 - new_weight)
                
                print("Updated traffic distribution:")
                for v, w in self.traffic_split.items():
                    print(f"  {v}: {w*100:.1f}%")
            
            def get_performance_report(self):
                """Generate performance comparison report"""
                print(f"\n=== Performance Report ===")
                
                for version, metrics in self.performance_metrics.items():
                    requests = metrics['requests']
                    avg_latency = np.mean(metrics['latency']) if metrics['latency'] else 0
                    error_rate = metrics['errors'] / requests if requests > 0 else 0
                    
                    print(f"\nModel {version}:")
                    print(f"  Requests: {requests}")
                    print(f"  Avg Latency: {avg_latency:.2f}ms")
                    print(f"  Error Rate: {error_rate:.2%}")
        
        # Example usage
        version_manager = ModelVersionManager()
        version_manager.register_model("v1.0", "models/baseline.pkl", 0.8)
        version_manager.register_model("v1.1", "models/optimized.pkl", 0.2)
        
        # Simulate request routing
        for request_id in range(100):
            selected_version = version_manager.route_request(request_id)
            # Simulate latency recording
            latency = np.random.gamma(2, 10)  # Simulated latency
            version_manager.performance_metrics[selected_version]['latency'].append(latency)
        
        version_manager.get_performance_report()
        
        # Gradually shift traffic to new model if performing well
        version_manager.update_traffic_split("v1.1", 0.5)
    
    def implement_auto_scaling(self):
        """Auto-scaling inference servers"""
        print("\n=== Auto-Scaling System ===")
        
        class AutoScaler:
            """Auto-scale model servers based on load"""
            
            def __init__(self):
                self.servers = {}
                self.target_utilization = 0.7  # 70% target CPU/GPU utilization
                self.min_replicas = 2
                self.max_replicas = 20
                
            def monitor_server_metrics(self, server_id):
                """Monitor server performance metrics"""
                # Simulated metrics
                cpu_usage = np.random.uniform(0.3, 0.9)
                memory_usage = np.random.uniform(0.4, 0.8) 
                gpu_usage = np.random.uniform(0.2, 0.95)
                queue_length = np.random.poisson(5)
                
                metrics = {
                    'cpu': cpu_usage,
                    'memory': memory_usage,
                    'gpu': gpu_usage,
                    'queue_length': queue_length,
                    'timestamp': time.time()
                }
                
                self.servers[server_id] = metrics
                return metrics
            
            def make_scaling_decision(self):
                """Decide whether to scale up or down"""
                if not self.servers:
                    return "no_action"
                
                # Calculate average utilization
                avg_cpu = np.mean([s['cpu'] for s in self.servers.values()])
                avg_gpu = np.mean([s['gpu'] for s in self.servers.values()])
                avg_queue = np.mean([s['queue_length'] for s in self.servers.values()])
                
                utilization = max(avg_cpu, avg_gpu)  # Use highest utilization
                current_replicas = len(self.servers)
                
                print(f"Current replicas: {current_replicas}")
                print(f"Average utilization: {utilization:.2%}")
                print(f"Average queue length: {avg_queue:.1f}")
                
                # Scaling logic
                if utilization > 0.8 or avg_queue > 10:
                    if current_replicas < self.max_replicas:
                        return "scale_up"
                elif utilization < 0.5 and avg_queue < 2:
                    if current_replicas > self.min_replicas:
                        return "scale_down"
                
                return "no_action"
            
            def execute_scaling(self, action):
                """Execute scaling action"""
                if action == "scale_up":
                    new_server_id = f"server_{len(self.servers) + 1}"
                    print(f"🔼 Scaling up: Adding {new_server_id}")
                    # Simulate new server startup
                    self.monitor_server_metrics(new_server_id)
                    
                elif action == "scale_down":
                    if len(self.servers) > self.min_replicas:
                        server_to_remove = list(self.servers.keys())[-1]
                        print(f"🔽 Scaling down: Removing {server_to_remove}")
                        del self.servers[server_to_remove]
                
                else:
                    print("➡️  No scaling action needed")
        
        # Simulate auto-scaling
        autoscaler = AutoScaler()
        
        # Start with minimum replicas
        for i in range(autoscaler.min_replicas):
            autoscaler.monitor_server_metrics(f"server_{i+1}")
        
        # Simulate monitoring and scaling over time
        for time_step in range(5):
            print(f"\n--- Time Step {time_step + 1} ---")
            
            # Update metrics for all servers
            for server_id in list(autoscaler.servers.keys()):
                autoscaler.monitor_server_metrics(server_id)
            
            # Make scaling decision
            action = autoscaler.make_scaling_decision()
            autoscaler.execute_scaling(action)
    
    def implement_monitoring_system(self):
        """Comprehensive monitoring and alerting"""
        print("\n=== Monitoring System ===")
        
        class ProductionMonitor:
            """Production monitoring and alerting system"""
            
            def __init__(self):
                self.metrics = {
                    'requests_per_second': [],
                    'latency_p50': [],
                    'latency_p95': [],
                    'latency_p99': [],
                    'error_rate': [],
                    'model_accuracy': [],
                    'gpu_utilization': [],
                    'memory_usage': []
                }
                self.alerts = []
                self.thresholds = {
                    'latency_p95': 100.0,  # 100ms
                    'error_rate': 0.01,    # 1%
                    'gpu_utilization': 0.9  # 90%
                }
            
            def collect_metrics(self):
                """Collect real-time metrics"""
                # Simulate metric collection
                rps = np.random.poisson(100)  # 100 RPS average
                latencies = np.random.gamma(2, 20, 1000)  # Simulated latency distribution
                
                current_metrics = {
                    'requests_per_second': rps,
                    'latency_p50': np.percentile(latencies, 50),
                    'latency_p95': np.percentile(latencies, 95),
                    'latency_p99': np.percentile(latencies, 99),
                    'error_rate': np.random.uniform(0, 0.02),
                    'model_accuracy': np.random.uniform(0.92, 0.98),
                    'gpu_utilization': np.random.uniform(0.6, 0.95),
                    'memory_usage': np.random.uniform(0.4, 0.8)
                }
                
                # Store metrics
                for key, value in current_metrics.items():
                    self.metrics[key].append(value)
                    
                return current_metrics
            
            def check_alerts(self, current_metrics):
                """Check for alert conditions"""
                alerts_triggered = []
                
                # Check each threshold
                for metric, threshold in self.thresholds.items():
                    if metric in current_metrics:
                        value = current_metrics[metric]
                        
                        if metric == 'error_rate' and value > threshold:
                            alerts_triggered.append(f"🚨 HIGH ERROR RATE: {value:.2%} > {threshold:.2%}")
                        elif metric == 'latency_p95' and value > threshold:
                            alerts_triggered.append(f"🚨 HIGH LATENCY: P95 {value:.1f}ms > {threshold}ms")
                        elif metric == 'gpu_utilization' and value > threshold:
                            alerts_triggered.append(f"⚠️  HIGH GPU USAGE: {value:.1%} > {threshold:.1%}")
                
                # Trend-based alerts
                if len(self.metrics['error_rate']) >= 5:
                    recent_errors = self.metrics['error_rate'][-5:]
                    if all(e > self.thresholds['error_rate'] * 0.5 for e in recent_errors):
                        alerts_triggered.append("📈 ERROR RATE TRENDING UP")
                
                if alerts_triggered:
                    self.alerts.extend(alerts_triggered)
                    for alert in alerts_triggered:
                        print(alert)
                
                return alerts_triggered
            
            def generate_dashboard_data(self):
                """Generate dashboard metrics"""
                if not any(self.metrics.values()):
                    return
                
                print(f"\n=== Production Dashboard ===")
                
                # Current status
                latest = {k: v[-1] if v else 0 for k, v in self.metrics.items()}
                
                print(f"🚀 Requests/sec: {latest['requests_per_second']}")
                print(f"⏱️  Latency P95: {latest['latency_p95']:.1f}ms")
                print(f"❌ Error Rate: {latest['error_rate']:.2%}")
                print(f"🎯 Model Accuracy: {latest['model_accuracy']:.2%}")
                print(f"🖥️  GPU Utilization: {latest['gpu_utilization']:.1%}")
                print(f"💾 Memory Usage: {latest['memory_usage']:.1%}")
                
                # System health
                health_score = (
                    (1.0 if latest['error_rate'] < self.thresholds['error_rate'] else 0.5) * 0.3 +
                    (1.0 if latest['latency_p95'] < self.thresholds['latency_p95'] else 0.5) * 0.3 +
                    (1.0 if latest['gpu_utilization'] < self.thresholds['gpu_utilization'] else 0.7) * 0.2 +
                    (latest['model_accuracy']) * 0.2
                )
                
                status_emoji = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"
                print(f"\n{status_emoji} System Health: {health_score:.1%}")
        
        # Simulate monitoring
        monitor = ProductionMonitor()
        
        for i in range(10):
            metrics = monitor.collect_metrics()
            alerts = monitor.check_alerts(metrics)
            
            if i % 3 == 0:  # Show dashboard every few iterations
                monitor.generate_dashboard_data()
                
            time.sleep(0.1)  # Simulate time passing

## Part 3: Team Development Workflows

class TeamDevelopmentWorkflow:
    """Team development best practices"""
    
    def implement_code_review_process(self):
        """Code review and collaboration workflow"""
        print("\n=== Code Review Process ===")
        
        review_checklist = {
            "Architecture & Design": [
                "Does the change follow tinygrad's 4-layer architecture?",
                "Are UOps used appropriately vs higher-level Tensor operations?",
                "Is the abstraction level appropriate for the change?",
                "Does it maintain the RISC-like philosophy of simplicity?"
            ],
            "Code Quality": [
                "Code follows existing style conventions",
                "Proper error handling and edge cases covered",
                "No hardcoded magic numbers or strings",
                "Clear variable and function names",
                "Appropriate code comments where needed"
            ],
            "Performance": [
                "No obvious performance regressions introduced",
                "Appropriate use of JIT compilation where beneficial",
                "Memory usage is reasonable",
                "GPU kernels are properly optimized"
            ],
            "Testing": [
                "Comprehensive test coverage for new functionality",
                "Edge cases and error conditions tested",
                "Tests are deterministic and not flaky",
                "Performance tests included where relevant",
                "Process replay tests for refactors"
            ],
            "Documentation": [
                "Public APIs are properly documented",
                "Complex algorithms have explanatory comments",
                "README and examples updated if needed",
                "Breaking changes are clearly documented"
            ]
        }
        
        print("Code Review Checklist:")
        for category, items in review_checklist.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  ☐ {item}")
    
    def setup_ci_cd_pipeline(self):
        """CI/CD pipeline configuration"""
        print("\n=== CI/CD Pipeline ===")
        
        # GitHub Actions workflow example
        github_workflow = '''
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e '.[testing,linting]'
    
    - name: Lint with ruff
      run: |
        ruff check .
    
    - name: Type check with mypy
      run: |
        mypy tinygrad/
    
    - name: Run unit tests
      run: |
        python -m pytest test/ -v --tb=short
    
    - name: Run external tests
      run: |
        python -m pytest test/external/ -v --tb=short
      continue-on-error: true
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
'''
        
        print("GitHub Actions Workflow:")
        print(github_workflow)
        
        # Pre-commit configuration
        precommit_config = '''
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]
    -   id: ruff-format

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
    -   id: mypy
        additional_dependencies: [types-all]

-   repo: local
    hooks:
    -   id: test-core
        name: Core Tests
        entry: python -m pytest test/test_ops.py test/test_tensor.py -x
        language: system
        pass_filenames: false
'''
        
        print("\nPre-commit Configuration:")
        print(precommit_config)

class AdvancedDebuggingExplorer:
    """Advanced debugging and profiling techniques"""
    
    def demonstrate_debugging_techniques(self):
        """Advanced debugging for tinygrad development"""
        print("\n=== Advanced Debugging ===")
        
        # Debug flag examples
        debug_flags = {
            "DEBUG=1": "Basic operation logging",
            "DEBUG=2": "Schedule and kernel information", 
            "DEBUG=3": "Generated code and compilation",
            "DEBUG=4": "Full kernel code with assembly",
            "DEBUG=5": "Buffer allocation and memory management",
            "DEBUG=6": "Graph rewriting and optimization passes"
        }
        
        print("Debug Environment Variables:")
        for flag, description in debug_flags.items():
            print(f"  {flag}: {description}")
        
        # Example debugging session
        print(f"\nDebugging Example:")
        debug_example = '''
# Debug a specific tensor operation
import os
os.environ["DEBUG"] = "3"

from tinygrad import Tensor

# This will show detailed compilation info
x = Tensor.randn(1000, 1000)
y = Tensor.randn(1000, 1000)
z = (x @ y).relu().sum()
z.realize()

# Output shows:
# - UOp graph construction
# - Kernel fusion decisions  
# - Generated GPU code
# - Compilation time and cache hits
'''
        print(debug_example)
    
    def implement_custom_profiler(self):
        """Custom profiling tools for performance analysis"""
        print("\n=== Custom Profiler ===")
        
        class TinygradProfiler:
            """Custom profiler for tinygrad operations"""
            
            def __init__(self):
                self.traces = []
                self.start_time = None
                
            def start_trace(self, operation_name):
                """Start tracing an operation"""
                trace = {
                    'operation': operation_name,
                    'start_time': time.time(),
                    'uops': [],
                    'kernels': [],
                    'memory_usage': self.get_memory_usage()
                }
                self.traces.append(trace)
                return len(self.traces) - 1
            
            def end_trace(self, trace_id):
                """End tracing and record results"""
                if trace_id < len(self.traces):
                    trace = self.traces[trace_id]
                    trace['end_time'] = time.time()
                    trace['duration'] = trace['end_time'] - trace['start_time']
                    trace['final_memory'] = self.get_memory_usage()
                    trace['memory_delta'] = trace['final_memory'] - trace['memory_usage']
            
            def get_memory_usage(self):
                """Get current memory usage (simplified)"""
                import psutil
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024  # MB
            
            def profile_operation(self, operation_name, operation_func):
                """Profile a complete operation"""
                trace_id = self.start_trace(operation_name)
                
                try:
                    start_time = time.time()
                    result = operation_func()
                    if hasattr(result, 'realize'):
                        result.realize()
                    end_time = time.time()
                    
                    self.end_trace(trace_id)
                    
                    trace = self.traces[trace_id]
                    print(f"Operation: {operation_name}")
                    print(f"  Duration: {trace['duration']*1000:.2f}ms")
                    print(f"  Memory Delta: {trace['memory_delta']:.2f}MB")
                    
                    return result
                    
                except Exception as e:
                    self.end_trace(trace_id)
                    print(f"Operation {operation_name} failed: {e}")
                    raise
            
            def generate_report(self):
                """Generate profiling report"""
                if not self.traces:
                    return
                
                print(f"\n=== Profiling Report ===")
                print(f"Total operations: {len(self.traces)}")
                
                total_time = sum(t.get('duration', 0) for t in self.traces)
                print(f"Total time: {total_time*1000:.2f}ms")
                
                # Sort by duration
                sorted_traces = sorted(self.traces, key=lambda t: t.get('duration', 0), reverse=True)
                
                print(f"\nTop operations by time:")
                for i, trace in enumerate(sorted_traces[:5]):
                    duration = trace.get('duration', 0)
                    memory_delta = trace.get('memory_delta', 0)
                    print(f"  {i+1}. {trace['operation']}: {duration*1000:.2f}ms, {memory_delta:.2f}MB")
        
        # Example usage
        profiler = TinygradProfiler()
        
        # Profile different operations
        profiler.profile_operation("Matrix Multiplication", 
                                 lambda: Tensor.randn(512, 512) @ Tensor.randn(512, 512))
        
        profiler.profile_operation("Convolution",
                                 lambda: Tensor.randn(1, 3, 224, 224).conv2d(Tensor.randn(64, 3, 3, 3)))
        
        profiler.profile_operation("Reduction",
                                 lambda: Tensor.randn(1000, 1000).sum())
        
        profiler.generate_report()

## Hands-On Exercises

### Exercise 1: Contribute a Feature
```python
# exercises/day7_exercise1_contribution.py
"""
Exercise: Contribute a new feature to tinygrad

Tasks:
1. Implement a new tensor operation (e.g., tensor.median())
2. Add comprehensive tests
3. Update documentation 
4. Follow the full contribution workflow
5. Submit a mock pull request
"""

def exercise_contribution():
    """Implement a new tensor operation"""
    
    # TODO: Choose an operation not yet implemented in tinygrad
    # TODO: Implement the operation following tinygrad patterns
    # TODO: Add unit tests with edge cases
    # TODO: Add integration tests
    # TODO: Benchmark against numpy implementation
    # TODO: Write documentation and examples
    
    print("Contribution Exercise")
    print("====================")
    
    # Example: Implement tensor.median()
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_contribution()
```

### Exercise 2: Production Deployment
```python  
# exercises/day7_exercise2_production.py
"""
Exercise: Design a complete production deployment system

Tasks:
1. Implement model versioning with A/B testing
2. Create auto-scaling based on load metrics  
3. Add comprehensive monitoring and alerting
4. Design failure recovery mechanisms
5. Implement gradual rollout strategies
"""

def exercise_production_deployment():
    """Design production deployment system"""
    
    # TODO: Implement ModelVersionManager with traffic splitting
    # TODO: Create AutoScaler with custom scaling policies
    # TODO: Build comprehensive monitoring dashboard
    # TODO: Add alert mechanisms for different failure modes
    # TODO: Implement blue-green deployment strategy
    
    print("Production Deployment Exercise")
    print("==============================")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_production_deployment()
```

### Exercise 3: Team Development Setup
```python
# exercises/day7_exercise3_team_workflow.py
"""
Exercise: Set up team development workflow

Tasks:
1. Create comprehensive CI/CD pipeline
2. Set up code review processes and templates
3. Implement custom debugging and profiling tools
4. Design team coding standards and guidelines
5. Create onboarding documentation for new team members
"""

def exercise_team_workflow():
    """Set up team development workflow"""
    
    # TODO: Write GitHub Actions workflow for testing
    # TODO: Create pre-commit hooks configuration
    # TODO: Design code review checklist and templates
    # TODO: Implement custom profiling and debugging tools
    # TODO: Write team coding standards document
    # TODO: Create new developer onboarding guide
    
    print("Team Workflow Exercise")
    print("======================")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_team_workflow()
```

## Final Project: End-to-End Contribution

### Capstone Project: Electrical Testing Integration
```python
# projects/electrical_testing_tinygrad_integration.py
"""
Capstone Project: Integrate electrical testing with tinygrad

This project combines everything learned in the 7-day course:
- Tinygrad architecture understanding
- Custom backend development  
- Neural network implementation
- Production deployment
- Team contribution workflow

Goal: Create a specialized tinygrad backend for electrical testing hardware
"""

class ElectricalTestingBackend:
    """Custom tinygrad backend for electrical testing equipment"""
    
    def __init__(self):
        # TODO: Initialize connection to electrical testing hardware
        # TODO: Implement buffer management for test data
        # TODO: Create device-specific optimization strategies
        pass
    
    def implement_custom_ops(self):
        """Implement electrical testing specific operations"""
        # TODO: FFT for signal analysis
        # TODO: Digital filter operations  
        # TODO: Statistical analysis operations
        # TODO: Anomaly detection primitives
        pass
    
    def optimize_for_realtime(self):
        """Optimize for real-time electrical testing"""
        # TODO: Minimize latency for live measurements
        # TODO: Implement streaming data processing
        # TODO: Add hardware-specific memory management
        pass

class ElectricalTestingModel:
    """Neural network model for electrical anomaly detection"""
    
    def __init__(self):
        # TODO: Design architecture for electrical signal processing
        # TODO: Implement custom layers for signal analysis
        # TODO: Add real-time inference capabilities
        pass
    
    def train_on_electrical_data(self, data):
        """Train model on electrical testing data"""
        # TODO: Implement training pipeline
        # TODO: Add domain-specific loss functions
        # TODO: Implement online learning for adaptation
        pass

# TODO: Complete implementation following all course learnings
# TODO: Add comprehensive testing
# TODO: Create documentation
# TODO: Prepare for contribution to tinygrad
```

## Summary and Course Conclusion

### 7-Day Learning Journey Complete

🎉 **Congratulations!** You've completed an intensive journey through the tinygrad ecosystem:

**Day 1**: Architecture mastery - Understanding the 4-layer system and execution flow
**Day 2**: UOp system expertise - Graph manipulation and custom optimizations  
**Day 3**: Engine deep dive - Scheduling, memory, and kernel fusion
**Day 4**: Backend development - Creating custom accelerator support
**Day 5**: Neural networks - High-level APIs and training systems
**Day 6**: Advanced optimization - JIT, quantization, and production deployment  
**Day 7**: Contributions and scale - Team workflows and production systems

### Key Competencies Achieved

✅ **Architectural Understanding**: Complete grasp of tinygrad's design philosophy
✅ **Implementation Skills**: Ability to extend and modify core functionality
✅ **Optimization Expertise**: Knowledge of performance optimization techniques
✅ **Production Readiness**: Skills for deploying models at scale
✅ **Team Collaboration**: Workflows for effective contribution to open source
✅ **Debugging Mastery**: Advanced techniques for troubleshooting complex issues

### Next Steps for Continued Growth

1. **Active Contribution**: Start contributing to tinygrad's GitHub repository
2. **Community Engagement**: Join discussions, help other developers
3. **Specialized Development**: Focus on areas like new backends or optimizations
4. **Production Experience**: Deploy tinygrad models in real-world scenarios
5. **Knowledge Sharing**: Teach others and create educational content

### The Tinygrad Advantage

Through this course, you've gained expertise in a framework that:
- **Prioritizes Simplicity**: ~10k lines vs 1M+ in other frameworks
- **Enables Understanding**: Every component is comprehensible
- **Supports Innovation**: Easy to extend and modify
- **Achieves Performance**: Competitive speed through intelligent optimization
- **Promotes Learning**: Educational value while being production-ready

You're now equipped to be both an effective user and contributor to the tinygrad ecosystem, with deep understanding of modern ML framework design and implementation.

**Welcome to the tinygrad community! 🚀**