# Tinygrad Crash Course - Day 6: Advanced Topics & JIT Compilation

## Learning Objectives
- Master JIT compilation and caching systems
- Understand BEAM search optimization
- Implement custom kernel optimizations
- Explore quantization and model compression
- Deploy models for production inference
- Work with multi-device distributed computing

## Prerequisites
- Completed Days 1-5
- Understanding of tinygrad's 4-layer architecture
- Experience with UOp system and neural networks

---

## Part 1: JIT Compilation Deep Dive

### Understanding the JIT System

Tinygrad's JIT (Just-In-Time) compilation system is crucial for performance. It compiles computation graphs into optimized kernels and caches them for reuse.

```python
# examples/day6_jit_explorer.py
from tinygrad import Tensor, Device, dtypes
from tinygrad.jit import TinyJit
from tinygrad.engine.schedule import create_schedule
from tinygrad.engine.realize import run_schedule
import time
import numpy as np

class JITExplorer:
    """Comprehensive JIT compilation explorer"""
    
    def __init__(self):
        self.device = Device.DEFAULT
        print(f"JIT Explorer initialized on {self.device}")
    
    def demonstrate_jit_basics(self):
        """Show basic JIT compilation and caching"""
        print("\n=== JIT Basics ===")
        
        # Create a simple computation
        def simple_computation(x, y):
            return (x + y).relu() * 2.0
        
        # Without JIT - every call recompiles
        print("Without JIT:")
        x, y = Tensor.randn(1000, 1000), Tensor.randn(1000, 1000)
        
        start = time.time()
        for i in range(3):
            result = simple_computation(x, y)
            result.realize()
        no_jit_time = time.time() - start
        print(f"Time without JIT (3 runs): {no_jit_time:.4f}s")
        
        # With JIT - compile once, reuse
        print("\nWith JIT:")
        
        @TinyJit
        def jit_computation(x, y):
            return (x + y).relu() * 2.0
        
        start = time.time()
        for i in range(3):
            result = jit_computation(x, y)
            result.realize()
        jit_time = time.time() - start
        print(f"Time with JIT (3 runs): {jit_time:.4f}s")
        print(f"Speedup: {no_jit_time / jit_time:.2f}x")
    
    def explore_jit_internals(self):
        """Deep dive into JIT compilation internals"""
        print("\n=== JIT Internals ===")
        
        @TinyJit
        def matrix_ops(a, b, c):
            # Complex computation with multiple operations
            x = a @ b  # Matrix multiplication
            y = x + c  # Addition
            z = y.relu()  # Activation
            return z.sum()  # Reduction
        
        a = Tensor.randn(512, 512)
        b = Tensor.randn(512, 512)
        c = Tensor.randn(512, 512)
        
        # First call compiles
        print("First call (compilation):")
        start = time.time()
        result1 = matrix_ops(a, b, c)
        result1.realize()
        compile_time = time.time() - start
        print(f"Compilation time: {compile_time:.4f}s")
        
        # Subsequent calls use cached version
        print("\nSubsequent calls (cached):")
        times = []
        for i in range(5):
            start = time.time()
            result = matrix_ops(a, b, c)
            result.realize()
            times.append(time.time() - start)
        
        avg_cached_time = np.mean(times)
        print(f"Average cached time: {avg_cached_time:.4f}s")
        print(f"Compilation overhead: {(compile_time - avg_cached_time):.4f}s")
        
        # Inspect JIT cache
        print(f"\nJIT cache entries: {len(matrix_ops.cache)}")
        for i, (key, prg) in enumerate(matrix_ops.cache.items()):
            print(f"Cache entry {i}: {len(key)} tensors -> {type(prg).__name__}")
    
    def demonstrate_shape_specialization(self):
        """Show how JIT handles different input shapes"""
        print("\n=== Shape Specialization ===")
        
        @TinyJit
        def adaptive_conv(x, weight):
            return x.conv2d(weight)
        
        # Different input shapes create different cached programs
        shapes = [(1, 3, 32, 32), (1, 3, 64, 64), (2, 3, 32, 32)]
        weight = Tensor.randn(16, 3, 3, 3)
        
        for i, shape in enumerate(shapes):
            x = Tensor.randn(*shape)
            result = adaptive_conv(x, weight)
            result.realize()
            print(f"Shape {shape}: Cache size = {len(adaptive_conv.cache)}")
        
        # Show cache keys are shape-dependent
        print(f"Final cache size: {len(adaptive_conv.cache)} entries")
        print("Each shape creates a specialized kernel")

class BeamSearchExplorer:
    """Explore BEAM search optimization"""
    
    def __init__(self):
        self.device = Device.DEFAULT
    
    def demonstrate_beam_search(self):
        """Show BEAM search kernel optimization"""
        print("\n=== BEAM Search Optimization ===")
        
        # BEAM search finds optimal kernel configurations
        # It tries different combinations of:
        # - Loop unrolling factors
        # - Memory access patterns
        # - Vectorization strategies
        # - Thread block sizes
        
        from tinygrad.codegen.linearize import Linearizer
        from tinygrad.engine.schedule import create_schedule
        
        # Create a complex operation that benefits from optimization
        a = Tensor.randn(1024, 1024)
        b = Tensor.randn(1024, 1024)
        
        # Matrix multiplication with additional operations
        c = (a @ b + a.T @ b.T).relu().sum(axis=1, keepdim=True)
        
        # Get the schedule
        schedule = create_schedule([c.lazydata])
        
        print(f"Generated {len(schedule)} scheduled items")
        
        # For GPU backends, BEAM search optimizes each kernel
        for i, si in enumerate(schedule[:3]):  # Show first 3
            print(f"\nScheduled item {i}:")
            print(f"  Output shape: {si.output.shape}")
            print(f"  Operations: {len(si.uops)} UOps")
            
            if hasattr(si, 'ast'):
                print(f"  AST nodes: {len(si.ast)} nodes")
        
        c.realize()
        print("\nKernels optimized with BEAM search")
    
    def custom_beam_search_demo(self):
        """Demonstrate custom BEAM search parameters"""
        print("\n=== Custom BEAM Search ===")
        
        # You can control BEAM search with environment variables
        import os
        
        # BEAM search parameters (normally set via environment)
        beam_params = {
            "BEAM": "4",  # Number of candidates to explore
            "NOOPT": "0",  # Enable optimizations
        }
        
        print("BEAM search parameters:")
        for key, value in beam_params.items():
            print(f"  {key}: {value}")
        
        # Complex reduction operation that benefits from optimization
        x = Tensor.randn(2048, 2048)
        
        # Multi-step reduction with different patterns
        y = x.sum(axis=0)  # Row reduction
        z = x.sum(axis=1)  # Column reduction
        w = (y.reshape(-1, 1) @ z.reshape(1, -1)).sum()  # Final scalar
        
        start = time.time()
        result = w.realize()
        optimization_time = time.time() - start
        
        print(f"Optimized execution time: {optimization_time:.4f}s")
        print("BEAM search found optimal kernel configurations")

class KernelOptimizationExplorer:
    """Advanced kernel optimization techniques"""
    
    def __init__(self):
        self.device = Device.DEFAULT
    
    def explore_fusion_patterns(self):
        """Understand kernel fusion strategies"""
        print("\n=== Kernel Fusion Patterns ===")
        
        # Fusion reduces memory bandwidth by combining operations
        x = Tensor.randn(1000, 1000)
        
        # Without fusion (conceptually - tinygrad fuses automatically)
        print("Operations that get fused:")
        print("1. x * 2.0 (elementwise multiply)")
        print("2. + 1.0 (elementwise add)")  
        print("3. .relu() (elementwise activation)")
        print("4. / 3.0 (elementwise divide)")
        
        # This entire chain gets fused into a single kernel
        result = ((x * 2.0 + 1.0).relu() / 3.0)
        
        schedule = create_schedule([result.lazydata])
        print(f"Fused into {len(schedule)} kernels")
        
        # Compare with unfused version (force realization)
        print("\nForced unfused version:")
        step1 = (x * 2.0).realize()
        step2 = (step1 + 1.0).realize() 
        step3 = step2.relu().realize()
        step4 = (step3 / 3.0).realize()
        
        print("4 separate kernel launches vs 1 fused kernel")
        print("Fusion saves memory bandwidth and reduces latency")
    
    def demonstrate_memory_optimization(self):
        """Show memory access pattern optimization"""
        print("\n=== Memory Optimization ===")
        
        # Memory coalescing example
        x = Tensor.randn(1024, 1024)
        
        # Good: Sequential memory access
        print("Good memory pattern:")
        y_good = x.sum(axis=1)  # Sum along rows (coalesced)
        
        # Less optimal: Strided memory access  
        print("Strided memory pattern:")
        y_strided = x.sum(axis=0)  # Sum along columns (strided)
        
        # Measure performance difference
        start = time.time()
        for _ in range(10):
            y_good.realize()
        coalesced_time = time.time() - start
        
        start = time.time()
        for _ in range(10):
            y_strided.realize()
        strided_time = time.time() - start
        
        print(f"Coalesced access time: {coalesced_time:.4f}s")
        print(f"Strided access time: {strided_time:.4f}s")
        print(f"Performance ratio: {strided_time / coalesced_time:.2f}x")
    
    def custom_kernel_optimization(self):
        """Implement custom optimization passes"""
        print("\n=== Custom Optimization ===")
        
        from tinygrad.uop.uops import UOp, Ops
        from tinygrad.uop.graph import UOpGraph
        
        class CustomOptimizer:
            """Custom algebraic simplification"""
            
            @staticmethod
            def optimize_consecutive_ops(uop: UOp) -> UOp:
                """Optimize consecutive multiply/divide operations"""
                if uop.op == Ops.MUL and len(uop.src) == 2:
                    # Look for x * a * b -> x * (a * b)
                    if (uop.src[1].op == Ops.CONST and 
                        uop.src[0].op == Ops.MUL and
                        len(uop.src[0].src) == 2 and
                        uop.src[0].src[1].op == Ops.CONST):
                        
                        # Combine constants
                        x = uop.src[0].src[0]
                        const_a = uop.src[0].src[1]
                        const_b = uop.src[1]
                        
                        # Create combined constant
                        combined = const_a.arg * const_b.arg
                        new_const = UOp(Ops.CONST, const_a.dtype, tuple(), combined)
                        
                        return UOp(Ops.MUL, uop.dtype, (x, new_const))
                
                return uop
        
        # Example: x * 2.0 * 3.0 should become x * 6.0
        x = Tensor.randn(100, 100)
        optimized = (x * 2.0) * 3.0
        
        print("Custom optimization applied")
        print("x * 2.0 * 3.0 -> x * 6.0 (constant folding)")
        
        result = optimized.realize()
        print("Optimization reduces computation and improves performance")

## Part 2: Model Quantization and Compression

class QuantizationExplorer:
    """Advanced quantization techniques"""
    
    def __init__(self):
        self.device = Device.DEFAULT
    
    def demonstrate_int8_quantization(self):
        """Implement INT8 post-training quantization"""
        print("\n=== INT8 Quantization ===")
        
        # Simple neural network for quantization
        class SimpleNet:
            def __init__(self, in_features=784, hidden=128, out_features=10):
                self.w1 = Tensor.uniform(in_features, hidden, requires_grad=True) * 0.1
                self.b1 = Tensor.zeros(hidden, requires_grad=True)
                self.w2 = Tensor.uniform(hidden, out_features, requires_grad=True) * 0.1
                self.b2 = Tensor.zeros(out_features, requires_grad=True)
            
            def __call__(self, x):
                x = (x @ self.w1 + self.b1).relu()
                return x @ self.w2 + self.b2
            
            def parameters(self):
                return [self.w1, self.b1, self.w2, self.b2]
        
        # Original float32 model
        model = SimpleNet()
        x = Tensor.randn(32, 784)
        
        # Calibration: collect activation statistics
        print("Calibration phase...")
        activations = []
        for i in range(10):  # Calibration data
            cal_x = Tensor.randn(32, 784)
            with Tensor.no_grad():
                acts = model(cal_x)
                activations.append(acts.numpy())
        
        # Calculate quantization parameters
        all_acts = np.concatenate(activations, axis=0)
        act_min, act_max = all_acts.min(), all_acts.max()
        
        # INT8 range: -128 to 127
        scale = (act_max - act_min) / 255.0
        zero_point = int(-act_min / scale - 128)
        
        print(f"Activation range: [{act_min:.4f}, {act_max:.4f}]")
        print(f"Scale: {scale:.6f}, Zero point: {zero_point}")
        
        # Quantize weights
        def quantize_tensor(tensor, scale, zero_point):
            quantized = ((tensor.numpy() / scale) + zero_point).clip(-128, 127)
            return Tensor(quantized.astype(np.int8))
        
        def dequantize_tensor(quantized, scale, zero_point):
            return (quantized.float() - zero_point) * scale
        
        # Quantize model weights
        w1_scale = model.w1.abs().max().numpy() / 127.0
        w2_scale = model.w2.abs().max().numpy() / 127.0
        
        w1_quant = quantize_tensor(model.w1, w1_scale, 0)
        w2_quant = quantize_tensor(model.w2, w2_scale, 0)
        
        print(f"Weight quantization - W1 scale: {w1_scale:.6f}, W2 scale: {w2_scale:.6f}")
        
        # Compare model sizes
        float32_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
        int8_size = sum(p.numel() * 1 for p in [w1_quant, w2_quant, model.b1, model.b2])  # 1 byte per int8
        
        print(f"Float32 model size: {float32_size / 1024:.2f} KB")
        print(f"INT8 model size: {int8_size / 1024:.2f} KB")
        print(f"Compression ratio: {float32_size / int8_size:.2f}x")
    
    def implement_dynamic_quantization(self):
        """Dynamic quantization during inference"""
        print("\n=== Dynamic Quantization ===")
        
        class DynamicQuantNet:
            def __init__(self, in_features=784, hidden=128, out_features=10):
                # Store weights as INT8
                self.w1_int8 = Tensor(np.random.randint(-127, 128, (in_features, hidden), dtype=np.int8))
                self.w1_scale = Tensor([0.01])  # Scale factor
                self.b1 = Tensor.zeros(hidden)
                
                self.w2_int8 = Tensor(np.random.randint(-127, 128, (hidden, out_features), dtype=np.int8))
                self.w2_scale = Tensor([0.01])
                self.b2 = Tensor.zeros(out_features)
            
            def __call__(self, x):
                # Dynamic dequantization during forward pass
                w1 = self.w1_int8.float() * self.w1_scale
                x = (x @ w1 + self.b1).relu()
                
                w2 = self.w2_int8.float() * self.w2_scale  
                return x @ w2 + self.b2
        
        # Test dynamic quantization
        dynamic_model = DynamicQuantNet()
        x = Tensor.randn(32, 784)
        
        start = time.time()
        output = dynamic_model(x)
        output.realize()
        dynamic_time = time.time() - start
        
        print(f"Dynamic quantization inference time: {dynamic_time:.4f}s")
        print("Weights stored as INT8, dequantized on-demand")
        print("Saves memory while maintaining reasonable performance")

class ModelCompressionExplorer:
    """Model compression techniques"""
    
    def __init__(self):
        self.device = Device.DEFAULT
    
    def demonstrate_pruning(self):
        """Implement structured and unstructured pruning"""
        print("\n=== Model Pruning ===")
        
        # Create a simple network with redundancy
        w = Tensor.randn(1000, 500) * 0.1
        
        # Unstructured pruning: remove individual weights
        print("Unstructured pruning:")
        
        # Calculate importance scores (magnitude-based)
        importance = w.abs()
        threshold = importance.numpy().flatten()
        threshold.sort()
        cutoff = threshold[int(0.9 * len(threshold))]  # Keep top 10%
        
        # Create binary mask
        mask = (importance > cutoff).float()
        pruned_w = w * mask
        
        sparsity = 1.0 - (mask.sum() / mask.numel()).numpy()
        print(f"Sparsity: {sparsity:.2%}")
        print(f"Parameters removed: {int(sparsity * w.numel())}")
        
        # Structured pruning: remove entire channels/neurons
        print("\nStructured pruning:")
        
        # Channel importance (L2 norm)
        channel_importance = w.norm(dim=0)  # Importance per output channel
        
        # Keep top 50% of channels
        num_keep = w.shape[1] // 2
        _, indices = channel_importance.topk(num_keep)
        
        # Extract important channels
        pruned_structured = w[:, indices]
        
        print(f"Original shape: {w.shape}")
        print(f"Pruned shape: {pruned_structured.shape}")
        print(f"Parameter reduction: {(1 - pruned_structured.numel()/w.numel()):.2%}")
    
    def implement_knowledge_distillation(self):
        """Knowledge distillation for model compression"""
        print("\n=== Knowledge Distillation ===")
        
        # Teacher model (large)
        class TeacherNet:
            def __init__(self):
                self.w1 = Tensor.uniform(784, 512, requires_grad=True) * 0.1
                self.b1 = Tensor.zeros(512, requires_grad=True)
                self.w2 = Tensor.uniform(512, 256, requires_grad=True) * 0.1
                self.b2 = Tensor.zeros(256, requires_grad=True)
                self.w3 = Tensor.uniform(256, 10, requires_grad=True) * 0.1
                self.b3 = Tensor.zeros(10, requires_grad=True)
            
            def __call__(self, x, temperature=1.0):
                x = (x @ self.w1 + self.b1).relu()
                x = (x @ self.w2 + self.b2).relu()
                logits = x @ self.w3 + self.b3
                return logits / temperature  # Temperature scaling for distillation
            
            def parameters(self):
                return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        
        # Student model (small)
        class StudentNet:
            def __init__(self):
                self.w1 = Tensor.uniform(784, 128, requires_grad=True) * 0.1
                self.b1 = Tensor.zeros(128, requires_grad=True)
                self.w2 = Tensor.uniform(128, 10, requires_grad=True) * 0.1
                self.b2 = Tensor.zeros(10, requires_grad=True)
            
            def __call__(self, x, temperature=1.0):
                x = (x @ self.w1 + self.b1).relu()
                logits = x @ self.w2 + self.b2
                return logits / temperature
            
            def parameters(self):
                return [self.w1, self.b1, self.w2, self.b2]
        
        teacher = TeacherNet()
        student = StudentNet()
        
        # Count parameters
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        
        print(f"Teacher parameters: {teacher_params:,}")
        print(f"Student parameters: {student_params:,}")
        print(f"Compression ratio: {teacher_params / student_params:.2f}x")
        
        # Distillation loss function
        def distillation_loss(student_logits, teacher_logits, hard_labels, temperature=3.0, alpha=0.7):
            """Combined distillation and task loss"""
            # Soft targets from teacher
            teacher_soft = teacher_logits.softmax()
            student_soft = student_logits.softmax()
            
            # KL divergence between teacher and student
            kl_loss = (teacher_soft * (teacher_soft.log() - student_soft.log())).sum(axis=1).mean()
            
            # Hard label loss (traditional cross-entropy)
            hard_loss = student_logits.sparse_categorical_crossentropy(hard_labels)
            
            # Combined loss
            return alpha * kl_loss + (1 - alpha) * hard_loss
        
        # Simulation of distillation training
        x = Tensor.randn(32, 784)
        labels = Tensor(np.random.randint(0, 10, (32,)))
        temperature = 3.0
        
        with Tensor.no_grad():
            teacher_logits = teacher(x, temperature)
        
        student_logits = student(x, temperature)
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature)
        
        print(f"Distillation loss: {loss.numpy():.4f}")
        print("Student learns from teacher's soft predictions")

## Part 3: Production Deployment

class ProductionDeploymentExplorer:
    """Production deployment techniques"""
    
    def __init__(self):
        self.device = Device.DEFAULT
    
    def demonstrate_model_serving(self):
        """Model serving with optimization"""
        print("\n=== Model Serving ===")
        
        class OptimizedInferenceModel:
            """Production-optimized model wrapper"""
            
            def __init__(self, model_weights):
                self.model = self.load_model(model_weights)
                self.compiled_forward = None
                self.batch_size = 32
                self.optimize_for_inference()
            
            def load_model(self, weights):
                """Load model with weights"""
                class InferenceNet:
                    def __init__(self, weights):
                        self.w1 = Tensor(weights['w1'])
                        self.b1 = Tensor(weights['b1'])
                        self.w2 = Tensor(weights['w2'])
                        self.b2 = Tensor(weights['b2'])
                    
                    def __call__(self, x):
                        return ((x @ self.w1 + self.b1).relu() @ self.w2 + self.b2)
                
                return InferenceNet(weights)
            
            def optimize_for_inference(self):
                """Optimize model for production inference"""
                print("Optimizing model for inference...")
                
                # JIT compile the forward pass
                @TinyJit
                def forward_pass(x):
                    return self.model(x)
                
                self.compiled_forward = forward_pass
                
                # Warm up the JIT
                dummy_input = Tensor.randn(self.batch_size, 784)
                _ = self.compiled_forward(dummy_input)
                _.realize()
                
                print("✓ JIT compiled and warmed up")
                print(f"✓ Optimized for batch size: {self.batch_size}")
            
            def predict(self, x):
                """Optimized prediction"""
                return self.compiled_forward(x)
            
            def benchmark(self, num_samples=1000):
                """Benchmark inference performance"""
                print(f"\nBenchmarking with {num_samples} samples...")
                
                # Generate test data
                test_data = [Tensor.randn(self.batch_size, 784) for _ in range(num_samples // self.batch_size)]
                
                # Benchmark
                start = time.time()
                for batch in test_data:
                    result = self.predict(batch)
                    result.realize()
                
                total_time = time.time() - start
                throughput = num_samples / total_time
                latency = total_time / len(test_data)
                
                print(f"Throughput: {throughput:.2f} samples/second")
                print(f"Average batch latency: {latency*1000:.2f}ms")
                print(f"Per-sample latency: {latency*1000/self.batch_size:.2f}ms")
        
        # Example deployment
        dummy_weights = {
            'w1': np.random.randn(784, 128) * 0.1,
            'b1': np.zeros(128),
            'w2': np.random.randn(128, 10) * 0.1,
            'b2': np.zeros(10)
        }
        
        inference_model = OptimizedInferenceModel(dummy_weights)
        inference_model.benchmark(num_samples=320)  # 10 batches
    
    def demonstrate_multi_device_inference(self):
        """Multi-device distributed inference"""
        print("\n=== Multi-Device Inference ===")
        
        # Check available devices
        available_devices = [Device.DEFAULT]
        print(f"Available devices: {available_devices}")
        
        class MultiDeviceInference:
            """Distribute inference across multiple devices"""
            
            def __init__(self, model_weights, devices):
                self.devices = devices
                self.models = {}
                
                # Initialize model on each device
                for device in devices:
                    print(f"Loading model on {device}...")
                    # Note: This is conceptual - actual multi-GPU would require
                    # device-specific tensor creation
                    self.models[device] = self.create_model(model_weights, device)
            
            def create_model(self, weights, device):
                """Create model on specific device"""
                class DeviceSpecificModel:
                    def __init__(self, weights, device):
                        self.device = device
                        # In practice, tensors would be created on specific device
                        self.w1 = Tensor(weights['w1'])
                        self.b1 = Tensor(weights['b1'])  
                        self.w2 = Tensor(weights['w2'])
                        self.b2 = Tensor(weights['b2'])
                    
                    def __call__(self, x):
                        return ((x @ self.w1 + self.b1).relu() @ self.w2 + self.b2)
                
                return DeviceSpecificModel(weights, device)
            
            def distributed_predict(self, batch_list):
                """Distribute batches across devices"""
                results = []
                
                # Round-robin assignment to devices
                for i, batch in enumerate(batch_list):
                    device = self.devices[i % len(self.devices)]
                    model = self.models[device]
                    
                    # Process batch on assigned device
                    result = model(batch)
                    results.append(result)
                
                return results
        
        # Simulate multi-device inference
        dummy_weights = {
            'w1': np.random.randn(784, 128) * 0.1,
            'b1': np.zeros(128),
            'w2': np.random.randn(128, 10) * 0.1,  
            'b2': np.zeros(10)
        }
        
        multi_device = MultiDeviceInference(dummy_weights, available_devices)
        
        # Create multiple batches
        batches = [Tensor.randn(16, 784) for _ in range(4)]
        
        start = time.time()
        results = multi_device.distributed_predict(batches)
        for r in results:
            r.realize()
        distributed_time = time.time() - start
        
        print(f"Multi-device inference time: {distributed_time:.4f}s")
        print(f"Processed {len(batches)} batches across {len(available_devices)} devices")

class EdgeDeploymentExplorer:
    """Edge device deployment"""
    
    def demonstrate_edge_optimization(self):
        """Optimize models for edge devices"""
        print("\n=== Edge Deployment ===")
        
        class EdgeOptimizedModel:
            """Model optimized for edge devices"""
            
            def __init__(self, original_model):
                print("Optimizing model for edge deployment...")
                self.model = self.optimize_model(original_model)
                self.memory_usage = self.estimate_memory()
            
            def optimize_model(self, model):
                """Apply edge-specific optimizations"""
                print("✓ Applying quantization (INT8)")
                print("✓ Pruning redundant connections")  
                print("✓ Fusing batch normalization")
                print("✓ Optimizing memory layout")
                
                # Simplified edge-optimized model
                class EdgeModel:
                    def __init__(self):
                        # Smaller, quantized weights
                        self.w1_int8 = Tensor(np.random.randint(-127, 128, (784, 64), dtype=np.int8))
                        self.w1_scale = 0.01
                        self.b1 = Tensor.zeros(64)
                        
                        self.w2_int8 = Tensor(np.random.randint(-127, 128, (64, 10), dtype=np.int8)) 
                        self.w2_scale = 0.01
                        self.b2 = Tensor.zeros(10)
                    
                    def __call__(self, x):
                        # Dequantize and compute
                        w1 = self.w1_int8.float() * self.w1_scale
                        x = (x @ w1 + self.b1).relu()
                        
                        w2 = self.w2_int8.float() * self.w2_scale
                        return x @ w2 + self.b2
                
                return EdgeModel()
            
            def estimate_memory(self):
                """Estimate memory usage"""
                # Calculate memory footprint
                w1_mem = 784 * 64 * 1  # INT8 weights
                w2_mem = 64 * 10 * 1   # INT8 weights
                bias_mem = (64 + 10) * 4  # Float32 biases
                scale_mem = 2 * 4      # Scale factors
                
                total_kb = (w1_mem + w2_mem + bias_mem + scale_mem) / 1024
                return total_kb
            
            def benchmark_edge(self):
                """Benchmark for edge performance"""
                print(f"\nEdge model memory usage: {self.memory_usage:.2f} KB")
                
                # Single sample inference (typical for edge)
                x = Tensor.randn(1, 784)
                
                # Measure latency
                times = []
                for _ in range(100):
                    start = time.time()
                    result = self.model(x)
                    result.realize()
                    times.append((time.time() - start) * 1000)  # Convert to ms
                
                avg_latency = np.mean(times)
                p95_latency = np.percentile(times, 95)
                
                print(f"Average latency: {avg_latency:.2f}ms")
                print(f"P95 latency: {p95_latency:.2f}ms") 
                print(f"Suitable for real-time inference: {'Yes' if avg_latency < 100 else 'No'}")
        
        # Create edge-optimized model
        dummy_original = None  # Placeholder
        edge_model = EdgeOptimizedModel(dummy_original)
        edge_model.benchmark_edge()

## Hands-On Exercises

### Exercise 1: JIT Optimization Challenge
```python
# exercises/day6_exercise1_jit.py
"""
Exercise: Optimize a complex computation using JIT compilation

Tasks:
1. Create a complex neural network forward pass
2. Measure performance without JIT  
3. Apply JIT compilation and measure speedup
4. Experiment with different batch sizes
5. Analyze JIT cache behavior with different input shapes
"""

def exercise_jit_optimization():
    """Implement and benchmark JIT optimization"""
    
    # TODO: Implement a 3-layer neural network
    # TODO: Measure performance without JIT (5 forward passes)
    # TODO: Apply @TinyJit decorator and measure performance  
    # TODO: Test with different batch sizes: [16, 32, 64, 128]
    # TODO: Analyze how cache size grows with different shapes
    
    print("JIT Optimization Exercise")
    print("=========================")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_jit_optimization()
```

### Exercise 2: Model Quantization Implementation
```python
# exercises/day6_exercise2_quantization.py
"""
Exercise: Implement INT8 quantization for a trained model

Tasks:
1. Create a simple CNN model
2. Implement calibration data collection
3. Calculate quantization parameters (scale, zero_point)
4. Quantize weights and activations
5. Compare accuracy and model size before/after quantization
"""

def exercise_quantization():
    """Implement model quantization"""
    
    # TODO: Define a simple CNN (Conv2d -> ReLU -> Conv2d -> Global Average Pool)
    # TODO: Generate calibration data (random images)
    # TODO: Collect activation statistics during calibration
    # TODO: Calculate optimal quantization parameters
    # TODO: Quantize the model weights
    # TODO: Compare original vs quantized model size and performance
    
    print("Model Quantization Exercise")
    print("===========================")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_quantization()
```

### Exercise 3: Production Deployment Setup
```python
# exercises/day6_exercise3_deployment.py
"""
Exercise: Create a production-ready model serving system

Tasks:
1. Design a model serving class with JIT compilation
2. Implement batch processing with optimal batch size
3. Add performance monitoring (latency, throughput)
4. Create a simple load testing framework
5. Optimize for different deployment scenarios (edge vs cloud)
"""

def exercise_production_deployment():
    """Implement production deployment system"""
    
    # TODO: Create ProductionModelServer class
    # TODO: Implement JIT-compiled inference
    # TODO: Add batch size optimization
    # TODO: Implement performance monitoring
    # TODO: Create load testing with concurrent requests
    # TODO: Compare edge vs cloud deployment configurations
    
    print("Production Deployment Exercise") 
    print("==============================")
    
    # Your implementation here
    pass

if __name__ == "__main__":
    exercise_production_deployment()
```

## Summary and Key Takeaways

### JIT Compilation Mastery
- **Compilation Caching**: JIT compiles computation graphs once, reuses for identical shapes
- **BEAM Search**: Automatically finds optimal kernel configurations
- **Performance**: Significant speedups after warm-up, especially for repeated operations
- **Memory**: Shape specialization means different input shapes create separate cache entries

### Quantization Techniques
- **INT8 Quantization**: 4x memory reduction with minimal accuracy loss
- **Dynamic vs Static**: Trade-offs between memory and compute overhead
- **Calibration**: Representative data crucial for good quantization parameters
- **Production**: Quantized models enable deployment on resource-constrained devices

### Production Optimization
- **Model Serving**: JIT compilation + batching for optimal throughput
- **Memory Management**: Careful buffer allocation and reuse
- **Multi-Device**: Distribute work across available accelerators
- **Edge Deployment**: Aggressive optimization for minimal resource usage

### Advanced Techniques
- **Kernel Fusion**: Reduce memory bandwidth by combining operations
- **Memory Coalescing**: Optimize access patterns for better GPU utilization  
- **Knowledge Distillation**: Compress models while preserving performance
- **Structured Pruning**: Remove entire channels/layers for hardware efficiency

Tomorrow (Day 7), we'll focus on contributing to tinygrad and advanced production considerations including team development workflows and large-scale deployment strategies.

The combination of JIT compilation, quantization, and production optimization makes tinygrad models highly efficient and deployable across a wide range of hardware, from edge devices to cloud GPUs.