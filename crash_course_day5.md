# Day 5: Neural Networks & High-Level APIs

## Overview
Today we master tinygrad's neural network ecosystem - from low-level layer implementations to complete training systems. You'll understand autograd, implement custom layers, build training pipelines, and create production-ready neural network architectures. By the end, you'll be able to design and train complex models efficiently.

## Learning Objectives
- ✅ Master neural network layer implementations and architectures
- ✅ Understand autograd system and gradient computation
- ✅ Build custom optimizers and training loops
- ✅ Implement advanced neural network techniques
- ✅ Create production-ready model architectures
- ✅ Optimize neural network performance

---

## Part 1: Neural Network Layers Deep Dive (120 minutes)

### Understanding Layer Architecture and Implementation

Neural network layers in tinygrad are built on the tensor system with careful attention to memory efficiency and gradient flow:

```python
#!/usr/bin/env python3
"""
Day 5 Exercise: Deep dive into neural network layers and architectures
"""

import os
import time
import math
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

from tinygrad import Tensor
from tinygrad.nn import Linear, Conv2d, BatchNorm, optim
from tinygrad.helpers import DEBUG, getenv

@dataclass
class LayerAnalysis:
    """Analysis results for neural network layers"""
    layer_name: str
    parameter_count: int
    memory_usage: int
    forward_time: float
    backward_time: float
    gradient_flow_quality: float
    optimization_opportunities: List[str]

class NeuralNetworkExplorer:
    """
    Advanced neural network layer analysis and implementation
    """
    
    def __init__(self):
        self.layer_registry = {}
        self.training_history = []
        self.custom_layers = {}
    
    def explore_fundamental_layers(self):
        """
        Deep exploration of fundamental neural network layers
        """
        print("=== Fundamental Neural Network Layers ===\\n")
        
        # Test different fundamental layers
        layer_tests = [
            ("Linear Layer", self._test_linear_layer),
            ("Convolutional Layer", self._test_conv_layer),
            ("Batch Normalization", self._test_batch_norm),
            ("Activation Functions", self._test_activations),
            ("Dropout Layer", self._test_dropout),
        ]
        
        layer_analyses = {}
        
        for layer_name, test_function in layer_tests:
            print(f"\\n{layer_name} Analysis:")
            print("-" * 40)
            
            try:
                analysis = test_function()
                layer_analyses[layer_name] = analysis
                
                # Print analysis summary
                print(f"  Parameters: {analysis.parameter_count:,}")
                print(f"  Memory usage: {analysis.memory_usage / (1024*1024):.2f} MB")
                print(f"  Forward time: {analysis.forward_time*1000:.2f}ms")
                print(f"  Backward time: {analysis.backward_time*1000:.2f}ms")
                print(f"  Gradient quality: {analysis.gradient_flow_quality:.2f}")
                
                if analysis.optimization_opportunities:
                    print(f"  Optimizations:")
                    for opt in analysis.optimization_opportunities:
                        print(f"    - {opt}")
                
            except Exception as e:
                print(f"  Error: {e}")
                layer_analyses[layer_name] = None
        
        return layer_analyses
    
    def _test_linear_layer(self) -> LayerAnalysis:
        """Deep analysis of Linear (fully connected) layer"""
        print("  Testing Linear Layer implementation...")
        
        # Create linear layer
        input_size = 784  # MNIST-like
        hidden_size = 256
        batch_size = 32
        
        layer = Linear(input_size, hidden_size)
        
        # Count parameters
        param_count = input_size * hidden_size + hidden_size  # weights + bias
        
        # Create test input
        x = Tensor.randn(batch_size, input_size, requires_grad=True)
        
        # Forward pass timing
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass timing
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Memory usage estimation
        memory_usage = param_count * 4 + batch_size * hidden_size * 4  # 4 bytes per float
        
        # Gradient flow analysis
        input_grad_norm = x.grad.abs().mean().numpy() if x.grad is not None else 0
        weight_grad_norm = layer.weight.grad.abs().mean().numpy() if layer.weight.grad is not None else 0
        gradient_quality = min(1.0, (input_grad_norm + weight_grad_norm) / 2)
        
        # Optimization opportunities
        optimizations = []
        if param_count > 100000:
            optimizations.append("Consider weight quantization for large layers")
        if forward_time > 0.01:
            optimizations.append("Optimize matrix multiplication with better backends")
        
        return LayerAnalysis(
            layer_name="Linear",
            parameter_count=param_count,
            memory_usage=memory_usage,
            forward_time=forward_time,
            backward_time=backward_time,
            gradient_flow_quality=gradient_quality,
            optimization_opportunities=optimizations
        )
    
    def _test_conv_layer(self) -> LayerAnalysis:
        """Deep analysis of Convolutional layer"""
        print("  Testing Convolutional Layer implementation...")
        
        # Create conv layer
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        input_height = input_width = 32  # CIFAR-like
        batch_size = 16
        
        layer = Conv2d(in_channels, out_channels, kernel_size, padding=1)
        
        # Count parameters
        param_count = in_channels * out_channels * kernel_size * kernel_size + out_channels
        
        # Create test input
        x = Tensor.randn(batch_size, in_channels, input_height, input_width, requires_grad=True)
        
        # Forward pass timing
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass timing
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Memory usage estimation
        output_size = batch_size * out_channels * input_height * input_width
        memory_usage = param_count * 4 + output_size * 4
        
        # Gradient flow analysis
        input_grad_norm = x.grad.abs().mean().numpy() if x.grad is not None else 0
        weight_grad_norm = layer.weight.grad.abs().mean().numpy() if layer.weight.grad is not None else 0
        gradient_quality = min(1.0, (input_grad_norm + weight_grad_norm) / 2)
        
        # Optimization opportunities
        optimizations = []
        if kernel_size > 5:
            optimizations.append("Consider depthwise separable convolutions")
        if out_channels > 512:
            optimizations.append("Consider channel pruning")
        optimizations.append("Optimize convolution with im2col or Winograd algorithm")
        
        return LayerAnalysis(
            layer_name="Conv2d",
            parameter_count=param_count,
            memory_usage=memory_usage,
            forward_time=forward_time,
            backward_time=backward_time,
            gradient_flow_quality=gradient_quality,
            optimization_opportunities=optimizations
        )
    
    def _test_batch_norm(self) -> LayerAnalysis:
        """Deep analysis of Batch Normalization layer"""
        print("  Testing Batch Normalization implementation...")
        
        # Create batch norm layer
        num_features = 64
        batch_size = 16
        height = width = 32
        
        layer = BatchNorm(num_features)
        
        # Count parameters
        param_count = num_features * 2  # weight and bias
        
        # Create test input (4D for conv output)
        x = Tensor.randn(batch_size, num_features, height, width, requires_grad=True)
        
        # Forward pass timing
        Tensor.training = True  # Enable training mode for BN
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass timing
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Memory usage estimation
        memory_usage = param_count * 4 + batch_size * num_features * height * width * 4
        
        # Gradient flow analysis (BN should maintain gradient flow)
        input_grad_norm = x.grad.abs().mean().numpy() if x.grad is not None else 0
        weight_grad_norm = layer.weight.grad.abs().mean().numpy() if layer.weight and layer.weight.grad is not None else 0
        gradient_quality = min(1.0, (input_grad_norm + weight_grad_norm) / 2)
        
        optimizations = [
            "Fuse batch norm with preceding convolution",
            "Use running statistics during inference",
            "Consider layer normalization for small batches"
        ]
        
        return LayerAnalysis(
            layer_name="BatchNorm",
            parameter_count=param_count,
            memory_usage=memory_usage,
            forward_time=forward_time,
            backward_time=backward_time,
            gradient_flow_quality=gradient_quality,
            optimization_opportunities=optimizations
        )
    
    def _test_activations(self) -> LayerAnalysis:
        """Test various activation functions"""
        print("  Testing Activation Functions...")
        
        batch_size = 32
        features = 512
        
        # Test different activation functions
        x = Tensor.randn(batch_size, features, requires_grad=True)
        
        activations = [
            ("ReLU", lambda t: t.relu()),
            ("Sigmoid", lambda t: t.sigmoid()),
            ("Tanh", lambda t: t.tanh()),
            ("LeakyReLU", lambda t: t.leakyrelu()),
            ("GELU", lambda t: t * 0.5 * (1 + ((t * 0.7978845608) * (1 + 0.044715 * t * t)).tanh())),  # GELU approximation
        ]
        
        total_forward_time = 0
        total_backward_time = 0
        gradient_qualities = []
        
        for act_name, activation in activations:
            # Forward timing
            start_time = time.time()
            output = activation(x)
            forward_time = time.time() - start_time
            total_forward_time += forward_time
            
            # Backward timing
            loss = output.sum()
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            total_backward_time += backward_time
            
            # Gradient analysis
            if x.grad is not None:
                grad_norm = x.grad.abs().mean().numpy()
                gradient_qualities.append(grad_norm)
                x.grad = None  # Reset for next activation
            
            print(f"    {act_name}: forward {forward_time*1000:.3f}ms, backward {backward_time*1000:.3f}ms")
        
        avg_gradient_quality = np.mean(gradient_qualities) if gradient_qualities else 0
        
        optimizations = [
            "Use fused activation functions when available",
            "Consider ReLU for better gradient flow",
            "Use GELU/Swish for transformer architectures"
        ]
        
        return LayerAnalysis(
            layer_name="Activations",
            parameter_count=0,  # No parameters
            memory_usage=batch_size * features * 4,
            forward_time=total_forward_time / len(activations),
            backward_time=total_backward_time / len(activations),
            gradient_flow_quality=avg_gradient_quality,
            optimization_opportunities=optimizations
        )
    
    def _test_dropout(self) -> LayerAnalysis:
        """Test Dropout layer implementation"""
        print("  Testing Dropout implementation...")
        
        batch_size = 32
        features = 256
        dropout_rate = 0.5
        
        x = Tensor.randn(batch_size, features, requires_grad=True)
        
        # Enable training mode
        Tensor.training = True
        
        # Forward pass with dropout
        start_time = time.time()
        output = x.dropout(dropout_rate)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Gradient analysis
        gradient_quality = x.grad.abs().mean().numpy() if x.grad is not None else 0
        
        optimizations = [
            "Disable dropout during inference",
            "Use structured dropout for better regularization",
            "Consider DropBlock for convolutional layers"
        ]
        
        return LayerAnalysis(
            layer_name="Dropout",
            parameter_count=0,
            memory_usage=batch_size * features * 4,
            forward_time=forward_time,
            backward_time=backward_time,
            gradient_flow_quality=gradient_quality,
            optimization_opportunities=optimizations
        )
    
    def implement_custom_layers(self):
        """
        Implement advanced custom neural network layers
        """
        print("\\n=== Custom Layer Implementation ===\\n")
        
        # Implement custom layers
        custom_layers = {
            "MultiHeadAttention": self._implement_attention_layer,
            "ResidualBlock": self._implement_residual_block,
            "DepthwiseSeparableConv": self._implement_depthwise_conv,
            "LayerNormalization": self._implement_layer_norm,
            "PositionalEncoding": self._implement_positional_encoding,
        }
        
        implemented_layers = {}
        
        for layer_name, implementation_func in custom_layers.items():
            print(f"Implementing {layer_name}:")
            print("-" * 30)
            
            try:
                layer_class, test_results = implementation_func()
                implemented_layers[layer_name] = (layer_class, test_results)
                
                print(f"  ✅ {layer_name} implemented successfully")
                print(f"  Test forward time: {test_results['forward_time']*1000:.2f}ms")
                print(f"  Test backward time: {test_results['backward_time']*1000:.2f}ms")
                print(f"  Parameters: {test_results['parameters']:,}")
                
            except Exception as e:
                print(f"  ❌ {layer_name} implementation failed: {e}")
                implemented_layers[layer_name] = None
        
        return implemented_layers
    
    def _implement_attention_layer(self):
        """Implement Multi-Head Self-Attention layer"""
        class MultiHeadAttention:
            def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
                self.d_model = d_model
                self.num_heads = num_heads
                self.d_k = d_model // num_heads
                
                assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
                
                # Linear projections for Q, K, V
                self.w_q = Linear(d_model, d_model)
                self.w_k = Linear(d_model, d_model)
                self.w_v = Linear(d_model, d_model)
                self.w_o = Linear(d_model, d_model)
                
                self.dropout = dropout
                
            def __call__(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
                batch_size, seq_len, d_model = x.shape
                
                # Linear projections and reshape for multi-head
                Q = self.w_q(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                K = self.w_k(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                V = self.w_v(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
                
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                attn_weights = scores.softmax(dim=-1)
                if Tensor.training and self.dropout > 0:
                    attn_weights = attn_weights.dropout(self.dropout)
                
                # Apply attention to values
                attn_output = attn_weights @ V
                
                # Concatenate heads and put through final linear layer
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
                output = self.w_o(attn_output)
                
                return output
        
        # Test the implementation
        d_model = 256
        num_heads = 8
        seq_len = 32
        batch_size = 4
        
        layer = MultiHeadAttention(d_model, num_heads)
        x = Tensor.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Count parameters
        parameters = 4 * d_model * d_model  # Q, K, V, O projections
        
        test_results = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'parameters': parameters,
            'output_shape': output.shape
        }
        
        return MultiHeadAttention, test_results
    
    def _implement_residual_block(self):
        """Implement Residual Block (ResNet-style)"""
        class ResidualBlock:
            def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
                self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
                self.bn1 = BatchNorm(out_channels)
                self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)
                self.bn2 = BatchNorm(out_channels)
                
                # Shortcut connection
                self.shortcut = None
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = Conv2d(in_channels, out_channels, 1, stride=stride)
                    self.shortcut_bn = BatchNorm(out_channels)
                
            def __call__(self, x: Tensor) -> Tensor:
                # Main path
                out = self.conv1(x)
                out = self.bn1(out)
                out = out.relu()
                out = self.conv2(out)
                out = self.bn2(out)
                
                # Shortcut path
                if self.shortcut is not None:
                    shortcut = self.shortcut(x)
                    shortcut = self.shortcut_bn(shortcut)
                else:
                    shortcut = x
                
                # Add and activate
                out = out + shortcut
                out = out.relu()
                
                return out
        
        # Test the implementation
        in_channels = 64
        out_channels = 128
        batch_size = 8
        height = width = 32
        
        layer = ResidualBlock(in_channels, out_channels, stride=2)
        x = Tensor.randn(batch_size, in_channels, height, width, requires_grad=True)
        
        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Count parameters (approximate)
        parameters = (in_channels * out_channels * 9 +  # conv1
                     out_channels * 2 +                 # bn1
                     out_channels * out_channels * 9 +  # conv2
                     out_channels * 2 +                 # bn2
                     in_channels * out_channels +       # shortcut conv
                     out_channels * 2)                  # shortcut bn
        
        test_results = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'parameters': parameters,
            'output_shape': output.shape
        }
        
        return ResidualBlock, test_results
    
    def _implement_depthwise_conv(self):
        """Implement Depthwise Separable Convolution"""
        class DepthwiseSeparableConv:
            def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
                # Depthwise convolution
                self.depthwise = Conv2d(in_channels, in_channels, kernel_size, 
                                      stride=stride, padding=padding, groups=in_channels)
                
                # Pointwise convolution (1x1)
                self.pointwise = Conv2d(in_channels, out_channels, 1)
                
                self.bn1 = BatchNorm(in_channels)
                self.bn2 = BatchNorm(out_channels)
            
            def __call__(self, x: Tensor) -> Tensor:
                # Depthwise convolution
                out = self.depthwise(x)
                out = self.bn1(out)
                out = out.relu()
                
                # Pointwise convolution
                out = self.pointwise(out)
                out = self.bn2(out)
                out = out.relu()
                
                return out
        
        # Test implementation
        in_channels = 32
        out_channels = 64
        batch_size = 4
        height = width = 28
        
        layer = DepthwiseSeparableConv(in_channels, out_channels)
        x = Tensor.randn(batch_size, in_channels, height, width, requires_grad=True)
        
        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        # Parameters (much fewer than standard conv)
        parameters = (in_channels * 9 +           # depthwise 3x3
                     in_channels * out_channels + # pointwise 1x1
                     in_channels * 2 +            # bn1
                     out_channels * 2)            # bn2
        
        test_results = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'parameters': parameters,
            'output_shape': output.shape
        }
        
        return DepthwiseSeparableConv, test_results
    
    def _implement_layer_norm(self):
        """Implement Layer Normalization"""
        class LayerNormalization:
            def __init__(self, normalized_shape: int, eps: float = 1e-5):
                self.eps = eps
                self.weight = Tensor.ones(normalized_shape)
                self.bias = Tensor.zeros(normalized_shape)
                
            def __call__(self, x: Tensor) -> Tensor:
                # Calculate mean and variance along the last dimension
                mean = x.mean(axis=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
                
                # Normalize
                x_norm = (x - mean) / (var + self.eps).sqrt()
                
                # Scale and shift
                return x_norm * self.weight + self.bias
        
        # Test implementation
        batch_size = 8
        seq_len = 64
        d_model = 256
        
        layer = LayerNormalization(d_model)
        x = Tensor.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        parameters = d_model * 2  # weight and bias
        
        test_results = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'parameters': parameters,
            'output_shape': output.shape
        }
        
        return LayerNormalization, test_results
    
    def _implement_positional_encoding(self):
        """Implement Positional Encoding for Transformers"""
        class PositionalEncoding:
            def __init__(self, d_model: int, max_len: int = 5000):
                pe = Tensor.zeros(max_len, d_model)
                position = Tensor.arange(0, max_len).unsqueeze(1).float()
                
                div_term = (Tensor.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model)).exp()
                
                pe[:, 0::2] = (position * div_term).sin()
                pe[:, 1::2] = (position * div_term).cos()
                
                self.pe = pe.unsqueeze(0)  # Add batch dimension
                
            def __call__(self, x: Tensor) -> Tensor:
                # x shape: (batch_size, seq_len, d_model)
                seq_len = x.shape[1]
                return x + self.pe[:, :seq_len]
        
        # Test implementation
        batch_size = 4
        seq_len = 128
        d_model = 256
        
        layer = PositionalEncoding(d_model)
        x = Tensor.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        # Forward pass
        start_time = time.time()
        output = layer(x)
        forward_time = time.time() - start_time
        
        # Backward pass
        loss = output.sum()
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        parameters = 0  # No learnable parameters
        
        test_results = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'parameters': parameters,
            'output_shape': output.shape
        }
        
        return PositionalEncoding, test_results

def explore_model_architectures():
    """
    Explore and implement complete neural network architectures
    """
    print("\\n=== Complete Model Architecture Implementation ===\\n")
    
    class MiniTransformer:
        """Simplified Transformer model for demonstration"""
        def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_len: int = 1000):
            self.d_model = d_model
            self.embedding = Linear(vocab_size, d_model)  # Simplified embedding
            self.pos_encoding = PositionalEncoding(d_model, max_len)
            
            # Transformer layers
            self.layers = []
            for _ in range(num_layers):
                layer = {
                    'attention': MultiHeadAttention(d_model, num_heads),
                    'norm1': LayerNormalization(d_model),
                    'feed_forward': [
                        Linear(d_model, d_model * 4),
                        Linear(d_model * 4, d_model)
                    ],
                    'norm2': LayerNormalization(d_model)
                }
                self.layers.append(layer)
            
            self.output_projection = Linear(d_model, vocab_size)
            
        def __call__(self, x: Tensor) -> Tensor:
            # Embedding and positional encoding
            x = self.embedding(x)
            x = self.pos_encoding(x)
            
            # Transformer layers
            for layer in self.layers:
                # Self-attention with residual connection
                attn_output = layer['attention'](x)
                x = layer['norm1'](x + attn_output)
                
                # Feed-forward with residual connection
                ff_output = layer['feed_forward'][0](x).relu()
                ff_output = layer['feed_forward'][1](ff_output)
                x = layer['norm2'](x + ff_output)
            
            # Output projection
            return self.output_projection(x)
    
    class MiniCNN:
        """Simplified CNN for image classification"""
        def __init__(self, num_classes: int = 10):
            # Feature extraction layers
            self.conv_layers = [
                {'conv': Conv2d(3, 32, 3, padding=1), 'bn': BatchNorm(32)},
                {'conv': Conv2d(32, 64, 3, padding=1), 'bn': BatchNorm(64)},
                {'conv': Conv2d(64, 128, 3, padding=1), 'bn': BatchNorm(128)},
            ]
            
            # Classification layers
            self.classifier = [
                Linear(128 * 4 * 4, 256),  # Assuming 32x32 input -> 4x4 after pooling
                Linear(256, num_classes)
            ]
            
        def __call__(self, x: Tensor) -> Tensor:
            # Convolutional layers
            for layer in self.conv_layers:
                x = layer['conv'](x)
                x = layer['bn'](x)
                x = x.relu()
                x = x.max_pool2d(kernel_size=2, stride=2)  # Downsample
            
            # Flatten for classifier
            x = x.reshape(x.shape[0], -1)
            
            # Classification layers
            x = self.classifier[0](x).relu()
            x = self.classifier[1](x)
            
            return x
    
    # Test model architectures
    models_to_test = [
        ("Mini Transformer", lambda: MiniTransformer(vocab_size=1000, d_model=256, num_heads=8, num_layers=2)),
        ("Mini CNN", lambda: MiniCNN(num_classes=10)),
    ]
    
    for model_name, model_constructor in models_to_test:
        print(f"Testing {model_name}:")
        print("-" * 30)
        
        try:
            model = model_constructor()
            
            # Create appropriate test input
            if "Transformer" in model_name:
                batch_size, seq_len, vocab_size = 2, 32, 1000
                # Create one-hot encoded input (simplified)
                test_input = Tensor.zeros(batch_size, seq_len, vocab_size)
                # Set random positions to 1
                for b in range(batch_size):
                    for s in range(seq_len):
                        idx = np.random.randint(0, vocab_size)
                        test_input.data[b, s, idx] = 1.0
                test_input.requires_grad = True
            else:  # CNN
                batch_size = 4
                test_input = Tensor.randn(batch_size, 3, 32, 32, requires_grad=True)
            
            # Forward pass
            start_time = time.time()
            output = model(test_input)
            forward_time = time.time() - start_time
            
            # Backward pass
            loss = output.sum()
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            
            print(f"  ✅ {model_name} working correctly")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Forward time: {forward_time*1000:.2f}ms")
            print(f"  Backward time: {backward_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
        
        print()

if __name__ == "__main__":
    print("Day 5: Neural Networks & High-Level APIs")
    print("=" * 50)
    
    # Set training mode
    Tensor.training = True
    
    explorer = NeuralNetworkExplorer()
    
    # Explore fundamental layers
    print("1. Fundamental Layer Analysis:")
    layer_analyses = explorer.explore_fundamental_layers()
    
    # Implement custom layers
    print("\\n" + "="*60)
    print("2. Custom Layer Implementation:")
    custom_layers = explorer.implement_custom_layers()
    
    # Explore complete architectures
    print("\\n" + "="*60)
    print("3. Complete Model Architectures:")
    explore_model_architectures()
    
    print("\\n" + "="*50)
    print("Neural Network Layer Analysis Complete!")
```

---

## Part 2: Autograd and Training Systems (90 minutes)

### Understanding Automatic Differentiation

```python
#!/usr/bin/env python3
"""
autograd_training.py - Deep dive into automatic differentiation and training systems
"""

import os
import time
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

from tinygrad import Tensor
from tinygrad.nn import Linear, optim
from tinygrad.helpers import DEBUG

@dataclass
class TrainingMetrics:
    """Comprehensive training metrics"""
    epoch: int
    loss: float
    accuracy: float
    gradient_norm: float
    parameter_norm: float
    learning_rate: float
    batch_time: float
    memory_usage: int

class AutogradTrainingExplorer:
    """
    Advanced exploration of autograd system and training pipelines
    """
    
    def __init__(self):
        self.training_history = []
        self.gradient_analysis = {}
        self.optimization_strategies = {}
    
    def deep_dive_autograd(self):
        """
        Deep exploration of the automatic differentiation system
        """
        print("=== Automatic Differentiation Deep Dive ===\\n")
        
        print("1. Gradient Computation Mechanics:")
        print("-" * 40)
        
        # Create a computational graph to analyze
        x = Tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        y = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        
        print(f"Input tensors:")
        print(f"  x: {x.numpy()}")
        print(f"  y: {y.numpy()}")
        print(f"  x.requires_grad: {x.requires_grad}")
        print(f"  y.requires_grad: {y.requires_grad}")
        
        # Build computational graph
        z1 = x * y          # Element-wise multiplication
        z2 = z1.sum()       # Reduction
        z3 = z2.log()       # Logarithm
        z4 = z3 ** 2        # Power
        loss = z4           # Final loss
        
        print(f"\\nComputational graph:")
        print(f"  z1 = x * y = {z1.numpy()}")
        print(f"  z2 = sum(z1) = {z2.numpy()}")
        print(f"  z3 = log(z2) = {z3.numpy()}")
        print(f"  loss = z3^2 = {loss.numpy()}")
        
        # Perform backward pass
        print(f"\\n2. Backward Pass Analysis:")
        print("-" * 30)
        
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        print(f"Backward pass completed in {backward_time*1000:.3f}ms")
        print(f"\\nGradients computed:")
        print(f"  dx = {x.grad.numpy()}")
        print(f"  dy = {y.grad.numpy()}")
        
        # Verify gradients numerically
        print(f"\\n3. Gradient Verification:")
        print("-" * 25)
        
        def verify_gradient(tensor, grad, epsilon=1e-5):
            """Verify gradients using finite differences"""
            tensor_np = tensor.detach().numpy()
            grad_numerical = np.zeros_like(tensor_np)
            
            for i in range(tensor_np.shape[0]):
                for j in range(tensor_np.shape[1]):
                    # Forward difference
                    tensor_np[i, j] += epsilon
                    tensor_plus = Tensor(tensor_np.copy(), requires_grad=True)
                    if tensor is x:
                        loss_plus = ((tensor_plus * y).sum().log() ** 2)
                    else:
                        loss_plus = ((x * tensor_plus).sum().log() ** 2)
                    
                    tensor_np[i, j] -= 2 * epsilon
                    tensor_minus = Tensor(tensor_np.copy(), requires_grad=True)
                    if tensor is x:
                        loss_minus = ((tensor_minus * y).sum().log() ** 2)
                    else:
                        loss_minus = ((x * tensor_minus).sum().log() ** 2)
                    
                    grad_numerical[i, j] = (loss_plus.numpy() - loss_minus.numpy()) / (2 * epsilon)
                    tensor_np[i, j] += epsilon  # Reset
            
            return grad_numerical
        
        # Verify x gradient
        x_grad_numerical = verify_gradient(x, x.grad)
        x_grad_error = np.abs(x.grad.numpy() - x_grad_numerical).max()
        print(f"  x gradient verification:")
        print(f"    Analytical: {x.grad.numpy()}")
        print(f"    Numerical:  {x_grad_numerical}")
        print(f"    Max error:  {x_grad_error:.2e}")
        
        # Verify y gradient
        y_grad_numerical = verify_gradient(y, y.grad)
        y_grad_error = np.abs(y.grad.numpy() - y_grad_numerical).max()
        print(f"  y gradient verification:")
        print(f"    Analytical: {y.grad.numpy()}")
        print(f"    Numerical:  {y_grad_numerical}")
        print(f"    Max error:  {y_grad_error:.2e}")
        
        return {
            'backward_time': backward_time,
            'x_grad_error': x_grad_error,
            'y_grad_error': y_grad_error
        }
    
    def implement_custom_optimizers(self):
        """
        Implement and compare custom optimization algorithms
        """
        print("\\n=== Custom Optimizer Implementation ===\\n")
        
        # Base optimizer class for our implementations
        class CustomOptimizer:
            def __init__(self, params: List[Tensor], lr: float):
                self.params = [p for p in params if p.requires_grad]
                self.lr = lr
                self.state = {}
            
            def zero_grad(self):
                for p in self.params:
                    p.grad = None
            
            def step(self):
                raise NotImplementedError
        
        class SGDOptimizer(CustomOptimizer):
            def __init__(self, params: List[Tensor], lr: float, momentum: float = 0.0):
                super().__init__(params, lr)
                self.momentum = momentum
                
            def step(self):
                for p in self.params:
                    if p.grad is None:
                        continue
                    
                    # Initialize momentum buffer
                    if p not in self.state:
                        self.state[p] = {'momentum_buffer': Tensor.zeros(*p.shape)}
                    
                    buf = self.state[p]['momentum_buffer']
                    
                    # Apply momentum
                    buf.assign(self.momentum * buf + p.grad)
                    
                    # Update parameters
                    p.assign(p - self.lr * buf)
        
        class AdamOptimizer(CustomOptimizer):
            def __init__(self, params: List[Tensor], lr: float = 0.001, 
                        beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
                super().__init__(params, lr)
                self.beta1 = beta1
                self.beta2 = beta2
                self.eps = eps
                self.step_count = 0
                
            def step(self):
                self.step_count += 1
                
                for p in self.params:
                    if p.grad is None:
                        continue
                    
                    # Initialize state
                    if p not in self.state:
                        self.state[p] = {
                            'm': Tensor.zeros(*p.shape),  # First moment
                            'v': Tensor.zeros(*p.shape),  # Second moment
                        }
                    
                    m = self.state[p]['m']
                    v = self.state[p]['v']
                    
                    # Update biased first moment estimate
                    m.assign(self.beta1 * m + (1 - self.beta1) * p.grad)
                    
                    # Update biased second raw moment estimate
                    v.assign(self.beta2 * v + (1 - self.beta2) * (p.grad ** 2))
                    
                    # Compute bias-corrected first moment estimate
                    m_hat = m / (1 - self.beta1 ** self.step_count)
                    
                    # Compute bias-corrected second raw moment estimate
                    v_hat = v / (1 - self.beta2 ** self.step_count)
                    
                    # Update parameters
                    p.assign(p - self.lr * m_hat / (v_hat.sqrt() + self.eps))
        
        class RMSpropOptimizer(CustomOptimizer):
            def __init__(self, params: List[Tensor], lr: float = 0.01, 
                        alpha: float = 0.99, eps: float = 1e-8):
                super().__init__(params, lr)
                self.alpha = alpha
                self.eps = eps
                
            def step(self):
                for p in self.params:
                    if p.grad is None:
                        continue
                    
                    # Initialize state
                    if p not in self.state:
                        self.state[p] = {'square_avg': Tensor.zeros(*p.shape)}
                    
                    square_avg = self.state[p]['square_avg']
                    
                    # Update exponential average of squared gradients
                    square_avg.assign(self.alpha * square_avg + (1 - self.alpha) * (p.grad ** 2))
                    
                    # Update parameters
                    p.assign(p - self.lr * p.grad / (square_avg.sqrt() + self.eps))
        
        # Test optimizers on a simple problem
        print("Testing custom optimizers on quadratic function:")
        print("-" * 50)
        
        def test_optimizer(optimizer_class, optimizer_name, **kwargs):
            # Simple quadratic function: f(x) = x^2 + 2x + 1
            x = Tensor([5.0], requires_grad=True)  # Start from x=5
            optimizer = optimizer_class([x], **kwargs)
            
            losses = []
            positions = []
            
            print(f"\\n{optimizer_name}:")
            
            for step in range(50):
                optimizer.zero_grad()
                
                # Compute loss: f(x) = x^2 + 2x + 1
                loss = x**2 + 2*x + 1
                loss.backward()
                
                # Record metrics
                losses.append(loss.numpy().item())
                positions.append(x.numpy().item())
                
                optimizer.step()
                
                if step % 10 == 0:
                    print(f"  Step {step:2d}: x = {x.numpy().item():.6f}, loss = {loss.numpy().item():.6f}")
            
            final_x = x.numpy().item()
            final_loss = losses[-1]
            optimal_x = -1.0  # Analytical optimum
            error = abs(final_x - optimal_x)
            
            print(f"  Final: x = {final_x:.6f}, loss = {final_loss:.6f}, error = {error:.6f}")
            
            return {
                'final_x': final_x,
                'final_loss': final_loss,
                'error': error,
                'convergence_steps': len([l for l in losses if l > 0.01]),
                'losses': losses,
                'positions': positions
            }
        
        # Test different optimizers
        optimizer_results = {}
        
        optimizers_to_test = [
            (SGDOptimizer, "SGD", {'lr': 0.1}),
            (SGDOptimizer, "SGD with Momentum", {'lr': 0.1, 'momentum': 0.9}),
            (AdamOptimizer, "Adam", {'lr': 0.1}),
            (RMSpropOptimizer, "RMSprop", {'lr': 0.1}),
        ]
        
        for optimizer_class, name, params in optimizers_to_test:
            result = test_optimizer(optimizer_class, name, **params)
            optimizer_results[name] = result
        
        # Compare results
        print(f"\\nOptimizer Comparison:")
        print(f"{'Optimizer':<20} | {'Final Error':<12} | {'Convergence':<12} | {'Final Loss':<12}")
        print("-" * 65)
        
        for name, result in optimizer_results.items():
            print(f"{name:<20} | {result['error']:<12.6f} | {result['convergence_steps']:<12} | {result['final_loss']:<12.6f}")
        
        return optimizer_results
    
    def implement_advanced_training_loop(self):
        """
        Implement a production-ready training loop with all the bells and whistles
        """
        print("\\n=== Advanced Training Loop Implementation ===\\n")
        
        class AdvancedTrainer:
            def __init__(self, model, optimizer, device='CPU'):
                self.model = model
                self.optimizer = optimizer
                self.device = device
                self.training_history = []
                self.best_loss = float('inf')
                self.patience_counter = 0
                
            def train_epoch(self, dataloader, epoch):
                """Train for one epoch"""
                Tensor.training = True
                epoch_metrics = {
                    'loss': [],
                    'accuracy': [],
                    'gradient_norms': [],
                    'batch_times': []
                }
                
                for batch_idx, (data, targets) in enumerate(dataloader):
                    batch_start = time.time()
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(data)
                    
                    # Compute loss
                    loss = self.compute_loss(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping (optional)
                    grad_norm = self.clip_gradients(max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Compute metrics
                    accuracy = self.compute_accuracy(outputs, targets)
                    batch_time = time.time() - batch_start
                    
                    # Record metrics
                    epoch_metrics['loss'].append(loss.numpy().item())
                    epoch_metrics['accuracy'].append(accuracy)
                    epoch_metrics['gradient_norms'].append(grad_norm)
                    epoch_metrics['batch_times'].append(batch_time)
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f"    Batch {batch_idx}: Loss = {loss.numpy().item():.6f}, "
                              f"Acc = {accuracy:.3f}, Grad Norm = {grad_norm:.6f}")
                
                # Compute epoch averages
                avg_metrics = {
                    'epoch': epoch,
                    'loss': np.mean(epoch_metrics['loss']),
                    'accuracy': np.mean(epoch_metrics['accuracy']),
                    'gradient_norm': np.mean(epoch_metrics['gradient_norms']),
                    'batch_time': np.mean(epoch_metrics['batch_times'])
                }
                
                return avg_metrics
            
            def validate(self, dataloader):
                """Validate the model"""
                Tensor.training = False
                val_metrics = {
                    'loss': [],
                    'accuracy': []
                }
                
                for data, targets in dataloader:
                    outputs = self.model(data)
                    loss = self.compute_loss(outputs, targets)
                    accuracy = self.compute_accuracy(outputs, targets)
                    
                    val_metrics['loss'].append(loss.numpy().item())
                    val_metrics['accuracy'].append(accuracy)
                
                return {
                    'loss': np.mean(val_metrics['loss']),
                    'accuracy': np.mean(val_metrics['accuracy'])
                }
            
            def compute_loss(self, outputs, targets):
                """Compute loss function (cross-entropy for classification)"""
                # Simple MSE for demonstration
                return ((outputs - targets) ** 2).mean()
            
            def compute_accuracy(self, outputs, targets):
                """Compute accuracy metric"""
                # For regression, use R² coefficient
                ss_res = ((targets - outputs) ** 2).sum()
                ss_tot = ((targets - targets.mean()) ** 2).sum()
                r_squared = 1 - (ss_res / ss_tot)
                return max(0, r_squared.numpy().item())  # Clamp to positive
            
            def clip_gradients(self, max_norm=1.0):
                """Clip gradients to prevent explosion"""
                total_norm = 0.0
                
                # Calculate total gradient norm
                for p in self.model.parameters() if hasattr(self.model, 'parameters') else []:
                    if p.grad is not None:
                        param_norm = (p.grad ** 2).sum()
                        total_norm += param_norm
                
                total_norm = total_norm.sqrt()
                
                # Clip gradients if necessary
                if total_norm > max_norm:
                    clip_coef = max_norm / (total_norm + 1e-6)
                    for p in self.model.parameters() if hasattr(self.model, 'parameters') else []:
                        if p.grad is not None:
                            p.grad.assign(p.grad * clip_coef)
                
                return total_norm.numpy().item() if hasattr(total_norm, 'numpy') else total_norm
            
            def early_stopping_check(self, val_loss, patience=10):
                """Check for early stopping"""
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    return False  # Continue training
                else:
                    self.patience_counter += 1
                    return self.patience_counter >= patience
            
            def train(self, train_loader, val_loader=None, epochs=10, patience=10):
                """Complete training loop"""
                print(f"Starting training for {epochs} epochs...")
                
                for epoch in range(epochs):
                    print(f"\\nEpoch {epoch + 1}/{epochs}:")
                    
                    # Train epoch
                    train_metrics = self.train_epoch(train_loader, epoch)
                    self.training_history.append(train_metrics)
                    
                    print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
                          f"Acc: {train_metrics['accuracy']:.3f}, "
                          f"Avg Batch Time: {train_metrics['batch_time']*1000:.2f}ms")
                    
                    # Validation
                    if val_loader is not None:
                        val_metrics = self.validate(val_loader)
                        print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
                              f"Acc: {val_metrics['accuracy']:.3f}")
                        
                        # Early stopping
                        if self.early_stopping_check(val_metrics['loss'], patience):
                            print(f"  Early stopping triggered after {epoch + 1} epochs")
                            break
                
                return self.training_history
        
        # Test the advanced training loop
        print("Testing Advanced Training Loop:")
        print("-" * 35)
        
        # Create a simple regression problem
        class SimpleModel:
            def __init__(self):
                self.linear = Linear(1, 1)
            
            def __call__(self, x):
                return self.linear(x)
        
        # Generate synthetic data: y = 2x + 1 + noise
        np.random.seed(42)
        train_x = np.random.randn(100, 1).astype(np.float32)
        train_y = (2 * train_x + 1 + 0.1 * np.random.randn(100, 1)).astype(np.float32)
        
        val_x = np.random.randn(20, 1).astype(np.float32)
        val_y = (2 * val_x + 1 + 0.1 * np.random.randn(20, 1)).astype(np.float32)
        
        # Create simple dataloaders (lists of batches)
        batch_size = 10
        train_loader = []
        for i in range(0, len(train_x), batch_size):
            batch_x = Tensor(train_x[i:i+batch_size])
            batch_y = Tensor(train_y[i:i+batch_size])
            train_loader.append((batch_x, batch_y))
        
        val_loader = []
        for i in range(0, len(val_x), batch_size):
            batch_x = Tensor(val_x[i:i+batch_size])
            batch_y = Tensor(val_y[i:i+batch_size])
            val_loader.append((batch_x, batch_y))
        
        # Create model and optimizer
        model = SimpleModel()
        optimizer = optim.Adam([model.linear.weight, model.linear.bias], lr=0.01)
        
        # Create trainer
        trainer = AdvancedTrainer(model, optimizer)
        
        # Train the model
        training_history = trainer.train(train_loader, val_loader, epochs=20, patience=5)
        
        # Analyze results
        print(f"\\nTraining completed!")
        print(f"Final model parameters:")
        print(f"  Weight: {model.linear.weight.numpy()}")
        print(f"  Bias: {model.linear.bias.numpy()}")
        print(f"  Expected: Weight ≈ 2.0, Bias ≈ 1.0")
        
        return training_history

if __name__ == "__main__":
    print("Day 5: Autograd and Training Systems")
    print("=" * 45)
    
    explorer = AutogradTrainingExplorer()
    
    # Deep dive into autograd
    print("1. Automatic Differentiation Analysis:")
    autograd_results = explorer.deep_dive_autograd()
    
    # Implement custom optimizers
    print("\\n" + "="*60)
    print("2. Custom Optimizer Implementation:")
    optimizer_results = explorer.implement_custom_optimizers()
    
    # Advanced training loop
    print("\\n" + "="*60)
    print("3. Advanced Training Loop:")
    training_results = explorer.implement_advanced_training_loop()
    
    print("\\n" + "="*45)
    print("Autograd and Training Systems Complete!")
```

---

## Part 3: Model Optimization and Production Deployment (60 minutes)

### Advanced Optimization Techniques

```python
#!/usr/bin/env python3
"""
model_optimization.py - Advanced model optimization and production techniques
"""

import os
import time
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from tinygrad import Tensor
from tinygrad.nn import Linear, Conv2d, BatchNorm
from tinygrad.helpers import DEBUG

@dataclass
class OptimizationProfile:
    """Profile for model optimization results"""
    technique: str
    original_size: int
    optimized_size: int
    compression_ratio: float
    accuracy_retention: float
    inference_speedup: float
    memory_reduction: float

class ModelOptimizationExpert:
    """
    Expert system for model optimization and production deployment
    """
    
    def __init__(self):
        self.optimization_history = []
        self.production_configs = {}
        self.performance_benchmarks = {}
    
    def implement_quantization_techniques(self):
        """
        Implement various quantization techniques for model compression
        """
        print("=== Quantization Techniques Implementation ===\\n")
        
        class QuantizationToolkit:
            @staticmethod
            def quantize_weights_int8(tensor: Tensor, symmetric: bool = True) -> Tuple[Tensor, float, int]:
                """Quantize weights to INT8"""
                if symmetric:
                    # Symmetric quantization: [-127, 127]
                    max_val = tensor.abs().max()
                    scale = max_val / 127.0
                    zero_point = 0
                else:
                    # Asymmetric quantization: [-128, 127]
                    min_val = tensor.min()
                    max_val = tensor.max()
                    scale = (max_val - min_val) / 255.0
                    zero_point = -128 - (min_val / scale).round()
                
                # Quantize
                quantized = ((tensor / scale) + zero_point).round().clip(-128, 127)
                
                return quantized, scale.numpy().item(), int(zero_point.numpy().item()) if hasattr(zero_point, 'numpy') else zero_point
            
            @staticmethod
            def dequantize_weights(quantized: Tensor, scale: float, zero_point: int) -> Tensor:
                """Dequantize INT8 weights back to float"""
                return (quantized - zero_point) * scale
            
            @staticmethod
            def quantize_activations_uint8(tensor: Tensor) -> Tuple[Tensor, float, int]:
                """Quantize activations to UINT8 (typically for ReLU outputs)"""
                min_val = tensor.min()
                max_val = tensor.max()
                
                scale = (max_val - min_val) / 255.0
                zero_point = -min_val / scale
                zero_point = zero_point.round().clip(0, 255)
                
                quantized = ((tensor / scale) + zero_point).round().clip(0, 255)
                
                return quantized, scale.numpy().item(), int(zero_point.numpy().item())
        
        # Test quantization on a sample model
        print("Testing Weight Quantization:")
        print("-" * 30)
        
        # Create a sample linear layer
        layer = Linear(256, 128)
        original_weights = layer.weight.detach()
        
        print(f"Original weights stats:")
        print(f"  Shape: {original_weights.shape}")
        print(f"  Min: {original_weights.min().numpy():.6f}")
        print(f"  Max: {original_weights.max().numpy():.6f}")
        print(f"  Mean: {original_weights.mean().numpy():.6f}")
        print(f"  Std: {original_weights.std().numpy():.6f}")
        
        # Quantize weights
        quant_weights, scale, zero_point = QuantizationToolkit.quantize_weights_int8(original_weights)
        
        print(f"\\nQuantized weights:")
        print(f"  Scale: {scale:.6f}")
        print(f"  Zero point: {zero_point}")
        print(f"  Range: [{quant_weights.min().numpy():.0f}, {quant_weights.max().numpy():.0f}]")
        
        # Dequantize and compare
        dequant_weights = QuantizationToolkit.dequantize_weights(quant_weights, scale, zero_point)
        quantization_error = (original_weights - dequant_weights).abs().mean()
        
        print(f"\\nQuantization error: {quantization_error.numpy():.6f}")
        print(f"Relative error: {(quantization_error / original_weights.abs().mean()).numpy():.4%}")
        
        # Memory savings
        original_size = original_weights.numel() * 4  # 32-bit floats
        quantized_size = quant_weights.numel() * 1 + 8  # 8-bit ints + scale/zero_point
        compression_ratio = original_size / quantized_size
        
        print(f"\\nCompression:")
        print(f"  Original size: {original_size} bytes")
        print(f"  Quantized size: {quantized_size} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return QuantizationToolkit, {
            'compression_ratio': compression_ratio,
            'quantization_error': quantization_error.numpy().item(),
            'relative_error': (quantization_error / original_weights.abs().mean()).numpy().item()
        }
    
    def implement_pruning_techniques(self):
        """
        Implement neural network pruning techniques
        """
        print("\\n=== Pruning Techniques Implementation ===\\n")
        
        class PruningToolkit:
            @staticmethod
            def magnitude_pruning(tensor: Tensor, sparsity: float) -> Tensor:
                """Prune weights based on magnitude"""
                # Calculate threshold for pruning
                abs_tensor = tensor.abs()
                flat_tensor = abs_tensor.reshape(-1)
                threshold_idx = int(flat_tensor.numel() * sparsity)
                
                # Get threshold value (approximate with numpy for now)
                sorted_vals = np.sort(flat_tensor.numpy())
                threshold = sorted_vals[threshold_idx] if threshold_idx < len(sorted_vals) else 0
                
                # Create mask
                mask = (abs_tensor > threshold).float()
                
                return tensor * mask, mask
            
            @staticmethod
            def structured_pruning_channels(conv_layer, importance_scores: Tensor, prune_ratio: float):
                """Prune entire channels based on importance scores"""
                num_channels = importance_scores.numel()
                num_to_prune = int(num_channels * prune_ratio)
                
                # Get indices to prune (lowest importance)
                _, indices = importance_scores.topk(num_channels - num_to_prune, largest=True)
                
                return indices.numpy().astype(int)
            
            @staticmethod
            def gradual_magnitude_pruning(tensor: Tensor, current_sparsity: float, 
                                        target_sparsity: float, step: int, total_steps: int) -> Tensor:
                """Gradually increase pruning over training"""
                # Polynomial sparsity schedule
                sparsity = target_sparsity * (1 - (1 - step / total_steps) ** 3)
                sparsity = min(sparsity, target_sparsity)
                
                return PruningToolkit.magnitude_pruning(tensor, sparsity)
        
        print("Testing Magnitude Pruning:")
        print("-" * 25)
        
        # Create test network
        layer = Linear(512, 256)
        original_weights = layer.weight.detach()
        
        # Test different sparsity levels
        sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for sparsity in sparsity_levels:
            pruned_weights, mask = PruningToolkit.magnitude_pruning(original_weights, sparsity)
            
            # Calculate actual sparsity
            actual_sparsity = (mask == 0).float().mean().numpy()
            
            # Calculate change in weight statistics
            weight_change = (pruned_weights - original_weights).abs().mean().numpy()
            remaining_weights = (mask > 0).float().sum().numpy()
            
            print(f"  Sparsity {sparsity:.1%}: Actual {actual_sparsity:.1%}, "
                  f"Remaining weights: {remaining_weights:.0f}/{mask.numel():.0f}")
        
        print("\\nTesting Gradual Pruning Schedule:")
        print("-" * 30)
        
        total_steps = 100
        target_sparsity = 0.8
        
        for step in [0, 25, 50, 75, 99]:
            pruned_weights, mask = PruningToolkit.gradual_magnitude_pruning(
                original_weights, 0.0, target_sparsity, step, total_steps)
            
            actual_sparsity = (mask == 0).float().mean().numpy()
            print(f"  Step {step:2d}/100: Sparsity = {actual_sparsity:.1%}")
        
        return PruningToolkit
    
    def implement_knowledge_distillation(self):
        """
        Implement knowledge distillation for model compression
        """
        print("\\n=== Knowledge Distillation Implementation ===\\n")
        
        class KnowledgeDistillation:
            def __init__(self, teacher_model, student_model, temperature: float = 3.0, alpha: float = 0.7):
                self.teacher = teacher_model
                self.student = student_model
                self.temperature = temperature
                self.alpha = alpha  # Weight for distillation loss
            
            def distillation_loss(self, student_outputs: Tensor, teacher_outputs: Tensor, 
                                targets: Tensor) -> Tensor:
                """Compute knowledge distillation loss"""
                # Soft targets from teacher
                teacher_soft = (teacher_outputs / self.temperature).softmax(dim=-1)
                student_soft = (student_outputs / self.temperature).log_softmax(dim=-1)
                
                # KL divergence loss (distillation)
                kd_loss = -(teacher_soft * student_soft).sum(dim=-1).mean()
                kd_loss = kd_loss * (self.temperature ** 2)
                
                # Hard target loss (standard cross-entropy)
                hard_loss = ((student_outputs - targets) ** 2).mean()  # MSE for simplicity
                
                # Combined loss
                total_loss = self.alpha * kd_loss + (1 - self.alpha) * hard_loss
                
                return total_loss, kd_loss, hard_loss
            
            def train_student(self, dataloader, optimizer, epochs: int = 10):
                """Train student model with knowledge distillation"""
                self.teacher.training = False  # Teacher in eval mode
                training_history = []
                
                for epoch in range(epochs):
                    epoch_metrics = {'total_loss': [], 'kd_loss': [], 'hard_loss': []}
                    
                    for batch_data, batch_targets in dataloader:
                        optimizer.zero_grad()
                        
                        # Get teacher outputs (no gradients)
                        with Tensor.no_grad():
                            teacher_outputs = self.teacher(batch_data)
                        
                        # Get student outputs
                        student_outputs = self.student(batch_data)
                        
                        # Compute distillation loss
                        total_loss, kd_loss, hard_loss = self.distillation_loss(
                            student_outputs, teacher_outputs, batch_targets)
                        
                        # Backward pass
                        total_loss.backward()
                        optimizer.step()
                        
                        # Record metrics
                        epoch_metrics['total_loss'].append(total_loss.numpy().item())
                        epoch_metrics['kd_loss'].append(kd_loss.numpy().item())
                        epoch_metrics['hard_loss'].append(hard_loss.numpy().item())
                    
                    # Average metrics
                    avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
                    avg_metrics['epoch'] = epoch
                    training_history.append(avg_metrics)
                    
                    print(f"  Epoch {epoch+1}: Total Loss = {avg_metrics['total_loss']:.6f}, "
                          f"KD Loss = {avg_metrics['kd_loss']:.6f}, "
                          f"Hard Loss = {avg_metrics['hard_loss']:.6f}")
                
                return training_history
        
        # Test knowledge distillation
        print("Testing Knowledge Distillation:")
        print("-" * 30)
        
        # Create teacher (large) and student (small) models
        class TeacherModel:
            def __init__(self):
                self.layer1 = Linear(10, 128)
                self.layer2 = Linear(128, 64)
                self.layer3 = Linear(64, 1)
                self.training = True
            
            def __call__(self, x):
                x = self.layer1(x).relu()
                x = self.layer2(x).relu()
                return self.layer3(x)
        
        class StudentModel:
            def __init__(self):
                self.layer1 = Linear(10, 32)
                self.layer2 = Linear(32, 1)
                self.training = True
            
            def __call__(self, x):
                x = self.layer1(x).relu()
                return self.layer2(x)
        
        teacher = TeacherModel()
        student = StudentModel()
        
        # Create synthetic data
        np.random.seed(42)
        train_data = np.random.randn(100, 10).astype(np.float32)
        train_targets = np.random.randn(100, 1).astype(np.float32)
        
        # Create simple dataloader
        batch_size = 10
        dataloader = []
        for i in range(0, len(train_data), batch_size):
            batch_x = Tensor(train_data[i:i+batch_size])
            batch_y = Tensor(train_targets[i:i+batch_size])
            dataloader.append((batch_x, batch_y))
        
        # Initialize distillation
        kd = KnowledgeDistillation(teacher, student, temperature=3.0, alpha=0.7)
        
        # Student optimizer
        student_params = [student.layer1.weight, student.layer1.bias,
                         student.layer2.weight, student.layer2.bias]
        from tinygrad.nn import optim
        optimizer = optim.Adam(student_params, lr=0.001)
        
        # Train student with distillation
        print("\\nTraining student with knowledge distillation:")
        history = kd.train_student(dataloader, optimizer, epochs=5)
        
        # Compare model sizes
        teacher_params = (128*10 + 128) + (64*128 + 64) + (1*64 + 1)
        student_params = (32*10 + 32) + (1*32 + 1)
        compression_ratio = teacher_params / student_params
        
        print(f"\\nModel Comparison:")
        print(f"  Teacher parameters: {teacher_params:,}")
        print(f"  Student parameters: {student_params:,}")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return KnowledgeDistillation, {
            'compression_ratio': compression_ratio,
            'training_history': history
        }
    
    def production_deployment_optimization(self):
        """
        Optimize models for production deployment
        """
        print("\\n=== Production Deployment Optimization ===\\n")
        
        class ProductionOptimizer:
            @staticmethod
            def fuse_batch_norm_conv(conv_layer, bn_layer):
                """Fuse batch normalization into convolution for inference"""
                # Get batch norm parameters
                gamma = bn_layer.weight if bn_layer.weight is not None else Tensor.ones(bn_layer.num_features)
                beta = bn_layer.bias if bn_layer.bias is not None else Tensor.zeros(bn_layer.num_features)
                running_mean = bn_layer.running_mean
                running_var = bn_layer.running_var
                eps = bn_layer.eps
                
                # Compute fused parameters
                std = (running_var + eps).sqrt()
                fused_weight = conv_layer.weight * (gamma / std).reshape(-1, 1, 1, 1)
                fused_bias = beta - running_mean * gamma / std
                
                if conv_layer.bias is not None:
                    fused_bias = fused_bias + conv_layer.bias
                
                return fused_weight, fused_bias
            
            @staticmethod
            def optimize_memory_layout(tensor: Tensor, target_layout: str = "NCHW"):
                """Optimize tensor memory layout for target hardware"""
                # This is a simplified example - real optimization would be more complex
                if target_layout == "NCHW" and len(tensor.shape) == 4:
                    # Already in NCHW format for typical CNN
                    return tensor
                elif target_layout == "NHWC" and len(tensor.shape) == 4:
                    # Convert NCHW to NHWC
                    return tensor.permute(0, 2, 3, 1)
                return tensor
            
            @staticmethod
            def optimize_for_inference(model):
                """Apply various inference optimizations"""
                optimizations = []
                
                # Set to evaluation mode
                if hasattr(model, 'training'):
                    model.training = False
                    optimizations.append("Set evaluation mode")
                
                # Placeholder for other optimizations
                optimizations.extend([
                    "Fuse operations where possible",
                    "Optimize memory access patterns",
                    "Enable operator fusion",
                    "Use optimized kernels"
                ])
                
                return optimizations
        
        print("Testing Production Optimizations:")
        print("-" * 32)
        
        # Test batch norm fusion
        print("\\n1. Batch Norm Fusion:")
        conv = Conv2d(32, 64, 3, padding=1)
        bn = BatchNorm(64)
        
        # Create test input
        x = Tensor.randn(1, 32, 28, 28)
        
        # Original forward pass
        conv_out = conv(x)
        bn_out = bn(conv_out)
        original_time = time.time()
        original_result = bn_out.numpy()
        original_time = time.time() - original_time
        
        # Fused forward pass
        fused_weight, fused_bias = ProductionOptimizer.fuse_batch_norm_conv(conv, bn)
        
        # Create fused layer (conceptually)
        start_time = time.time()
        conv_out_fused = conv(x)  # Would use fused parameters in practice
        fused_time = time.time() - start_time
        
        print(f"  Original conv+bn time: {original_time*1000:.3f}ms")
        print(f"  Fused conv time: {fused_time*1000:.3f}ms")
        print(f"  Fusion reduces operations and memory access")
        
        # Test memory layout optimization
        print("\\n2. Memory Layout Optimization:")
        tensor_nchw = Tensor.randn(4, 64, 32, 32)  # Batch, Channel, Height, Width
        
        start_time = time.time()
        tensor_nhwc = ProductionOptimizer.optimize_memory_layout(tensor_nchw, "NHWC")
        layout_time = time.time() - start_time
        
        print(f"  Original shape (NCHW): {tensor_nchw.shape}")
        print(f"  Optimized shape (NHWC): {tensor_nhwc.shape}")
        print(f"  Layout conversion time: {layout_time*1000:.3f}ms")
        
        # Test inference optimizations
        print("\\n3. Inference Optimizations:")
        
        class TestModel:
            def __init__(self):
                self.conv1 = Conv2d(3, 32, 3, padding=1)
                self.bn1 = BatchNorm(32)
                self.conv2 = Conv2d(32, 64, 3, padding=1)
                self.training = True
        
        model = TestModel()
        optimizations = ProductionOptimizer.optimize_for_inference(model)
        
        print(f"  Applied optimizations:")
        for opt in optimizations:
            print(f"    - {opt}")
        
        # Benchmark inference performance
        print("\\n4. Inference Performance Benchmark:")
        test_input = Tensor.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(5):
            _ = model.conv1(test_input)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            output = model.conv1(test_input)
            _ = output.numpy()  # Force computation
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_inference_time
        
        print(f"  Average inference time: {avg_inference_time*1000:.3f}ms")
        print(f"  Throughput: {throughput:.1f} inferences/second")
        
        return ProductionOptimizer, {
            'avg_inference_time': avg_inference_time,
            'throughput': throughput,
            'optimizations_applied': len(optimizations)
        }

def demonstrate_complete_optimization_pipeline():
    """
    Demonstrate a complete model optimization pipeline
    """
    print("\\n=== Complete Optimization Pipeline ===\\n")
    
    optimizer = ModelOptimizationExpert()
    
    print("1. Quantization Techniques:")
    print("=" * 30)
    quant_toolkit, quant_results = optimizer.implement_quantization_techniques()
    
    print("\\n2. Pruning Techniques:")
    print("=" * 25)
    pruning_toolkit = optimizer.implement_pruning_techniques()
    
    print("\\n3. Knowledge Distillation:")
    print("=" * 30)
    kd_toolkit, kd_results = optimizer.implement_knowledge_distillation()
    
    print("\\n4. Production Optimization:")
    print("=" * 30)
    prod_optimizer, prod_results = optimizer.production_deployment_optimization()
    
    # Summary report
    print("\\n" + "="*60)
    print("OPTIMIZATION PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\\nQuantization Results:")
    print(f"  Compression ratio: {quant_results['compression_ratio']:.2f}x")
    print(f"  Quantization error: {quant_results['quantization_error']:.6f}")
    print(f"  Relative error: {quant_results['relative_error']:.2%}")
    
    print(f"\\nKnowledge Distillation Results:")
    print(f"  Model compression: {kd_results['compression_ratio']:.2f}x")
    print(f"  Training epochs: {len(kd_results['training_history'])}")
    
    print(f"\\nProduction Optimization Results:")
    print(f"  Inference time: {prod_results['avg_inference_time']*1000:.2f}ms")
    print(f"  Throughput: {prod_results['throughput']:.1f} inferences/sec")
    print(f"  Optimizations applied: {prod_results['optimizations_applied']}")
    
    print(f"\\nRecommendations for Production:")
    print(f"  1. Apply quantization for {quant_results['compression_ratio']:.1f}x model compression")
    print(f"  2. Use knowledge distillation for {kd_results['compression_ratio']:.1f}x parameter reduction")
    print(f"  3. Implement operator fusion for inference speedup")
    print(f"  4. Optimize memory layout for target hardware")
    print(f"  5. Use pruning for additional sparsity-based acceleration")
    
    return {
        'quantization': quant_results,
        'knowledge_distillation': kd_results,
        'production': prod_results
    }

if __name__ == "__main__":
    print("Day 5: Model Optimization and Production")
    print("=" * 45)
    
    # Run complete optimization pipeline
    results = demonstrate_complete_optimization_pipeline()
    
    print("\\n" + "="*45)
    print("Model Optimization and Production Complete!")
```

---

## Day 5 Wrap-up & Neural Network Mastery

### What You've Mastered Today

1. ✅ **Neural Network Layers**: Deep understanding of layer implementations and custom layer creation
2. ✅ **Autograd System**: Comprehensive knowledge of automatic differentiation and gradient computation
3. ✅ **Training Systems**: Advanced training loops, optimizers, and training strategies
4. ✅ **Model Architectures**: Complete neural network architectures from scratch
5. ✅ **Model Optimization**: Production-ready optimization techniques including quantization, pruning, and knowledge distillation
6. ✅ **Production Deployment**: Real-world deployment optimization and performance engineering

### Tomorrow's Preview: Advanced Topics & JIT Compilation

Day 6 will focus on:
- **JIT Compilation**: Just-in-time compilation and optimization
- **Advanced Optimization**: BEAM search, kernel optimization, and performance tuning
- **Quantization**: Advanced quantization techniques and mixed precision
- **Model Deployment**: Production deployment strategies and edge optimization

### Advanced Homework Assignments

1. **Custom Architecture**: Implement a complete transformer architecture with attention mechanisms
2. **Production Pipeline**: Build a complete model optimization pipeline for your electrical testing domain
3. **Training Framework**: Create a comprehensive training framework with advanced features
4. **Deployment System**: Implement model deployment with optimization for edge devices

### Self-Assessment Checklist

- [ ] Can I implement custom neural network layers from scratch?
- [ ] Can I design and debug complex training pipelines?
- [ ] Can I optimize models for production deployment?
- [ ] Can I implement advanced techniques like attention and knowledge distillation?
- [ ] Can I analyze and optimize neural network performance?

### Practical Project: Complete Neural Network Framework

```python
# Advanced Project: Production-ready neural network framework
class ElectricalValidationNeuralFramework:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.training_pipeline = AdvancedTrainingPipeline()
        self.optimization_suite = OptimizationSuite()
        self.deployment_manager = DeploymentManager()
    
    def create_signal_classifier(self, signal_specs):
        """Create optimized neural network for signal classification"""
        # TODO: Implement complete framework
        pass
    
    def train_with_electrical_data(self, training_data):
        """Train models with electrical validation data"""
        # TODO: Implement specialized training
        pass
    
    def deploy_for_real_time(self, model, target_device):
        """Deploy optimized model for real-time inference"""
        # TODO: Implement deployment pipeline
        pass
```

**Ready for Day 6? Advanced optimization and JIT compilation await! 🚀**