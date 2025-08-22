#!/usr/bin/env python3
"""
Electrical Data Anomaly Detection System using Tinygrad
======================================================

This module provides real-time anomaly detection for electrical validation data
that doesn't meet specifications. Supports CSV input, plotting, and live monitoring.

Key Features:
- Real-time anomaly detection during data collection
- CSV file processing with plotting
- Multiple anomaly detection methods (z-score, statistical bounds, spec compliance)
- Live visualization during testing
- Configurable thresholds and specifications
"""

import os
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from queue import Queue, Empty

import numpy as np
from tinygrad import Tensor
from tinygrad.helpers import getenv

# Optional imports for plotting (install with: pip install matplotlib pandas)
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/pandas not installed. Plotting disabled.")
    print("Install with: pip install matplotlib pandas")

@dataclass
class ElectricalSpec:
    """Define electrical specifications for anomaly detection"""
    min_voltage: float = 0.0
    max_voltage: float = 5.0
    min_current: float = 0.0
    max_current: float = 1.0
    min_power: float = 0.0
    max_power: float = 5.0
    voltage_tolerance: float = 0.1  # ±10%
    current_tolerance: float = 0.05  # ±5%
    power_tolerance: float = 0.15   # ±15%

@dataclass
class AnomalyResult:
    """Results from anomaly detection"""
    timestamp: float
    anomalies_detected: bool
    anomaly_indices: List[int]
    anomaly_values: List[float]
    anomaly_types: List[str]
    confidence_scores: List[float]
    data_stats: Dict[str, float]

class ElectricalAnomalyDetector:
    """
    Advanced anomaly detection system for electrical validation data
    """
    
    def __init__(self, spec: ElectricalSpec, 
                 z_score_threshold: float = 3.0,
                 statistical_window: int = 100,
                 enable_plotting: bool = True):
        """
        Initialize the anomaly detection system
        
        Args:
            spec: Electrical specifications
            z_score_threshold: Z-score threshold for statistical anomalies
            statistical_window: Window size for rolling statistics
            enable_plotting: Enable real-time plotting
        """
        self.spec = spec
        self.z_threshold = z_score_threshold
        self.window_size = statistical_window
        self.enable_plotting = enable_plotting and HAS_PLOTTING
        
        # Data storage
        self.voltage_history = []
        self.current_history = []
        self.power_history = []
        self.timestamp_history = []
        self.anomaly_history = []
        
        # Real-time monitoring
        self.monitoring = False
        self.data_queue = Queue()
        self.monitor_thread = None
        
        # Plotting setup
        if self.enable_plotting:
            self.setup_plotting()
    
    def setup_plotting(self):
        """Setup real-time plotting"""
        if not HAS_PLOTTING:
            return
            
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Real-Time Electrical Data Monitoring', fontsize=16)
        
        # Setup subplots
        self.ax1.set_title('Voltage (V)')
        self.ax1.set_ylabel('Voltage')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axhline(y=self.spec.min_voltage, color='r', linestyle='--', alpha=0.7, label='Min Spec')
        self.ax1.axhline(y=self.spec.max_voltage, color='r', linestyle='--', alpha=0.7, label='Max Spec')
        
        self.ax2.set_title('Current (A)')
        self.ax2.set_ylabel('Current')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=self.spec.min_current, color='r', linestyle='--', alpha=0.7)
        self.ax2.axhline(y=self.spec.max_current, color='r', linestyle='--', alpha=0.7)
        
        self.ax3.set_title('Power (W)')
        self.ax3.set_ylabel('Power')
        self.ax3.set_xlabel('Sample Number')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=self.spec.min_power, color='r', linestyle='--', alpha=0.7)
        self.ax3.axhline(y=self.spec.max_power, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
    
    def load_csv_data(self, csv_path: str) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Load electrical data from CSV file
        
        Expected CSV format:
        timestamp,voltage,current,power
        0.0,3.3,0.5,1.65
        0.1,3.2,0.48,1.536
        ...
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Tuple of (voltage_tensor, current_tensor, power_tensor)
        """
        if not HAS_PLOTTING:
            raise ImportError("pandas required for CSV loading. Install with: pip install pandas")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Extract columns
        voltage_data = df['voltage'].values.astype(np.float32)
        current_data = df['current'].values.astype(np.float32)
        power_data = df['power'].values.astype(np.float32)
        
        # Convert to tinygrad tensors
        voltage_tensor = Tensor(voltage_data)
        current_tensor = Tensor(current_data)
        power_tensor = Tensor(power_data)
        
        # Store for plotting
        if 'timestamp' in df.columns:
            self.timestamp_history = df['timestamp'].values.tolist()
        else:
            self.timestamp_history = list(range(len(voltage_data)))
        
        self.voltage_history = voltage_data.tolist()
        self.current_history = current_data.tolist()
        self.power_history = power_data.tolist()
        
        return voltage_tensor, current_tensor, power_tensor
    
    def detect_spec_violations(self, voltage: Tensor, current: Tensor, power: Tensor) -> AnomalyResult:
        """
        Detect specification violations
        
        Args:
            voltage: Voltage measurements tensor
            current: Current measurements tensor  
            power: Power measurements tensor
            
        Returns:
            AnomalyResult with detected violations
        """
        anomaly_indices = []
        anomaly_values = []
        anomaly_types = []
        confidence_scores = []
        
        # Convert to numpy for indexing (tinygrad doesn't have full boolean indexing yet)
        v_np = voltage.numpy()
        c_np = current.numpy()
        p_np = power.numpy()
        
        # Check voltage violations
        v_low_violations = np.where(v_np < self.spec.min_voltage)[0]
        v_high_violations = np.where(v_np > self.spec.max_voltage)[0]
        
        for idx in v_low_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(v_np[idx])
            anomaly_types.append("voltage_too_low")
            confidence_scores.append(1.0)  # Spec violations are 100% confident
        
        for idx in v_high_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(v_np[idx])
            anomaly_types.append("voltage_too_high")
            confidence_scores.append(1.0)
        
        # Check current violations
        c_low_violations = np.where(c_np < self.spec.min_current)[0]
        c_high_violations = np.where(c_np > self.spec.max_current)[0]
        
        for idx in c_low_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(c_np[idx])
            anomaly_types.append("current_too_low")
            confidence_scores.append(1.0)
        
        for idx in c_high_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(c_np[idx])
            anomaly_types.append("current_too_high")
            confidence_scores.append(1.0)
        
        # Check power violations
        p_low_violations = np.where(p_np < self.spec.min_power)[0]
        p_high_violations = np.where(p_np > self.spec.max_power)[0]
        
        for idx in p_low_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(p_np[idx])
            anomaly_types.append("power_too_low")
            confidence_scores.append(1.0)
        
        for idx in p_high_violations:
            anomaly_indices.append(idx)
            anomaly_values.append(p_np[idx])
            anomaly_types.append("power_too_high")
            confidence_scores.append(1.0)
        
        # Calculate data statistics
        data_stats = {
            'voltage_mean': voltage.mean().item(),
            'voltage_std': voltage.std().item(),
            'current_mean': current.mean().item(),
            'current_std': current.std().item(),
            'power_mean': power.mean().item(),
            'power_std': power.std().item(),
        }
        
        return AnomalyResult(
            timestamp=time.time(),
            anomalies_detected=len(anomaly_indices) > 0,
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_types=anomaly_types,
            confidence_scores=confidence_scores,
            data_stats=data_stats
        )
    
    def detect_statistical_anomalies(self, voltage: Tensor, current: Tensor, power: Tensor) -> AnomalyResult:
        """
        Detect statistical anomalies using z-score method
        
        Args:
            voltage: Voltage measurements tensor
            current: Current measurements tensor
            power: Power measurements tensor
            
        Returns:
            AnomalyResult with detected statistical anomalies
        """
        anomaly_indices = []
        anomaly_values = []
        anomaly_types = []
        confidence_scores = []
        
        # Z-score anomaly detection for each signal
        signals = [
            (voltage, "voltage"),
            (current, "current"), 
            (power, "power")
        ]
        
        for signal, signal_name in signals:
            # Calculate z-scores using tinygrad
            mean_val = signal.mean()
            std_val = signal.std()
            z_scores = (signal - mean_val) / std_val
            z_abs = z_scores.abs()
            
            # Find anomalies (convert to numpy for indexing)
            z_np = z_abs.numpy()
            anomaly_mask = z_np > self.z_threshold
            signal_anomaly_indices = np.where(anomaly_mask)[0]
            
            for idx in signal_anomaly_indices:
                anomaly_indices.append(idx)
                anomaly_values.append(signal.numpy()[idx])
                anomaly_types.append(f"{signal_name}_statistical")
                confidence_scores.append(min(z_np[idx] / self.z_threshold, 1.0))
        
        # Calculate data statistics  
        data_stats = {
            'voltage_mean': voltage.mean().item(),
            'voltage_std': voltage.std().item(),
            'current_mean': current.mean().item(),
            'current_std': current.std().item(),
            'power_mean': power.mean().item(),
            'power_std': power.std().item(),
        }
        
        return AnomalyResult(
            timestamp=time.time(),
            anomalies_detected=len(anomaly_indices) > 0,
            anomaly_indices=anomaly_indices,
            anomaly_values=anomaly_values,
            anomaly_types=anomaly_types,
            confidence_scores=confidence_scores,
            data_stats=data_stats
        )
    
    def analyze_csv_file(self, csv_path: str, plot_results: bool = True) -> Tuple[AnomalyResult, AnomalyResult]:
        """
        Complete analysis of CSV file with both spec and statistical anomaly detection
        
        Args:
            csv_path: Path to CSV file
            plot_results: Whether to plot the results
            
        Returns:
            Tuple of (spec_violations, statistical_anomalies)
        """
        print(f"Analyzing CSV file: {csv_path}")
        
        # Load data
        voltage, current, power = self.load_csv_data(csv_path)
        print(f"Loaded {len(voltage)} data points")
        
        # Detect anomalies
        spec_violations = self.detect_spec_violations(voltage, current, power)
        statistical_anomalies = self.detect_statistical_anomalies(voltage, current, power)
        
        # Print results
        print(f"\\nSpec Violations: {len(spec_violations.anomaly_indices)} detected")
        print(f"Statistical Anomalies: {len(statistical_anomalies.anomaly_indices)} detected")
        
        # Plot results
        if plot_results and self.enable_plotting:
            self.plot_analysis_results(spec_violations, statistical_anomalies)
        
        return spec_violations, statistical_anomalies
    
    def plot_analysis_results(self, spec_violations: AnomalyResult, statistical_anomalies: AnomalyResult):
        """Plot analysis results with anomalies highlighted"""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Electrical Data Analysis Results', fontsize=16)
        
        # Plot voltage
        axes[0].plot(self.timestamp_history, self.voltage_history, 'b-', alpha=0.7, label='Voltage')
        axes[0].axhline(y=self.spec.min_voltage, color='r', linestyle='--', alpha=0.7, label='Min Spec')
        axes[0].axhline(y=self.spec.max_voltage, color='r', linestyle='--', alpha=0.7, label='Max Spec')
        
        # Highlight anomalies
        for idx, anom_type in zip(spec_violations.anomaly_indices, spec_violations.anomaly_types):
            if 'voltage' in anom_type:
                axes[0].scatter(self.timestamp_history[idx], self.voltage_history[idx], 
                              color='red', s=100, marker='x', alpha=0.8)
        
        for idx, anom_type in zip(statistical_anomalies.anomaly_indices, statistical_anomalies.anomaly_types):
            if 'voltage' in anom_type:
                axes[0].scatter(self.timestamp_history[idx], self.voltage_history[idx], 
                              color='orange', s=50, marker='o', alpha=0.6)
        
        axes[0].set_title('Voltage Analysis')
        axes[0].set_ylabel('Voltage (V)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Similar plots for current and power...
        # (truncated for brevity - full implementation would include current and power plots)
        
        plt.tight_layout()
        plt.show()
    
    def start_real_time_monitoring(self, data_callback: Callable[[float, float, float], None]):
        """
        Start real-time monitoring thread
        
        Args:
            data_callback: Function that provides (voltage, current, power) data points
        """
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(data_callback,))
        self.monitor_thread.start()
        print("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Real-time monitoring stopped")
    
    def _monitor_loop(self, data_callback: Callable[[float, float, float], None]):
        """Internal monitoring loop"""
        buffer_size = 50  # Process in chunks for efficiency
        voltage_buffer = []
        current_buffer = []
        power_buffer = []
        
        while self.monitoring:
            try:
                # Get new data point
                voltage, current, power = data_callback()
                
                # Add to buffers
                voltage_buffer.append(voltage)
                current_buffer.append(current)
                power_buffer.append(power)
                
                # Process when buffer is full
                if len(voltage_buffer) >= buffer_size:
                    v_tensor = Tensor(voltage_buffer)
                    c_tensor = Tensor(current_buffer)
                    p_tensor = Tensor(power_buffer)
                    
                    # Detect anomalies
                    spec_result = self.detect_spec_violations(v_tensor, c_tensor, p_tensor)
                    stat_result = self.detect_statistical_anomalies(v_tensor, c_tensor, p_tensor)
                    
                    # Report anomalies
                    if spec_result.anomalies_detected or stat_result.anomalies_detected:
                        print(f"ANOMALY DETECTED at {time.time():.2f}")
                        print(f"  Spec violations: {len(spec_result.anomaly_indices)}")
                        print(f"  Statistical anomalies: {len(stat_result.anomaly_indices)}")
                    
                    # Update history for plotting
                    self.voltage_history.extend(voltage_buffer)
                    self.current_history.extend(current_buffer)
                    self.power_history.extend(power_buffer)
                    self.timestamp_history.extend([time.time() + i*0.1 for i in range(len(voltage_buffer))])
                    
                    # Update real-time plot
                    if self.enable_plotting:
                        self._update_realtime_plot()
                    
                    # Clear buffers
                    voltage_buffer.clear()
                    current_buffer.clear()
                    power_buffer.clear()
                
                time.sleep(0.1)  # 10 Hz sampling
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                continue
    
    def _update_realtime_plot(self):
        """Update real-time plot"""
        if not HAS_PLOTTING or not hasattr(self, 'ax1'):
            return
            
        # Keep only last 1000 points for performance
        max_points = 1000
        if len(self.voltage_history) > max_points:
            self.voltage_history = self.voltage_history[-max_points:]
            self.current_history = self.current_history[-max_points:]
            self.power_history = self.power_history[-max_points:]
            self.timestamp_history = self.timestamp_history[-max_points:]
        
        # Update plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Replot data
        self.ax1.plot(self.timestamp_history, self.voltage_history, 'b-', alpha=0.7)
        self.ax1.axhline(y=self.spec.min_voltage, color='r', linestyle='--', alpha=0.7)
        self.ax1.axhline(y=self.spec.max_voltage, color='r', linestyle='--', alpha=0.7)
        self.ax1.set_title('Voltage (V)')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.plot(self.timestamp_history, self.current_history, 'g-', alpha=0.7)
        self.ax2.axhline(y=self.spec.min_current, color='r', linestyle='--', alpha=0.7)
        self.ax2.axhline(y=self.spec.max_current, color='r', linestyle='--', alpha=0.7)
        self.ax2.set_title('Current (A)')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.plot(self.timestamp_history, self.power_history, 'm-', alpha=0.7)
        self.ax3.axhline(y=self.spec.min_power, color='r', linestyle='--', alpha=0.7)
        self.ax3.axhline(y=self.spec.max_power, color='r', linestyle='--', alpha=0.7)
        self.ax3.set_title('Power (W)')
        self.ax3.set_xlabel('Timestamp')
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

# Example usage and demonstration
def main():
    """Demonstration of the anomaly detection system"""
    
    # Define electrical specifications
    spec = ElectricalSpec(
        min_voltage=2.8,
        max_voltage=3.6,
        min_current=0.1,
        max_current=0.8,
        min_power=0.5,
        max_power=3.0
    )
    
    # Create detector
    detector = ElectricalAnomalyDetector(
        spec=spec,
        z_score_threshold=2.5,
        statistical_window=100,
        enable_plotting=True
    )
    
    # Example 1: Generate sample CSV data for testing
    print("Generating sample data...")
    sample_data = generate_sample_csv("sample_electrical_data.csv")
    
    # Example 2: Analyze CSV file
    print("\\nAnalyzing CSV file...")
    spec_violations, statistical_anomalies = detector.analyze_csv_file("sample_electrical_data.csv")
    
    # Example 3: Real-time monitoring simulation
    print("\\nStarting real-time monitoring simulation (10 seconds)...")
    
    def simulated_data_source():
        """Simulate real-time data acquisition"""
        import random
        # Normal data with occasional anomalies
        base_voltage = 3.3
        base_current = 0.5
        base_power = base_voltage * base_current
        
        # Add noise and occasional anomalies
        voltage = base_voltage + random.gauss(0, 0.1)
        current = base_current + random.gauss(0, 0.05)
        power = voltage * current
        
        # Inject anomalies 5% of the time
        if random.random() < 0.05:
            if random.random() < 0.5:
                voltage = random.choice([1.0, 4.5])  # Spec violation
            else:
                current = random.choice([0.05, 1.2])  # Spec violation
            power = voltage * current
        
        return voltage, current, power
    
    # Start monitoring
    detector.start_real_time_monitoring(simulated_data_source)
    time.sleep(10)  # Monitor for 10 seconds
    detector.stop_monitoring()
    
    print("\\nDemo completed!")

def generate_sample_csv(filename: str) -> str:
    """Generate sample CSV data for testing"""
    import random
    
    # Generate 1000 data points
    timestamps = [i * 0.1 for i in range(1000)]  # 100ms intervals
    voltages = []
    currents = []
    powers = []
    
    for i, t in enumerate(timestamps):
        # Base values with noise
        base_v = 3.3 + 0.1 * np.sin(0.1 * t)  # Slight sine wave
        base_c = 0.5 + 0.05 * np.cos(0.15 * t)
        
        # Add random noise
        voltage = base_v + random.gauss(0, 0.05)
        current = base_c + random.gauss(0, 0.02)
        power = voltage * current
        
        # Inject some anomalies
        if i in [100, 250, 400, 750, 900]:  # Specific anomaly points
            if i == 100:
                voltage = 4.2  # High voltage
            elif i == 250:
                current = 0.02  # Low current
            elif i == 400:
                voltage = 2.1  # Low voltage
            elif i == 750:
                current = 0.95  # High current
            elif i == 900:
                voltage = 1.8  # Very low voltage
            power = voltage * current
        
        voltages.append(voltage)
        currents.append(current)
        powers.append(power)
    
    # Write to CSV
    with open(filename, 'w') as f:
        f.write("timestamp,voltage,current,power\\n")
        for t, v, c, p in zip(timestamps, voltages, currents, powers):
            f.write(f"{t:.3f},{v:.4f},{c:.4f},{p:.4f}\\n")
    
    print(f"Sample data written to {filename}")
    return filename

if __name__ == "__main__":
    main()