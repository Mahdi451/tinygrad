#!/usr/bin/env python3
"""
Simple CSV Anomaly Detection for Electrical Data using Tinygrad
===============================================================

This is a simplified version focused on quick anomaly detection in CSV files
with plotting support. Perfect for getting started with electrical validation.

Usage:
    python simple_csv_anomaly_detector.py your_data.csv
"""

import sys
import numpy as np
from tinygrad import Tensor

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Install matplotlib and pandas for plotting: pip install matplotlib pandas")

class SimpleAnomalyDetector:
    """
    Simple anomaly detector for CSV electrical data
    """
    
    def __init__(self, 
                 min_voltage: float = 0.0, 
                 max_voltage: float = 5.0,
                 z_threshold: float = 3.0):
        """
        Initialize detector with specifications
        
        Args:
            min_voltage: Minimum acceptable voltage
            max_voltage: Maximum acceptable voltage  
            z_threshold: Z-score threshold for statistical anomalies
        """
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
        self.z_threshold = z_threshold
    
    def load_csv(self, csv_path: str):
        """
        Load data from CSV file
        
        Expected CSV format (flexible column names):
        - Must have a column with 'voltage' in the name
        - Can have 'current', 'power', 'temperature', etc.
        - First column is assumed to be timestamp/index
        """
        if HAS_PLOTTING:
            # Use pandas for easy CSV loading
            df = pd.read_csv(csv_path)
            
            # Find voltage column (flexible naming)
            voltage_col = None
            for col in df.columns:
                if 'voltage' in col.lower() or 'volt' in col.lower():
                    voltage_col = col
                    break
            
            if voltage_col is None:
                raise ValueError("No voltage column found. Column must contain 'voltage' or 'volt' in name")
            
            voltage_data = df[voltage_col].values.astype(np.float32)
            
            # Get other numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if voltage_col in numeric_cols:
                numeric_cols.remove(voltage_col)
            
            other_data = {}
            for col in numeric_cols:
                other_data[col] = df[col].values.astype(np.float32)
            
            return voltage_data, other_data, df
        else:
            # Manual CSV parsing without pandas
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Find voltage column
                voltage_idx = None
                for i, header in enumerate(headers):
                    if 'voltage' in header.lower() or 'volt' in header.lower():
                        voltage_idx = i
                        break
                
                if voltage_idx is None:
                    raise ValueError("No voltage column found")
                
                # Read data
                voltage_data = []
                for row in reader:
                    try:
                        voltage_data.append(float(row[voltage_idx]))
                    except (ValueError, IndexError):
                        continue
                
                return np.array(voltage_data, dtype=np.float32), {}, None
    
    def detect_anomalies(self, data: np.ndarray):
        """
        Detect anomalies in data using both spec compliance and statistical methods
        
        Args:
            data: numpy array of measurements
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Convert to tinygrad tensor
        tensor_data = Tensor(data)
        
        # 1. Specification-based anomalies
        spec_violations = []
        
        # Check each data point against specs
        for i, value in enumerate(data):
            if value < self.min_voltage:
                spec_violations.append({
                    'index': i,
                    'value': value,
                    'type': 'below_min_spec',
                    'severity': 'high'
                })
            elif value > self.max_voltage:
                spec_violations.append({
                    'index': i,
                    'value': value,
                    'type': 'above_max_spec', 
                    'severity': 'high'
                })
        
        # 2. Statistical anomalies using z-score
        mean_val = tensor_data.mean()
        std_val = tensor_data.std()
        z_scores = (tensor_data - mean_val) / std_val
        z_abs = z_scores.abs()
        
        # Find statistical outliers
        statistical_anomalies = []
        z_np = z_abs.numpy()
        
        for i, z_score in enumerate(z_np):
            if z_score > self.z_threshold:
                statistical_anomalies.append({
                    'index': i,
                    'value': data[i],
                    'z_score': z_score,
                    'type': 'statistical_outlier',
                    'severity': 'medium' if z_score < self.z_threshold * 1.5 else 'high'
                })
        
        # 3. Calculate statistics
        stats = {
            'mean': mean_val.item(),
            'std': std_val.item(),
            'min': tensor_data.min().item(),
            'max': tensor_data.max().item(),
            'total_points': len(data),
            'spec_violations': len(spec_violations),
            'statistical_anomalies': len(statistical_anomalies)
        }
        
        return {
            'spec_violations': spec_violations,
            'statistical_anomalies': statistical_anomalies,
            'stats': stats,
            'z_scores': z_np
        }
    
    def plot_results(self, data: np.ndarray, results: dict, title: str = "Anomaly Detection Results"):
        """
        Plot the data with anomalies highlighted
        """
        if not HAS_PLOTTING:
            print("Plotting not available. Install matplotlib: pip install matplotlib")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Data with spec limits and anomalies
        ax1.plot(data, 'b-', alpha=0.7, label='Data')
        ax1.axhline(y=self.min_voltage, color='r', linestyle='--', alpha=0.7, label='Min Spec')
        ax1.axhline(y=self.max_voltage, color='r', linestyle='--', alpha=0.7, label='Max Spec')
        
        # Highlight spec violations
        for anomaly in results['spec_violations']:
            ax1.scatter(anomaly['index'], anomaly['value'], 
                       color='red', s=100, marker='x', alpha=0.8, zorder=5)
        
        # Highlight statistical anomalies
        for anomaly in results['statistical_anomalies']:
            ax1.scatter(anomaly['index'], anomaly['value'], 
                       color='orange', s=60, marker='o', alpha=0.6, zorder=4)
        
        ax1.set_title('Data with Anomalies (Red X = Spec Violation, Orange O = Statistical Outlier)')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Z-scores
        ax2.plot(results['z_scores'], 'g-', alpha=0.7, label='Z-scores')
        ax2.axhline(y=self.z_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Z-threshold ({self.z_threshold})')
        ax2.axhline(y=-self.z_threshold, color='orange', linestyle='--', alpha=0.7)
        
        # Highlight statistical anomalies
        for anomaly in results['statistical_anomalies']:
            ax2.scatter(anomaly['index'], anomaly['z_score'], 
                       color='orange', s=60, marker='o', alpha=0.8)
        
        ax2.set_title('Z-Scores (Statistical Anomaly Detection)')
        ax2.set_xlabel('Sample Number')
        ax2.set_ylabel('Z-Score')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        stats = results['stats']
        print(f"\\n=== ANOMALY DETECTION SUMMARY ===")
        print(f"Total data points: {stats['total_points']}")
        print(f"Data range: {stats['min']:.3f} to {stats['max']:.3f}")
        print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"\\nSpec violations: {stats['spec_violations']}")
        print(f"Statistical anomalies: {stats['statistical_anomalies']}")
        print(f"Total anomalies: {stats['spec_violations'] + stats['statistical_anomalies']}")
        
        if stats['spec_violations'] > 0:
            print(f"\\n⚠️  SPEC VIOLATIONS DETECTED!")
            for violation in results['spec_violations']:
                print(f"  Sample {violation['index']}: {violation['value']:.3f}V ({violation['type']})")
    
    def analyze_file(self, csv_path: str, plot: bool = True):
        """
        Complete analysis of a CSV file
        
        Args:
            csv_path: Path to CSV file
            plot: Whether to show plots
            
        Returns:
            Analysis results dictionary
        """
        print(f"Analyzing: {csv_path}")
        
        # Load data
        voltage_data, other_data, df = self.load_csv(csv_path)
        print(f"Loaded {len(voltage_data)} voltage measurements")
        
        # Detect anomalies
        results = self.detect_anomalies(voltage_data)
        
        # Plot if requested
        if plot:
            self.plot_results(voltage_data, results, f"Analysis: {csv_path}")
        
        # Analyze other columns if available
        other_results = {}
        for col_name, col_data in other_data.items():
            if len(col_data) > 0:
                # Simple statistical analysis for other columns
                col_tensor = Tensor(col_data)
                other_results[col_name] = {
                    'mean': col_tensor.mean().item(),
                    'std': col_tensor.std().item(),
                    'min': col_tensor.min().item(),
                    'max': col_tensor.max().item()
                }
        
        if other_results:
            print(f"\\n=== OTHER COLUMNS STATISTICS ===")
            for col, stats in other_results.items():
                print(f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        return {
            'voltage_results': results,
            'other_results': other_results,
            'voltage_data': voltage_data,
            'other_data': other_data
        }

def create_sample_data(filename: str = "sample_electrical_data.csv"):
    """Create sample CSV data for testing"""
    import random
    
    print(f"Creating sample data: {filename}")
    
    # Generate 500 data points
    timestamps = [i * 0.1 for i in range(500)]  # 100ms intervals
    voltages = []
    currents = []
    temperatures = []
    
    for i, t in enumerate(timestamps):
        # Normal voltage around 3.3V with small variations
        base_voltage = 3.3 + 0.05 * np.sin(0.1 * t)  # Slight sine wave
        voltage = base_voltage + random.gauss(0, 0.02)  # Add noise
        
        # Inject some anomalies
        if i in [50, 150, 300, 450]:  # Specific anomaly points
            if i == 50:
                voltage = 4.8  # High voltage violation
            elif i == 150:
                voltage = 1.2  # Low voltage violation
            elif i == 300:
                voltage = 3.9  # Borderline high
            elif i == 450:
                voltage = 2.1  # Low voltage violation
        
        # Generate corresponding current and temperature
        current = 0.5 + random.gauss(0, 0.05)
        temperature = 25 + random.gauss(0, 2.0)
        
        voltages.append(voltage)
        currents.append(current)
        temperatures.append(temperature)
    
    # Write CSV
    with open(filename, 'w') as f:
        f.write("timestamp,voltage,current,temperature\\n")
        for t, v, c, temp in zip(timestamps, voltages, currents, temperatures):
            f.write(f"{t:.3f},{v:.4f},{c:.4f},{temp:.2f}\\n")
    
    print(f"Sample data created with {len(voltages)} points and 4 injected anomalies")
    return filename

def main():
    """Main function for command line usage"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Create sample data for demonstration
        print("No CSV file provided. Creating sample data for demonstration...")
        csv_file = create_sample_data()
    
    # Define your specifications
    detector = SimpleAnomalyDetector(
        min_voltage=2.5,  # Minimum acceptable voltage
        max_voltage=4.0,  # Maximum acceptable voltage
        z_threshold=2.5   # Z-score threshold for statistical anomalies
    )
    
    try:
        # Analyze the file
        results = detector.analyze_file(csv_file, plot=True)
        
        # Provide recommendations
        voltage_results = results['voltage_results']
        stats = voltage_results['stats']
        
        print(f"\\n=== RECOMMENDATIONS ===")
        
        if stats['spec_violations'] > 0:
            print("🔴 IMMEDIATE ACTION REQUIRED:")
            print("  - Spec violations detected - check equipment calibration")
            print("  - Review test setup and power supply stability")
            print("  - Consider tightening quality control procedures")
        
        if stats['statistical_anomalies'] > 0:
            print("🟡 INVESTIGATION RECOMMENDED:")
            print("  - Statistical outliers detected - may indicate:")
            print("    * Intermittent connections")
            print("    * Environmental factors (temperature, vibration)")
            print("    * Component aging or drift")
        
        if stats['spec_violations'] == 0 and stats['statistical_anomalies'] == 0:
            print("✅ NO ANOMALIES DETECTED:")
            print("  - All measurements within specifications")
            print("  - Statistical distribution appears normal")
            print("  - System appears to be operating correctly")
        
        # Data quality assessment
        variation_coefficient = stats['std'] / stats['mean'] * 100
        print(f"\\n=== DATA QUALITY ASSESSMENT ===")
        print(f"Coefficient of variation: {variation_coefficient:.2f}%")
        
        if variation_coefficient < 5:
            print("📊 Data quality: EXCELLENT (very stable)")
        elif variation_coefficient < 10:
            print("📊 Data quality: GOOD (acceptable variation)")
        elif variation_coefficient < 20:
            print("📊 Data quality: FAIR (moderate variation)")
        else:
            print("📊 Data quality: POOR (high variation)")
            print("   Consider investigating sources of variability")
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        print("\\nPlease ensure your CSV file has a column containing 'voltage' in the name")
        print("Example format:")
        print("timestamp,voltage,current")
        print("0.0,3.3,0.5")
        print("0.1,3.2,0.48")

if __name__ == "__main__":
    main()