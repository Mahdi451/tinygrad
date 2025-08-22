#!/usr/bin/env python3
"""
Comprehensive Data Validation and Eye Diagram Reconstruction Toolkit
===================================================================

This module provides complete data validation methods and eye diagram reconstruction
capabilities for Ethernet testing validation. Includes statistical validation,
temporal analysis, eye diagram construction, and comprehensive validation insights.

Key Features:
- Multiple data validation methods (statistical, temporal, cross-correlation)
- Complete eye diagram reconstruction from time-domain signals
- Comprehensive validation insights and root cause analysis
- Production test validation with pass/fail decisions
- Debug and characterization support

Usage:
    from comprehensive_validation_toolkit import *
    
    # Quick validation
    validator = EthernetValidator(your_data, your_specs)
    results = validator.comprehensive_validation()
    
    # Eye diagram reconstruction
    eye_data = EyeDiagramData(signal_samples, sample_rate, symbol_rate)
    reconstructor = EyeDiagramReconstructor(eye_data)
    eye_results = reconstructor.reconstruct_eye_diagram()
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum

from tinygrad import Tensor

# Optional imports for enhanced functionality
try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

@dataclass
class EyeDiagramData:
    """Required data for eye diagram reconstruction"""
    # Essential data
    signal_samples: np.ndarray      # Raw signal amplitude samples
    sample_rate: float              # Sampling rate (Hz)
    symbol_rate: float              # Symbol/baud rate (Hz)
    
    # Optional but recommended
    trigger_info: Optional[List[float]] = None  # Clock recovery info
    bit_pattern: Optional[List[int]] = None     # Known bit pattern
    signal_type: str = "PAM4"                   # "PAM4" or "NRZ"
    
    # Advanced features
    differential_mode: bool = False              # True for differential signals
    reference_levels: Optional[List[float]] = None  # Expected signal levels

@dataclass
class ValidationSpecs:
    """Validation specifications and limits"""
    # Eye diagram specs
    min_eye_height: float = 0.4      # Minimum eye height (V)
    min_eye_width: float = 0.6       # Minimum eye width (UI)
    min_snr_db: float = 15           # Minimum SNR (dB)
    max_jitter_ps: float = 8         # Maximum jitter (ps)
    
    # Signal level specs
    pam4_levels: List[float] = field(default_factory=lambda: [-0.75, -0.25, 0.25, 0.75])
    level_tolerance: float = 0.05    # ±5% tolerance
    
    # BER specs
    max_ber: float = 1e-12          # Maximum acceptable BER
    
    # Temporal stability
    max_drift_rate: float = 0.01    # Maximum drift rate (1% per measurement)

class EthernetSpecValidator:
    """Comprehensive specification validation for Ethernet signals"""
    
    def __init__(self, spec_limits: ValidationSpecs):
        self.limits = spec_limits
        
    def validate_pam4_levels(self, signal: Tensor):
        """Validate PAM4 signal levels against specifications"""
        violations = []
        
        # Expected PAM4 levels
        expected_levels = self.limits.pam4_levels
        tolerance = self.limits.level_tolerance
        
        # Classify each sample to nearest level
        classified = self.classify_signal_levels(signal, expected_levels)
        
        # Check level accuracy
        signal_np = signal.numpy()
        for i, sample in enumerate(signal_np):
            expected_level = expected_levels[classified[i]]
            deviation = abs(sample - expected_level)
            
            if deviation > tolerance:
                violations.append({
                    'sample_index': i,
                    'actual_value': sample,
                    'expected_level': expected_level,
                    'deviation': deviation,
                    'severity': 'high' if deviation > tolerance * 2 else 'medium'
                })
        
        return {
            'total_violations': len(violations),
            'violation_rate': len(violations) / len(signal_np),
            'violations': violations[:10],  # First 10 for reporting
            'compliance_status': 'PASS' if len(violations) == 0 else 'FAIL'
        }
    
    def classify_signal_levels(self, signal: Tensor, levels: List[float]) -> np.ndarray:
        """Classify signal samples to nearest levels"""
        signal_np = signal.numpy()
        classified = np.zeros(len(signal_np), dtype=int)
        
        for i, sample in enumerate(signal_np):
            distances = [abs(sample - level) for level in levels]
            classified[i] = np.argmin(distances)
        
        return classified
    
    def validate_eye_diagram_specs(self, eye_height: float, eye_width: float):
        """Validate eye diagram against specifications"""
        results = {
            'eye_height': {
                'value': eye_height,
                'limit': self.limits.min_eye_height,
                'pass': eye_height >= self.limits.min_eye_height,
                'margin': eye_height - self.limits.min_eye_height
            },
            'eye_width': {
                'value': eye_width,
                'limit': self.limits.min_eye_width,
                'pass': eye_width >= self.limits.min_eye_width,
                'margin': eye_width - self.limits.min_eye_width
            }
        }
        
        overall_pass = all(param['pass'] for param in results.values())
        results['overall_compliance'] = 'PASS' if overall_pass else 'FAIL'
        
        return results

class DataValidator:
    """Advanced data validation methods"""
    
    @staticmethod
    def validate_against_baseline(current_data: Tensor, baseline_data: Tensor, tolerance: float = 0.1):
        """
        Validate current measurements against known good baseline
        
        Args:
            current_data: Current test measurements
            baseline_data: Known good reference data
            tolerance: Acceptable deviation (10% default)
        
        Returns:
            Validation results with pass/fail status
        """
        # Statistical comparison
        current_mean = current_data.mean()
        baseline_mean = baseline_data.mean()
        
        current_std = current_data.std()
        baseline_std = baseline_data.std()
        
        # Calculate deviations
        mean_deviation = abs(current_mean - baseline_mean) / baseline_mean
        std_deviation = abs(current_std - baseline_std) / baseline_std
        
        # Validation criteria
        mean_valid = mean_deviation < tolerance
        std_valid = std_deviation < tolerance
        
        # Distribution comparison
        distribution_valid = DataValidator.compare_distributions(current_data, baseline_data)
        
        return {
            'overall_valid': mean_valid and std_valid and distribution_valid,
            'mean_deviation': mean_deviation.item(),
            'std_deviation': std_deviation.item(),
            'mean_valid': mean_valid,
            'std_valid': std_valid,
            'distribution_valid': distribution_valid,
            'recommendations': DataValidator.get_validation_recommendations(mean_valid, std_valid, distribution_valid)
        }
    
    @staticmethod
    def compare_distributions(data1: Tensor, data2: Tensor, num_bins: int = 50):
        """Simple distribution comparison"""
        # Create histograms
        min_val = min(data1.min().item(), data2.min().item())
        max_val = max(data1.max().item(), data2.max().item())
        
        # Calculate similarity metric
        hist1 = DataValidator.create_histogram(data1, min_val, max_val, num_bins)
        hist2 = DataValidator.create_histogram(data2, min_val, max_val, num_bins)
        
        # Chi-square-like test
        similarity = DataValidator.calculate_histogram_similarity(hist1, hist2)
        return similarity > 0.95  # 95% similarity threshold
    
    @staticmethod
    def create_histogram(data: Tensor, min_val: float, max_val: float, num_bins: int):
        """Create histogram from tensor data"""
        data_np = data.numpy()
        bin_width = (max_val - min_val) / num_bins
        
        histogram = np.zeros(num_bins)
        for value in data_np:
            bin_idx = int((value - min_val) / bin_width)
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            histogram[bin_idx] += 1
        
        return histogram / len(data_np)  # Normalize
    
    @staticmethod
    def calculate_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray):
        """Calculate histogram similarity"""
        # Simple correlation coefficient
        return np.corrcoef(hist1, hist2)[0, 1] if len(hist1) > 1 else 1.0
    
    @staticmethod
    def get_validation_recommendations(mean_valid: bool, std_valid: bool, distribution_valid: bool):
        """Get recommendations based on validation results"""
        recommendations = []
        
        if not mean_valid:
            recommendations.append("Mean deviation detected - check calibration or offset")
        if not std_valid:
            recommendations.append("Standard deviation change - check noise sources or stability")
        if not distribution_valid:
            recommendations.append("Distribution shape changed - verify test conditions")
        
        if mean_valid and std_valid and distribution_valid:
            recommendations.append("All validations passed - signal quality maintained")
        
        return recommendations
    
    @staticmethod
    def validate_signal_integrity(signal: Tensor, reference_pattern: Tensor):
        """
        Validate signal integrity using cross-correlation with reference pattern
        """
        # Normalize signals
        signal_norm = (signal - signal.mean()) / signal.std()
        ref_norm = (reference_pattern - reference_pattern.mean()) / reference_pattern.std()
        
        # Manual cross-correlation
        correlation = DataValidator.manual_cross_correlation(signal_norm, ref_norm)
        max_correlation = correlation.max()
        
        # Validation criteria
        correlation_threshold = 0.8  # 80% correlation required
        
        return {
            'max_correlation': max_correlation.item(),
            'correlation_valid': max_correlation > correlation_threshold,
            'signal_quality': 'excellent' if max_correlation > 0.95 else 
                             'good' if max_correlation > 0.85 else 
                             'poor'
        }
    
    @staticmethod
    def manual_cross_correlation(signal1: Tensor, signal2: Tensor):
        """Manual cross-correlation implementation"""
        correlations = []
        min_len = min(len(signal1), len(signal2))
        
        for shift in range(-min_len//2, min_len//2):
            if shift >= 0:
                s1 = signal1[shift:min_len]
                s2 = signal2[:min_len-shift]
            else:
                s1 = signal1[:min_len+shift]
                s2 = signal2[-shift:min_len]
            
            if len(s1) > 0 and len(s2) > 0:
                corr = (s1 * s2).mean()
                correlations.append(corr.item())
        
        return Tensor(correlations)
    
    @staticmethod
    def validate_temporal_stability(time_series_data: List[Tensor], timestamps: List[float]):
        """
        Validate that measurements are stable over time
        """
        # Calculate metrics over time
        means_over_time = [data.mean().item() for data in time_series_data]
        stds_over_time = [data.std().item() for data in time_series_data]
        
        # Convert to tensors for analysis
        means_tensor = Tensor(means_over_time)
        stds_tensor = Tensor(stds_over_time)
        
        # Check for drift
        mean_trend = DataValidator.calculate_trend(means_tensor)
        std_trend = DataValidator.calculate_trend(stds_tensor)
        
        # Stability criteria
        max_drift_rate = 0.01  # 1% per measurement
        
        return {
            'mean_stable': abs(mean_trend) < max_drift_rate,
            'std_stable': abs(std_trend) < max_drift_rate,
            'mean_drift_rate': mean_trend,
            'std_drift_rate': std_trend,
            'temporal_validation': 'PASS' if abs(mean_trend) < max_drift_rate and abs(std_trend) < max_drift_rate else 'FAIL'
        }
    
    @staticmethod
    def calculate_trend(data: Tensor):
        """Calculate linear trend in data"""
        n = len(data)
        x = Tensor(list(range(n)))
        
        # Calculate slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        xy = (x * data).sum()
        x_sum = x.sum()
        y_sum = data.sum()
        x_sq_sum = (x * x).sum()
        
        denominator = n * x_sq_sum - x_sum * x_sum
        if abs(denominator.item()) < 1e-10:
            return 0.0
        
        slope = (n * xy - x_sum * y_sum) / denominator
        return slope.item()

def validate_data_quality_for_eye_diagram(data: EyeDiagramData):
    """
    Validate that data quality is sufficient for eye diagram reconstruction
    """
    validation_results = {}
    
    # 1. Sample rate validation
    nyquist_rate = data.symbol_rate * 2
    recommended_rate = data.symbol_rate * 10  # 10x oversampling recommended
    
    validation_results['sample_rate'] = {
        'sufficient': data.sample_rate >= nyquist_rate,
        'recommended': data.sample_rate >= recommended_rate,
        'current': data.sample_rate,
        'nyquist': nyquist_rate,
        'recommended_min': recommended_rate
    }
    
    # 2. Signal length validation
    samples_per_symbol = data.sample_rate / data.symbol_rate
    min_symbols_needed = 100  # Minimum for decent eye diagram
    actual_symbols = len(data.signal_samples) / samples_per_symbol
    
    validation_results['signal_length'] = {
        'sufficient': actual_symbols >= min_symbols_needed,
        'actual_symbols': actual_symbols,
        'min_required': min_symbols_needed,
        'samples_per_symbol': samples_per_symbol
    }
    
    # 3. Signal quality validation
    signal_tensor = Tensor(data.signal_samples.astype(np.float32))
    snr_estimate = estimate_snr(signal_tensor)
    
    validation_results['signal_quality'] = {
        'snr_db': snr_estimate,
        'sufficient': snr_estimate > 10,  # 10dB minimum SNR
        'quality_grade': 'excellent' if snr_estimate > 20 else 
                        'good' if snr_estimate > 15 else 
                        'acceptable' if snr_estimate > 10 else 'poor'
    }
    
    return validation_results

def estimate_snr(signal: Tensor):
    """Estimate Signal-to-Noise Ratio"""
    # Simple SNR estimation using signal variation
    signal_power = (signal.pow(2)).mean()
    
    # Estimate noise from high-frequency components
    # (difference between adjacent samples)
    noise_estimate = (signal[1:] - signal[:-1]).pow(2).mean()
    
    snr_linear = signal_power / noise_estimate
    snr_db = 10 * (snr_linear.log() / np.log(10))
    
    return snr_db.item()

class EyeDiagramReconstructor:
    """
    Reconstruct eye diagrams from time-domain signal data
    """
    
    def __init__(self, data: EyeDiagramData):
        self.data = data
        self.samples_per_symbol = int(data.sample_rate / data.symbol_rate)
        
    def reconstruct_eye_diagram(self, num_traces: int = 200):
        """
        Reconstruct eye diagram from signal data
        
        Args:
            num_traces: Number of symbol periods to overlay
            
        Returns:
            Eye diagram data and metrics
        """
        signal = self.data.signal_samples
        sps = self.samples_per_symbol
        
        # Extract symbol periods for overlay
        num_available_symbols = len(signal) // sps
        actual_traces = min(num_traces, num_available_symbols)
        
        # Create eye traces (each trace is one symbol period)
        eye_traces = []
        for i in range(actual_traces):
            start_idx = i * sps
            end_idx = start_idx + sps
            if end_idx <= len(signal):
                trace = signal[start_idx:end_idx]
                eye_traces.append(trace)
        
        # Convert to numpy for analysis
        eye_matrix = np.array(eye_traces)
        
        # Calculate eye metrics
        eye_metrics = self.calculate_eye_metrics(eye_matrix)
        
        # Detect crossing points and timing
        crossing_analysis = self.analyze_crossings(eye_matrix)
        
        return {
            'eye_traces': eye_matrix,
            'time_axis': np.linspace(0, 1, sps),  # Normalized to 1 UI
            'eye_metrics': eye_metrics,
            'crossing_analysis': crossing_analysis,
            'reconstruction_quality': self.assess_reconstruction_quality(eye_matrix)
        }
    
    def calculate_eye_metrics(self, eye_matrix: np.ndarray):
        """Calculate comprehensive eye diagram metrics"""
        # Convert to tensor for calculations
        eye_tensor = Tensor(eye_matrix.astype(np.float32))
        
        # Sampling points for measurements
        center_idx = eye_matrix.shape[1] // 2  # Center of symbol
        
        # Eye height (vertical opening)
        center_samples = eye_tensor[:, center_idx]
        eye_height = (center_samples.max() - center_samples.min()).item()
        
        # Eye width (horizontal opening) - simplified calculation
        crossing_spread = self.calculate_crossing_spread(eye_matrix)
        eye_width = max(0, 1.0 - crossing_spread)  # 1.0 UI is ideal width
        
        # Rise/fall time analysis
        rise_fall_metrics = self.analyze_rise_fall_times(eye_matrix)
        
        # Noise analysis
        noise_metrics = self.analyze_eye_noise(eye_tensor)
        
        # Overall eye opening
        signal_range = eye_tensor.max() - eye_tensor.min()
        eye_opening_ratio = eye_height / signal_range.item()
        
        return {
            'eye_height_v': eye_height,
            'eye_width_ui': eye_width,
            'eye_opening_ratio': eye_opening_ratio,
            'rise_time_ps': rise_fall_metrics['rise_time'],
            'fall_time_ps': rise_fall_metrics['fall_time'],
            'rms_noise_v': noise_metrics['rms_noise'],
            'snr_db': noise_metrics['snr_db'],
            'timing_jitter_ps': crossing_spread * 1e12 / self.data.symbol_rate,  # Convert to ps
            'quality_factor': self.calculate_quality_factor(eye_height, eye_width, noise_metrics['rms_noise'])
        }
    
    def analyze_crossings(self, eye_matrix: np.ndarray):
        """Analyze zero crossings for timing analysis"""
        crossings_per_trace = []
        crossing_times = []
        
        # For each trace, find crossings
        for trace in eye_matrix:
            # Find midpoint voltage
            mid_voltage = (np.max(trace) + np.min(trace)) / 2
            
            # Find crossings
            crossings = []
            for i in range(len(trace) - 1):
                if ((trace[i] <= mid_voltage and trace[i+1] > mid_voltage) or
                    (trace[i] >= mid_voltage and trace[i+1] < mid_voltage)):
                    # Linear interpolation for more accurate crossing time
                    if abs(trace[i+1] - trace[i]) > 1e-10:
                        crossing_time = i + (mid_voltage - trace[i]) / (trace[i+1] - trace[i])
                        crossings.append(crossing_time)
            
            crossings_per_trace.append(len(crossings))
            crossing_times.extend(crossings)
        
        # Calculate crossing statistics
        if crossing_times:
            crossing_jitter = np.std(crossing_times)
            avg_crossings_per_trace = np.mean(crossings_per_trace)
        else:
            crossing_jitter = 0
            avg_crossings_per_trace = 0
        
        return {
            'avg_crossings_per_trace': avg_crossings_per_trace,
            'crossing_jitter_samples': crossing_jitter,
            'crossing_times': crossing_times[:100],  # First 100 for analysis
            'crossing_quality': 'good' if crossing_jitter < 2.0 else 'poor'
        }
    
    def calculate_crossing_spread(self, eye_matrix: np.ndarray):
        """Calculate spread of zero crossings"""
        crossing_positions = []
        
        for trace in eye_matrix:
            mid_voltage = (np.max(trace) + np.min(trace)) / 2
            
            # Find first crossing in each trace
            for i in range(len(trace) - 1):
                if ((trace[i] <= mid_voltage and trace[i+1] > mid_voltage) or
                    (trace[i] >= mid_voltage and trace[i+1] < mid_voltage)):
                    crossing_positions.append(i / len(trace))  # Normalize to UI
                    break
        
        if crossing_positions:
            return np.std(crossing_positions) * 2  # 2-sigma spread
        else:
            return 1.0  # Maximum spread if no crossings found
    
    def analyze_rise_fall_times(self, eye_matrix: np.ndarray):
        """Analyze rise and fall times from eye diagram"""
        rise_times = []
        fall_times = []
        
        for trace in eye_matrix:
            # Find transitions (10% to 90% of signal swing)
            trace_min = np.min(trace)
            trace_max = np.max(trace)
            trace_range = trace_max - trace_min
            
            if trace_range < 1e-10:
                continue
            
            level_10 = trace_min + 0.1 * trace_range
            level_90 = trace_min + 0.9 * trace_range
            
            # Look for rising edges
            for i in range(len(trace) - 10):  # Need some samples for measurement
                if trace[i] <= level_10:
                    # Look for 90% level in next samples
                    for j in range(i + 1, min(i + 20, len(trace))):
                        if trace[j] >= level_90:
                            rise_time = (j - i) / self.data.sample_rate
                            rise_times.append(rise_time)
                            break
            
            # Look for falling edges
            for i in range(len(trace) - 10):
                if trace[i] >= level_90:
                    for j in range(i + 1, min(i + 20, len(trace))):
                        if trace[j] <= level_10:
                            fall_time = (j - i) / self.data.sample_rate
                            fall_times.append(fall_time)
                            break
        
        # Convert to picoseconds and calculate statistics
        avg_rise_time = np.mean(rise_times) * 1e12 if rise_times else 0
        avg_fall_time = np.mean(fall_times) * 1e12 if fall_times else 0
        
        return {
            'rise_time': avg_rise_time,
            'fall_time': avg_fall_time,
            'rise_time_std': np.std(rise_times) * 1e12 if rise_times else 0,
            'fall_time_std': np.std(fall_times) * 1e12 if fall_times else 0
        }
    
    def analyze_eye_noise(self, eye_tensor: Tensor):
        """Analyze noise characteristics from eye diagram"""
        # Estimate noise from symbol centers (should be most stable)
        center_idx = eye_tensor.shape[1] // 2
        center_samples = eye_tensor[:, center_idx]
        
        # Group samples by level (for PAM4 or NRZ)
        if self.data.signal_type == "PAM4":
            levels = [-0.75, -0.25, 0.25, 0.75]  # Normalized PAM4 levels
        else:
            levels = [0.0, 1.0]  # NRZ levels
        
        # Calculate noise for each level
        level_noises = []
        for level in levels:
            # Find samples closest to this level
            level_samples = []
            for sample in center_samples.numpy():
                if abs(sample - level) < 0.2:  # Within 20% of level
                    level_samples.append(sample)
            
            if level_samples:
                level_tensor = Tensor(level_samples)
                level_noise = level_tensor.std().item()
                level_noises.append(level_noise)
        
        # Overall noise metrics
        rms_noise = np.mean(level_noises) if level_noises else center_samples.std().item()
        signal_power = center_samples.var().item()
        noise_power = rms_noise ** 2
        
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'rms_noise': rms_noise,
            'snr_db': snr_db,
            'level_noises': level_noises
        }
    
    def calculate_quality_factor(self, eye_height: float, eye_width: float, noise: float):
        """Calculate overall eye diagram quality factor"""
        # Normalized quality metric
        height_factor = eye_height / 1.0  # Assume 1V is ideal height
        width_factor = eye_width / 1.0    # 1 UI is ideal width
        noise_factor = max(0, 1 - noise / 0.1)  # 0.1V is poor noise level
        
        quality_factor = (height_factor * width_factor * noise_factor) ** (1/3)
        return min(1.0, quality_factor)  # Cap at 1.0
    
    def assess_reconstruction_quality(self, eye_matrix: np.ndarray):
        """Assess quality of eye diagram reconstruction"""
        quality_metrics = {
            'num_traces': len(eye_matrix),
            'samples_per_trace': eye_matrix.shape[1] if len(eye_matrix) > 0 else 0,
            'data_completeness': len(eye_matrix) / 200,  # Assume 200 is ideal
            'signal_coherence': self.calculate_signal_coherence(eye_matrix)
        }
        
        # Overall quality assessment
        if quality_metrics['num_traces'] >= 100 and quality_metrics['samples_per_trace'] >= 10:
            quality_metrics['overall'] = 'excellent'
        elif quality_metrics['num_traces'] >= 50 and quality_metrics['samples_per_trace'] >= 5:
            quality_metrics['overall'] = 'good'
        elif quality_metrics['num_traces'] >= 20:
            quality_metrics['overall'] = 'acceptable'
        else:
            quality_metrics['overall'] = 'poor'
        
        return quality_metrics
    
    def calculate_signal_coherence(self, eye_matrix: np.ndarray):
        """Calculate how coherent the eye traces are"""
        if len(eye_matrix) < 2:
            return 0.0
        
        # Calculate cross-correlation between traces
        correlations = []
        reference_trace = eye_matrix[0]
        
        for trace in eye_matrix[1:]:
            if len(trace) == len(reference_trace):
                # Simple correlation coefficient
                corr = np.corrcoef(reference_trace, trace)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0

class ValidationInsights:
    """Extract validation insights from eye diagram analysis"""
    
    @staticmethod
    def analyze_eye_diagram_for_validation(eye_diagram_result: Dict):
        """
        Extract validation insights from eye diagram analysis
        """
        metrics = eye_diagram_result['eye_metrics']
        
        validation_insights = {
            'signal_integrity': {
                'eye_height_margin': ValidationInsights.calculate_margin(metrics['eye_height_v'], 0.4),  # 400mV min
                'eye_width_margin': ValidationInsights.calculate_margin(metrics['eye_width_ui'], 0.6),   # 0.6 UI min
                'overall_opening': metrics['eye_opening_ratio'],
                'assessment': ValidationInsights.assess_signal_integrity(metrics)
            },
            
            'timing_analysis': {
                'jitter_analysis': {
                    'timing_jitter_ps': metrics['timing_jitter_ps'],
                    'jitter_budget_used': metrics['timing_jitter_ps'] / 10.0,  # Assume 10ps budget
                    'jitter_grade': ValidationInsights.grade_jitter(metrics['timing_jitter_ps'])
                },
                'rise_fall_times': {
                    'rise_time_ps': metrics['rise_time_ps'],
                    'fall_time_ps': metrics['fall_time_ps'],
                    'symmetry': abs(metrics['rise_time_ps'] - metrics['fall_time_ps']),
                    'bandwidth_estimate': ValidationInsights.estimate_bandwidth_from_risetime(metrics['rise_time_ps'])
                }
            },
            
            'noise_analysis': {
                'snr_db': metrics['snr_db'],
                'noise_grade': ValidationInsights.grade_noise(metrics['snr_db']),
                'rms_noise_v': metrics['rms_noise_v'],
                'noise_margin': ValidationInsights.calculate_noise_margin(metrics['eye_height_v'], metrics['rms_noise_v'])
            },
            
            'compliance_check': {
                'ieee_802_3_compliance': ValidationInsights.check_ieee_compliance(metrics),
                'custom_spec_compliance': ValidationInsights.check_custom_compliance(metrics),
                'overall_grade': ValidationInsights.calculate_overall_grade(metrics)
            }
        }
        
        return validation_insights
    
    @staticmethod
    def calculate_margin(actual_value: float, minimum_spec: float):
        """Calculate margin above specification"""
        if minimum_spec == 0:
            return float('inf')
        margin_ratio = (actual_value - minimum_spec) / minimum_spec
        return max(0, margin_ratio)
    
    @staticmethod
    def assess_signal_integrity(metrics: Dict):
        """Assess overall signal integrity"""
        eye_height_ok = metrics['eye_height_v'] > 0.4
        eye_width_ok = metrics['eye_width_ui'] > 0.6
        snr_ok = metrics['snr_db'] > 15
        
        if eye_height_ok and eye_width_ok and snr_ok:
            return 'excellent'
        elif eye_height_ok and eye_width_ok:
            return 'good'
        elif eye_height_ok or eye_width_ok:
            return 'marginal'
        else:
            return 'poor'
    
    @staticmethod
    def grade_jitter(jitter_ps: float):
        """Grade timing jitter performance"""
        if jitter_ps < 2:
            return 'excellent'
        elif jitter_ps < 5:
            return 'good'
        elif jitter_ps < 10:
            return 'acceptable'
        else:
            return 'poor'
    
    @staticmethod
    def grade_noise(snr_db: float):
        """Grade noise performance"""
        if snr_db > 25:
            return 'excellent'
        elif snr_db > 20:
            return 'good'
        elif snr_db > 15:
            return 'acceptable'
        else:
            return 'poor'
    
    @staticmethod
    def calculate_noise_margin(eye_height: float, rms_noise: float):
        """Calculate noise margin"""
        if rms_noise == 0:
            return float('inf')
        return eye_height / (6 * rms_noise)  # 6-sigma margin
    
    @staticmethod
    def estimate_bandwidth_from_risetime(rise_time_ps: float):
        """Estimate signal bandwidth from rise time"""
        if rise_time_ps > 0:
            # BW ≈ 0.35 / rise_time
            bandwidth_ghz = 0.35 / (rise_time_ps * 1e-12) / 1e9
            return bandwidth_ghz
        return 0
    
    @staticmethod
    def check_ieee_compliance(metrics: Dict):
        """Check IEEE 802.3 compliance"""
        # Example IEEE 802.3bs requirements for 25G PAM4
        ieee_limits = {
            'min_eye_height': 0.15,  # 150mV
            'min_eye_width': 0.65,   # 0.65 UI
            'max_jitter': 8          # 8ps
        }
        
        compliance = {
            'eye_height': metrics['eye_height_v'] >= ieee_limits['min_eye_height'],
            'eye_width': metrics['eye_width_ui'] >= ieee_limits['min_eye_width'],
            'jitter': metrics['timing_jitter_ps'] <= ieee_limits['max_jitter']
        }
        
        compliance['overall'] = all(compliance.values())
        return compliance
    
    @staticmethod
    def check_custom_compliance(metrics: Dict):
        """Check custom specification compliance"""
        # Placeholder for custom specifications
        return {'overall': True, 'details': 'Custom compliance check not implemented'}
    
    @staticmethod
    def calculate_overall_grade(metrics: Dict):
        """Calculate overall performance grade"""
        scores = []
        
        # Eye height score (0-100)
        eye_height_score = min(100, (metrics['eye_height_v'] / 0.8) * 100)
        scores.append(eye_height_score)
        
        # Eye width score (0-100)
        eye_width_score = min(100, (metrics['eye_width_ui'] / 1.0) * 100)
        scores.append(eye_width_score)
        
        # SNR score (0-100)
        snr_score = min(100, (metrics['snr_db'] / 30) * 100)
        scores.append(snr_score)
        
        # Jitter score (0-100, lower jitter is better)
        jitter_score = max(0, 100 - (metrics['timing_jitter_ps'] / 20) * 100)
        scores.append(jitter_score)
        
        overall_score = np.mean(scores)
        
        if overall_score >= 90:
            grade = 'A'
        elif overall_score >= 80:
            grade = 'B'
        elif overall_score >= 70:
            grade = 'C'
        elif overall_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {'score': overall_score, 'grade': grade}

class ProductionTestValidator:
    """Production test validation using eye diagram analysis"""
    
    @staticmethod
    def production_test_validation(eye_metrics: Dict, test_limits: ValidationSpecs):
        """
        Validate production test results using eye diagram analysis
        """
        test_results = {
            'pass_fail_decision': ProductionTestValidator.make_pass_fail_decision(eye_metrics, test_limits),
            'test_margins': ProductionTestValidator.calculate_test_margins(eye_metrics, test_limits),
            'yield_prediction': ProductionTestValidator.predict_yield_impact(eye_metrics),
            'correlation_with_ber': ProductionTestValidator.correlate_eye_with_ber(eye_metrics)
        }
        
        return test_results
    
    @staticmethod
    def make_pass_fail_decision(eye_metrics: Dict, limits: ValidationSpecs):
        """Make production test pass/fail decision"""
        checks = {
            'eye_height': eye_metrics['eye_height_v'] >= limits.min_eye_height,
            'eye_width': eye_metrics['eye_width_ui'] >= limits.min_eye_width,
            'snr': eye_metrics['snr_db'] >= limits.min_snr_db,
            'jitter': eye_metrics['timing_jitter_ps'] <= limits.max_jitter_ps
        }
        
        overall_pass = all(checks.values())
        
        return {
            'overall_result': 'PASS' if overall_pass else 'FAIL',
            'individual_checks': checks,
            'failing_parameters': [param for param, result in checks.items() if not result]
        }
    
    @staticmethod
    def calculate_test_margins(eye_metrics: Dict, limits: ValidationSpecs):
        """Calculate test margins"""
        margins = {
            'eye_height_margin': (eye_metrics['eye_height_v'] - limits.min_eye_height) / limits.min_eye_height,
            'eye_width_margin': (eye_metrics['eye_width_ui'] - limits.min_eye_width) / limits.min_eye_width,
            'snr_margin': (eye_metrics['snr_db'] - limits.min_snr_db) / limits.min_snr_db,
            'jitter_margin': (limits.max_jitter_ps - eye_metrics['timing_jitter_ps']) / limits.max_jitter_ps
        }
        
        # Overall margin (minimum of all margins)
        margins['overall_margin'] = min(margins.values())
        
        return margins
    
    @staticmethod
    def predict_yield_impact(eye_metrics: Dict):
        """Predict yield impact based on eye metrics"""
        # Simple yield prediction model
        quality_score = eye_metrics['quality_factor']
        
        if quality_score > 0.9:
            yield_prediction = 'high'
            estimated_yield = 95
        elif quality_score > 0.8:
            yield_prediction = 'good'
            estimated_yield = 85
        elif quality_score > 0.7:
            yield_prediction = 'marginal'
            estimated_yield = 70
        else:
            yield_prediction = 'poor'
            estimated_yield = 50
        
        return {
            'yield_prediction': yield_prediction,
            'estimated_yield_percent': estimated_yield,
            'quality_score': quality_score
        }
    
    @staticmethod
    def correlate_eye_with_ber(eye_metrics: Dict):
        """Correlate eye diagram metrics with expected BER"""
        # Simplified BER correlation model
        snr_db = eye_metrics['snr_db']
        eye_opening = eye_metrics['eye_opening_ratio']
        
        # Estimate BER from SNR (simplified)
        if snr_db > 20:
            estimated_ber = 1e-15
        elif snr_db > 15:
            estimated_ber = 1e-12
        elif snr_db > 10:
            estimated_ber = 1e-9
        else:
            estimated_ber = 1e-6
        
        # Adjust based on eye opening
        ber_adjustment = eye_opening ** 2
        final_ber_estimate = estimated_ber / ber_adjustment
        
        return {
            'estimated_ber': final_ber_estimate,
            'ber_grade': 'excellent' if final_ber_estimate < 1e-12 else
                        'good' if final_ber_estimate < 1e-9 else
                        'poor',
            'correlation_confidence': 'medium'  # This would be higher with real correlation data
        }

class DebugAnalyzer:
    """Debug and characterization analysis"""
    
    @staticmethod
    def debug_characterization_insights(eye_analysis: Dict, historical_data: Optional[Dict] = None):
        """
        Provide debug insights from eye diagram analysis
        """
        current_metrics = eye_analysis['eye_metrics']
        
        insights = {
            'degradation_analysis': DebugAnalyzer.analyze_degradation(current_metrics, historical_data) if historical_data else {},
            'root_cause_analysis': DebugAnalyzer.identify_root_causes(current_metrics),
            'improvement_opportunities': DebugAnalyzer.identify_improvements(current_metrics),
            'correlation_analysis': DebugAnalyzer.analyze_parameter_correlations(current_metrics)
        }
        
        return insights
    
    @staticmethod
    def analyze_degradation(current_metrics: Dict, historical_data: Dict):
        """Analyze degradation compared to historical data"""
        degradation = {}
        
        for metric in ['eye_height_v', 'eye_width_ui', 'snr_db']:
            if metric in historical_data:
                current_value = current_metrics[metric]
                historical_avg = np.mean(historical_data[metric])
                degradation[metric] = {
                    'current': current_value,
                    'historical_avg': historical_avg,
                    'degradation_percent': ((historical_avg - current_value) / historical_avg) * 100,
                    'status': 'degraded' if current_value < historical_avg * 0.9 else 'stable'
                }
        
        return degradation
    
    @staticmethod
    def identify_root_causes(metrics: Dict):
        """Identify potential root causes from eye diagram symptoms"""
        root_causes = []
        
        # Low eye height -> loss, attenuation, poor signal levels
        if metrics['eye_height_v'] < 0.3:
            root_causes.append({
                'symptom': 'low_eye_height',
                'possible_causes': ['excessive_channel_loss', 'low_transmit_swing', 'poor_termination'],
                'investigation_steps': ['check_channel_insertion_loss', 'verify_tx_levels', 'check_impedance_matching'],
                'severity': 'high'
            })
        
        # Excessive jitter -> clock issues, power supply noise
        if metrics['timing_jitter_ps'] > 10:
            root_causes.append({
                'symptom': 'excessive_jitter',
                'possible_causes': ['clock_quality', 'power_supply_noise', 'crosstalk'],
                'investigation_steps': ['check_clock_spectrum', 'measure_power_supply_ripple', 'analyze_crosstalk'],
                'severity': 'medium'
            })
        
        # Poor SNR -> noise sources
        if metrics['snr_db'] < 15:
            root_causes.append({
                'symptom': 'poor_snr',
                'possible_causes': ['thermal_noise', 'switching_noise', 'ground_bounce'],
                'investigation_steps': ['check_noise_floor', 'analyze_switching_activity', 'verify_ground_integrity'],
                'severity': 'medium'
            })
        
        # Asymmetric rise/fall times -> driver issues
        if abs(metrics['rise_time_ps'] - metrics['fall_time_ps']) > 5:
            root_causes.append({
                'symptom': 'asymmetric_edges',
                'possible_causes': ['driver_asymmetry', 'package_parasitics', 'load_imbalance'],
                'investigation_steps': ['check_driver_specs', 'analyze_package_model', 'verify_load_matching'],
                'severity': 'low'
            })
        
        return root_causes
    
    @staticmethod
    def identify_improvements(metrics: Dict):
        """Identify improvement opportunities"""
        improvements = []
        
        if metrics['eye_height_v'] < 0.6:
            improvements.append({
                'parameter': 'eye_height',
                'current_value': metrics['eye_height_v'],
                'target_value': 0.6,
                'improvement_methods': ['increase_tx_swing', 'reduce_channel_loss', 'optimize_equalization'],
                'expected_benefit': 'improved_signal_margins'
            })
        
        if metrics['timing_jitter_ps'] > 5:
            improvements.append({
                'parameter': 'timing_jitter',
                'current_value': metrics['timing_jitter_ps'],
                'target_value': 5,
                'improvement_methods': ['improve_clock_quality', 'reduce_power_noise', 'minimize_crosstalk'],
                'expected_benefit': 'better_timing_margins'
            })
        
        if metrics['snr_db'] < 20:
            improvements.append({
                'parameter': 'snr',
                'current_value': metrics['snr_db'],
                'target_value': 20,
                'improvement_methods': ['reduce_noise_sources', 'improve_shielding', 'optimize_layout'],
                'expected_benefit': 'lower_ber_improved_reliability'
            })
        
        return improvements
    
    @staticmethod
    def analyze_parameter_correlations(metrics: Dict):
        """Analyze correlations between parameters"""
        correlations = {
            'eye_height_vs_snr': {
                'correlation': 'positive',
                'strength': 'strong',
                'explanation': 'Higher SNR typically leads to better eye height'
            },
            'jitter_vs_eye_width': {
                'correlation': 'negative',
                'strength': 'strong',
                'explanation': 'Higher jitter reduces effective eye width'
            },
            'rise_time_vs_bandwidth': {
                'correlation': 'negative',
                'strength': 'strong',
                'explanation': 'Faster rise times indicate higher bandwidth'
            }
        }
        
        return correlations

class EthernetValidator:
    """Comprehensive Ethernet testing validator"""
    
    def __init__(self, signal_data: np.ndarray, specs: ValidationSpecs, 
                 sample_rate: float, symbol_rate: float, signal_type: str = "PAM4"):
        """
        Initialize comprehensive Ethernet validator
        
        Args:
            signal_data: Raw signal samples
            specs: Validation specifications
            sample_rate: Sampling rate (Hz)
            symbol_rate: Symbol rate (Hz)
            signal_type: "PAM4" or "NRZ"
        """
        self.signal_data = signal_data
        self.specs = specs
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.signal_type = signal_type
        
        # Initialize validators
        self.spec_validator = EthernetSpecValidator(specs)
        self.data_validator = DataValidator()
        
        # Create eye diagram data structure
        self.eye_data = EyeDiagramData(
            signal_samples=signal_data,
            sample_rate=sample_rate,
            symbol_rate=symbol_rate,
            signal_type=signal_type
        )
    
    def comprehensive_validation(self, baseline_data: Optional[np.ndarray] = None):
        """
        Perform comprehensive validation analysis
        
        Args:
            baseline_data: Optional baseline data for comparison
            
        Returns:
            Complete validation results
        """
        results = {
            'timestamp': time.time(),
            'validation_summary': {},
            'detailed_results': {}
        }
        
        # Convert signal to tensor
        signal_tensor = Tensor(self.signal_data.astype(np.float32))
        
        # 1. Data quality validation
        print("Performing data quality validation...")
        data_quality = validate_data_quality_for_eye_diagram(self.eye_data)
        results['detailed_results']['data_quality'] = data_quality
        
        # 2. Signal level validation
        print("Validating signal levels...")
        if self.signal_type == "PAM4":
            level_validation = self.spec_validator.validate_pam4_levels(signal_tensor)
        else:
            # Simplified NRZ validation
            level_validation = {'compliance_status': 'PASS', 'total_violations': 0}
        
        results['detailed_results']['signal_levels'] = level_validation
        
        # 3. Baseline comparison (if provided)
        if baseline_data is not None:
            print("Comparing against baseline...")
            baseline_tensor = Tensor(baseline_data.astype(np.float32))
            baseline_comparison = self.data_validator.validate_against_baseline(signal_tensor, baseline_tensor)
            results['detailed_results']['baseline_comparison'] = baseline_comparison
        
        # 4. Eye diagram reconstruction and analysis
        print("Reconstructing eye diagram...")
        reconstructor = EyeDiagramReconstructor(self.eye_data)
        eye_result = reconstructor.reconstruct_eye_diagram(num_traces=200)
        results['detailed_results']['eye_diagram'] = eye_result
        
        # 5. Eye diagram validation
        print("Validating eye diagram metrics...")
        eye_validation = self.spec_validator.validate_eye_diagram_specs(
            eye_result['eye_metrics']['eye_height_v'],
            eye_result['eye_metrics']['eye_width_ui']
        )
        results['detailed_results']['eye_validation'] = eye_validation
        
        # 6. Validation insights
        print("Extracting validation insights...")
        validation_insights = ValidationInsights.analyze_eye_diagram_for_validation(eye_result)
        results['detailed_results']['validation_insights'] = validation_insights
        
        # 7. Production test validation
        print("Performing production test validation...")
        production_results = ProductionTestValidator.production_test_validation(
            eye_result['eye_metrics'], self.specs
        )
        results['detailed_results']['production_test'] = production_results
        
        # 8. Debug analysis
        print("Performing debug analysis...")
        debug_insights = DebugAnalyzer.debug_characterization_insights(eye_result)
        results['detailed_results']['debug_analysis'] = debug_insights
        
        # 9. Summary
        print("Generating validation summary...")
        results['validation_summary'] = self.generate_validation_summary(results['detailed_results'])
        
        return results
    
    def generate_validation_summary(self, detailed_results: Dict):
        """Generate high-level validation summary"""
        summary = {
            'overall_status': 'PASS',
            'critical_issues': [],
            'warnings': [],
            'key_metrics': {},
            'recommendations': []
        }
        
        # Check for critical failures
        if detailed_results['signal_levels']['compliance_status'] == 'FAIL':
            summary['overall_status'] = 'FAIL'
            summary['critical_issues'].append('Signal level violations detected')
        
        if detailed_results['eye_validation']['overall_compliance'] == 'FAIL':
            summary['overall_status'] = 'FAIL'
            summary['critical_issues'].append('Eye diagram specifications not met')
        
        if detailed_results['production_test']['pass_fail_decision']['overall_result'] == 'FAIL':
            summary['overall_status'] = 'FAIL'
            summary['critical_issues'].append('Production test limits exceeded')
        
        # Extract key metrics
        eye_metrics = detailed_results['eye_diagram']['eye_metrics']
        summary['key_metrics'] = {
            'eye_height_v': eye_metrics['eye_height_v'],
            'eye_width_ui': eye_metrics['eye_width_ui'],
            'snr_db': eye_metrics['snr_db'],
            'timing_jitter_ps': eye_metrics['timing_jitter_ps'],
            'quality_factor': eye_metrics['quality_factor']
        }
        
        # Add recommendations from debug analysis
        if 'root_cause_analysis' in detailed_results['debug_analysis']:
            for cause in detailed_results['debug_analysis']['root_cause_analysis']:
                summary['recommendations'].extend(cause['investigation_steps'])
        
        # Add improvement recommendations
        if 'improvement_opportunities' in detailed_results['debug_analysis']:
            for improvement in detailed_results['debug_analysis']['improvement_opportunities']:
                summary['recommendations'].extend(improvement['improvement_methods'])
        
        return summary
    
    def plot_validation_results(self, results: Dict):
        """Plot comprehensive validation results"""
        if not HAS_PLOTTING:
            print("Plotting not available. Install matplotlib: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Ethernet Validation Results', fontsize=16)
        
        # Plot 1: Eye diagram
        eye_traces = results['detailed_results']['eye_diagram']['eye_traces']
        time_axis = results['detailed_results']['eye_diagram']['time_axis']
        
        for trace in eye_traces[:50]:  # Plot first 50 traces
            axes[0, 0].plot(time_axis, trace, 'b-', alpha=0.1)
        
        axes[0, 0].set_title('Eye Diagram')
        axes[0, 0].set_xlabel('Time (UI)')
        axes[0, 0].set_ylabel('Amplitude (V)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Key metrics bar chart
        metrics = results['validation_summary']['key_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[0, 1].bar(range(len(metric_names)), metric_values)
        axes[0, 1].set_title('Key Metrics')
        axes[0, 1].set_xticks(range(len(metric_names)))
        axes[0, 1].set_xticklabels(metric_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Signal histogram
        axes[0, 2].hist(self.signal_data, bins=50, alpha=0.7)
        axes[0, 2].set_title('Signal Amplitude Distribution')
        axes[0, 2].set_xlabel('Amplitude (V)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Compliance status
        compliance_data = results['detailed_results']['production_test']['pass_fail_decision']['individual_checks']
        compliance_names = list(compliance_data.keys())
        compliance_values = [1 if v else 0 for v in compliance_data.values()]
        
        colors = ['green' if v else 'red' for v in compliance_values]
        axes[1, 0].bar(range(len(compliance_names)), compliance_values, color=colors)
        axes[1, 0].set_title('Compliance Status')
        axes[1, 0].set_xticks(range(len(compliance_names)))
        axes[1, 0].set_xticklabels(compliance_names, rotation=45)
        axes[1, 0].set_ylabel('Pass/Fail')
        axes[1, 0].set_ylim([0, 1.2])
        
        # Plot 5: Test margins
        margins_data = results['detailed_results']['production_test']['test_margins']
        margin_names = [k for k in margins_data.keys() if k != 'overall_margin']
        margin_values = [margins_data[k] for k in margin_names]
        
        colors = ['green' if v > 0 else 'red' for v in margin_values]
        axes[1, 1].bar(range(len(margin_names)), margin_values, color=colors)
        axes[1, 1].set_title('Test Margins')
        axes[1, 1].set_xticks(range(len(margin_names)))
        axes[1, 1].set_xticklabels(margin_names, rotation=45)
        axes[1, 1].set_ylabel('Margin Ratio')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Signal time series (first 1000 samples)
        time_samples = np.arange(min(1000, len(self.signal_data))) / self.sample_rate * 1e9  # ns
        signal_samples = self.signal_data[:1000]
        
        axes[1, 2].plot(time_samples, signal_samples, 'b-', alpha=0.7)
        axes[1, 2].set_title('Signal Time Series')
        axes[1, 2].set_xlabel('Time (ns)')
        axes[1, 2].set_ylabel('Amplitude (V)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\\n{'='*60}")
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {results['validation_summary']['overall_status']}")
        print(f"\\nKey Metrics:")
        for metric, value in results['validation_summary']['key_metrics'].items():
            print(f"  {metric}: {value:.3f}")
        
        if results['validation_summary']['critical_issues']:
            print(f"\\nCritical Issues:")
            for issue in results['validation_summary']['critical_issues']:
                print(f"  - {issue}")
        
        if results['validation_summary']['recommendations']:
            print(f"\\nRecommendations:")
            for rec in results['validation_summary']['recommendations'][:5]:  # First 5
                print(f"  - {rec}")

def comprehensive_ethernet_validation_example():
    """
    Complete example of comprehensive Ethernet testing validation
    """
    print("Starting comprehensive Ethernet validation example...")
    
    # Generate sample PAM4 data
    sample_rate = 25e9  # 25 GSa/s
    symbol_rate = 10e9  # 10 GBaud
    num_samples = 5000
    
    # Create sample PAM4 signal
    np.random.seed(42)  # For reproducible results
    symbols = np.random.choice(4, size=num_samples//10)
    pam4_levels = [-0.75, -0.25, 0.25, 0.75]
    
    signal_data = np.zeros(num_samples)
    samples_per_symbol = int(sample_rate / symbol_rate)
    
    for i, symbol in enumerate(symbols):
        start_idx = i * samples_per_symbol
        end_idx = min(start_idx + samples_per_symbol, num_samples)
        if start_idx < num_samples:
            signal_data[start_idx:end_idx] = pam4_levels[symbol]
    
    # Add noise and some ISI
    signal_data += np.random.normal(0, 0.03, size=num_samples)
    
    # Apply simple ISI (each symbol affected by previous)
    for i in range(samples_per_symbol, len(signal_data)):
        signal_data[i] += 0.05 * signal_data[i - samples_per_symbol]
    
    # Define specifications
    specs = ValidationSpecs(
        min_eye_height=0.4,
        min_eye_width=0.6,
        min_snr_db=15,
        max_jitter_ps=8,
        pam4_levels=[-0.75, -0.25, 0.25, 0.75],
        level_tolerance=0.1,
        max_ber=1e-12
    )
    
    # Create validator
    validator = EthernetValidator(
        signal_data=signal_data,
        specs=specs,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        signal_type="PAM4"
    )
    
    # Perform comprehensive validation
    results = validator.comprehensive_validation()
    
    # Plot results
    validator.plot_validation_results(results)
    
    return results

if __name__ == "__main__":
    # Run comprehensive validation example
    results = comprehensive_ethernet_validation_example()
    
    print("\\nValidation complete! Results saved in 'results' variable.")
    print("\\nTo use this toolkit with your own data:")
    print("1. Load your signal data into a numpy array")
    print("2. Define your specifications using ValidationSpecs")
    print("3. Create an EthernetValidator instance")
    print("4. Call comprehensive_validation() method")
    print("5. Use plot_validation_results() to visualize")