#!/usr/bin/env python3
"""
Ethernet Testing Anomaly Detection System using Tinygrad
=======================================================

Specialized system for PAM4/NRZ, CTLE, ISI, and BER analysis in Ethernet testing.
Designed to work within tinygrad's capabilities while providing meaningful analysis.

Key Features:
- PAM4/NRZ signal level analysis and eye diagram metrics
- CTLE performance monitoring through gain measurements
- ISI detection via eye closure and timing analysis  
- BER statistical analysis and pattern detection
- Real-time anomaly detection for Ethernet test parameters

Note: This implementation works around tinygrad's lack of complex number and FFT support
by focusing on time-domain and statistical analysis techniques.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum

from tinygrad import Tensor
from tinygrad.helpers import getenv

# Optional imports for enhanced functionality
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

class SignalType(Enum):
    """Signal encoding types"""
    NRZ = "NRZ"
    PAM4 = "PAM4"

class CTLEMode(Enum):
    """CTLE operation modes"""
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    DISABLED = "disabled"

@dataclass
class EthernetTestSpec:
    """Ethernet test specifications and thresholds"""
    # Signal levels (for PAM4: 4 levels, for NRZ: 2 levels)
    signal_type: SignalType = SignalType.PAM4
    
    # PAM4 signal levels (in volts)
    pam4_levels: List[float] = None
    pam4_tolerance: float = 0.05  # ±50mV tolerance
    
    # NRZ signal levels
    nrz_high: float = 1.0
    nrz_low: float = 0.0
    nrz_tolerance: float = 0.1
    
    # Eye diagram specifications
    eye_height_min: float = 0.4  # Minimum eye height (V)
    eye_width_min: float = 0.6   # Minimum eye width (UI - Unit Interval)
    
    # BER specifications
    ber_threshold: float = 1e-12  # Maximum acceptable BER
    ber_sample_size: int = 1000000  # Minimum samples for BER calculation
    
    # CTLE specifications
    ctle_gain_min: float = 0.0    # Minimum CTLE gain (dB)
    ctle_gain_max: float = 20.0   # Maximum CTLE gain (dB)
    ctle_adaptation_rate: float = 0.1  # Expected adaptation rate
    
    # ISI specifications
    isi_threshold: float = 0.15   # Maximum acceptable ISI ratio
    timing_margin_min: float = 0.3  # Minimum timing margin (UI)
    
    def __post_init__(self):
        if self.pam4_levels is None:
            # Default PAM4 levels: -3, -1, +1, +3 (normalized)
            self.pam4_levels = [-0.75, -0.25, 0.25, 0.75]

class EyeDiagramMetrics(NamedTuple):
    """Eye diagram analysis results"""
    eye_height: float
    eye_width: float
    eye_opening: float
    crossing_percentage: float
    amplitude_imbalance: float
    timing_jitter: float

class CTLEMetrics(NamedTuple):
    """CTLE performance metrics"""
    gain_setting: float
    adaptation_speed: float
    frequency_response: List[float]
    gain_stability: float
    convergence_status: bool

class BERMetrics(NamedTuple):
    """BER analysis results"""
    bit_error_rate: float
    total_bits: int
    error_count: int
    error_burst_count: int
    error_distribution: List[int]
    link_margin_db: float

class ISIMetrics(NamedTuple):
    """ISI analysis results"""
    isi_ratio: float
    timing_margin: float
    pattern_dependency: Dict[str, float]
    worst_case_isi: float
    crosstalk_estimate: float

class EthernetTestAnalyzer:
    """
    Comprehensive Ethernet testing analyzer for PAM4/NRZ signals
    """
    
    def __init__(self, spec: EthernetTestSpec):
        """
        Initialize the Ethernet test analyzer
        
        Args:
            spec: Ethernet test specifications
        """
        self.spec = spec
        self.signal_history = []
        self.timestamp_history = []
        self.ber_history = []
        self.ctle_history = []
        
        # Pre-calculate some values for efficiency
        self._setup_signal_parameters()
        
    def _setup_signal_parameters(self):
        """Setup signal-specific parameters"""
        if self.spec.signal_type == SignalType.PAM4:
            self.signal_levels = self.spec.pam4_levels
            self.level_tolerance = self.spec.pam4_tolerance
            self.num_levels = 4
        else:  # NRZ
            self.signal_levels = [self.spec.nrz_low, self.spec.nrz_high]
            self.level_tolerance = self.spec.nrz_tolerance
            self.num_levels = 2
    
    def classify_signal_levels(self, signal: Tensor) -> Tensor:
        """
        Classify signal samples into discrete levels (PAM4 or NRZ)
        
        Args:
            signal: Raw signal samples
            
        Returns:
            Tensor of classified level indices
        """
        # Convert signal to numpy for level classification
        signal_np = signal.numpy()
        
        # Find closest signal level for each sample
        level_indices = np.zeros(len(signal_np), dtype=np.int32)
        
        for i, sample in enumerate(signal_np):
            # Find closest level
            distances = [abs(sample - level) for level in self.signal_levels]
            level_indices[i] = np.argmin(distances)
        
        return Tensor(level_indices)
    
    def analyze_eye_diagram(self, signal: Tensor, sample_rate: float, 
                          symbol_rate: float) -> EyeDiagramMetrics:
        """
        Analyze eye diagram metrics from signal samples
        
        Args:
            signal: Signal samples
            sample_rate: Sampling rate (Hz)
            symbol_rate: Symbol rate (baud)
            
        Returns:
            Eye diagram analysis results
        """
        samples_per_symbol = int(sample_rate / symbol_rate)
        signal_np = signal.numpy()
        
        # Reshape signal into symbol periods for eye diagram construction
        num_symbols = len(signal_np) // samples_per_symbol
        eye_data = signal_np[:num_symbols * samples_per_symbol].reshape(
            num_symbols, samples_per_symbol)
        
        # Calculate eye metrics using tinygrad operations
        eye_tensor = Tensor(eye_data)
        
        # Eye height: difference between max and min at center
        center_idx = samples_per_symbol // 2
        center_samples = eye_tensor[:, center_idx]
        eye_height = (center_samples.max() - center_samples.min()).item()
        
        # Eye width: measure at crossing points
        # Find zero crossings (simplified for this implementation)
        crossing_threshold = (center_samples.max() + center_samples.min()) / 2
        
        # Timing jitter: std deviation of zero crossing times
        crossings = []
        for i in range(num_symbols - 1):
            symbol_data = eye_data[i]
            # Find approximate crossing point
            for j in range(len(symbol_data) - 1):
                if ((symbol_data[j] <= crossing_threshold.item() and 
                     symbol_data[j+1] > crossing_threshold.item()) or
                    (symbol_data[j] >= crossing_threshold.item() and 
                     symbol_data[j+1] < crossing_threshold.item())):
                    crossings.append(j)
                    break
        
        timing_jitter = np.std(crossings) if crossings else 0.0
        
        # Eye opening: ratio of usable eye area
        eye_opening = eye_height / (self.signal_levels[-1] - self.signal_levels[0])
        
        # Crossing percentage: ratio of actual to ideal crossing
        crossing_percentage = len(crossings) / num_symbols if num_symbols > 0 else 0.0
        
        # Amplitude imbalance (for PAM4)
        if self.spec.signal_type == SignalType.PAM4:
            level_counts = np.zeros(4)
            classified_levels = self.classify_signal_levels(signal).numpy()
            for level in range(4):
                level_counts[level] = np.sum(classified_levels == level)
            amplitude_imbalance = np.std(level_counts) / np.mean(level_counts)
        else:
            amplitude_imbalance = 0.0
        
        # Eye width calculation (simplified)
        eye_width = max(0.0, 1.0 - timing_jitter / samples_per_symbol)
        
        return EyeDiagramMetrics(
            eye_height=eye_height,
            eye_width=eye_width,
            eye_opening=eye_opening,
            crossing_percentage=crossing_percentage,
            amplitude_imbalance=amplitude_imbalance,
            timing_jitter=timing_jitter
        )
    
    def analyze_ctle_performance(self, input_signal: Tensor, 
                               output_signal: Tensor, 
                               ctle_gain: float) -> CTLEMetrics:
        """
        Analyze CTLE (Continuous Time Linear Equalizer) performance
        
        Args:
            input_signal: Signal before CTLE
            output_signal: Signal after CTLE
            ctle_gain: Current CTLE gain setting
            
        Returns:
            CTLE performance metrics
        """
        # Calculate gain through power ratio (simplified)
        input_power = (input_signal.pow(2)).mean()
        output_power = (output_signal.pow(2)).mean()
        measured_gain = 10 * (output_power / input_power).log() / np.log(10)
        
        # Adaptation speed: change in gain over time
        if len(self.ctle_history) > 0:
            last_gain = self.ctle_history[-1]['gain']
            adaptation_speed = abs(ctle_gain - last_gain)
        else:
            adaptation_speed = 0.0
        
        # Gain stability: variance over recent history
        recent_gains = [entry['gain'] for entry in self.ctle_history[-10:]]
        gain_stability = np.std(recent_gains) if len(recent_gains) > 1 else 0.0
        
        # Frequency response estimation (simplified time-domain approach)
        # This is a limitation without FFT - we'll estimate based on signal characteristics
        frequency_response = []
        
        # Estimate response at different "frequencies" by analyzing signal patterns
        for window_size in [5, 10, 20, 50]:  # Different time scales
            if len(output_signal) > window_size:
                # Analyze signal variation at different time scales
                windowed_var = []
                for i in range(0, len(output_signal) - window_size, window_size):
                    window = output_signal[i:i+window_size]
                    windowed_var.append(window.var().item())
                frequency_response.append(np.mean(windowed_var))
        
        # Convergence status: is CTLE adaptation stable?
        convergence_status = (adaptation_speed < self.spec.ctle_adaptation_rate and
                            gain_stability < 0.5)
        
        return CTLEMetrics(
            gain_setting=ctle_gain,
            adaptation_speed=adaptation_speed,
            frequency_response=frequency_response,
            gain_stability=gain_stability,
            convergence_status=convergence_status
        )
    
    def analyze_ber(self, received_bits: List[int], 
                   transmitted_bits: List[int]) -> BERMetrics:
        """
        Analyze Bit Error Rate and error patterns
        
        Args:
            received_bits: Received bit sequence
            transmitted_bits: Transmitted bit sequence (reference)
            
        Returns:
            BER analysis results
        """
        if len(received_bits) != len(transmitted_bits):
            raise ValueError("Received and transmitted sequences must be same length")
        
        # Convert to tensors for efficient processing
        rx_tensor = Tensor(received_bits)
        tx_tensor = Tensor(transmitted_bits)
        
        # Calculate errors
        errors = (rx_tensor != tx_tensor).numpy().astype(int)
        total_bits = len(received_bits)
        error_count = int(np.sum(errors))
        
        # Bit error rate
        bit_error_rate = error_count / total_bits if total_bits > 0 else 0.0
        
        # Error burst analysis
        error_bursts = []
        in_burst = False
        burst_length = 0
        
        for error in errors:
            if error:
                if not in_burst:
                    in_burst = True
                    burst_length = 1
                else:
                    burst_length += 1
            else:
                if in_burst:
                    error_bursts.append(burst_length)
                    in_burst = False
                    burst_length = 0
        
        error_burst_count = len(error_bursts)
        
        # Error distribution (histogram of error positions modulo pattern length)
        pattern_length = 64  # Assume 64-bit test pattern
        error_positions = np.where(errors)[0]
        error_distribution = np.zeros(pattern_length)
        for pos in error_positions:
            error_distribution[pos % pattern_length] += 1
        
        # Link margin calculation
        if bit_error_rate > 0:
            link_margin_db = -20 * np.log10(bit_error_rate / self.spec.ber_threshold)
        else:
            link_margin_db = float('inf')
        
        return BERMetrics(
            bit_error_rate=bit_error_rate,
            total_bits=total_bits,
            error_count=error_count,
            error_burst_count=error_burst_count,
            error_distribution=error_distribution.tolist(),
            link_margin_db=link_margin_db
        )
    
    def analyze_isi(self, signal: Tensor, symbol_rate: float, 
                   sample_rate: float) -> ISIMetrics:
        """
        Analyze Inter-Symbol Interference
        
        Args:
            signal: Signal samples
            symbol_rate: Symbol rate (baud)
            sample_rate: Sampling rate (Hz)
            
        Returns:
            ISI analysis results
        """
        samples_per_symbol = int(sample_rate / symbol_rate)
        signal_np = signal.numpy()
        
        # Classify signal levels
        classified_levels = self.classify_signal_levels(signal).numpy()
        
        # ISI analysis using pattern dependency
        pattern_dependency = {}
        
        # Analyze 3-symbol patterns for ISI effects
        for i in range(len(classified_levels) - 2):
            pattern = f"{classified_levels[i]}{classified_levels[i+1]}{classified_levels[i+2]}"
            if pattern not in pattern_dependency:
                pattern_dependency[pattern] = []
            
            # Look at the middle symbol's actual value vs expected
            middle_idx = (i + 1) * samples_per_symbol + samples_per_symbol // 2
            if middle_idx < len(signal_np):
                expected_level = self.signal_levels[classified_levels[i+1]]
                actual_value = signal_np[middle_idx]
                deviation = abs(actual_value - expected_level)
                pattern_dependency[pattern].append(deviation)
        
        # Calculate average ISI for each pattern
        pattern_isi = {}
        for pattern, deviations in pattern_dependency.items():
            if deviations:
                pattern_isi[pattern] = np.mean(deviations)
            else:
                pattern_isi[pattern] = 0.0
        
        # Overall ISI ratio
        all_deviations = [dev for devs in pattern_dependency.values() for dev in devs]
        if all_deviations:
            signal_range = max(self.signal_levels) - min(self.signal_levels)
            isi_ratio = np.mean(all_deviations) / signal_range
            worst_case_isi = max(all_deviations) / signal_range
        else:
            isi_ratio = 0.0
            worst_case_isi = 0.0
        
        # Timing margin estimation
        eye_metrics = self.analyze_eye_diagram(signal, sample_rate, symbol_rate)
        timing_margin = eye_metrics.eye_width
        
        # Crosstalk estimation (simplified)
        # Look for correlations between adjacent symbols
        crosstalk_samples = []
        for i in range(1, len(classified_levels)):
            if classified_levels[i] != classified_levels[i-1]:
                # Transition - look for overshoot/undershoot
                start_idx = i * samples_per_symbol
                end_idx = start_idx + samples_per_symbol
                if end_idx < len(signal_np):
                    transition_signal = signal_np[start_idx:end_idx]
                    expected_final = self.signal_levels[classified_levels[i]]
                    actual_final = transition_signal[-1]
                    crosstalk_samples.append(abs(actual_final - expected_final))
        
        crosstalk_estimate = np.mean(crosstalk_samples) if crosstalk_samples else 0.0
        
        return ISIMetrics(
            isi_ratio=isi_ratio,
            timing_margin=timing_margin,
            pattern_dependency=pattern_isi,
            worst_case_isi=worst_case_isi,
            crosstalk_estimate=crosstalk_estimate
        )
    
    def detect_ethernet_anomalies(self, signal: Tensor, 
                                sample_rate: float, 
                                symbol_rate: float,
                                received_bits: Optional[List[int]] = None,
                                transmitted_bits: Optional[List[int]] = None,
                                ctle_gain: Optional[float] = None,
                                input_signal: Optional[Tensor] = None) -> Dict[str, any]:
        """
        Comprehensive anomaly detection for Ethernet testing parameters
        
        Args:
            signal: Main signal to analyze
            sample_rate: Sampling rate (Hz)
            symbol_rate: Symbol rate (baud)
            received_bits: Received bit sequence (for BER analysis)
            transmitted_bits: Transmitted bit sequence (for BER analysis)
            ctle_gain: Current CTLE gain setting
            input_signal: Signal before CTLE (for CTLE analysis)
            
        Returns:
            Dictionary containing all analysis results and anomalies
        """
        results = {
            'timestamp': time.time(),
            'anomalies_detected': False,
            'anomaly_details': [],
            'analysis_results': {}
        }
        
        # 1. Eye Diagram Analysis
        eye_metrics = self.analyze_eye_diagram(signal, sample_rate, symbol_rate)
        results['analysis_results']['eye_diagram'] = eye_metrics
        
        # Check for eye diagram anomalies
        if eye_metrics.eye_height < self.spec.eye_height_min:
            results['anomaly_details'].append({
                'type': 'eye_height_violation',
                'severity': 'high',
                'value': eye_metrics.eye_height,
                'threshold': self.spec.eye_height_min,
                'description': f"Eye height {eye_metrics.eye_height:.3f}V below minimum {self.spec.eye_height_min:.3f}V"
            })
            results['anomalies_detected'] = True
        
        if eye_metrics.eye_width < self.spec.eye_width_min:
            results['anomaly_details'].append({
                'type': 'eye_width_violation',
                'severity': 'high',
                'value': eye_metrics.eye_width,
                'threshold': self.spec.eye_width_min,
                'description': f"Eye width {eye_metrics.eye_width:.3f}UI below minimum {self.spec.eye_width_min:.3f}UI"
            })
            results['anomalies_detected'] = True
        
        # 2. ISI Analysis
        isi_metrics = self.analyze_isi(signal, symbol_rate, sample_rate)
        results['analysis_results']['isi'] = isi_metrics
        
        if isi_metrics.isi_ratio > self.spec.isi_threshold:
            results['anomaly_details'].append({
                'type': 'isi_violation',
                'severity': 'medium',
                'value': isi_metrics.isi_ratio,
                'threshold': self.spec.isi_threshold,
                'description': f"ISI ratio {isi_metrics.isi_ratio:.3f} exceeds threshold {self.spec.isi_threshold:.3f}"
            })
            results['anomalies_detected'] = True
        
        if isi_metrics.timing_margin < self.spec.timing_margin_min:
            results['anomaly_details'].append({
                'type': 'timing_margin_violation',
                'severity': 'high',
                'value': isi_metrics.timing_margin,
                'threshold': self.spec.timing_margin_min,
                'description': f"Timing margin {isi_metrics.timing_margin:.3f}UI below minimum {self.spec.timing_margin_min:.3f}UI"
            })
            results['anomalies_detected'] = True
        
        # 3. BER Analysis (if data provided)
        if received_bits is not None and transmitted_bits is not None:
            ber_metrics = self.analyze_ber(received_bits, transmitted_bits)
            results['analysis_results']['ber'] = ber_metrics
            
            if ber_metrics.bit_error_rate > self.spec.ber_threshold:
                results['anomaly_details'].append({
                    'type': 'ber_violation',
                    'severity': 'critical',
                    'value': ber_metrics.bit_error_rate,
                    'threshold': self.spec.ber_threshold,
                    'description': f"BER {ber_metrics.bit_error_rate:.2e} exceeds threshold {self.spec.ber_threshold:.2e}"
                })
                results['anomalies_detected'] = True
            
            if ber_metrics.error_burst_count > 10:  # Arbitrary threshold
                results['anomaly_details'].append({
                    'type': 'error_burst_detected',
                    'severity': 'medium',
                    'value': ber_metrics.error_burst_count,
                    'threshold': 10,
                    'description': f"Error bursts detected: {ber_metrics.error_burst_count}"
                })
                results['anomalies_detected'] = True
        
        # 4. CTLE Analysis (if data provided)
        if ctle_gain is not None and input_signal is not None:
            ctle_metrics = self.analyze_ctle_performance(input_signal, signal, ctle_gain)
            results['analysis_results']['ctle'] = ctle_metrics
            
            if not ctle_metrics.convergence_status:
                results['anomaly_details'].append({
                    'type': 'ctle_convergence_failure',
                    'severity': 'medium',
                    'value': ctle_metrics.adaptation_speed,
                    'threshold': self.spec.ctle_adaptation_rate,
                    'description': f"CTLE not converged, adaptation speed: {ctle_metrics.adaptation_speed:.3f}"
                })
                results['anomalies_detected'] = True
            
            if ctle_gain < self.spec.ctle_gain_min or ctle_gain > self.spec.ctle_gain_max:
                results['anomaly_details'].append({
                    'type': 'ctle_gain_out_of_range',
                    'severity': 'high',
                    'value': ctle_gain,
                    'threshold': f"{self.spec.ctle_gain_min}-{self.spec.ctle_gain_max}",
                    'description': f"CTLE gain {ctle_gain:.1f}dB outside acceptable range"
                })
                results['anomalies_detected'] = True
        
        return results
    
    def plot_analysis_results(self, signal: Tensor, analysis_results: Dict[str, any]):
        """Plot comprehensive analysis results"""
        if not HAS_PLOTTING:
            print("Plotting not available. Install matplotlib: pip install matplotlib")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ethernet Testing Analysis Results', fontsize=16)
        
        signal_np = signal.numpy()
        
        # Plot 1: Raw signal with anomaly markers
        axes[0, 0].plot(signal_np, 'b-', alpha=0.7, label='Signal')
        
        # Add signal level lines
        for i, level in enumerate(self.signal_levels):
            axes[0, 0].axhline(y=level, color='g', linestyle='--', alpha=0.5, 
                             label=f'Level {i}' if i == 0 else "")
        
        axes[0, 0].set_title('Signal with Reference Levels')
        axes[0, 0].set_ylabel('Amplitude (V)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Eye diagram (simplified)
        if 'eye_diagram' in analysis_results['analysis_results']:
            eye_metrics = analysis_results['analysis_results']['eye_diagram']
            
            # Create a simplified eye diagram representation
            sample_rate = 25e9  # Assume 25 GSa/s
            symbol_rate = 10e9  # Assume 10 GBaud
            samples_per_symbol = int(sample_rate / symbol_rate)
            
            if len(signal_np) >= samples_per_symbol * 10:
                eye_data = signal_np[:samples_per_symbol * 10].reshape(10, samples_per_symbol)
                
                for trace in eye_data:
                    axes[0, 1].plot(trace, 'b-', alpha=0.3)
                
                axes[0, 1].set_title(f'Eye Diagram (Height: {eye_metrics.eye_height:.3f}V, Width: {eye_metrics.eye_width:.3f}UI)')
                axes[0, 1].set_ylabel('Amplitude (V)')
                axes[0, 1].set_xlabel('Sample')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: BER analysis (if available)
        if 'ber' in analysis_results['analysis_results']:
            ber_metrics = analysis_results['analysis_results']['ber']
            
            # Plot error distribution
            error_dist = ber_metrics.error_distribution
            axes[1, 0].bar(range(len(error_dist)), error_dist, alpha=0.7)
            axes[1, 0].set_title(f'Error Distribution (BER: {ber_metrics.bit_error_rate:.2e})')
            axes[1, 0].set_ylabel('Error Count')
            axes[1, 0].set_xlabel('Pattern Position')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'BER Analysis\\nNot Available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('BER Analysis')
        
        # Plot 4: ISI analysis
        if 'isi' in analysis_results['analysis_results']:
            isi_metrics = analysis_results['analysis_results']['isi']
            
            # Plot pattern dependency
            patterns = list(isi_metrics.pattern_dependency.keys())[:10]  # Show first 10
            values = [isi_metrics.pattern_dependency[p] for p in patterns]
            
            axes[1, 1].bar(range(len(patterns)), values, alpha=0.7)
            axes[1, 1].set_title(f'ISI Pattern Dependency (ISI Ratio: {isi_metrics.isi_ratio:.3f})')
            axes[1, 1].set_ylabel('Deviation (V)')
            axes[1, 1].set_xlabel('Pattern')
            axes[1, 1].set_xticks(range(len(patterns)))
            axes[1, 1].set_xticklabels(patterns, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'ISI Analysis\\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('ISI Analysis')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\\n=== ETHERNET TESTING ANALYSIS SUMMARY ===")
        print(f"Signal Type: {self.spec.signal_type.value}")
        print(f"Anomalies Detected: {'YES' if analysis_results['anomalies_detected'] else 'NO'}")
        
        if analysis_results['anomalies_detected']:
            print(f"\\n⚠️  ANOMALIES DETECTED:")
            for anomaly in analysis_results['anomaly_details']:
                print(f"  {anomaly['severity'].upper()}: {anomaly['description']}")

def create_sample_ethernet_data(filename: str = "sample_ethernet_data.csv",
                              signal_type: SignalType = SignalType.PAM4,
                              num_samples: int = 10000):
    """Create sample Ethernet testing data for demonstration"""
    print(f"Creating sample {signal_type.value} data: {filename}")
    
    # Generate time vector
    sample_rate = 25e9  # 25 GSa/s
    symbol_rate = 10e9  # 10 GBaud 
    dt = 1.0 / sample_rate
    times = np.arange(num_samples) * dt
    
    # Generate signal
    if signal_type == SignalType.PAM4:
        # PAM4 signal with 4 levels
        levels = [-0.75, -0.25, 0.25, 0.75]
        symbols = np.random.choice(4, size=num_samples//10)
        
        # Create signal with some ISI and noise
        signal = np.zeros(num_samples)
        samples_per_symbol = int(sample_rate / symbol_rate)
        
        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min(start_idx + samples_per_symbol, num_samples)
            if start_idx < num_samples:
                # Add some ISI from previous symbol
                base_level = levels[symbol]
                if i > 0:
                    prev_level = levels[symbols[i-1]]
                    isi_factor = 0.1 * (prev_level - base_level)
                else:
                    isi_factor = 0.0
                
                # Create symbol with rise/fall times
                symbol_samples = np.linspace(
                    base_level + isi_factor,
                    base_level,
                    end_idx - start_idx
                )
                signal[start_idx:end_idx] = symbol_samples
        
        # Add noise
        signal += np.random.normal(0, 0.02, size=num_samples)
        
        # Generate corresponding bits (2 bits per PAM4 symbol)
        tx_bits = []
        rx_bits = []
        for symbol in symbols:
            # Convert symbol to 2 bits
            bits = [(symbol >> 1) & 1, symbol & 1]
            tx_bits.extend(bits)
            rx_bits.extend(bits)
        
        # Inject some bit errors
        error_positions = np.random.choice(len(rx_bits), size=len(rx_bits)//10000, replace=False)
        for pos in error_positions:
            rx_bits[pos] = 1 - rx_bits[pos]  # Flip bit
    
    else:  # NRZ
        # NRZ signal
        symbols = np.random.choice(2, size=num_samples//10)
        signal = np.zeros(num_samples)
        samples_per_symbol = int(sample_rate / symbol_rate)
        
        for i, symbol in enumerate(symbols):
            start_idx = i * samples_per_symbol
            end_idx = min(start_idx + samples_per_symbol, num_samples)
            if start_idx < num_samples:
                level = 1.0 if symbol else 0.0
                signal[start_idx:end_idx] = level
        
        # Add noise
        signal += np.random.normal(0, 0.05, size=num_samples)
        
        tx_bits = symbols.tolist()
        rx_bits = symbols.tolist()
        
        # Inject some bit errors
        error_positions = np.random.choice(len(rx_bits), size=len(rx_bits)//10000, replace=False)
        for pos in error_positions:
            rx_bits[pos] = 1 - rx_bits[pos]
    
    # Generate CTLE data
    ctle_gains = 5.0 + np.random.normal(0, 0.5, size=num_samples//100)
    input_signal = signal * 0.7  # Attenuated input
    
    # Write to CSV
    with open(filename, 'w') as f:
        f.write("timestamp,signal,input_signal,ctle_gain,tx_bit,rx_bit\\n")
        
        bit_idx = 0
        ctle_idx = 0
        
        for i, (t, sig, input_sig) in enumerate(zip(times, signal, input_signal)):
            # Get corresponding data
            if i % 100 == 0 and ctle_idx < len(ctle_gains):
                current_ctle_gain = ctle_gains[ctle_idx]
                ctle_idx += 1
            
            if bit_idx < len(tx_bits):
                tx_bit = tx_bits[bit_idx]
                rx_bit = rx_bits[bit_idx]
            else:
                tx_bit = 0
                rx_bit = 0
            
            if i % (samples_per_symbol // 2) == 0:  # Advance bit every half symbol
                bit_idx = min(bit_idx + 1, len(tx_bits) - 1)
            
            f.write(f"{t:.9e},{sig:.6f},{input_sig:.6f},{current_ctle_gain:.3f},{tx_bit},{rx_bit}\\n")
    
    print(f"Sample {signal_type.value} data created with {num_samples} samples")
    return filename

def main():
    """Demonstration of the Ethernet testing analyzer"""
    
    # Define Ethernet test specifications
    spec = EthernetTestSpec(
        signal_type=SignalType.PAM4,
        pam4_levels=[-0.75, -0.25, 0.25, 0.75],
        pam4_tolerance=0.05,
        eye_height_min=0.5,
        eye_width_min=0.7,
        ber_threshold=1e-12,
        ctle_gain_min=0.0,
        ctle_gain_max=15.0,
        isi_threshold=0.1,
        timing_margin_min=0.4
    )
    
    # Create analyzer
    analyzer = EthernetTestAnalyzer(spec)
    
    # Generate sample data
    print("Generating sample Ethernet testing data...")
    sample_file = create_sample_ethernet_data("sample_ethernet_pam4.csv", SignalType.PAM4, 5000)
    
    # Load and analyze data
    print("\\nAnalyzing Ethernet testing data...")
    
    if HAS_PLOTTING:
        df = pd.read_csv(sample_file)
        signal_data = df['signal'].values.astype(np.float32)
        input_signal_data = df['input_signal'].values.astype(np.float32)
        ctle_gains = df['ctle_gain'].values
        tx_bits = df['tx_bit'].values.astype(int).tolist()
        rx_bits = df['rx_bit'].values.astype(int).tolist()
        
        # Convert to tensors
        signal_tensor = Tensor(signal_data)
        input_tensor = Tensor(input_signal_data)
        
        # Analyze
        results = analyzer.detect_ethernet_anomalies(
            signal=signal_tensor,
            sample_rate=25e9,
            symbol_rate=10e9,
            received_bits=rx_bits[:1000],  # Analyze first 1000 bits
            transmitted_bits=tx_bits[:1000],
            ctle_gain=np.mean(ctle_gains),
            input_signal=input_tensor
        )
        
        # Plot results
        analyzer.plot_analysis_results(signal_tensor, results)
        
        print(f"\\nAnalysis completed. Results:")
        print(f"- Eye Height: {results['analysis_results']['eye_diagram'].eye_height:.3f}V")
        print(f"- Eye Width: {results['analysis_results']['eye_diagram'].eye_width:.3f}UI")
        print(f"- ISI Ratio: {results['analysis_results']['isi'].isi_ratio:.3f}")
        print(f"- BER: {results['analysis_results']['ber'].bit_error_rate:.2e}")
        
    else:
        print("Install matplotlib and pandas for full functionality:")
        print("pip install matplotlib pandas")

if __name__ == "__main__":
    main()